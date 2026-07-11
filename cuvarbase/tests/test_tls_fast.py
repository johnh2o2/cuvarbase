"""GPU tests for the fast (batched, phase-binned) TLS path.

The fast path is the default for tls_search_gpu/tls_transit; these
tests cover what the legacy-oriented suites do not: batch consistency,
the coarse/refined statistics separation, adaptive binning, chunking,
and the removal of the legacy ndata cap.
"""
import numpy as np
import pytest

try:
    import pycuda.driver  # noqa: F401
    PYCUDA_AVAILABLE = True
except Exception:
    PYCUDA_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYCUDA_AVAILABLE,
                                reason="pycuda unavailable")


def make_transit_lc(period, q, depth, ndata=1500, baseline=27.0,
                    noise=2e-3, seed=42, t0_frac=0.3):
    rng = np.random.RandomState(seed)
    t = np.sort(rng.uniform(0, baseline, ndata))
    y = 1.0 + rng.randn(ndata) * noise
    t0 = t0_frac * period
    rel = np.abs(((t - t0 + 0.5 * period) % period) - 0.5 * period)
    y[rel < 0.5 * q * period] -= depth
    dy = np.full(ndata, noise)
    return t, y, dy


def shared_grid(baseline=27.0, period_min=1.0, period_max=12.0):
    from cuvarbase import tls_grids
    t_ref = np.linspace(0, baseline, 500)
    return tls_grids.period_grid_ofir(
        t_ref, R_star=1.0, M_star=1.0, oversampling_factor=3,
        period_min=period_min, period_max=period_max)


class TestBatchConsistency:
    def test_batch_matches_single(self):
        from cuvarbase import tls
        periods = shared_grid()
        lcs = [make_transit_lc(3.3, 0.03, 0.012, seed=1),
               make_transit_lc(7.7, 0.02, 0.012, ndata=2500, seed=2)]
        batch = tls.tls_search_batch(lcs, periods=periods)
        singles = [tls.tls_search_batch([lc], periods=periods)[0]
                   for lc in lcs]
        for b, s in zip(batch, singles):
            # atomics make near-tied neighbors non-deterministic;
            # a few grid steps of slack
            assert abs(b['period'] - s['period']) / s['period'] < 5e-3
            assert abs(b['SDE'] - s['SDE']) < 1.0

    def test_recovers_injected_periods(self):
        from cuvarbase import tls
        periods = shared_grid()
        p_injs = [3.3, 7.7]
        lcs = [make_transit_lc(p, 0.03, 0.012, seed=10 + i)
               for i, p in enumerate(p_injs)]
        results = tls.tls_search_batch(lcs, periods=periods)
        for r, p in zip(results, p_injs):
            assert abs(r['period'] - p) / p < 0.01
            assert r['SDE'] > 5

    def test_noise_lc_scores_below_signal(self):
        from cuvarbase import tls
        periods = shared_grid()
        rng = np.random.RandomState(3)
        t = np.sort(rng.uniform(0, 27.0, 1500))
        noise_lc = (t, 1.0 + 2e-3 * rng.randn(1500),
                    np.full(1500, 2e-3))
        sig_lc = make_transit_lc(3.3, 0.03, 0.012, seed=4)
        r_noise, r_sig = tls.tls_search_batch([noise_lc, sig_lc],
                                              periods=periods)
        assert r_noise['SDE'] < r_sig['SDE']


class TestStatisticsSeparation:
    def test_spectrum_is_coarse_and_uniform(self):
        """Refinement must not touch the per-period spectrum: SDE
        computed with refine on and off must agree."""
        from cuvarbase import tls
        periods = shared_grid()
        lc = make_transit_lc(3.3, 0.03, 0.012, seed=5)
        r_ref = tls.tls_search_batch([lc], periods=periods,
                                     refine_top_k=200,
                                     return_arrays=True)[0]
        r_none = tls.tls_search_batch([lc], periods=periods,
                                      refine_top_k=0,
                                      return_arrays=True)[0]
        ok = (np.isfinite(r_ref['chi2']) & np.isfinite(r_none['chi2']))
        np.testing.assert_allclose(r_ref['chi2'][ok],
                                   r_none['chi2'][ok], rtol=1e-2)
        assert abs(r_ref['SDE'] - r_none['SDE']) < 0.5

    def test_refined_chi2_min_not_above_coarse(self):
        """The exact refinement searches a finer local grid around the
        coarse optimum, so the reported chi2_min should be at or below
        the coarse spectrum minimum (up to float noise)."""
        from cuvarbase import tls
        periods = shared_grid()
        lc = make_transit_lc(3.3, 0.03, 0.012, seed=6)
        r = tls.tls_search_batch([lc], periods=periods,
                                 return_arrays=True)[0]
        coarse_min = np.nanmin(r['chi2'])
        assert r['chi2_min'] <= coarse_min * (1 + 1e-3)


class TestScalability:
    def test_ndata_beyond_legacy_cap(self):
        from cuvarbase import tls
        periods = shared_grid()
        lc = make_transit_lc(4.56, 0.025, 0.008, ndata=20000, seed=7)
        r = tls.tls_search_batch([lc], periods=periods)[0]
        assert abs(r['period'] - 4.56) / 4.56 < 0.01

    def test_bjd_scale_times(self):
        from cuvarbase import tls
        periods = shared_grid()
        t, y, dy = make_transit_lc(4.56, 0.025, 0.008, ndata=5000,
                                   seed=8)
        r = tls.tls_search_batch([(t + 2457000.0, y, dy)],
                                 periods=periods)[0]
        assert abs(r['period'] - 4.56) / 4.56 < 0.01
        # T0 reported near the (shifted) epoch
        assert r['T0'] >= 2457000.0
        assert r['T0'] <= 2457000.0 + 27.0 + r['period']

    def test_chunking_many_small_lcs(self):
        """Force multiple chunks via the LC-count ceiling and check
        every LC still gets a result."""
        from cuvarbase import tls
        periods = shared_grid()
        old = tls._TLS_FAST_MAX_OUT_FLOATS
        tls._TLS_FAST_MAX_OUT_FLOATS = 3 * len(periods)  # 3 LCs/chunk
        try:
            lcs = [make_transit_lc(3.3, 0.03, 0.012, ndata=400,
                                   seed=20 + i) for i in range(8)]
            results = tls.tls_search_batch(lcs, periods=periods)
        finally:
            tls._TLS_FAST_MAX_OUT_FLOATS = old
        assert len(results) == 8
        for r in results:
            assert 'error' not in r
            assert abs(r['period'] - 3.3) / 3.3 < 0.02

    def test_mixed_lengths_offsets(self):
        from cuvarbase import tls
        periods = shared_grid()
        lcs = [make_transit_lc(3.3, 0.03, 0.015, ndata=n, seed=30 + i)
               for i, n in enumerate((300, 4000, 1100))]
        results = tls.tls_search_batch(lcs, periods=periods)
        for r in results:
            assert abs(r['period'] - 3.3) / 3.3 < 0.02


class TestValidation:
    def test_empty_batch(self):
        from cuvarbase import tls
        assert tls.tls_search_batch([]) == []

    def test_mismatched_qmin_qmax(self):
        from cuvarbase import tls
        lc = make_transit_lc(3.3, 0.03, 0.012, ndata=300)
        with pytest.raises(ValueError):
            tls.tls_search_batch([lc], periods=np.linspace(2, 5, 50),
                                 qmin=np.full(10, 0.01),
                                 qmax=np.full(10, 0.05))

    def test_qmin_only_rejected(self):
        from cuvarbase import tls
        lc = make_transit_lc(3.3, 0.03, 0.012, ndata=300)
        with pytest.raises(ValueError, match="both qmin and qmax"):
            tls.tls_search_batch([lc], periods=np.linspace(2, 5, 50),
                                 qmin=np.full(50, 0.01))

    def test_bad_n_durations(self):
        from cuvarbase import tls
        lc = make_transit_lc(3.3, 0.03, 0.012, ndata=300)
        with pytest.raises(ValueError, match="n_durations"):
            tls.tls_search_batch([lc], n_durations=100)


class TestBanding:
    """The period grid is banded by required bin count (NBINS variants
    + period_map scatter); banded results must match a single-band
    (fixed nbins) run over the identical trial grid."""

    def test_banded_matches_single_band(self):
        from cuvarbase import tls
        periods = np.asarray(shared_grid(), dtype=np.float64)
        n = len(periods)
        # interleaved qmin values straddle a power-of-two boundary in
        # need = t0_oversample/qmin (3/0.02 -> 256 bins, 3/0.008 -> 512
        # bins), so the banded run launches two NBINS variants with a
        # non-contiguous period_map scatter; the duration and t0 trial
        # grids depend only on qmin/qmax and are identical in both runs,
        # and both duration windows bracket the injected q = 0.03
        qmin = np.where(np.arange(n) % 2 == 0, 0.02, 0.008)
        qmax = np.full(n, 0.09)
        lc = make_transit_lc(3.3, 0.03, 0.012, seed=11)

        r_banded = tls.tls_search_batch([lc], periods=periods,
                                        qmin=qmin, qmax=qmax,
                                        return_arrays=True)[0]
        r_fixed = tls.tls_search_batch([lc], periods=periods,
                                       qmin=qmin, qmax=qmax,
                                       nbins=512,
                                       return_arrays=True)[0]

        assert abs(r_banded['period'] - 3.3) / 3.3 < 0.01
        assert abs(r_banded['period'] - r_fixed['period']) / 3.3 < 5e-3
        ok = (np.isfinite(r_banded['chi2'])
              & np.isfinite(r_fixed['chi2']))
        assert ok.sum() > 0.9 * n
        # the odd-index periods run at 512 bins in BOTH configurations;
        # the even-index ones differ only in bin resolution (256 vs
        # 512), so the spectra must agree closely everywhere
        corr = np.corrcoef(r_banded['chi2'][ok],
                           r_fixed['chi2'][ok])[0, 1]
        assert corr > 0.99


class TestRefinementFallback:
    """PR #68 review regression: the coarse-parameter fallback in _finish_lc
    must not depend on return_arrays being set."""

    def test_all_refinements_fail_falls_back_no_crash(self, monkeypatch):
        # Force every top-K exact refinement to return the failure sentinel
        # (rscore <= 0) while the coarse phase-binned scan still finds valid
        # periods. With the default batch args (refine_top_k > 0 and
        # return_arrays=False) the else-branch in _finish_lc must fall back to
        # the coarse best-fit t0/duration/depth — it must NOT raise
        # UnboundLocalError because those coarse arrays were fetched only under
        # `return_arrays or not K`.
        from cuvarbase import tls
        orig = tls._get_cached_fast_kernels

        def patched(*a, **k):
            kern = dict(orig(*a, **k))          # copy cached {'search','refine'}

            def fail_refine(*args, **kwargs):   # rscore_g is positional arg 16
                args[16].fill(np.float32(-1.0))

            kern['refine'] = fail_refine
            return kern

        monkeypatch.setattr(tls, '_get_cached_fast_kernels', patched)
        periods = shared_grid()
        lcs = [make_transit_lc(3.3, 0.03, 0.012, seed=1),
               make_transit_lc(7.7, 0.02, 0.012, seed=2)]
        res = tls.tls_search_batch(lcs, periods=periods)   # defaults
        assert len(res) == 2
        for r in res:
            assert 'error' not in r
            assert np.isfinite(r['period']) and r['period'] > 0
            assert np.isfinite(r['duration']) and np.isfinite(r['depth'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
