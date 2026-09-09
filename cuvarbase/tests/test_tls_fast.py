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
        computed with refine on and off must agree.

        Tolerances: the coarse spectrum is accumulated with float32
        shared-memory atomics (one block per (lightcurve, period); see
        kernels/tls_fast.cu), whose summation order is not deterministic
        between launches, so two runs of the SAME configuration are not
        bit-identical. rtol=1e-2 on chi2 and |dSDE| < 0.5 are the
        run-to-run envelope observed on the Jul-2026 gate hardware (A5000)
        with a wide margin -- NOT a statement that refinement may perturb
        the spectrum by that much. Tightening to ~10x the measured
        run-to-run floor is a device task (release finding 84).
        """
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
        # T0 is the first mid-transit at or after the first observation
        tmin = t.min() + 2457000.0
        assert tmin <= r['T0'] < tmin + r['period']
        assert 0.0 <= r['t0_phase'] < 1.0

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


class TestEmptyBinTraversal:
    """Skipping empty phase bins must preserve the numerical search.

    Compare the optimized kernel with its dense reference traversal on
    identical grids, including bins straddling phase zero, capped narrow
    durations, long empty phase intervals, and the conservative sparse
    dispatch threshold and small-block fallback.
    These are numerical regressions, not evidence about astrophysical
    recovery or false-positive calibration.
    """

    @staticmethod
    def _search_pair(tls, monkeypatch, lightcurves, **kwargs):
        reader = tls._module_reader
        compiled = {}
        mode = 0

        def get_kernels(bs, nb, oversample, refine_nd=3):
            key = (mode, bs, nb, oversample, refine_nd)
            if key not in compiled:
                def read_variant(*args, **kw):
                    return ('#define TLS_SKIP_EMPTY_BINS %d\n' % mode
                            + reader(*args, **kw))
                with monkeypatch.context() as patch:
                    patch.setattr(tls, '_module_reader', read_variant)
                    compiled[key] = tls.compile_tls_fast(
                        bs, nb, oversample, refine_nd)
            return compiled[key]

        monkeypatch.setattr(tls, '_get_cached_fast_kernels', get_kernels)
        outputs = []
        for mode in (0, 1):
            outputs.append(tls.tls_search_batch(lightcurves, **kwargs))
        return outputs

    @pytest.mark.parametrize('nbins,ndata,q,center,clustered,block_size', [
        (8192, 700, .03, .9999, False, 512),
        (8192, 256, .0001, .99999, False, 512),
        (8192, 300, .03, .01, True, 512),
        (1024, 3000, .03, .3, False, 256),
        (8192, 2200, .03, .3, False, 512),
        (8192, 300, .03, .99, False, 32),
    ])
    def test_same_scores_and_signal_candidate(
            self, monkeypatch, nbins, ndata, q, center, clustered,
            block_size):
        from cuvarbase import tls

        tls.ensure_context()
        if tls._tls_fast_shared_size(block_size, nbins) > tls._device_max_shared():
            pytest.skip('device cannot fit the requested fine-bin kernel')
        rng = np.random.RandomState(841)
        cycles = rng.randint(0, 2744, ndata)
        phase = rng.uniform(0, 1, ndata)
        if clustered:
            phase = np.mod(center + rng.uniform(-.02, .02, ndata), 1.)
        # Ensure that even the tiny-duty-cycle case contains measured
        # transits. The aim is traversal parity, not random observability.
        phase[:24] = np.mod(center + np.linspace(-.3 * q, .3 * q, 24), 1.)
        t = cycles + phase
        rel = (phase - center + .5) % 1. - .5
        shape = np.maximum(0., 1. - (2. * rel / q) ** 2)
        dy = rng.uniform(.001, .003, ndata)
        y = 1. - .03 * shape + rng.randn(ndata) * dy
        order = np.argsort(t)
        lc = (t[order] + 2457000., y[order], dy[order])
        periods = np.array([.701, .913, 1., 1.127, 1.701])
        qmin = np.full(len(periods), q)
        qmax = np.full(len(periods), 1.5 * q)
        old_list, new_list = self._search_pair(
            tls, monkeypatch, [lc], periods=periods, qmin=qmin, qmax=qmax,
            n_durations=4, t0_oversample=8., nbins=nbins,
            block_size=block_size, refine_top_k=0, return_arrays=True)
        old, new = old_list[0], new_list[0]
        np.testing.assert_array_equal(old['valid_periods'], new['valid_periods'])
        chi2_0 = tls._preprocess_batch([lc])[6][0]
        old_score = chi2_0 - old['chi2']
        new_score = chi2_0 - new['chi2']
        # Atomic histogram sums vary in order across launches. The
        # sparse scan retains the dense scan's coordinate arithmetic.
        np.testing.assert_allclose(new_score, old_score,
                                   rtol=2e-5, atol=1e-5, equal_nan=True)
        assert old['period'] == new['period']

    def test_sparse_template_tail(self, monkeypatch):
        """Tiny coordinate shifts can amplify integral-subtraction error.

        Most phases are unobserved. A single downward fluctuation can
        sit in a transit's faint tail, where S2(right)-S2(left) is tiny.
        Multiplying a skip distance by dc instead of repeating the
        dense coordinate additions perturbs that subtraction and can
        change the winning score. The synthetic fixture is deliberately
        small and needs no survey archive.
        """
        from cuvarbase import tls

        tls.ensure_context()
        if tls._tls_fast_shared_size(512, 8192) > tls._device_max_shared():
            pytest.skip('device cannot fit the requested fine-bin kernel')
        t = np.r_[np.linspace(0., .5, 128), 231.5752637386322]
        y = np.r_[np.full(128, 1.001), .9982297870702772]
        dy = np.r_[np.full(128, .001), .0006100752167838939]
        lc = (t, y, dy)
        old_list, new_list = self._search_pair(
            tls, monkeypatch, [lc], periods=np.array([.9998087951893398]),
            qmin=np.array([.038196352656710154]),
            qmax=np.array([.15278541062684062]),
            n_durations=32, t0_oversample=16., nbins=8192,
            block_size=512, refine_top_k=0, return_arrays=True)
        old, new = old_list[0], new_list[0]
        chi2_0 = tls._preprocess_batch([lc])[6][0]
        old_score, new_score = chi2_0 - old['chi2'], chi2_0 - new['chi2']
        assert old['valid_periods'].all() and new['valid_periods'].all()
        assert np.isfinite(old_score).all() and np.isfinite(new_score).all()
        assert (old_score > 0).all() and (new_score > 0).all()
        np.testing.assert_allclose(new_score, old_score, rtol=2e-5, atol=1e-5)


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


# ---------------------------------------------------------------------
# 1.0 correctness fixes (Sep 2026 audit), GPU regressions on all paths
# ---------------------------------------------------------------------

import warnings as _w


def _call_expect_warning(fn, match):
    """Run fn() and assert a UserWarning containing `match` was emitted.
    (pytest.warns around a GPU call would report DID NOT WARN instead of
    letting the conftest's GPUStubError skip through on CPU-only hosts.)"""
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        r = fn()
    msgs = [str(x.message) for x in rec if issubclass(x.category, UserWarning)]
    assert any(match in m for m in msgs), msgs
    return r


def _three_paths(t, y, dy, periods, **kw):
    """(fast, legacy, batch) results for one light curve."""
    from cuvarbase import tls
    fast = tls.tls_search_gpu(t, y, dy, periods=periods, **kw)
    legacy = tls.tls_search_gpu(t, y, dy, periods=periods, use_fast=False,
                                **kw)
    batch = tls.tls_search_batch([(t, y, dy)], periods=periods,
                                 return_arrays=True)[0]
    return {'fast': fast, 'legacy': legacy, 'batch': batch}


def _box_lc(period, q, depth, t0, t_start, baseline=40.0, ndata=2000,
            noise=2e-3, seed=3):
    rng = np.random.RandomState(seed)
    t = t_start + np.sort(rng.uniform(0, baseline, ndata))
    y = 1.0 + rng.randn(ndata) * noise
    rel = np.abs(((t - t0 + 0.5 * period) % period) - 0.5 * period)
    in_tr = rel < 0.5 * q * period
    y[in_tr] -= depth
    return t, y, np.full(ndata, noise), in_tr


class TestT0Semantics:
    """Defect 11 (tls-T0, audit ids 11/145): 'T0' was a fold phase on
    the fast path (relative to floor(min t)), a phase relative to t = 0
    on the legacy path, and an absolute time that could precede the
    first observation on the batch path. It is now the absolute time of
    the first mid-transit at or after min(t) on every path, with the
    phase under 't0_phase'."""

    @pytest.mark.parametrize("t_start,frac", [(100.3, 0.37), (100.9, 0.8),
                                              (2457000.3, 0.37)])
    def test_T0_first_transit_after_min_t_on_all_paths(self, t_start, frac):
        P, q, depth = 3.0, 0.03, 0.01
        t0_true = t_start + frac * P
        t, y, dy, in_tr = _box_lc(P, q, depth, t0_true, t_start)
        periods = np.linspace(2.9, 3.1, 300)
        res = _three_paths(t, y, dy, periods)
        tmin = t.min()
        dur_true = q * P
        for name, r in res.items():
            assert abs(r['period'] - P) / P < 0.01, name
            # absolute time in [min(t), min(t) + P)
            assert tmin <= r['T0'] < tmin + r['period'], (name, r['T0'])
            assert 0.0 <= r['t0_phase'] < 1.0, name
            # T0 and t0_phase describe the same epoch (relative to
            # floor(min t)), up to whole periods
            t_from_phase = np.floor(tmin) + r['t0_phase'] * r['period']
            frac_diff = ((r['T0'] - t_from_phase) / r['period']) % 1.0
            assert min(frac_diff, 1.0 - frac_diff) < 1e-4, name
            # folding the data at T0 puts the injected transit at phase 0
            # (legacy coarse t0 stride is q/3 -> up to 0.17 durations off)
            nearest = np.min(np.abs(r['T0'] - (t0_true + P * np.arange(-2, 20))))
            assert nearest < 0.4 * dur_true, (name, nearest)
            ph = ((t - r['T0']) / r['period'] + 0.5) % 1.0 - 0.5
            sel = np.abs(ph) < 0.4 * q
            assert sel.sum() > 20, name
            assert y[sel].mean() < 1.0 - 0.7 * depth, name
            assert 'FAP' not in r, name

    def test_paths_agree_on_T0(self):
        P, q, depth = 3.0, 0.03, 0.01
        t, y, dy, _ = _box_lc(P, q, depth, 100.9 + 0.8 * P, 100.9)
        res = _three_paths(t, y, dy, np.linspace(2.9, 3.1, 300))
        assert res['fast']['T0'] == pytest.approx(res['batch']['T0'], abs=1e-6)
        assert res['fast']['T0'] == pytest.approx(res['legacy']['T0'],
                                                  abs=0.4 * q * P)


class TestUnsortedPeriodGrid:
    """id 82: a descending (what transitleastsquares returns) or shuffled
    user grid gave a negative period_uncertainty and a changed SDE."""

    def test_descending_and_shuffled_match_ascending(self):
        from cuvarbase import tls
        periods = np.asarray(shared_grid(), dtype=np.float64)
        lc = make_transit_lc(3.3, 0.03, 0.012, seed=1)
        ref = tls.tls_search_gpu(*lc, periods=periods)
        assert ref['period_uncertainty'] > 0
        rng = np.random.RandomState(0)
        for label, grid in (('descending', periods[::-1].copy()),
                            ('shuffled', periods[rng.permutation(len(periods))])):
            for path in ('fast', 'legacy'):
                r = tls.tls_search_gpu(*lc, periods=grid,
                                       use_fast=(path == 'fast'))
                assert r['period'] == pytest.approx(ref['period'], rel=5e-3), (label, path)
                assert r['period_uncertainty'] > 0, (label, path)
                # per-period arrays come back in the caller's order
                np.testing.assert_array_equal(r['periods'],
                                              grid.astype(np.float32))
                back = np.argsort(grid)
                if path == 'fast':
                    assert abs(r['SDE'] - ref['SDE']) < 0.05, label
                    ok = np.isfinite(r['chi2'][back]) & np.isfinite(ref['chi2'])
                    np.testing.assert_allclose(r['chi2'][back][ok],
                                               ref['chi2'][ok], rtol=1e-4)
                    np.testing.assert_array_equal(r['valid_periods'][back],
                                                  ref['valid_periods'])
            rb = tls.tls_search_batch([lc], periods=grid,
                                      return_arrays=True)[0]
            assert rb['period_uncertainty'] > 0
            np.testing.assert_array_equal(rb['periods'], grid.astype(np.float32))
            assert abs(rb['SDE'] - ref['SDE']) < 0.05


class TestFlatLightCurve:
    """id 89: a flat/noiseless light curve fails every trial period; the
    reference returns SDE = 0 with a warning, cuvarbase used to raise."""

    def test_sde_zero_on_all_paths(self):
        from cuvarbase import tls
        t = np.linspace(0, 30, 1000)
        y = np.ones(1000)
        dy = np.full(1000, 1e-3)
        periods = np.linspace(2, 5, 200)
        calls = {
            'fast': lambda: tls.tls_search_gpu(t, y, dy, periods=periods),
            'legacy': lambda: tls.tls_search_gpu(t, y, dy, periods=periods,
                                                 use_fast=False),
            'batch': lambda: tls.tls_search_batch([(t, y, dy)],
                                                  periods=periods)[0],
        }
        for name, fn in calls.items():
            r = _call_expect_warning(fn, "no valid solution")
            assert r['SDE'] == 0.0 and r['SDE_raw'] == 0.0, name
            assert np.isnan(r['period']) and np.isnan(r['T0']), name
            assert r['n_failed_periods'] == 200, name
            assert 'error' in r and 'FAP' not in r, name
        # a flat light curve in a batch does not poison its neighbours
        good = make_transit_lc(3.3, 0.03, 0.012, seed=1)
        rs = _call_expect_warning(
            lambda: tls.tls_search_batch([(t, y, dy), good],
                                         periods=shared_grid()),
            "no valid solution")
        assert rs[0]['SDE'] == 0.0
        assert abs(rs[1]['period'] - 3.3) / 3.3 < 0.01 and rs[1]['SDE'] > 5


class TestFAPKey:
    """Defect 10 (tls-fap): no result carries a FAP unless a null
    bootstrap was requested; the bootstrap is uniform under the null."""

    def test_no_fap_without_calibration(self):
        lc = make_transit_lc(3.3, 0.03, 0.012, seed=1)
        for r in _three_paths(*lc, periods=shared_grid()).values():
            assert 'FAP' not in r and 'SDE_null' not in r

    def test_null_bootstrap(self):
        from cuvarbase import tls
        periods = shared_grid()
        rng = np.random.RandomState(3)
        t = np.sort(rng.uniform(0, 27.0, 1500))
        noise_lc = (t, 1.0 + 2e-3 * rng.randn(1500), np.full(1500, 2e-3))
        sig_lc = make_transit_lc(3.3, 0.03, 0.012, seed=4)
        r_noise, r_sig = tls.tls_search_batch(
            [noise_lc, sig_lc], periods=periods, fap_null_draws=40,
            fap_seed=7)
        for r in (r_noise, r_sig):
            assert 0 < r['FAP'] <= 1.0
            assert r['SDE_null'].shape == (40,)
            assert np.all(np.isfinite(r['SDE_null']))
            # null SDEs sit in the expected range for this grid
            assert 3 < r['SDE_null'].mean() < 10
        # the signal beats every permutation: minimum resolvable FAP
        assert r_sig['FAP'] == pytest.approx(1.0 / 41.0)
        assert r_sig['SDE'] > r_sig['SDE_null'].max()
        # the noise light curve is not significant
        assert r_noise['FAP'] > 0.05
        # the observed SDE is unchanged by the bootstrap
        plain = tls.tls_search_batch([noise_lc, sig_lc], periods=periods)
        assert plain[1]['SDE'] == pytest.approx(r_sig['SDE'], abs=1e-3)
        # seeded -> reproducible null
        again = tls.tls_search_batch([noise_lc], periods=periods,
                                     fap_null_draws=40, fap_seed=7)[0]
        np.testing.assert_allclose(again['SDE_null'], r_noise['SDE_null'],
                                   atol=1e-2)

    def test_bad_draw_count(self):
        from cuvarbase import tls
        lc = make_transit_lc(3.3, 0.03, 0.012, ndata=300)
        with pytest.raises(ValueError, match="fap_null_draws"):
            tls.tls_search_batch([lc], periods=shared_grid(),
                                 fap_null_draws=-1)


class TestSNRDefinition:
    """id 85: SNR is sqrt(chi2_0 - chi2_min) with the float64
    constant-model chi2_0 and the refined chi2_min (was max(chi2) over
    the grid and the coarse chi2)."""

    def test_snr_is_delta_chi2_over_constant_model(self):
        t, y, dy = make_transit_lc(3.3, 0.03, 0.012, seed=6)
        chi2_0 = np.sum((1.0 - y) ** 2 / (dy ** 2 + 1e-10))
        res = _three_paths(t, y, dy, shared_grid())
        for name, r in res.items():
            assert r['SNR'] == pytest.approx(np.sqrt(chi2_0 - r['chi2_min']),
                                             rel=1e-5), name
            assert r['SNR'] > 10, name


class TestDurationWindowDefault:
    """Defect 2 on device: the default window of tls_search_gpu equals
    the explicit Keplerian window (identical trial grid), 'fixed' is an
    opt-in that warns, and the legacy path honours the same window."""

    def test_default_equals_explicit_keplerian(self):
        from cuvarbase import tls, tls_grids
        lc = make_transit_lc(3.3, 0.03, 0.012, seed=1)
        periods = np.asarray(shared_grid(), dtype=np.float64)
        q = tls_grids.q_transit(periods)
        r_def = tls.tls_search_gpu(*lc, periods=periods)
        r_exp = tls.tls_search_gpu(*lc, periods=periods, qmin=0.5 * q,
                                   qmax=2.0 * q)
        ok = np.isfinite(r_def['chi2']) & np.isfinite(r_exp['chi2'])
        np.testing.assert_allclose(r_def['chi2'][ok], r_exp['chi2'][ok],
                                   rtol=1e-5)
        assert r_def['period'] == pytest.approx(r_exp['period'], rel=1e-3)
        # tls_transit builds its own Ofir grid from the data's span and
        # the same Keplerian window; it must find the same transit
        r_tr = tls.tls_transit(*lc, period_min=1.0, period_max=12.0)
        assert r_tr['period'] == pytest.approx(3.3, rel=0.01)
        assert r_tr['depth'] == pytest.approx(r_def['depth'], rel=0.1)

    def test_fixed_window_optin_warns_and_default_does_not(self):
        from cuvarbase import tls
        rng = np.random.RandomState(9)
        t = np.sort(rng.uniform(0, 700.0, 2000))
        y = 1.0 + 1e-3 * rng.randn(2000)
        dy = np.full(2000, 1e-3)
        periods = np.linspace(100.0, 300.0, 50)
        with _w.catch_warnings():
            _w.simplefilter("error")
            tls.tls_search_gpu(t, y, dy, periods=periods)
            tls.tls_search_gpu(t, y, dy, periods=periods, use_fast=False)
        for path in ('fast', 'legacy'):
            r = _call_expect_warning(
                lambda: tls.tls_search_gpu(t, y, dy, periods=periods,
                                           duration_window='fixed',
                                           use_fast=(path == 'fast')),
                "excludes the Keplerian")
            assert np.isfinite(r['SDE'])

    def test_legacy_path_uses_keplerian_kernel(self, monkeypatch):
        from cuvarbase import tls
        seen = []
        orig = tls._get_cached_kernels

        def spy(*a, **k):
            kern = dict(orig(*a, **k))
            real = kern['keplerian']

            def kep(*args, **kwargs):
                seen.append('keplerian')
                return real(*args, **kwargs)

            def std(*args, **kwargs):
                seen.append('standard')
                return kern['standard'](*args, **kwargs)

            kern['keplerian'] = kep
            kern['standard'] = std
            return kern

        monkeypatch.setattr(tls, '_get_cached_kernels', spy)
        lc = make_transit_lc(3.3, 0.03, 0.012, ndata=800, seed=2)
        r = tls.tls_search_gpu(*lc, periods=shared_grid(), use_fast=False)
        assert seen == ['keplerian']
        assert abs(r['period'] - 3.3) / 3.3 < 0.02


class TestBatchStatisticsAreSequential:
    """Phase 2 TLS-2 (audit section 5, id 53): the per-light-curve
    statistics ran on a ThreadPoolExecutor on the assumption that
    scipy released the GIL in the running-median detrend. It does not:
    measured on an A40 (shared), 64 tess-ffi light curves took 166 ms
    with the pool and 77 ms without it, and the statistics alone cost
    21.7 ms sequentially versus 40.3 ms on 8 threads. Bit-neutral --
    only the executor changed -- and the light-curve order (and hence
    the order of any per-light-curve warning) is now deterministic."""

    def test_statistics_run_on_the_calling_thread_in_order(self, monkeypatch):
        import threading
        from cuvarbase import tls, tls_stats
        seen = []
        real = tls_stats.compute_all_statistics

        def spy(*a, **k):
            seen.append(threading.current_thread().name)
            return real(*a, **k)

        monkeypatch.setattr(tls_stats, 'compute_all_statistics', spy)
        periods = shared_grid()
        lcs = [make_transit_lc(2.5 + 0.7 * i, 0.03, 0.012, ndata=600,
                               seed=30 + i) for i in range(6)]
        results = tls.tls_search_batch(lcs, periods=periods)
        assert len(seen) == len(lcs)
        assert set(seen) == {threading.current_thread().name}
        assert all(r is not None for r in results)

    def test_results_are_returned_in_lightcurve_order(self):
        from cuvarbase import tls
        periods = shared_grid()
        p_injs = [2.6, 4.1, 6.3, 9.5]
        lcs = [make_transit_lc(p, 0.03, 0.015, ndata=900, seed=40 + i)
               for i, p in enumerate(p_injs)]
        results = tls.tls_search_batch(lcs, periods=periods)
        for r, p in zip(results, p_injs):
            assert abs(r['period'] - p) / p < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestFastLegacyParity:
    """Fast (phase-binned scan + exact top-K refinement) versus legacy
    (per-point template) kernel on ONE explicit trial grid -- the parity
    check ported from ``scripts/tls_fast_smoke.py`` (section 2). The
    smoke script's other sections already have suite equivalents:
    batch-vs-single agreement (``TestBatchConsistency``), ndata beyond
    the legacy 3,500-point cap and BJD-scale times
    (``TestScalability``), and the auto-grid ``tls_transit`` recovery
    (``TestDurationWindowDefault``, ``TestTlsTransitSmoke``)."""

    P_INJ, DEPTH = 5.123, 0.01

    def _grid_and_data(self):
        from cuvarbase import tls_grids
        t, y, dy = make_transit_lc(self.P_INJ,
                                   0.0763 * self.P_INJ ** (-2.0 / 3.0),
                                   self.DEPTH, ndata=1200, seed=42)
        periods = tls_grids.period_grid_ofir(
            t, R_star=1.0, M_star=1.0, oversampling_factor=3,
            period_min=1.0, period_max=12.0).astype(np.float64)
        _, _, qv = tls_grids.duration_grid_keplerian(
            periods, R_star=1.0, M_star=1.0, R_planet=1.0,
            qmin_fac=0.5, qmax_fac=2.0, n_durations=15)
        return (t, y, dy), periods, 0.5 * qv, 2.0 * qv

    def test_fast_matches_legacy_on_an_explicit_grid(self):
        from cuvarbase import tls
        lc, periods, qmin, qmax = self._grid_and_data()
        kw = dict(periods=periods, qmin=qmin, qmax=qmax, n_durations=15)
        r_old = tls.tls_search_gpu(*lc, use_fast=False, **kw)
        r_new = tls.tls_search_gpu(*lc, use_fast=True, **kw)

        c_old, c_new = r_old['chi2'], r_new['chi2']
        both = np.isfinite(c_old) & np.isfinite(c_new)
        assert both.sum() > 0.9 * len(periods)
        corr = np.corrcoef(c_old[both], c_new[both])[0, 1]
        assert corr > 0.99, corr
        # same chi2 scale (the binned scan is not a different statistic)
        med_old, med_new = np.median(c_old[both]), np.median(c_new[both])
        assert abs(med_new / med_old - 1.0) < 0.05, (med_old, med_new)
        # same best period, which is the injected one
        assert abs(r_new['period'] - r_old['period']) / r_old['period'] < 0.01
        assert abs(r_new['period'] - self.P_INJ) / self.P_INJ < 0.01
        assert r_new['SDE'] > 0.8 * r_old['SDE'], (r_old['SDE'], r_new['SDE'])
        assert abs(r_new['depth'] - self.DEPTH) / self.DEPTH < 0.5


class TestTlsTransitSmoke:
    """``tls.tls_transit`` end to end on an injected transit (release
    finding 71/149: the Keplerian wrapper had no test of its own
    result). Period within two steps of the grid it builds itself,
    ``T0`` the first mid-transit at or after ``min(t)``, ``t0_phase`` a
    fold phase in [0, 1)."""

    def test_recovers_injected_transit(self):
        from cuvarbase import tls
        P, q, depth = 4.56, 0.025, 0.008
        t, y, dy = make_transit_lc(P, q, depth, ndata=3000, seed=11,
                                   t0_frac=0.3)
        res = tls.tls_transit(t, y, dy, R_star=1.0, M_star=1.0,
                              period_min=1.0, period_max=12.0)
        grid = np.sort(np.asarray(res['periods'], dtype=np.float64))
        i = int(np.argmin(np.abs(grid - P)))
        step = grid[min(i + 1, len(grid) - 1)] - grid[max(i - 1, 0)]
        assert abs(res['period'] - P) <= 2 * step + 1e-9, (res['period'], step)
        assert res['SDE'] > 5.0
        tmin = t.min()
        assert tmin <= res['T0'] < tmin + res['period']
        assert 0.0 <= res['t0_phase'] < 1.0
        # T0 lands on the injected mid-transit (modulo whole periods)
        t0_true = 0.3 * P
        nearest = np.min(np.abs(res['T0'] - (t0_true + P * np.arange(-2, 20))))
        assert nearest < 0.5 * q * P, nearest
        assert abs(res['depth'] - depth) / depth < 0.5
