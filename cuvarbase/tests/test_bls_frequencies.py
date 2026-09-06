"""
Tests for the Keplerian/uniform frequency grid utilities and the
per-frequency q-bound wiring into the batch BLS API.
"""
import numpy as np
import pytest

from ..bls_frequencies import (_q_transit, keplerian_freq_grid,
                               uniform_freq_grid, freq_grid_stats)


class TestKeplerianFreqGrid:

    def test_grid_covers_range(self):
        freqs = keplerian_freq_grid(1.0, 10.0, baseline=365.0)
        assert freqs[0] == pytest.approx(0.1, rel=1e-5)
        assert freqs[-1] >= 1.0 * 0.999
        assert np.all(np.diff(freqs) > 0)

    def test_fewer_freqs_than_uniform(self):
        kep = keplerian_freq_grid(1.0, 50.0, baseline=1000.0)
        uni = uniform_freq_grid(1.0, 50.0, baseline=1000.0)
        assert len(kep) < len(uni)

    def test_return_qvals(self):
        freqs, qvals = keplerian_freq_grid(1.0, 10.0, baseline=365.0,
                                           return_qvals=True)
        assert len(qvals) == len(freqs)
        assert qvals.dtype == np.float32
        assert np.all(qvals > 0)
        assert np.all(qvals <= 0.5)
        # Keplerian q grows with frequency (shorter periods -> larger
        # duration fraction)
        assert np.all(np.diff(qvals) >= 0)
        # consistent with the q model used to build the grid
        np.testing.assert_allclose(
            qvals, _q_transit(freqs.astype(np.float64)), rtol=1e-5)

    def test_default_return_unchanged(self):
        out = keplerian_freq_grid(1.0, 10.0, baseline=365.0)
        assert isinstance(out, np.ndarray)

    def test_grid_stats(self):
        freqs = keplerian_freq_grid(1.0, 50.0, baseline=1000.0)
        stats = freq_grid_stats(freqs, 1000.0)
        assert stats['nfreq'] == len(freqs)
        assert stats['reduction_factor'] > 1


class TestBatchPerFrequencyQBounds:
    """eebls_gpu_batch accepts per-frequency qmin/qmax arrays (the
    batch kernel reads per-frequency bin counts); combined with
    keplerian_freq_grid(return_qvals=True) this enables
    duration-constrained Keplerian searches in batch mode."""

    def test_batch_keplerian_q_bounds(self):
        # GPU only: skipped on CPU machines via the conftest stub.
        from ..bls import eebls_gpu_batch

        freq_inj, q_inj, delta = 0.5, 0.03, 0.05
        ndata, baseline = 300, 365.0

        lcs = []
        for seed in (1, 2):
            r = np.random.RandomState(seed)
            t = np.sort(baseline * r.rand(ndata))
            phase = (t * freq_inj) % 1.0
            y = 12.0 - delta * (phase < q_inj)
            y += 0.01 * r.randn(ndata)
            lcs.append((t, y, 0.01 * np.ones(ndata)))

        freqs, qvals = keplerian_freq_grid(1.5, 3.0, baseline,
                                           return_qvals=True)
        results = eebls_gpu_batch(lcs, freqs,
                                  qmin=0.5 * qvals, qmax=2.0 * qvals)
        assert len(results) == 2
        for power in results:
            best = freqs[int(np.argmax(power))]
            assert abs(best - freq_inj) / freq_inj < 0.02


class TestVectorizedGridRecursion:
    """Sep 2026 audit, ids 4/44 (plan item BLS-4).

    ``keplerian_freq_grid`` and ``cuvarbase.bls.transit_autofreq`` built
    the Ofir (2014) duty-cycle grid with a scalar Python ``while`` loop,
    one ``q`` evaluation per frequency: 0.2-1.0 s per call at survey
    grid sizes, which dwarfed the GPU search that followed. They now
    solve the same recursion with numpy (``method='vectorized'``, the
    default) and keep the loop as ``method='recursion'``.

    The vectorized form converges to a fixed point of the *same*
    recursion (seed from the continuum integral, then defect
    correction), so these tests pin agreement, not a tolerance chosen to
    accommodate a different grid.
    """

    CASES = [
        dict(period_min=0.5, period_max=100., baseline=730.),
        dict(period_min=0.5, period_max=13.5, baseline=27.),
        dict(period_min=0.5, period_max=300., baseline=1400.),
        dict(period_min=0.3, period_max=50., baseline=200.,
             R_star=0.3, M_star=0.3),
        dict(period_min=1.0, period_max=200., baseline=500., R_star=3.0),
        dict(period_min=0.5, period_max=100., baseline=365.,
             oversampling=10),
        dict(period_min=0.5, period_max=100., baseline=365.,
             oversampling=0.5),
        dict(period_min=9.0, period_max=10., baseline=100.),
        # degenerate: period_min > period_max
        dict(period_min=100., period_max=0.5, baseline=100.),
    ]

    @pytest.mark.parametrize("case", CASES)
    def test_keplerian_grid_matches_the_recursion(self, case):
        rec = keplerian_freq_grid(method='recursion', **case)
        vec = keplerian_freq_grid(method='vectorized', **case)
        # float32 output: bit-identical
        assert rec.shape == vec.shape
        assert np.array_equal(rec, vec)

    @pytest.mark.parametrize("case", CASES[:4])
    def test_keplerian_qvals_match(self, case):
        fr, qr = keplerian_freq_grid(method='recursion', return_qvals=True,
                                     **case)
        fv, qv = keplerian_freq_grid(method='vectorized', return_qvals=True,
                                     **case)
        assert np.array_equal(qr, qv)

    def test_bad_method_raises(self):
        with pytest.raises(ValueError, match="grid method"):
            keplerian_freq_grid(1.0, 10.0, 365.0, method='euler')

    def test_recursion_still_reproduces_its_own_definition(self):
        # guard against the shared helper drifting from the scalar loop
        # it replaced (this is the literal pre-1.0 body)
        from ..bls_frequencies import _recursion_transit_grid
        rho, oversampling, T = 1.0, 2, 365.0
        f_min, f_max = 1. / 100., 1. / 0.5
        freqs = [f_min]
        while freqs[-1] < f_max:
            q = float(_q_transit(freqs[-1], rho=rho))
            q = max(q, 1e-6)
            freqs.append(freqs[-1] + q / (oversampling * T))
        ref = np.array(freqs)
        got = _recursion_transit_grid(f_min, f_max, 1.0, oversampling * T,
                                      rho=rho, q_floor=1e-6)
        assert np.array_equal(ref, got)

    @pytest.mark.parametrize("kw", [
        {}, dict(samples_per_peak=5), dict(rho=5.), dict(rho=0.05),
        dict(qmin_fac=0.5), dict(qmin_fac=0.1, qmax_fac=4.),
    ])
    def test_transit_autofreq_matches_the_recursion(self, kw):
        from ..bls import transit_autofreq
        rand = np.random.RandomState(21)
        t = np.sort(180. * rand.rand(400))

        fr, qr = transit_autofreq(t, method='recursion', **kw)
        fv, qv = transit_autofreq(t, method='vectorized', **kw)
        assert len(fr) == len(fv), "grid length changed"
        # float64 output: the accumulated Euler sum agrees to rounding
        np.testing.assert_allclose(fv, fr, rtol=1e-13, atol=0.)
        # the trial frequencies the kernels actually search (float32)
        # are bit-identical
        assert np.array_equal(fr.astype(np.float32),
                              fv.astype(np.float32))
        np.testing.assert_allclose(qv, qr, rtol=1e-12, atol=0.)

    def test_transit_autofreq_bad_method_raises(self):
        from ..bls import transit_autofreq
        rand = np.random.RandomState(3)
        t = np.sort(180. * rand.rand(200))
        with pytest.raises(ValueError, match="grid method"):
            transit_autofreq(t, method='integral')

    def test_vectorized_grid_satisfies_the_recursion(self):
        # the defining property, checked directly on the returned grid
        from ..bls import transit_autofreq, q_transit
        rand = np.random.RandomState(5)
        t = np.sort(365. * rand.rand(500))
        T = float(np.max(t) - np.min(t))
        freqs, _ = transit_autofreq(t, samples_per_peak=2, qmin_fac=0.2)
        step = (0.2 * q_transit(freqs[:-1], rho=1.)) / (2 * T)
        np.testing.assert_allclose(freqs[1:], freqs[:-1] + step,
                                   rtol=1e-13, atol=0.)
        # ... and it stops exactly where the loop would
        assert freqs[-1] >= freqs[-2] or len(freqs) == 1
