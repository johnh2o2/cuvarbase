"""
Golden accuracy tests for the GPU TLS implementation.

The reference is the original CPU `transitleastsquares` package
(Hippke & Heller 2019). These tests need a GPU (the conftest stub
converts them to skips on CPU-only machines) and, for the comparison
tests, the optional `transitleastsquares` package — install both
`batman-package` and `transitleastsquares` on the validation pod.

The narrow-transit recovery test reproduces the audit scenario that
the pre-rework fixed 30-epoch t0 grid failed (8/8 injected epochs
missed): it requires no reference package and documents that the
duration-scaled grid actually finds what the old grid could not.

1.0 (Sep 2026 audit): the default duration window is now Keplerian
(defect 2), the SDE uses the reference's ``SR = chi2_min / chi2``
(ids 81/146) and its edge-extended running median (id 83). The
recovery-level expectations below were re-checked on an A40 after
those changes (period/depth unchanged; SDE values move -- the
thresholds are on the reference's own scale now).
"""
import numpy as np
import pytest


def make_transit_lc(period, q, depth=0.01, ndata=1500, baseline=60.0,
                    sigma=0.002, phase0=0.25, seed=0):
    """Synthetic light curve with an injected box transit.

    The box shape is deliberately simple: both implementations fit
    limb-darkened templates, and recovery-level assertions (period,
    depth, SDE) are insensitive to the exact ingress shape.
    """
    rand = np.random.RandomState(seed)
    t = np.sort(baseline * rand.rand(ndata))
    phase = (t / period) % 1.0
    in_transit = np.abs(((phase - phase0 + 0.5) % 1.0) - 0.5) < q / 2
    y = np.ones(ndata) - depth * in_transit
    y += sigma * rand.randn(ndata)
    dy = sigma * np.ones(ndata)
    return t, y, dy


class TestNarrowTransitRecovery:
    """The audit scenario: a transit with duration < 1/30 of the
    period, which the old fixed 30-epoch grid missed entirely."""

    def test_long_period_narrow_transit(self):
        from cuvarbase.tls import tls_search_gpu
        period, q, depth = 15.0, 0.012, 0.012
        t, y, dy = make_transit_lc(period, q, depth=depth)
        periods = np.linspace(14.0, 16.0, 400).astype(np.float32)

        results = tls_search_gpu(t, y, dy, periods=periods)

        assert abs(results['period'] - period) / period < 0.01
        # SDE > 5 is a clear detection; the absolute value depends on
        # the trial-period range (measured 5.75 here on an A5000)
        assert results['SDE'] > 5
        assert results['depth'] == pytest.approx(depth, rel=0.5)

    def test_short_period_regression(self):
        # Wide-transit case the old grid handled; must keep working.
        from cuvarbase.tls import tls_search_gpu
        period, q, depth = 3.0, 0.04, 0.01
        t, y, dy = make_transit_lc(period, q, depth=depth, baseline=30.0)
        periods = np.linspace(2.8, 3.2, 400).astype(np.float32)

        results = tls_search_gpu(t, y, dy, periods=periods)

        assert abs(results['period'] - period) / period < 0.01
        # 1.0: measured 6.90 on an A40 with the default Keplerian
        # duration window ([0.018, 0.073] at 3 d; 8.55 with the retired
        # fixed window, 6.55 / 8.00 under the pre-1.0 SR definition on
        # the same spectra). The null on this 400-period grid is
        # 4.2 +/- 0.6 (max 5.7 over 40 noise light curves), so > 6 is
        # still a clear detection; the old threshold of 7 was set on
        # the wider-window spectrum.
        assert results['SDE'] > 6


class TestGoldenVsTransitLeastSquares:
    """Direct comparison against the reference CPU implementation on
    identical data, over the same period range."""

    @pytest.mark.parametrize("period,q,depth", [
        (3.0, 0.04, 0.01),    # short period, wide transit
        (12.0, 0.014, 0.012),  # long period, narrow transit
    ])
    def test_recovery_matches_reference(self, period, q, depth):
        ref = pytest.importorskip('transitleastsquares')
        from cuvarbase.tls import tls_search_gpu

        t, y, dy = make_transit_lc(period, q, depth=depth)

        # cuvarbase (GPU)
        periods = np.linspace(0.9 * period, 1.1 * period,
                              500).astype(np.float32)
        res_gpu = tls_search_gpu(t, y, dy, periods=periods)

        # reference (CPU); same period range to bound runtime
        model = ref.transitleastsquares(t, y, dy)
        res_cpu = model.power(period_min=0.9 * period,
                              period_max=1.1 * period,
                              show_progress_bar=False,
                              use_threads=2)

        # Both must find the injected signal...
        assert abs(res_gpu['period'] - period) / period < 0.01
        assert abs(res_cpu.period - period) / period < 0.01
        # ...agree with each other on the period...
        assert (abs(res_gpu['period'] - res_cpu.period)
                / res_cpu.period < 0.01)
        # ...and roughly on the depth (reference reports flux level)
        ref_depth = 1.0 - res_cpu.depth
        assert res_gpu['depth'] == pytest.approx(ref_depth, rel=0.5)
        # both detections must be significant (SDE > 5; the reference
        # itself measured 6.3 on the narrow-transit configuration)
        assert res_gpu['SDE'] > 5
        assert res_cpu.SDE > 5


def _batman_lc(period, rp, t0, baseline, cadence_min, sigma, seed,
               R_star=1.0, M_star=1.0):
    """Limb-darkened batman transit on a regular cadence (the audit's
    make_lc); returns t, y, dy, true depth, T14 (days)."""
    batman = pytest.importorskip('batman')
    G, R_sun, M_sun = 6.67430e-11, 6.957e8, 1.9884e30
    rng = np.random.RandomState(seed)
    n = int(baseline * 1440 / cadence_min)
    t = np.arange(n) * cadence_min / 1440.0
    a = (G * M_star * M_sun * (period * 86400.0) ** 2
         / (4 * np.pi ** 2)) ** (1 / 3) / (R_star * R_sun)
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = t0, period, rp, a, 90.0
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [0.4804, 0.1867], 'quadratic'
    f = batman.TransitModel(p, t).light_curve(p)
    y = f + sigma * rng.randn(n)
    t14 = period / np.pi * np.arcsin(min(1.0, (1 + rp) / a))
    return t, y, sigma * np.ones(n), 1.0 - f.min(), t14


class TestLongPeriodDurationWindow:
    """Defect 2 (tls-duration-window, audit id 9): with the pre-1.0
    constant q window [0.005, 0.15] a P = 365 d transit on a 1400-d
    baseline (30-min cadence, sigma 3e-4, rp = 0.04) came back at
    182.5 d with half the depth (q_true = 0.00154 is 3.2x below the old
    qmin). The default window is now Keplerian, so the default call
    recovers it; the old window is an opt-in that warns and still
    shows the alias."""

    def test_p365_on_1400d_baseline(self):
        from cuvarbase.tls import tls_search_gpu
        from cuvarbase import tls_grids
        P = 365.0
        t, y, dy, depth_true, t14 = _batman_lc(
            P, 0.04, 0.41 * P, baseline=1400.0, cadence_min=30.0,
            sigma=3e-4, seed=11)
        assert tls_grids.q_transit(P) < tls_grids.FIXED_QMIN / 3
        # reduced Ofir grid around the truth (~9000 periods, ~1.5 s);
        # the failure is in the duration window, not the grid
        periods = tls_grids.period_grid_ofir(t, period_min=0.5 * P,
                                             period_max=1.5 * P)

        r = tls_search_gpu(t, y, dy, periods=periods)   # default window
        assert abs(r['period'] - P) / P < 0.01, r['period']
        assert r['depth'] == pytest.approx(depth_true, rel=0.10)
        assert r['duration'] == pytest.approx(t14, rel=0.25)
        assert t.min() <= r['T0'] < t.min() + r['period']
        assert abs(r['T0'] - 0.41 * P) < 0.25 * t14

        # the retired window reproduces the audit's failure and warns
        import warnings
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            rf = tls_search_gpu(t, y, dy, periods=periods,
                                duration_window='fixed')
        assert any("excludes the Keplerian" in str(w.message) for w in rec)
        assert abs(rf['period'] - 0.5 * P) / (0.5 * P) < 0.01, rf['period']
        assert rf['depth'] < 0.6 * depth_true

    def test_true_default_grid_recovers_p365(self):
        # the actual default call: no period grid at all (Ofir grid to
        # span/2, ~180k periods; ~2-3 s on an A40)
        from cuvarbase.tls import tls_search_gpu
        P = 365.0
        t, y, dy, depth_true, t14 = _batman_lc(
            P, 0.04, 0.41 * P, baseline=1400.0, cadence_min=30.0,
            sigma=3e-4, seed=11)
        r = tls_search_gpu(t, y, dy)
        assert len(r['periods']) > 100000
        assert abs(r['period'] - P) / P < 0.01, r['period']
        assert r['depth'] == pytest.approx(depth_true, rel=0.10)


class TestSDEParityWithReference:
    """ids 81/146: cuvarbase's SDE now uses the reference definition
    (SR = chi2_min / chi2, edge-extended running median), so on the
    reference's own period grid the two packages report the same SDE
    for the same detection to within the coarse-vs-fine t0 grid
    difference. The pre-1.0 SR (1 - chi2 / max chi2) gave about half
    the SDE for strong signals."""

    def test_strong_signal_sde_matches_reference(self):
        ref = pytest.importorskip('transitleastsquares')
        from cuvarbase.tls import tls_search_gpu

        period, q, depth = 3.0, 0.04, 0.02
        t, y, dy = make_transit_lc(period, q, depth=depth, sigma=0.002,
                                   seed=3)
        model = ref.transitleastsquares(t, y, dy)
        res_cpu = model.power(period_min=0.8 * period,
                              period_max=1.25 * period,
                              oversampling_factor=3,
                              show_progress_bar=False, use_threads=2)
        periods = np.sort(np.asarray(res_cpu.periods, dtype=np.float64))
        # both packages detrend with the 91-point kernel (reference:
        # oversampling 3 x 30 + 1; cuvarbase's automatic kernel is
        # length-scaled below 910 periods, so pin it)
        res_gpu = tls_search_gpu(t, y, dy, periods=periods,
                                 sde_kernel_size=91)

        assert abs(res_gpu['period'] - period) / period < 0.01
        assert abs(res_cpu.period - period) / period < 0.01
        assert res_gpu['SDE'] == pytest.approx(res_cpu.SDE, rel=0.15), (
            res_gpu['SDE'], res_cpu.SDE)
        # the old definition on the same cuvarbase spectrum
        chi2 = res_gpu['chi2'][np.isfinite(res_gpu['chi2'])]
        sr_old = 1.0 - chi2 / chi2.max()
        sde_old_raw = (sr_old.max() - sr_old.mean()) / sr_old.std()
        assert sde_old_raw < 0.75 * res_cpu.SDE, (sde_old_raw, res_cpu.SDE)
        # SNR is the delta-chi2 significance, not the reference's snr
        chi2_0 = np.sum((1.0 - y) ** 2 / (dy ** 2 + 1e-10))
        assert res_gpu['SNR'] == pytest.approx(
            np.sqrt(chi2_0 - res_gpu['chi2_min']), rel=1e-5)
