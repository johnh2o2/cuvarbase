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
        assert results['SDE'] > 7


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
