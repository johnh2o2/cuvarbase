"""CPU-only tests for the Rust-backed period-finding algorithms.

These tests mirror the GPU integration tests in test_periodfind.py but use
the periodfind.cpu backend so no GPU is required.

Run with: pytest tests/test_cpu_standalone.py -v
"""

import warnings

import numpy as np
import pytest

from periodfind import Periodogram, Statistics
from periodfind.cpu import (
    AOV,
    FPW,
    BoxLeastSquares,
    ConditionalEntropy,
    LombScargle,
    find_top_peaks_batched,
)

# ---------------------------------------------------------------------------
# Helpers (same as test_periodfind.py)
# ---------------------------------------------------------------------------


def make_sinusoidal_lightcurve(
    period, n_points=500, amplitude=1.0, noise_std=0.05, t_span=100.0, seed=42
):
    """Generate a synthetic sinusoidal light curve with known period."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, t_span, n_points)).astype(np.float32)
    phase = 2.0 * np.pi * times / period
    mags = (amplitude * np.sin(phase) + amplitude).astype(np.float32)
    mags += rng.normal(0, noise_std, n_points).astype(np.float32)
    return times, mags


def make_trial_periods(true_period, n_periods=200, margin=0.5):
    """Create a trial period grid centered around the true period."""
    lo = max(true_period - margin * true_period, 0.01)
    hi = true_period + margin * true_period
    return np.linspace(lo, hi, n_periods, dtype=np.float32)


def make_eclipsing_binary(
    period,
    n_points=500,
    eclipse_depth=0.5,
    eclipse_width=0.1,
    noise_std=0.02,
    t_span=200.0,
    seed=42,
):
    """Generate an eclipsing binary light curve with periodic V-shaped dips."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, t_span, n_points)).astype(np.float32)
    phase = (times / period) % 1.0
    dist = np.minimum(phase, 1.0 - phase)
    mags = np.ones(n_points, dtype=np.float32)
    in_eclipse = dist < eclipse_width / 2
    mags[in_eclipse] = 1.0 - eclipse_depth * (1.0 - dist[in_eclipse] / (eclipse_width / 2))
    mags += rng.normal(0, noise_std, n_points).astype(np.float32)
    return times, mags


def make_rr_lyrae(period, n_points=500, amplitude=0.8, noise_std=0.02, t_span=200.0, seed=42):
    """Generate an RR Lyrae-like sawtooth light curve."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, t_span, n_points)).astype(np.float32)
    phase = (times / period) % 1.0
    rise_end = 0.2
    mags = np.where(
        phase < rise_end,
        amplitude * (phase / rise_end),
        amplitude * (1.0 - (phase - rise_end) / (1.0 - rise_end)),
    ).astype(np.float32)
    mags += rng.normal(0, noise_std, n_points).astype(np.float32)
    return times, mags


def period_matches(detected, true_period, tol=0.05, harmonics=None):
    """Check if detected period matches true period or a known harmonic."""
    candidates = [true_period]
    if harmonics is not None:
        candidates.extend(harmonics)
    else:
        candidates.extend([true_period / 2, 2 * true_period])
    return any(abs(detected - c) / c < tol for c in candidates)


# ---------------------------------------------------------------------------
# Conditional Entropy tests
# ---------------------------------------------------------------------------


class TestCPUConditionalEntropy:
    def test_basic_stats_output(self):
        """CE should return a list of Statistics, one per light curve."""
        t, m = make_sinusoidal_lightcurve(period=2.5)
        periods = make_trial_periods(2.5)
        period_dts = np.array([0.0], dtype=np.float32)

        ce = ConditionalEntropy(n_phase=10, n_mag=10)
        result = ce.calc([t], [m], periods, period_dts, output="stats")

        assert isinstance(result, list)
        assert len(result) == 1
        stat = result[0]
        assert isinstance(stat, Statistics)
        assert stat.significance > 0

    def test_detects_known_period(self):
        """CE should find the correct period for a clean sinusoidal signal."""
        true_period = 5.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=800, noise_std=0.02, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        ce = ConditionalEntropy(n_phase=15, n_mag=10)
        result = ce.calc([t], [m], periods, period_dts, output="stats")

        detected = result[0].params[0]
        assert abs(detected - true_period) / true_period < 0.05, (
            f"Expected ~{true_period}, got {detected}"
        )

    def test_periodogram_output(self):
        """CE periodogram output should have correct shape."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 100, dtype=np.float32)
        period_dts = np.array([0.0, 0.001], dtype=np.float32)

        ce = ConditionalEntropy()
        result = ce.calc([t], [m], periods, period_dts, output="periodogram")

        assert isinstance(result, list)
        assert len(result) == 1
        pgram = result[0]
        assert isinstance(pgram, Periodogram)
        assert pgram.data.shape == (100, 2)
        assert pgram.use_max is False  # CE uses minima

    def test_multiple_lightcurves(self):
        """CE should handle batched light curves."""
        lcs = [make_sinusoidal_lightcurve(period=p, seed=i) for i, p in enumerate([2.0, 4.0, 6.0])]
        times = [lc[0] for lc in lcs]
        mag_list = [lc[1] for lc in lcs]
        periods = np.linspace(1.0, 10.0, 300, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        ce = ConditionalEntropy(n_phase=10, n_mag=10)
        result = ce.calc(times, mag_list, periods, period_dts, output="stats")
        assert len(result) == 3

    def test_n_stats_multiple(self):
        """Requesting n_stats > 1 should return a list of Statistics."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 200, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        ce = ConditionalEntropy()
        result = ce.calc([t], [m], periods, period_dts, output="stats", n_stats=5)
        assert isinstance(result[0], list)
        assert len(result[0]) == 5

    def test_custom_bin_params(self):
        """CE with different bin settings should still run."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 50, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        ce = ConditionalEntropy(n_phase=20, n_mag=5, phase_bin_extent=2, mag_bin_extent=2)
        result = ce.calc([t], [m], periods, period_dts, output="stats")
        assert isinstance(result[0], Statistics)

    def test_mismatched_times_mags_raises(self):
        """Mismatched times/mags list lengths should raise ValueError."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 50, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        ce = ConditionalEntropy()
        with pytest.raises(ValueError):
            ce.calc([t, t], [m], periods, period_dts)

    def test_center_and_normalize_warns(self):
        """Setting both center and normalize should issue a warning."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 50, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        ce = ConditionalEntropy()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ce.calc([t], [m], periods, period_dts, normalize=True, center=True)
            runtime_warns = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warns) >= 1

    def test_float64_input_raises(self):
        """Passing float64 arrays should raise TypeError."""
        t = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        m = np.array([1.0, 0.5, 1.0], dtype=np.float64)
        periods = np.linspace(1.0, 5.0, 20, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        ce = ConditionalEntropy()
        with pytest.raises(TypeError, match="expected float32"):
            ce.calc([t], [m], periods, period_dts)


# ---------------------------------------------------------------------------
# Analysis of Variance tests
# ---------------------------------------------------------------------------


class TestCPUAOV:
    def test_basic_stats_output(self):
        """AOV should return Statistics objects."""
        t, m = make_sinusoidal_lightcurve(period=2.5)
        periods = make_trial_periods(2.5)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV(n_phase=10)
        result = aov.calc([t], [m], periods, period_dts, output="stats")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Statistics)

    def test_detects_known_period(self):
        """AOV should find the correct period for a clean sinusoidal signal."""
        true_period = 5.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=800, noise_std=0.02, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV(n_phase=15)
        result = aov.calc([t], [m], periods, period_dts, output="stats")

        detected = result[0].params[0]
        assert abs(detected - true_period) / true_period < 0.05, (
            f"Expected ~{true_period}, got {detected}"
        )

    def test_periodogram_output(self):
        """AOV periodogram output should have correct shape and use maxima."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 100, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV()
        result = aov.calc([t], [m], periods, period_dts, output="periodogram")

        pgram = result[0]
        assert isinstance(pgram, Periodogram)
        assert pgram.data.shape == (100, 1)
        assert pgram.use_max is True

    def test_multiple_lightcurves(self):
        """AOV should handle batched light curves."""
        lcs = [make_sinusoidal_lightcurve(period=p, seed=i) for i, p in enumerate([2.0, 4.0])]
        times = [lc[0] for lc in lcs]
        mag_list = [lc[1] for lc in lcs]
        periods = np.linspace(1.0, 10.0, 200, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV()
        result = aov.calc(times, mag_list, periods, period_dts, output="stats")
        assert len(result) == 2

    def test_normalize_flag(self):
        """AOV with normalize=True should still produce valid output."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        m_shifted = m + 1000.0
        periods = np.linspace(1.0, 10.0, 100, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV()
        result = aov.calc([t], [m_shifted], periods, period_dts, normalize=True, output="stats")
        assert isinstance(result[0], Statistics)

    def test_overlap_still_detects_period(self):
        """AOV with overlap should still find the correct period."""
        true_period = 4.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=800, noise_std=0.02, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV(n_phase=15, phase_bin_extent=3)
        result = aov.calc([t], [m], periods, period_dts, output="stats")

        detected = result[0].params[0]
        assert abs(detected - true_period) / true_period < 0.05, (
            f"Expected ~{true_period}, got {detected}"
        )


# ---------------------------------------------------------------------------
# Lomb-Scargle tests
# ---------------------------------------------------------------------------


class TestCPULombScargle:
    def test_basic_stats_output(self):
        """LS should return Statistics objects."""
        t, m = make_sinusoidal_lightcurve(period=2.5)
        periods = make_trial_periods(2.5)
        period_dts = np.array([0.0], dtype=np.float32)

        ls = LombScargle()
        result = ls.calc([t], [m], periods, period_dts, output="stats")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Statistics)

    def test_detects_known_period(self):
        """LS should find the correct period for a clean sinusoidal signal."""
        true_period = 5.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=800, noise_std=0.02, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        ls = LombScargle()
        result = ls.calc([t], [m], periods, period_dts, output="stats")

        detected = result[0].params[0]
        assert abs(detected - true_period) / true_period < 0.05, (
            f"Expected ~{true_period}, got {detected}"
        )

    def test_periodogram_output(self):
        """LS periodogram output should have correct shape and use maxima."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 100, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        ls = LombScargle()
        result = ls.calc([t], [m], periods, period_dts, output="periodogram")

        pgram = result[0]
        assert isinstance(pgram, Periodogram)
        assert pgram.data.shape == (100, 1)
        assert pgram.use_max is True

    def test_multiple_lightcurves(self):
        """LS should handle batched light curves."""
        lcs = [make_sinusoidal_lightcurve(period=p, seed=i) for i, p in enumerate([2.0, 4.0])]
        times = [lc[0] for lc in lcs]
        mag_list = [lc[1] for lc in lcs]
        periods = np.linspace(1.0, 10.0, 200, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        ls = LombScargle()
        result = ls.calc(times, mag_list, periods, period_dts, output="stats")
        assert len(result) == 2

    def test_center_default(self):
        """LS defaults to center=True; result should still be valid."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 100, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        ls = LombScargle()
        result = ls.calc([t], [m], periods, period_dts, center=True, output="stats")
        assert result[0].significance > 0

    def test_without_centering(self):
        """LS with center=False should still produce output."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 100, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        ls = LombScargle()
        result = ls.calc(
            [t], [m], periods, period_dts, center=False, normalize=False, output="stats"
        )
        assert isinstance(result[0], Statistics)


# ---------------------------------------------------------------------------
# FPW tests
# ---------------------------------------------------------------------------


class TestCPUFPW:
    def test_basic_stats_output(self):
        """FPW should return Statistics objects."""
        t, m = make_sinusoidal_lightcurve(period=2.5)
        periods = make_trial_periods(2.5)
        period_dts = np.array([0.0], dtype=np.float32)

        fpw = FPW(n_bins=10)
        result = fpw.calc([t], [m], periods, period_dts, output="stats")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Statistics)
        assert result[0].significance > 0

    def test_detects_known_period(self):
        """FPW should find the correct period for a clean sinusoidal signal."""
        true_period = 5.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=800, noise_std=0.02, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        fpw = FPW(n_bins=20)
        result = fpw.calc([t], [m], periods, period_dts, output="stats")

        detected = result[0].params[0]
        assert abs(detected - true_period) / true_period < 0.05, (
            f"Expected ~{true_period}, got {detected}"
        )

    def test_with_uncertainties(self):
        """FPW with explicit uncertainties should detect the period."""
        true_period = 3.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=600, noise_std=0.05, t_span=150.0
        )
        errs = np.full(len(t), 0.05, dtype=np.float32)
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        fpw = FPW(n_bins=15)
        result = fpw.calc([t], [m], periods, period_dts, errs=[errs], output="stats")

        detected = result[0].params[0]
        assert abs(detected - true_period) / true_period < 0.05, (
            f"Expected ~{true_period}, got {detected}"
        )

    def test_without_uncertainties(self):
        """FPW without uncertainties (uniform weights) should still work."""
        true_period = 2.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=500, noise_std=0.03, t_span=100.0
        )
        periods = make_trial_periods(true_period, n_periods=400)
        period_dts = np.array([0.0], dtype=np.float32)

        fpw = FPW(n_bins=10)
        result = fpw.calc([t], [m], periods, period_dts, output="stats")

        detected = result[0].params[0]
        assert abs(detected - true_period) / true_period < 0.05, (
            f"Expected ~{true_period}, got {detected}"
        )

    def test_periodogram_output(self):
        """FPW periodogram should have correct shape and use maxima."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 100, dtype=np.float32)
        period_dts = np.array([0.0, 0.001], dtype=np.float32)

        fpw = FPW(n_bins=10)
        result = fpw.calc([t], [m], periods, period_dts, output="periodogram")

        pgram = result[0]
        assert isinstance(pgram, Periodogram)
        assert pgram.data.shape == (100, 2)
        assert pgram.use_max is True

    def test_multiple_lightcurves(self):
        """FPW should handle batched light curves."""
        lcs = [make_sinusoidal_lightcurve(period=p, seed=i) for i, p in enumerate([2.0, 4.0, 6.0])]
        times = [lc[0] for lc in lcs]
        mag_list = [lc[1] for lc in lcs]
        periods = np.linspace(1.0, 10.0, 300, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        fpw = FPW(n_bins=15)
        result = fpw.calc(times, mag_list, periods, period_dts, output="stats")
        assert len(result) == 3

    def test_n_stats_multiple(self):
        """Requesting n_stats > 1 should return a list of Statistics."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 200, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        fpw = FPW(n_bins=10)
        result = fpw.calc([t], [m], periods, period_dts, output="stats", n_stats=5)
        assert isinstance(result[0], list)
        assert len(result[0]) == 5

    def test_mismatched_errs_raises(self):
        """Mismatched errs list length should raise ValueError."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        errs = np.ones(len(t), dtype=np.float32)
        periods = np.linspace(1.0, 10.0, 50, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        fpw = FPW()
        with pytest.raises(ValueError):
            fpw.calc([t, t], [m, m], periods, period_dts, errs=[errs])

    def test_float64_errs_raises(self):
        """Passing float64 errs should raise TypeError."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        errs = np.ones(len(t), dtype=np.float64)
        periods = np.linspace(1.0, 10.0, 50, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        fpw = FPW()
        with pytest.raises(TypeError, match="expected float32"):
            fpw.calc([t], [m], periods, period_dts, errs=[errs])

    def test_variable_uncertainties(self):
        """FPW with non-uniform uncertainties should still find period."""
        true_period = 4.0
        rng = np.random.default_rng(42)
        n_points = 800
        times = np.sort(rng.uniform(0, 200, n_points)).astype(np.float32)
        errs = rng.uniform(0.01, 0.2, n_points).astype(np.float32)
        noise = (rng.normal(0, 1, n_points) * errs).astype(np.float32)
        mags = (np.sin(2 * np.pi * times / true_period) + noise).astype(np.float32)

        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        fpw = FPW(n_bins=20)
        result = fpw.calc([times], [mags], periods, period_dts, errs=[errs], output="stats")

        detected = result[0].params[0]
        assert abs(detected - true_period) / true_period < 0.05, (
            f"Expected ~{true_period}, got {detected}"
        )

    def test_factory_function(self):
        """periodfind.FPW() factory should produce a working FPW object."""
        import periodfind

        fpw = periodfind.FPW(n_bins=10, device="cpu")
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 100, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)
        result = fpw.calc([t], [m], periods, period_dts, output="stats")
        assert isinstance(result[0], Statistics)


# ---------------------------------------------------------------------------
# Box Least Squares tests
# ---------------------------------------------------------------------------


class TestCPUBoxLeastSquares:
    def test_basic_stats_output(self):
        """BLS should return Statistics objects."""
        t, m = make_eclipsing_binary(period=2.5)
        periods = make_trial_periods(2.5)
        period_dts = np.array([0.0], dtype=np.float32)

        bls = BoxLeastSquares(n_bins=50, qmin=0.01, qmax=0.5)
        result = bls.calc([t], [m], periods, period_dts, output="stats")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Statistics)
        assert result[0].significance > 0

    def test_detects_eclipsing_binary(self):
        """BLS should find the correct period for an eclipsing binary."""
        true_period = 2.5
        t, m = make_eclipsing_binary(
            period=true_period, n_points=800, eclipse_depth=0.5, noise_std=0.02, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        bls = BoxLeastSquares(n_bins=50, qmin=0.01, qmax=0.3)
        result = bls.calc([t], [m], periods, period_dts, output="stats")

        detected = result[0].params[0]
        assert period_matches(detected, true_period), (
            f"Expected ~{true_period}, got {detected}"
        )

    def test_detects_known_period_sinusoidal(self):
        """BLS should find the correct period for a sinusoidal signal."""
        true_period = 5.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=800, noise_std=0.02, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        bls = BoxLeastSquares(n_bins=50, qmin=0.01, qmax=0.5)
        result = bls.calc([t], [m], periods, period_dts, output="stats")

        detected = result[0].params[0]
        assert period_matches(detected, true_period), (
            f"Expected ~{true_period}, got {detected}"
        )

    def test_with_uncertainties(self):
        """BLS with explicit uncertainties should detect the period."""
        true_period = 3.0
        t, m = make_eclipsing_binary(
            period=true_period, n_points=600, eclipse_depth=0.5, noise_std=0.05, t_span=150.0
        )
        errs = np.full(len(t), 0.05, dtype=np.float32)
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        bls = BoxLeastSquares(n_bins=50, qmin=0.01, qmax=0.3)
        result = bls.calc([t], [m], periods, period_dts, errs=[errs], output="stats")

        detected = result[0].params[0]
        assert period_matches(detected, true_period), (
            f"Expected ~{true_period}, got {detected}"
        )

    def test_without_uncertainties(self):
        """BLS without uncertainties (uniform weights) should still work."""
        true_period = 2.0
        t, m = make_eclipsing_binary(
            period=true_period, n_points=500, eclipse_depth=0.5, noise_std=0.03, t_span=100.0
        )
        periods = make_trial_periods(true_period, n_periods=400)
        period_dts = np.array([0.0], dtype=np.float32)

        bls = BoxLeastSquares(n_bins=50, qmin=0.01, qmax=0.3)
        result = bls.calc([t], [m], periods, period_dts, output="stats")

        detected = result[0].params[0]
        assert period_matches(detected, true_period), (
            f"Expected ~{true_period}, got {detected}"
        )

    def test_periodogram_output(self):
        """BLS periodogram should have correct shape and use maxima."""
        t, m = make_eclipsing_binary(period=3.0)
        periods = np.linspace(1.0, 10.0, 100, dtype=np.float32)
        period_dts = np.array([0.0, 0.001], dtype=np.float32)

        bls = BoxLeastSquares(n_bins=50, qmin=0.01, qmax=0.5)
        result = bls.calc([t], [m], periods, period_dts, output="periodogram")

        pgram = result[0]
        assert isinstance(pgram, Periodogram)
        assert pgram.data.shape == (100, 2)
        assert pgram.use_max is True

    def test_multiple_lightcurves(self):
        """BLS should handle batched light curves."""
        lcs = [
            make_eclipsing_binary(period=p, seed=i) for i, p in enumerate([2.0, 4.0, 6.0])
        ]
        times = [lc[0] for lc in lcs]
        mag_list = [lc[1] for lc in lcs]
        periods = np.linspace(1.0, 10.0, 300, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        bls = BoxLeastSquares(n_bins=50, qmin=0.01, qmax=0.3)
        result = bls.calc(times, mag_list, periods, period_dts, output="stats")
        assert len(result) == 3

    def test_n_stats_multiple(self):
        """Requesting n_stats > 1 should return a list of Statistics."""
        t, m = make_eclipsing_binary(period=3.0)
        periods = np.linspace(1.0, 10.0, 200, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        bls = BoxLeastSquares(n_bins=50, qmin=0.01, qmax=0.5)
        result = bls.calc([t], [m], periods, period_dts, output="stats", n_stats=5)
        assert isinstance(result[0], list)
        assert len(result[0]) == 5

    def test_mismatched_errs_raises(self):
        """Mismatched errs list length should raise ValueError."""
        t, m = make_eclipsing_binary(period=3.0)
        errs = np.ones(len(t), dtype=np.float32)
        periods = np.linspace(1.0, 10.0, 50, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        bls = BoxLeastSquares()
        with pytest.raises(ValueError):
            bls.calc([t, t], [m, m], periods, period_dts, errs=[errs])

    def test_float64_errs_raises(self):
        """Passing float64 errs should raise TypeError."""
        t, m = make_eclipsing_binary(period=3.0)
        errs = np.ones(len(t), dtype=np.float64)
        periods = np.linspace(1.0, 10.0, 50, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        bls = BoxLeastSquares()
        with pytest.raises(TypeError, match="expected float32"):
            bls.calc([t], [m], periods, period_dts, errs=[errs])

    def test_factory_function(self):
        """periodfind.BoxLeastSquares() factory should produce a working object."""
        import periodfind

        bls = periodfind.BoxLeastSquares(n_bins=50, qmin=0.01, qmax=0.5, device="cpu")
        t, m = make_eclipsing_binary(period=3.0)
        periods = np.linspace(1.0, 10.0, 100, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)
        result = bls.calc([t], [m], periods, period_dts, output="stats")
        assert isinstance(result[0], Statistics)

    def test_peaks_output(self):
        """BLS peak finding should work."""
        true_period = 2.5
        t, m = make_eclipsing_binary(
            period=true_period, n_points=800, eclipse_depth=0.5, noise_std=0.02, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        bls = BoxLeastSquares(n_bins=50, qmin=0.01, qmax=0.3)
        peaks = bls.calc([t], [m], periods, period_dts, output="peaks", n_peaks=16)

        assert isinstance(peaks, list)
        assert len(peaks) == 1
        assert isinstance(peaks[0], list)
        assert len(peaks[0]) <= 16
        assert isinstance(peaks[0][0], Statistics)

    def test_peaks_best_matches_stats(self):
        """The best BLS peak should match the stats output."""
        true_period = 3.0
        t, m = make_eclipsing_binary(
            period=true_period, n_points=600, eclipse_depth=0.5, t_span=150.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        bls = BoxLeastSquares(n_bins=50, qmin=0.01, qmax=0.3)
        stats = bls.calc([t], [m], periods, period_dts, output="stats")
        peaks = bls.calc(
            [t], [m], periods, period_dts, output="peaks", n_peaks=32, min_distance=1
        )

        stats_period = stats[0].params[0]
        peaks_period = peaks[0][0].params[0]
        assert stats_period == pytest.approx(peaks_period)


# ---------------------------------------------------------------------------
# Cross-algorithm consistency tests
# ---------------------------------------------------------------------------


class TestCPUCrossAlgorithm:
    def test_all_algorithms_find_same_period(self):
        """All three algorithms should agree on the period for clean signals."""
        true_period = 4.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=1000, noise_std=0.01, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        ce = ConditionalEntropy(n_phase=15, n_mag=10)
        aov = AOV(n_phase=15)
        ls_algo = LombScargle()
        fpw = FPW(n_bins=15)

        ce_result = ce.calc([t], [m], periods, period_dts, output="stats")
        aov_result = aov.calc([t], [m], periods, period_dts, output="stats")
        ls_result = ls_algo.calc([t], [m], periods, period_dts, output="stats")
        fpw_result = fpw.calc([t], [m], periods, period_dts, output="stats")

        tol = 0.05
        for name, res in [
            ("CE", ce_result),
            ("AOV", aov_result),
            ("LS", ls_result),
            ("FPW", fpw_result),
        ]:
            detected = res[0].params[0]
            assert abs(detected - true_period) / true_period < tol, (
                f"{name} detected {detected}, expected ~{true_period}"
            )

    def test_periodogram_best_matches_stats(self):
        """Periodogram.best_params should agree with stats output."""
        true_period = 3.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=500, noise_std=0.02, t_span=100.0
        )
        periods = make_trial_periods(true_period, n_periods=300)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV(n_phase=10)
        stats_result = aov.calc([t], [m], periods, period_dts, output="stats", n_stats=1)
        pgram_result = aov.calc([t], [m], periods, period_dts, output="periodogram")

        stats_period = stats_result[0].params[0]
        pgram_period = pgram_result[0].best_params(n=1).params[0]
        assert stats_period == pytest.approx(pgram_period)


# ---------------------------------------------------------------------------
# Edge case / robustness tests
# ---------------------------------------------------------------------------


class TestCPUEdgeCases:
    def test_single_period_dt(self):
        """Should work with a single period_dt = 0."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 50, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        ce = ConditionalEntropy()
        result = ce.calc([t], [m], periods, period_dts, output="stats")
        assert isinstance(result[0], Statistics)

    def test_multiple_period_dts(self):
        """Should work with multiple period derivative values."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 50, dtype=np.float32)
        period_dts = np.linspace(-0.01, 0.01, 5, dtype=np.float32)

        ce = ConditionalEntropy()
        result = ce.calc([t], [m], periods, period_dts, output="periodogram")
        assert result[0].data.shape == (50, 5)

    def test_short_lightcurve(self):
        """Should handle a very short light curve without crashing."""
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        m = np.array([1.0, 0.5, 1.0, 0.5, 1.0], dtype=np.float32)
        periods = np.linspace(1.0, 5.0, 20, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        for AlgoCls in [ConditionalEntropy, AOV, LombScargle]:
            algo = AlgoCls()
            result = algo.calc([t], [m], periods, period_dts, output="stats")
            assert isinstance(result[0], Statistics)

    def test_invalid_output_type_raises(self):
        """Invalid output type should raise NotImplementedError."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 50, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        ce = ConditionalEntropy()
        with pytest.raises(NotImplementedError):
            ce.calc([t], [m], periods, period_dts, output="invalid")

    def test_significance_types_both_work(self):
        """Both significance types should produce valid results."""
        t, m = make_sinusoidal_lightcurve(period=3.0, n_points=300)
        periods = np.linspace(1.0, 10.0, 100, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV()
        for sig_type in ["stdmean", "madmedian"]:
            result = aov.calc(
                [t], [m], periods, period_dts, output="stats", significance_type=sig_type
            )
            assert result[0].significance_type == sig_type
            assert result[0].significance > 0


# ---------------------------------------------------------------------------
# LSST-like scenarios
# ---------------------------------------------------------------------------


class TestCPULSSTScenarios:
    def test_eclipsing_binary_detection(self):
        """Algorithms should detect an eclipsing binary period."""
        true_period = 2.5
        t, m = make_eclipsing_binary(
            period=true_period, n_points=800, eclipse_depth=0.5, noise_std=0.02, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        for AlgoCls in [AOV, LombScargle]:
            algo = AlgoCls() if AlgoCls == LombScargle else AlgoCls(n_phase=20)
            result = algo.calc([t], [m], periods, period_dts, output="stats")
            detected = result[0].params[0]
            assert period_matches(detected, true_period), (
                f"{AlgoCls.__name__} detected {detected}, expected ~{true_period}"
            )

    def test_rr_lyrae_detection(self):
        """Algorithms should detect an RR Lyrae period."""
        true_period = 0.6
        t, m = make_rr_lyrae(
            period=true_period, n_points=600, amplitude=0.8, noise_std=0.03, t_span=50.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        for AlgoCls in [AOV, LombScargle]:
            algo = AlgoCls() if AlgoCls == LombScargle else AlgoCls(n_phase=20)
            result = algo.calc([t], [m], periods, period_dts, output="stats")
            detected = result[0].params[0]
            assert period_matches(detected, true_period), (
                f"{AlgoCls.__name__} detected {detected}, expected ~{true_period}"
            )


# ---------------------------------------------------------------------------
# Large-scale tests
# ---------------------------------------------------------------------------


class TestCPULargeScale:
    def test_large_period_grid(self):
        """Correctness with a large (5000+) period grid."""
        true_period = 5.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=500, noise_std=0.05, t_span=200.0
        )
        periods = np.linspace(0.5, 15.0, 5000, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        ls = LombScargle()
        result = ls.calc([t], [m], periods, period_dts, output="stats")
        detected = result[0].params[0]
        assert period_matches(detected, true_period), (
            f"LS large-grid: detected {detected}, expected ~{true_period}"
        )

    def test_large_batch(self):
        """Correctness with a batch of 10+ light curves."""
        n_curves = 12
        true_periods = np.linspace(2.0, 8.0, n_curves)
        times_list = []
        mags_list = []
        for i, p in enumerate(true_periods):
            t, m = make_sinusoidal_lightcurve(
                period=p, n_points=400, noise_std=0.03, t_span=200.0, seed=i
            )
            times_list.append(t)
            mags_list.append(m)

        periods = np.linspace(1.0, 12.0, 1000, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV(n_phase=15)
        results = aov.calc(times_list, mags_list, periods, period_dts, output="stats")
        assert len(results) == n_curves

        n_correct = 0
        for i, res in enumerate(results):
            detected = res.params[0]
            if period_matches(detected, true_periods[i]):
                n_correct += 1
        assert n_correct >= int(0.8 * n_curves), (
            f"Only {n_correct}/{n_curves} periods detected correctly"
        )

    def test_large_grid_with_period_dts(self):
        """Large period grid with multiple period derivatives."""
        true_period = 4.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=600, noise_std=0.03, t_span=200.0
        )
        periods = np.linspace(1.0, 10.0, 5000, dtype=np.float32)
        period_dts = np.linspace(-0.001, 0.001, 3, dtype=np.float32)

        ls = LombScargle()
        result = ls.calc([t], [m], periods, period_dts, output="periodogram")
        pgram = result[0]
        assert pgram.data.shape == (5000, 3)

        pdt_zero_col = pgram.data[:, 1]
        best_idx = np.argmax(pdt_zero_col)
        detected = periods[best_idx]
        assert period_matches(detected, true_period), (
            f"LS large-grid+dts: detected {detected}, expected ~{true_period}"
        )


# ---------------------------------------------------------------------------
# Peak finding tests
# ---------------------------------------------------------------------------


class TestPeakFinding:
    """Tests for the chunked greedy peak finder (output='peaks')."""

    def test_peaks_returns_list_of_lists(self):
        """output='peaks' should return a list of lists of Statistics."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 200, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV(n_phase=10)
        result = aov.calc([t], [m], periods, period_dts, output="peaks", n_peaks=8)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) <= 8
        assert isinstance(result[0][0], Statistics)

    def test_peaks_best_matches_stats(self):
        """The best peak should match the stats output for all algorithms."""
        true_period = 5.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=800, noise_std=0.02, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        for name, algo, _use_max in [
            ("CE", ConditionalEntropy(n_phase=15, n_mag=10), False),
            ("AOV", AOV(n_phase=15), True),
            ("LS", LombScargle(), True),
            ("FPW", FPW(n_bins=15), True),
        ]:
            stats = algo.calc([t], [m], periods, period_dts, output="stats")
            peaks = algo.calc(
                [t], [m], periods, period_dts, output="peaks", n_peaks=32, min_distance=1
            )

            stats_period = stats[0].params[0]
            peaks_period = peaks[0][0].params[0]
            assert stats_period == pytest.approx(peaks_period), (
                f"{name}: stats={stats_period}, peaks={peaks_period}"
            )

    def test_peaks_detects_known_period(self):
        """Peak finder should find the correct period for a clean signal."""
        true_period = 4.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=800, noise_std=0.02, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV(n_phase=15)
        peaks = aov.calc([t], [m], periods, period_dts, output="peaks", n_peaks=32, min_distance=5)
        detected = peaks[0][0].params[0]
        assert abs(detected - true_period) / true_period < 0.05, (
            f"Expected ~{true_period}, got {detected}"
        )

    def test_peaks_sorted_best_first(self):
        """Peaks should be sorted by value (best first)."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 300, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        # AOV uses maxima
        aov = AOV(n_phase=10)
        peaks = aov.calc([t], [m], periods, period_dts, output="peaks", n_peaks=16, min_distance=1)
        values = [p.value for p in peaks[0]]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1], "Peaks not sorted best-first"

        # CE uses minima
        ce = ConditionalEntropy(n_phase=10, n_mag=10)
        peaks = ce.calc([t], [m], periods, period_dts, output="peaks", n_peaks=16, min_distance=1)
        values = [p.value for p in peaks[0]]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], "CE peaks not sorted best-first (min)"

    def test_peaks_n_peaks_32_default(self):
        """Default n_peaks=32 should return up to 32 peaks."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 500, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV(n_phase=10)
        peaks = aov.calc([t], [m], periods, period_dts, output="peaks")
        assert len(peaks[0]) <= 32
        assert len(peaks[0]) > 0

    def test_peaks_min_distance(self):
        """Larger min_distance should produce fewer, more spread out peaks."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 500, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV(n_phase=10)
        peaks_close = aov.calc(
            [t], [m], periods, period_dts, output="peaks", n_peaks=32, min_distance=1
        )
        peaks_far = aov.calc(
            [t], [m], periods, period_dts, output="peaks", n_peaks=32, min_distance=20
        )
        assert len(peaks_far[0]) <= len(peaks_close[0])

    def test_peaks_batched(self):
        """Peak finding should work on multiple light curves."""
        lcs = [make_sinusoidal_lightcurve(period=p, seed=i) for i, p in enumerate([2.0, 4.0, 6.0])]
        times = [lc[0] for lc in lcs]
        mag_list = [lc[1] for lc in lcs]
        periods = np.linspace(1.0, 10.0, 300, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV(n_phase=10)
        peaks = aov.calc(times, mag_list, periods, period_dts, output="peaks", n_peaks=32)
        assert len(peaks) == 3
        for curve_peaks in peaks:
            assert len(curve_peaks) > 0

    def test_peaks_with_multiple_period_dts(self):
        """Peak finding should work with a 2D period grid."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 100, dtype=np.float32)
        period_dts = np.array([0.0, 0.001], dtype=np.float32)

        aov = AOV(n_phase=10)
        peaks = aov.calc([t], [m], periods, period_dts, output="peaks", n_peaks=16, min_distance=1)
        assert len(peaks[0]) > 0
        # Each peak should have [period, period_dt] params
        assert len(peaks[0][0].params) == 2

    def test_find_top_peaks_batched_standalone(self):
        """Standalone find_top_peaks_batched on pre-computed periodograms."""
        t, m = make_sinusoidal_lightcurve(period=5.0, n_points=500)
        periods = np.linspace(1.0, 10.0, 200, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV(n_phase=10)
        pgrams = aov.calc([t], [m], periods, period_dts, output="periodogram")
        data_3d = np.expand_dims(pgrams[0].data, axis=0)  # (1, n_periods, n_pdts)

        idx, val = find_top_peaks_batched(data_3d, n_peaks=8, min_distance=3, use_max=True)
        assert idx.shape == (1, 8)
        assert val.shape == (1, 8)
        assert idx[0, 0] >= 0  # at least one peak found
        assert val[0, 0] >= val[0, 1]  # sorted

    def test_peaks_fpw_with_errs(self):
        """FPW peak finding should work with explicit uncertainties."""
        true_period = 3.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=600, noise_std=0.05, t_span=150.0
        )
        errs = np.full(len(t), 0.05, dtype=np.float32)
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        fpw = FPW(n_bins=15)
        peaks = fpw.calc([t], [m], periods, period_dts, errs=[errs], output="peaks", n_peaks=32)
        detected = peaks[0][0].params[0]
        assert abs(detected - true_period) / true_period < 0.05, (
            f"Expected ~{true_period}, got {detected}"
        )
