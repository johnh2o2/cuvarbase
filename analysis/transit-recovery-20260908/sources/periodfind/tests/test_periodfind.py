"""Tests for the periodfind package.

Tests are split into two categories:
- Unit tests for Statistics/Periodogram (no GPU required)
- Integration tests for CE/AOV/LS algorithms (require GPU + CUDA)

Run with: pytest tests/ -v
GPU tests are skipped automatically if CUDA extensions aren't available.
"""

import subprocess
import warnings

import numpy as np
import pytest

from periodfind import Periodogram, Statistics
from periodfind._utils import ensure_float32, prepare_magnitudes, validate_inputs

# Try importing CUDA-backed modules; skip GPU tests if unavailable
HAS_GPU = False
try:
    from periodfind.gpu import AOV, ConditionalEntropy, LombScargle

    # Verify an actual GPU is reachable by doing a trivial CUDA operation
    ret = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
    if ret.returncode == 0:
        HAS_GPU = True
except (ImportError, FileNotFoundError, subprocess.TimeoutExpired):
    pass

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="CUDA GPU not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sinusoidal_lightcurve(
    period, n_points=500, amplitude=1.0, noise_std=0.05, t_span=100.0, seed=42
):
    """Generate a synthetic sinusoidal light curve with known period."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, t_span, n_points)).astype(np.float32)
    phase = 2.0 * np.pi * times / period
    mags = (amplitude * np.sin(phase) + amplitude).astype(np.float32)  # shift to positive
    mags += rng.normal(0, noise_std, n_points).astype(np.float32)
    return times, mags


def make_trial_periods(true_period, n_periods=200, margin=0.5):
    """Create a trial period grid centered around the true period."""
    lo = max(true_period - margin * true_period, 0.01)
    hi = true_period + margin * true_period
    return np.linspace(lo, hi, n_periods, dtype=np.float32)


# ---------------------------------------------------------------------------
# Utils unit tests
# ---------------------------------------------------------------------------


class TestUtils:
    def test_prepare_magnitudes_normalize(self):
        mags = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        result = prepare_magnitudes(mags, center=False, normalize=True)
        assert len(result) == 1
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0

    def test_prepare_magnitudes_center(self):
        mags = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        result = prepare_magnitudes(mags, center=True, normalize=False)
        assert np.abs(np.mean(result[0])) < 1e-6

    def test_prepare_magnitudes_passthrough(self):
        mags = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        result = prepare_magnitudes(mags, center=False, normalize=False)
        np.testing.assert_array_equal(result[0], mags[0])

    def test_prepare_magnitudes_conflict_warns(self):
        mags = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = prepare_magnitudes(mags, center=True, normalize=True)
            runtime_warns = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warns) >= 1
        # center wins when both are set
        assert np.abs(np.mean(result[0])) < 1e-6

    def test_validate_inputs_ok(self):
        t = [np.array([1.0, 2.0], dtype=np.float32)]
        m = [np.array([3.0, 4.0], dtype=np.float32)]
        validate_inputs(t, m)  # should not raise

    def test_validate_inputs_list_length_mismatch(self):
        t = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
        m = [np.array([5.0, 6.0], dtype=np.float32)]
        with pytest.raises(ValueError, match="same number of light curves"):
            validate_inputs(t, m)

    def test_validate_inputs_array_length_mismatch(self):
        t = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        m = [np.array([4.0, 5.0], dtype=np.float32)]
        with pytest.raises(ValueError, match="different lengths"):
            validate_inputs(t, m)

    def test_ensure_float32_ok(self):
        arrays = [np.array([1.0, 2.0], dtype=np.float32)]
        ensure_float32(arrays, "test")  # should not raise

    def test_ensure_float32_wrong_dtype(self):
        arrays = [np.array([1.0, 2.0], dtype=np.float64)]
        with pytest.raises(TypeError, match="float64.*expected float32"):
            ensure_float32(arrays, "test")


# ---------------------------------------------------------------------------
# Statistics unit tests
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_construction(self):
        s = Statistics([1.0, 0.0], 0.5, mean=1.0, std=0.2, median=1.0, mad=0.15)
        assert s.params == [1.0, 0.0]
        assert s.value == 0.5
        assert s.mean == 1.0
        assert s.std == 0.2
        assert s.median == 1.0
        assert s.mad == 0.15
        assert s.significance_type == "stdmean"

    def test_repr(self):
        s = Statistics([1.0, 0.0], 0.5, mean=1.0, std=0.2, median=1.0, mad=0.15)
        r = repr(s)
        assert "Statistics" in r
        assert "0.5" in r
        assert "stdmean" in r

    def test_significance_stdmean(self):
        s = Statistics([1.0], 3.0, mean=1.0, std=0.5, median=1.0, mad=0.4)
        # |3.0 - 1.0| / 0.5 = 4.0
        assert s.significance == pytest.approx(4.0)

    def test_significance_madmedian(self):
        s = Statistics(
            [1.0], 3.0, mean=1.0, std=0.5, median=1.2, mad=0.3, significance_type="madmedian"
        )
        # |3.0 - 1.2| / 0.3 = 6.0
        assert s.significance == pytest.approx(6.0)

    def test_significance_unknown_type(self):
        s = Statistics(
            [1.0], 3.0, mean=1.0, std=0.5, median=1.0, mad=0.4, significance_type="bogus"
        )
        with pytest.raises(NotImplementedError):
            _ = s.significance

    def test_statistics_from_data_single(self):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 0.5, 6.0]], dtype=np.float32)
        params = [
            np.array([10.0, 20.0], dtype=np.float32),
            np.array([0.0, 0.1, 0.2], dtype=np.float32),
        ]

        # use_max=True => should find 6.0 at (1, 2)
        s = Statistics.statistics_from_data(data, params, use_max=True, n=1)
        assert s.value == pytest.approx(6.0)
        assert s.params == pytest.approx([20.0, 0.2])

    def test_statistics_from_data_min(self):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 0.5, 6.0]], dtype=np.float32)
        params = [
            np.array([10.0, 20.0], dtype=np.float32),
            np.array([0.0, 0.1, 0.2], dtype=np.float32),
        ]

        # use_max=False => should find 0.5 at (1, 1)
        s = Statistics.statistics_from_data(data, params, use_max=False, n=1)
        assert s.value == pytest.approx(0.5)
        assert s.params == pytest.approx([20.0, 0.1])

    def test_statistics_from_data_multiple(self):
        data = np.array([5.0, 1.0, 9.0, 3.0, 7.0], dtype=np.float32)
        params = [np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)]

        results = Statistics.statistics_from_data(data, params, use_max=True, n=3)
        assert isinstance(results, list)
        assert len(results) == 3
        # Should be sorted descending: 9.0, 7.0, 5.0
        assert results[0].value == pytest.approx(9.0)
        assert results[1].value == pytest.approx(7.0)
        assert results[2].value == pytest.approx(5.0)

    def test_statistics_from_data_precomputed_stats(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        params = [np.array([10.0, 20.0, 30.0], dtype=np.float32)]

        s = Statistics.statistics_from_data(
            data, params, use_max=True, n=1, mean=100.0, std=50.0, median=99.0, mad=45.0
        )
        # Should use the precomputed values, not recalculate
        assert s.mean == pytest.approx(100.0)
        assert s.std == pytest.approx(50.0)
        assert s.median == pytest.approx(99.0)
        assert s.mad == pytest.approx(45.0)

    def test_statistics_from_data_computes_stats(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        params = [np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)]

        s = Statistics.statistics_from_data(data, params, use_max=True, n=1)
        assert s.mean == pytest.approx(np.mean(data))
        assert s.std == pytest.approx(np.std(data))
        assert s.median == pytest.approx(np.median(data))


# ---------------------------------------------------------------------------
# Periodogram unit tests
# ---------------------------------------------------------------------------


class TestPeriodogram:
    def test_construction(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        params = [np.array([0.1, 0.2], dtype=np.float32), np.array([0.0, 0.01], dtype=np.float32)]
        p = Periodogram(data, params, use_max=True)
        assert p.use_max is True
        np.testing.assert_array_equal(p.data, data)
        assert len(p.params) == 2

    def test_repr(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        params = [np.array([0.1, 0.2], dtype=np.float32), np.array([0.0, 0.01], dtype=np.float32)]
        p = Periodogram(data, params, use_max=True)
        r = repr(p)
        assert "Periodogram" in r
        assert "2, 2" in r or "(2, 2)" in r
        assert "True" in r

    def test_best_params_max(self):
        data = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        params = [np.array([0.1, 0.2], dtype=np.float32), np.array([0.0, 0.01], dtype=np.float32)]
        p = Periodogram(data, params, use_max=True)
        best = p.best_params(n=1)
        assert best.value == pytest.approx(5.0)
        assert best.params == pytest.approx([0.1, 0.01])

    def test_best_params_min(self):
        data = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        params = [np.array([0.1, 0.2], dtype=np.float32), np.array([0.0, 0.01], dtype=np.float32)]
        p = Periodogram(data, params, use_max=False)
        best = p.best_params(n=1)
        assert best.value == pytest.approx(1.0)
        assert best.params == pytest.approx([0.1, 0.0])

    def test_best_params_multiple(self):
        data = np.array([1.0, 5.0, 3.0, 9.0, 2.0], dtype=np.float32)
        params = [np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)]
        p = Periodogram(data, params, use_max=True)
        results = p.best_params(n=2)
        assert len(results) == 2
        assert results[0].value == pytest.approx(9.0)
        assert results[1].value == pytest.approx(5.0)

    def test_best_params_populates_stats(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        params = [np.array([0.1, 0.2, 0.3], dtype=np.float32)]
        p = Periodogram(data, params, use_max=True)
        assert p.mean is None  # not computed yet
        _ = p.best_params(n=1)
        assert p.mean is not None
        assert p.std is not None
        assert p.median is not None
        assert p.mad is not None

    def test_madmedian_significance(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 100.0], dtype=np.float32)
        params = [np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)]
        p = Periodogram(data, params, use_max=True)
        best = p.best_params(n=1, significance_type="madmedian")
        assert best.significance_type == "madmedian"
        assert best.significance > 0


# ---------------------------------------------------------------------------
# Device API unit tests
# ---------------------------------------------------------------------------


class TestDeviceAPI:
    """Tests for the PyTorch-style device abstraction (set_device / get_device / factories)."""

    def setup_method(self):
        """Reset global device state before each test."""
        import periodfind

        periodfind._default_device = None

    def teardown_method(self):
        """Reset global device state after each test."""
        import periodfind

        periodfind._default_device = None

    def test_set_device_cpu(self):
        import periodfind

        periodfind.set_device("cpu")
        assert periodfind.get_device() == "cpu"

    def test_set_device_gpu(self):
        import periodfind

        periodfind.set_device("gpu")
        assert periodfind.get_device() == "gpu"

    def test_set_device_case_insensitive(self):
        import periodfind

        periodfind.set_device("CPU")
        assert periodfind.get_device() == "cpu"
        periodfind.set_device("Gpu")
        assert periodfind.get_device() == "gpu"

    def test_set_device_invalid_raises(self):
        import periodfind

        with pytest.raises(ValueError, match="Unknown device"):
            periodfind.set_device("tpu")

    def test_resolve_device_explicit(self):
        import periodfind

        assert periodfind._resolve_device("cpu") == "cpu"
        assert periodfind._resolve_device("gpu") == "gpu"
        assert periodfind._resolve_device("CPU") == "cpu"

    def test_resolve_device_invalid_raises(self):
        import periodfind

        with pytest.raises(ValueError, match="Unknown device"):
            periodfind._resolve_device("tpu")

    def test_resolve_device_uses_global(self):
        import periodfind

        periodfind.set_device("cpu")
        assert periodfind._resolve_device() == "cpu"

    def test_factory_cpu(self):
        """Factory functions with device='cpu' should return CPU backend instances."""
        import periodfind
        from periodfind.cpu import (
            AOV as CpuAOV,
        )
        from periodfind.cpu import (
            ConditionalEntropy as CpuCE,
        )
        from periodfind.cpu import (
            LombScargle as CpuLS,
        )

        ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10, device="cpu")
        aov = periodfind.AOV(n_phase=10, device="cpu")
        ls = periodfind.LombScargle(device="cpu")
        assert isinstance(ce, CpuCE)
        assert isinstance(aov, CpuAOV)
        assert isinstance(ls, CpuLS)

    def test_factory_uses_global_device(self):
        """Factory functions should use the global default when no device= given."""
        import periodfind
        from periodfind.cpu import ConditionalEntropy as CpuCE

        periodfind.set_device("cpu")
        ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10)
        assert isinstance(ce, CpuCE)

    def test_factory_override_beats_global(self):
        """Per-call device= should override the global default."""
        import periodfind
        from periodfind.cpu import ConditionalEntropy as CpuCE

        periodfind.set_device("gpu")  # global says gpu
        ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10, device="cpu")
        assert isinstance(ce, CpuCE)

    @requires_gpu
    def test_factory_gpu(self):
        """Factory functions with device='gpu' should return GPU backend instances."""
        import periodfind
        from periodfind.gpu import (
            AOV as GpuAOV,
        )
        from periodfind.gpu import (
            ConditionalEntropy as GpuCE,
        )
        from periodfind.gpu import (
            LombScargle as GpuLS,
        )

        ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10, device="gpu")
        aov = periodfind.AOV(n_phase=10, device="gpu")
        ls = periodfind.LombScargle(device="gpu")
        assert isinstance(ce, GpuCE)
        assert isinstance(aov, GpuAOV)
        assert isinstance(ls, GpuLS)

    def test_factory_forwards_kwargs(self):
        """Factory should forward constructor kwargs to backend class."""
        import periodfind

        periodfind.set_device("cpu")
        ce = periodfind.ConditionalEntropy(
            n_phase=20, n_mag=5, phase_bin_extent=2, mag_bin_extent=3
        )
        assert ce.n_phase == 20
        assert ce.n_mag == 5
        assert ce.phase_bin_extent == 2
        assert ce.mag_bin_extent == 3

    def test_factory_ce_cpu_runs(self):
        """CPU ConditionalEntropy via factory should produce valid results."""
        import periodfind

        periodfind.set_device("cpu")
        ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10)
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 50, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)
        result = ce.calc([t], [m], periods, period_dts, output="stats")
        assert isinstance(result[0], Statistics)

    def test_auto_detect_returns_valid_device(self):
        """Auto-detect (no explicit device) should return 'cpu' or 'gpu'."""
        import periodfind

        periodfind._default_device = None
        device = periodfind.get_device()
        assert device in ("cpu", "gpu")


# ---------------------------------------------------------------------------
# GPU integration tests — Conditional Entropy
# ---------------------------------------------------------------------------


@requires_gpu
class TestConditionalEntropy:
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
        # Allow 5% tolerance
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
        # n_stats > 1 means each entry is a list
        assert isinstance(result[0], list)
        assert len(result[0]) == 5

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

    def test_custom_bin_params(self):
        """CE with different bin settings should still run."""
        t, m = make_sinusoidal_lightcurve(period=3.0)
        periods = np.linspace(1.0, 10.0, 50, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        ce = ConditionalEntropy(n_phase=20, n_mag=5, phase_bin_extent=2, mag_bin_extent=2)
        result = ce.calc([t], [m], periods, period_dts, output="stats")
        assert isinstance(result[0], Statistics)


# ---------------------------------------------------------------------------
# GPU integration tests — Analysis of Variance
# ---------------------------------------------------------------------------


@requires_gpu
class TestAOV:
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
        assert pgram.use_max is True  # AOV uses maxima

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
        # Shift mags to a large range to test normalization
        m_shifted = m + 1000.0
        periods = np.linspace(1.0, 10.0, 100, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV()
        result = aov.calc([t], [m_shifted], periods, period_dts, normalize=True, output="stats")
        assert isinstance(result[0], Statistics)


# ---------------------------------------------------------------------------
# GPU integration tests — Lomb-Scargle
# ---------------------------------------------------------------------------


@requires_gpu
class TestLombScargle:
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
        assert pgram.use_max is True  # LS uses maxima

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
# Cross-algorithm consistency tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestCrossAlgorithm:
    """All three algorithms should agree on the period for clean signals."""

    def test_all_algorithms_find_same_period(self):
        true_period = 4.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=1000, noise_std=0.01, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        ce = ConditionalEntropy(n_phase=15, n_mag=10)
        aov = AOV(n_phase=15)
        ls_algo = LombScargle()

        ce_result = ce.calc([t], [m], periods, period_dts, output="stats")
        aov_result = aov.calc([t], [m], periods, period_dts, output="stats")
        ls_result = ls_algo.calc([t], [m], periods, period_dts, output="stats")

        tol = 0.05  # 5% tolerance
        for name, res in [("CE", ce_result), ("AOV", aov_result), ("LS", ls_result)]:
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


@requires_gpu
class TestEdgeCases:
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
# Realistic light curve generators (LSST-relevant)
# ---------------------------------------------------------------------------


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
    # V-shaped primary eclipse centered at phase=0
    dist = np.minimum(phase, 1.0 - phase)  # distance from phase 0
    mags = np.ones(n_points, dtype=np.float32)
    in_eclipse = dist < eclipse_width / 2
    mags[in_eclipse] = 1.0 - eclipse_depth * (1.0 - dist[in_eclipse] / (eclipse_width / 2))
    mags += rng.normal(0, noise_std, n_points).astype(np.float32)
    return times, mags


def make_rr_lyrae(period, n_points=500, amplitude=0.8, noise_std=0.02, t_span=200.0, seed=42):
    """Generate an RR Lyrae-like sawtooth light curve (fast rise, slow decline)."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, t_span, n_points)).astype(np.float32)
    phase = (times / period) % 1.0
    # Sawtooth: rapid rise (phase 0.0 to 0.2), slow decline (0.2 to 1.0)
    rise_end = 0.2
    mags = np.where(
        phase < rise_end,
        amplitude * (phase / rise_end),
        amplitude * (1.0 - (phase - rise_end) / (1.0 - rise_end)),
    ).astype(np.float32)
    mags += rng.normal(0, noise_std, n_points).astype(np.float32)
    return times, mags


def make_lsst_cadence(
    period,
    waveform="sinusoidal",
    n_years=10,
    points_per_season=60,
    amplitude=1.0,
    noise_std=0.05,
    seed=42,
):
    """Generate a light curve with LSST-like irregular cadence and seasonal gaps.

    Simulates ~6 months on / ~6 months off per year, with irregular spacing
    within observing seasons. Typical total: 200-600 points over 10 years.
    """
    rng = np.random.default_rng(seed)
    all_times = []
    for year in range(n_years):
        # Observing season: ~180 days starting around day 60 of each year
        season_start = year * 365.25 + 60
        season_end = season_start + 180
        n_pts = rng.poisson(points_per_season)
        n_pts = max(n_pts, 10)
        season_times = np.sort(rng.uniform(season_start, season_end, n_pts))
        all_times.append(season_times)
    times = np.concatenate(all_times).astype(np.float32)
    phase = 2.0 * np.pi * times / period
    if waveform == "sinusoidal":
        mags = (amplitude * np.sin(phase) + amplitude).astype(np.float32)
    elif waveform == "eclipsing_binary":
        ph = (times / period) % 1.0
        dist = np.minimum(ph, 1.0 - ph)
        mags = np.ones(len(times), dtype=np.float32)
        eclipse_width = 0.1
        in_eclipse = dist < eclipse_width / 2
        mags[in_eclipse] = (
            1.0 - amplitude * (1.0 - dist[in_eclipse] / (eclipse_width / 2))
        ).astype(np.float32)
    elif waveform == "rr_lyrae":
        ph = (times / period) % 1.0
        rise_end = 0.2
        mags = np.where(
            ph < rise_end,
            amplitude * (ph / rise_end),
            amplitude * (1.0 - (ph - rise_end) / (1.0 - rise_end)),
        ).astype(np.float32)
    else:
        raise ValueError(f"Unknown waveform: {waveform}")
    mags += rng.normal(0, noise_std, len(times)).astype(np.float32)
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
# AOV overlap bug fix test
# ---------------------------------------------------------------------------


@requires_gpu
class TestAOVOverlap:
    def test_overlap_differs_from_no_overlap(self):
        """AOV with phase_bin_extent > 1 should produce different (smoother) results."""
        true_period = 3.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=800, noise_std=0.03, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=300)
        period_dts = np.array([0.0], dtype=np.float32)

        aov_no_overlap = AOV(n_phase=10, phase_bin_extent=1)
        aov_overlap = AOV(n_phase=10, phase_bin_extent=3)

        pgram_no = aov_no_overlap.calc([t], [m], periods, period_dts, output="periodogram")
        pgram_ov = aov_overlap.calc([t], [m], periods, period_dts, output="periodogram")

        data_no = pgram_no[0].data.ravel()
        data_ov = pgram_ov[0].data.ravel()

        # They should not be identical
        assert not np.allclose(data_no, data_ov, atol=1e-6), (
            "Overlap and no-overlap periodograms should differ"
        )

        # The overlap version should be smoother (lower variance of diff)
        diff_no = np.diff(data_no)
        diff_ov = np.diff(data_ov)
        assert np.std(diff_ov) < np.std(diff_no), (
            "Overlap periodogram should be smoother (lower diff variance)"
        )

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
# LSST Scenarios
# ---------------------------------------------------------------------------


@requires_gpu
class TestLSSTScenarios:
    """Period detection on realistic LSST-like light curves."""

    def test_eclipsing_binary_detection(self):
        """All algorithms should detect an eclipsing binary period."""
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
        """All algorithms should detect an RR Lyrae period."""
        true_period = 0.6  # typical RR Lyrae period in days
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

    def test_lsst_cadence_sinusoidal(self):
        """Period detection with LSST-like cadence (seasonal gaps)."""
        true_period = 5.0
        t, m = make_lsst_cadence(
            period=true_period,
            waveform="sinusoidal",
            n_years=10,
            points_per_season=80,
            amplitude=1.0,
            noise_std=0.05,
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV(n_phase=15)
        result = aov.calc([t], [m], periods, period_dts, output="stats")
        detected = result[0].params[0]
        assert period_matches(detected, true_period), (
            f"AOV detected {detected}, expected ~{true_period}"
        )

    def test_lsst_cadence_eclipsing_binary(self):
        """Eclipsing binary detection with LSST cadence."""
        true_period = 3.0
        t, m = make_lsst_cadence(
            period=true_period,
            waveform="eclipsing_binary",
            n_years=10,
            points_per_season=80,
            amplitude=0.5,
            noise_std=0.02,
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV(n_phase=20)
        result = aov.calc([t], [m], periods, period_dts, output="stats")
        detected = result[0].params[0]
        assert period_matches(detected, true_period), (
            f"AOV detected {detected}, expected ~{true_period}"
        )

    def test_lsst_cadence_rr_lyrae(self):
        """RR Lyrae detection with LSST cadence."""
        true_period = 0.55
        t, m = make_lsst_cadence(
            period=true_period,
            waveform="rr_lyrae",
            n_years=10,
            points_per_season=80,
            amplitude=0.8,
            noise_std=0.03,
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV(n_phase=20)
        result = aov.calc([t], [m], periods, period_dts, output="stats")
        detected = result[0].params[0]
        assert period_matches(detected, true_period), (
            f"AOV detected {detected}, expected ~{true_period}"
        )


# ---------------------------------------------------------------------------
# Harmonics and aliasing tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestHarmonicsAndAliasing:
    """Verify behavior at harmonic and alias periods."""

    def test_harmonics_detected(self):
        """Best period should be the true period or a known harmonic (P/2, 2P)."""
        true_period = 4.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=800, noise_std=0.02, t_span=200.0
        )
        # Wide search grid covering harmonics
        periods = np.linspace(1.0, 12.0, 1000, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        for AlgoCls in [ConditionalEntropy, AOV, LombScargle]:
            if AlgoCls == ConditionalEntropy:
                algo = AlgoCls(n_phase=15, n_mag=10)
            elif AlgoCls == AOV:
                algo = AlgoCls(n_phase=15)
            else:
                algo = AlgoCls()
            result = algo.calc([t], [m], periods, period_dts, output="stats")
            detected = result[0].params[0]
            assert period_matches(detected, true_period), (
                f"{AlgoCls.__name__}: detected {detected}, expected ~{true_period} or harmonic"
            )

    def test_half_period_harmonic(self):
        """Non-sinusoidal waveforms may show power at P/2."""
        true_period = 3.0
        t, m = make_eclipsing_binary(
            period=true_period, n_points=800, eclipse_depth=0.5, noise_std=0.02, t_span=200.0
        )
        # Search grid includes both P and P/2
        periods = np.linspace(0.5, 6.0, 1000, dtype=np.float32)
        period_dts = np.array([0.0], dtype=np.float32)

        aov = AOV(n_phase=20)
        result = aov.calc([t], [m], periods, period_dts, output="stats")
        detected = result[0].params[0]
        # Should find true period or P/2 harmonic
        assert period_matches(
            detected, true_period, harmonics=[true_period / 2, 2 * true_period]
        ), f"AOV detected {detected}, expected ~{true_period} or harmonic"


# ---------------------------------------------------------------------------
# Noise robustness tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestNoiseRobustness:
    """Test period recovery across a range of signal-to-noise ratios."""

    def test_high_snr(self):
        """High SNR: all algorithms should recover the period easily."""
        true_period = 4.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=800, amplitude=1.0, noise_std=0.01, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        for AlgoCls in [ConditionalEntropy, AOV, LombScargle]:
            if AlgoCls == ConditionalEntropy:
                algo = AlgoCls(n_phase=15, n_mag=10)
            elif AlgoCls == AOV:
                algo = AlgoCls(n_phase=15)
            else:
                algo = AlgoCls()
            result = algo.calc([t], [m], periods, period_dts, output="stats")
            detected = result[0].params[0]
            assert abs(detected - true_period) / true_period < 0.02, (
                f"{AlgoCls.__name__} high-SNR: detected {detected}"
            )

    def test_medium_snr(self):
        """Medium SNR: algorithms should still recover the period."""
        true_period = 4.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=800, amplitude=1.0, noise_std=0.2, t_span=200.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        for AlgoCls in [AOV, LombScargle]:
            algo = AlgoCls() if AlgoCls == LombScargle else AlgoCls(n_phase=15)
            result = algo.calc([t], [m], periods, period_dts, output="stats")
            detected = result[0].params[0]
            assert period_matches(detected, true_period), (
                f"{AlgoCls.__name__} medium-SNR: detected {detected}"
            )

    def test_low_snr(self):
        """Low SNR: the best period should still be in the right neighborhood."""
        true_period = 4.0
        t, m = make_sinusoidal_lightcurve(
            period=true_period, n_points=1000, amplitude=1.0, noise_std=0.5, t_span=300.0
        )
        periods = make_trial_periods(true_period, n_periods=500)
        period_dts = np.array([0.0], dtype=np.float32)

        ls = LombScargle()
        result = ls.calc([t], [m], periods, period_dts, output="stats")
        detected = result[0].params[0]
        # Relax tolerance for low SNR — 10% or harmonic
        assert period_matches(detected, true_period, tol=0.10), (
            f"LS low-SNR: detected {detected}, expected ~{true_period}"
        )


# ---------------------------------------------------------------------------
# Large-scale tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestLargeScale:
    """Benchmark-style tests with large period grids and batched light curves."""

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

        # At least 80% of curves should have their period detected
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

        # Check that the periodogram at pdt=0 (middle column) peaks near true_period
        pdt_zero_col = pgram.data[:, 1]  # middle column = period_dt=0
        best_idx = np.argmax(pdt_zero_col)
        detected = periods[best_idx]
        assert period_matches(detected, true_period), (
            f"LS large-grid+dts: detected {detected}, expected ~{true_period}"
        )
