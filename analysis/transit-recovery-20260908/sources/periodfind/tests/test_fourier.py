"""Tests for the Fourier decomposition feature extraction (CPU backend).

Run with: pytest tests/test_fourier.py -v
"""

import numpy as np
import pytest

import periodfind
from periodfind.cpu import FourierDecomposition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_FEATURES = 14  # [power, BIC, offset, slope, A1, B1, ..., A5, B5]


def make_sinusoidal(period, n=200, offset=15.0, A1=1.0, B1=0.5, noise=0.0, seed=42):
    """Generate y = offset + A1*cos(phi) + B1*sin(phi) + noise."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0, 10 * period, n)).astype(np.float32)
    phi = 2.0 * np.pi * times / period
    mags = (offset + A1 * np.cos(phi) + B1 * np.sin(phi)).astype(np.float32)
    if noise > 0:
        mags += rng.normal(0, noise, n).astype(np.float32)
    errs = np.full(n, max(noise, 0.01), dtype=np.float32)
    return times, mags, errs


# ---------------------------------------------------------------------------
# FourierDecomposition class tests
# ---------------------------------------------------------------------------


class TestFourierDecompositionOutputShape:
    """Verify output shapes and dtypes."""

    def test_single_curve(self):
        fd = FourierDecomposition()
        t, m, e = make_sinusoidal(3.0)
        result = fd.calc([t], [m], [e], np.array([3.0], dtype=np.float32))
        assert result.shape == (1, NUM_FEATURES)
        assert result.dtype == np.float32

    def test_batch_of_curves(self):
        fd = FourierDecomposition()
        curves = [make_sinusoidal(3.0, seed=i) for i in range(5)]
        times = [c[0] for c in curves]
        mags = [c[1] for c in curves]
        errs = [c[2] for c in curves]
        periods = np.array([3.0] * 5, dtype=np.float32)

        result = fd.calc(times, mags, errs, periods)
        assert result.shape == (5, NUM_FEATURES)

    def test_empty_batch(self):
        fd = FourierDecomposition()
        result = fd.calc([], [], [], np.array([], dtype=np.float32))
        assert result.shape == (0, NUM_FEATURES)


class TestFourierDecompositionKnownSignals:
    """Test recovery of known Fourier coefficients."""

    def test_pure_cosine(self):
        """y = 10 + 2*cos(phi): should recover offset=10, A1~2, B1~0."""
        fd = FourierDecomposition()
        t, m, e = make_sinusoidal(5.0, offset=10.0, A1=2.0, B1=0.0, n=300)
        result = fd.calc([t], [m], [e], np.array([5.0], dtype=np.float32))
        r = result[0]

        assert r[0] > 0.9, f"power = {r[0]}"  # high power
        assert abs(r[2] - 10.0) < 0.1, f"offset = {r[2]}"
        assert abs(r[3]) < 0.01, f"slope = {r[3]}"
        assert abs(r[4] - 2.0) < 0.1, f"A1 = {r[4]}"
        assert abs(r[5]) < 0.1, f"B1 = {r[5]}"

    def test_pure_sine(self):
        """y = 15 + 1.5*sin(phi): offset=15, A1~0, B1~1.5."""
        fd = FourierDecomposition()
        t, m, e = make_sinusoidal(4.0, offset=15.0, A1=0.0, B1=1.5, n=300)
        result = fd.calc([t], [m], [e], np.array([4.0], dtype=np.float32))
        r = result[0]

        assert r[0] > 0.9
        assert abs(r[2] - 15.0) < 0.1
        assert abs(r[4]) < 0.1, f"A1 = {r[4]}"
        assert abs(r[5] - 1.5) < 0.1, f"B1 = {r[5]}"

    def test_constant_signal(self):
        """Constant magnitude: power should be ~0."""
        fd = FourierDecomposition()
        n = 100
        times = np.arange(n, dtype=np.float32) * 0.5
        mags = np.full(n, 18.0, dtype=np.float32)
        errs = np.full(n, 0.01, dtype=np.float32)

        result = fd.calc([times], [mags], [errs], np.array([2.0], dtype=np.float32))
        r = result[0]

        assert abs(r[0]) < 0.01, f"power = {r[0]}"
        assert abs(r[2] - 18.0) < 0.1, f"offset = {r[2]}"

    def test_two_harmonics(self):
        """y = 10 + cos(phi) + 0.3*cos(2*phi): should recover both harmonics."""
        fd = FourierDecomposition()
        n = 400
        period = 3.0
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(0, 30.0, n)).astype(np.float32)
        phi = 2.0 * np.pi * times / period
        mags = (10.0 + 1.0 * np.cos(phi) + 0.3 * np.cos(2 * phi)).astype(np.float32)
        errs = np.full(n, 0.01, dtype=np.float32)

        result = fd.calc([times], [mags], [errs], np.array([period], dtype=np.float32))
        r = result[0]

        assert r[0] > 0.9
        assert abs(r[4] - 1.0) < 0.1, f"A1 = {r[4]}"
        assert abs(r[5]) < 0.1, f"B1 = {r[5]}"
        assert abs(r[6] - 0.3) < 0.1, f"A2 = {r[6]}"
        assert abs(r[7]) < 0.1, f"B2 = {r[7]}"


class TestFourierDecompositionEdgeCases:
    """Edge cases and degenerate inputs."""

    def test_too_few_points_returns_nan(self):
        """Fewer than 3 points should return all NaN."""
        fd = FourierDecomposition()
        t = np.array([1.0, 2.0], dtype=np.float32)
        m = np.array([10.0, 11.0], dtype=np.float32)
        e = np.array([0.1, 0.1], dtype=np.float32)

        result = fd.calc([t], [m], [e], np.array([1.0], dtype=np.float32))
        assert np.all(np.isnan(result[0]))

    def test_nan_in_batch_does_not_corrupt_others(self):
        """A degenerate curve in a batch shouldn't affect good curves."""
        fd = FourierDecomposition()
        t_good, m_good, e_good = make_sinusoidal(3.0, n=100)
        t_bad = np.array([1.0], dtype=np.float32)
        m_bad = np.array([10.0], dtype=np.float32)
        e_bad = np.array([0.1], dtype=np.float32)

        result = fd.calc(
            [t_good, t_bad, t_good],
            [m_good, m_bad, m_good],
            [e_good, e_bad, e_good],
            np.array([3.0, 1.0, 3.0], dtype=np.float32),
        )

        assert result.shape == (3, NUM_FEATURES)
        assert np.all(np.isnan(result[1]))
        assert np.all(np.isfinite(result[0]))
        assert np.all(np.isfinite(result[2]))

    def test_noisy_signal(self):
        """With noise, Fourier should still detect the dominant signal."""
        fd = FourierDecomposition()
        t, m, e = make_sinusoidal(3.0, offset=15.0, A1=1.0, B1=0.0, noise=0.1, n=500)
        result = fd.calc([t], [m], [e], np.array([3.0], dtype=np.float32))

        assert result[0, 0] > 0.5, f"power with noise = {result[0, 0]}"
        assert abs(result[0, 2] - 15.0) < 0.5


class TestFourierDecompositionInputValidation:
    """Input validation in the Python wrapper."""

    def test_mismatched_lengths_raises(self):
        fd = FourierDecomposition()
        t = [np.array([1, 2, 3], dtype=np.float32)]
        m = [np.array([1, 2, 3], dtype=np.float32)]
        e = [np.array([0.1, 0.1], dtype=np.float32)]  # wrong length

        with pytest.raises(ValueError, match="different lengths"):
            fd.calc(t, m, e, np.array([1.0], dtype=np.float32))

    def test_wrong_number_of_errs_raises(self):
        fd = FourierDecomposition()
        t = [np.array([1, 2, 3], dtype=np.float32)]
        m = [np.array([1, 2, 3], dtype=np.float32)]
        e = []  # no errs

        with pytest.raises(ValueError, match="same number"):
            fd.calc(t, m, e, np.array([1.0], dtype=np.float32))

    def test_wrong_dtype_raises(self):
        fd = FourierDecomposition()
        t = [np.array([1, 2, 3], dtype=np.float64)]  # wrong dtype
        m = [np.array([1, 2, 3], dtype=np.float32)]
        e = [np.array([0.1, 0.1, 0.1], dtype=np.float32)]

        with pytest.raises(TypeError, match="float32"):
            fd.calc(t, m, e, np.array([1.0], dtype=np.float32))

    def test_periods_shape_mismatch_raises(self):
        fd = FourierDecomposition()
        t, m, e = make_sinusoidal(3.0)

        with pytest.raises(ValueError, match="one entry per curve"):
            fd.calc([t], [m], [e], np.array([1.0, 2.0], dtype=np.float32))


class TestFourierDecompositionFactory:
    """Test the top-level periodfind.FourierDecomposition factory."""

    def test_cpu_factory(self):
        fd = periodfind.FourierDecomposition(device="cpu")
        assert isinstance(fd, FourierDecomposition)

    def test_gpu_falls_back_with_warning(self):
        with pytest.warns(RuntimeWarning, match="falling back to CPU"):
            fd = periodfind.FourierDecomposition(device="gpu")
        assert isinstance(fd, FourierDecomposition)
