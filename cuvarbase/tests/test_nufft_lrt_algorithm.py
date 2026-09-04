"""
Test NUFFT LRT algorithm logic without requiring GPU.

These tests exercise the *shipped* template-generation and matched-filter
code in cuvarbase.nufft_lrt (both are pure numpy). An earlier version of
this file defined local copies of the algorithms and tested those, which
validated nothing about the package.
"""
import numpy as np
import pytest

from ..nufft_lrt import NUFFTLRTAsyncProcess

pytestmark = pytest.mark.filterwarnings(
    "ignore:cuvarbase.nufft_lrt is EXPERIMENTAL")


@pytest.fixture(scope='module')
def proc():
    return NUFFTLRTAsyncProcess()


def test_smoothed_periodogram_clamps_the_window():
    # CPU-only: np.convolve(..., 'same') returns max(len, window)
    # samples, so an unclamped window > nf lengthened the PSD and blew
    # up downstream with a raw numpy broadcast error.
    from ..nufft_lrt import _smoothed_periodogram
    for n in (1, 2, 3, 4, 5, 6, 7, 33):
        p = np.arange(1.0, n + 1.0)
        for window in (1, 2, 5, 1000):
            out = _smoothed_periodogram(p, window)
            assert len(out) == n, (n, window)
            assert np.all(np.isfinite(out))
        # any window >= n gives the same (fully clamped) result
        assert np.allclose(_smoothed_periodogram(p, 1000),
                           _smoothed_periodogram(p, n))
        # ... and smoothing preserves the total (edge-corrected mean of
        # the available neighbours, never zero-padded)
        assert _smoothed_periodogram(p, 3).min() >= p.min()
        assert _smoothed_periodogram(p, 3).max() <= p.max()


def test_run_docstring_warns_about_the_duration_outer_product():
    doc = ' '.join(NUFFTLRTAsyncProcess.run.__doc__.split())
    assert 'outer product' in doc
    assert 'len(periods)**2' in doc
    assert '0.1 * periods' in doc


class TestNUFFTLRTAlgorithm:
    """Test NUFFT LRT algorithm logic (CPU-only, real implementation)"""

    def test_template_generation(self, proc):
        """Test transit template generation"""
        t = np.linspace(0, 10, 100)
        period = 2.0
        epoch = 0.0
        duration = 0.2
        depth = 1.0

        template = proc._generate_template(t, period, epoch, duration, depth)

        # Check properties
        assert len(template) == len(t)
        assert np.min(template) == -depth
        assert np.max(template) == 0.0

        # Check that some points are in transit
        in_transit = template < 0
        assert np.sum(in_transit) > 0
        assert np.sum(in_transit) < len(template)

        # Check expected number of points in transit
        expected_fraction = duration / period
        actual_fraction = np.sum(in_transit) / len(template)

        # Should be roughly correct (within factor of 2)
        assert 0.5 * expected_fraction < actual_fraction < 2.0 * expected_fraction

    def test_matched_filter_perfect_match(self, proc):
        """Test matched filter with perfect match gives high SNR"""
        nf = 100

        # Perfect match should give high SNR
        rng = np.random.RandomState(0)
        T = rng.randn(nf) + 1j * rng.randn(nf)
        Y = T.copy()  # Perfect match
        P_s = np.ones(nf)
        weights = np.ones(nf)

        snr = proc._compute_matched_filter_snr(Y, T, P_s, weights, 1e-12)

        # Perfect match should give SNR ~ sqrt(sum(|T|^2))
        expected_snr = np.sqrt(np.sum(np.abs(T) ** 2))
        assert np.abs(snr - expected_snr) / expected_snr < 0.01

    def test_matched_filter_orthogonal_signals(self, proc):
        """Test matched filter with orthogonal signals gives low SNR"""
        nf = 100

        rng = np.random.RandomState(1)
        T = rng.randn(nf) + 1j * rng.randn(nf)
        Y = rng.randn(nf) + 1j * rng.randn(nf)
        Y = Y - np.vdot(Y, T) * T / np.vdot(T, T)  # Make orthogonal

        P_s = np.ones(nf)
        weights = np.ones(nf)

        snr = proc._compute_matched_filter_snr(Y, T, P_s, weights, 1e-12)

        # Orthogonal signals should give SNR ~ 0
        assert np.abs(snr) < 1.0

    def test_matched_filter_scale_invariance(self, proc):
        """Test matched filter is invariant to template scaling"""
        nf = 100

        rng = np.random.RandomState(2)
        T = rng.randn(nf) + 1j * rng.randn(nf)
        Y = 2.0 * T  # Scaled version
        P_s = np.ones(nf)
        weights = np.ones(nf)

        snr1 = proc._compute_matched_filter_snr(Y, T, P_s, weights, 1e-12)
        snr2 = proc._compute_matched_filter_snr(Y, 0.5 * T, P_s, weights,
                                                1e-12)

        # SNR should be invariant to template scaling
        assert np.abs(snr1 - snr2) < 0.01

    def test_matched_filter_noise_distribution(self, proc):
        """Test matched filter gives reasonable SNR distribution for noise"""
        nf = 100
        P_s = np.ones(nf)
        weights = np.ones(nf)

        snrs = []
        rng = np.random.RandomState(42)
        for _ in range(50):
            Y = rng.randn(nf) + 1j * rng.randn(nf)
            T = rng.randn(nf) + 1j * rng.randn(nf)
            snr = proc._compute_matched_filter_snr(Y, T, P_s, weights, 1e-12)
            snrs.append(snr)

        mean_snr = np.mean(snrs)
        std_snr = np.std(snrs)

        # Mean should be close to 0, std should be reasonable
        assert np.abs(mean_snr) < 2.0
        assert std_snr > 0

    def test_power_spectrum_floor_prevents_blowup(self, proc):
        """Zero entries in the power spectrum must not produce inf/nan"""
        nf = 100
        rng = np.random.RandomState(3)
        T = rng.randn(nf) + 1j * rng.randn(nf)
        Y = T.copy()
        weights = np.ones(nf)

        P_s = np.ones(nf)
        P_s[::7] = 0.0  # exact zeros, would divide-by-zero without floor

        snr = proc._compute_matched_filter_snr(Y, T, P_s, weights, 1e-6)
        assert np.isfinite(snr)
        assert snr > 0

    def test_matched_filter_with_colored_noise(self, proc):
        """Test matched filter with non-uniform power spectrum"""
        nf = 100

        rng = np.random.RandomState(4)
        # Create frequency-dependent noise (colored noise)
        P_s = np.linspace(0.5, 2.0, nf)  # Varying power
        weights = np.ones(nf)

        T = rng.randn(nf) + 1j * rng.randn(nf)
        Y = T + np.sqrt(P_s) * (rng.randn(nf) + 1j * rng.randn(nf))

        snr = proc._compute_matched_filter_snr(Y, T, P_s, weights, 1e-12)

        # SNR should be positive and finite
        assert snr > 0
        assert np.isfinite(snr)
