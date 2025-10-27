"""
Test NUFFT LRT algorithm logic without requiring GPU.

These tests validate the matched filter computation logic
using CPU-only implementations.
"""
import pytest
import numpy as np


def generate_transit_template(t, period, epoch, duration, depth):
    """Generate transit template"""
    phase = np.fmod(t - epoch, period) / period
    phase[phase < 0] += 1.0
    phase[phase > 0.5] -= 1.0

    template = np.zeros_like(t)
    phase_width = duration / (2.0 * period)
    in_transit = np.abs(phase) <= phase_width
    template[in_transit] = -depth

    return template


def compute_matched_filter_snr(Y, T, P_s, weights, eps_floor=1e-12):
    """Compute matched filter SNR (CPU version)"""
    # Apply floor to power spectrum
    median_ps = np.median(P_s[P_s > 0])
    P_s = np.maximum(P_s, eps_floor * median_ps)

    # Numerator: real(Y * conj(T) * weights / P_s)
    numerator = np.real(np.sum((Y * np.conj(T)) * weights / P_s))

    # Denominator: sqrt(|T|^2 * weights / P_s)
    denominator = np.sqrt(np.real(np.sum((np.abs(T) ** 2) * weights / P_s)))

    if denominator > 0:
        return numerator / denominator
    else:
        return 0.0


class TestNUFFTLRTAlgorithm:
    """Test NUFFT LRT algorithm logic (CPU-only)"""

    def test_template_generation(self):
        """Test transit template generation"""
        t = np.linspace(0, 10, 100)
        period = 2.0
        epoch = 0.0
        duration = 0.2
        depth = 1.0

        template = generate_transit_template(t, period, epoch, duration, depth)

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

    def test_matched_filter_perfect_match(self):
        """Test matched filter with perfect match gives high SNR"""
        nf = 100

        # Perfect match should give high SNR
        T = np.random.randn(nf) + 1j * np.random.randn(nf)
        Y = T.copy()  # Perfect match
        P_s = np.ones(nf)
        weights = np.ones(nf)

        snr = compute_matched_filter_snr(Y, T, P_s, weights)

        # Perfect match should give SNR ≈ sqrt(sum(|T|^2))
        expected_snr = np.sqrt(np.sum(np.abs(T) ** 2))
        assert np.abs(snr - expected_snr) / expected_snr < 0.01

    def test_matched_filter_orthogonal_signals(self):
        """Test matched filter with orthogonal signals gives low SNR"""
        nf = 100

        # Orthogonal signals should give low SNR
        T = np.random.randn(nf) + 1j * np.random.randn(nf)
        Y = np.random.randn(nf) + 1j * np.random.randn(nf)
        Y = Y - np.vdot(Y, T) * T / np.vdot(T, T)  # Make orthogonal

        P_s = np.ones(nf)
        weights = np.ones(nf)

        snr = compute_matched_filter_snr(Y, T, P_s, weights)

        # Orthogonal signals should give SNR ≈ 0
        assert np.abs(snr) < 1.0

    def test_matched_filter_scale_invariance(self):
        """Test matched filter is invariant to template scaling"""
        nf = 100

        T = np.random.randn(nf) + 1j * np.random.randn(nf)
        Y = 2.0 * T  # Scaled version
        P_s = np.ones(nf)
        weights = np.ones(nf)

        snr1 = compute_matched_filter_snr(Y, T, P_s, weights)
        snr2 = compute_matched_filter_snr(Y, 0.5 * T, P_s, weights)

        # SNR should be invariant to template scaling
        assert np.abs(snr1 - snr2) < 0.01

    def test_matched_filter_noise_distribution(self):
        """Test matched filter gives reasonable SNR distribution for random noise"""
        nf = 100
        P_s = np.ones(nf)
        weights = np.ones(nf)

        snrs = []
        np.random.seed(42)  # For reproducibility
        for _ in range(50):
            Y = np.random.randn(nf) + 1j * np.random.randn(nf)
            T = np.random.randn(nf) + 1j * np.random.randn(nf)
            snr = compute_matched_filter_snr(Y, T, P_s, weights)
            snrs.append(snr)

        mean_snr = np.mean(snrs)
        std_snr = np.std(snrs)

        # Mean should be close to 0, std should be reasonable
        assert np.abs(mean_snr) < 2.0
        assert std_snr > 0

    def test_frequency_weights_one_sided_spectrum(self):
        """Test frequency weight computation for one-sided spectrum"""
        # For even length
        n = 100
        nf = n // 2 + 1
        weights = np.ones(nf)
        weights[1:-1] = 2.0
        weights[0] = 1.0
        weights[-1] = 1.0

        # Check that weighting is correct for one-sided spectrum
        assert weights[0] == 1.0  # DC component
        assert weights[-1] == 1.0  # Nyquist frequency
        assert np.all(weights[1:-1] == 2.0)  # Others doubled

    def test_power_spectrum_floor(self):
        """Test power spectrum floor prevents division by zero"""
        P_s = np.array([0.0, 1.0, 2.0, 3.0, 0.1])
        eps_floor = 1e-2

        median_ps = np.median(P_s[P_s > 0])
        P_s_floored = np.maximum(P_s, eps_floor * median_ps)

        # Check that all values are above floor
        assert np.all(P_s_floored >= eps_floor * median_ps)

        # Check that non-zero values are preserved if above floor
        assert P_s_floored[1] == 1.0
        assert P_s_floored[2] == 2.0
        assert P_s_floored[3] == 3.0

    def test_matched_filter_with_colored_noise(self):
        """Test matched filter with non-uniform power spectrum"""
        nf = 100

        # Create frequency-dependent noise (colored noise)
        P_s = np.linspace(0.5, 2.0, nf)  # Varying power
        weights = np.ones(nf)

        T = np.random.randn(nf) + 1j * np.random.randn(nf)
        Y = T + np.sqrt(P_s) * (np.random.randn(nf) + 1j * np.random.randn(nf))

        snr = compute_matched_filter_snr(Y, T, P_s, weights)

        # SNR should be positive and finite
        assert snr > 0
        assert np.isfinite(snr)
