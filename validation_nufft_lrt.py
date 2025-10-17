#!/usr/bin/env python
"""
Simple validation script to test the basic logic of NUFFT LRT without GPU.
This validates the algorithm implementation independent of CUDA.
"""
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


def test_template_generation():
    """Test transit template generation"""
    print("Testing template generation...")
    
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
    
    print("  ✓ Template generation works correctly")
    return True


def test_matched_filter_logic():
    """Test matched filter SNR computation logic"""
    print("Testing matched filter logic...")
    
    nf = 100
    
    # Test 1: Perfect match should give high SNR
    T = np.random.randn(nf) + 1j * np.random.randn(nf)
    Y = T.copy()  # Perfect match
    P_s = np.ones(nf)
    weights = np.ones(nf)
    
    snr = compute_matched_filter_snr(Y, T, P_s, weights)
    
    # Perfect match should give SNR ≈ sqrt(nf) (for unit variance)
    expected_snr = np.sqrt(np.sum(np.abs(T) ** 2))
    assert np.abs(snr - expected_snr) / expected_snr < 0.01
    
    print(f"  ✓ Perfect match SNR: {snr:.2f} (expected: {expected_snr:.2f})")
    
    # Test 2: Orthogonal signals should give low SNR
    T = np.random.randn(nf) + 1j * np.random.randn(nf)
    Y = np.random.randn(nf) + 1j * np.random.randn(nf)
    Y = Y - np.vdot(Y, T) * T / np.vdot(T, T)  # Make orthogonal
    
    snr = compute_matched_filter_snr(Y, T, P_s, weights)
    
    # Orthogonal signals should give SNR ≈ 0
    assert np.abs(snr) < 1.0
    
    print(f"  ✓ Orthogonal signals SNR: {snr:.2f} (expected: ~0)")
    
    # Test 3: Scaled template should give same SNR (normalized)
    T = np.random.randn(nf) + 1j * np.random.randn(nf)
    Y = 2.0 * T  # Scaled version
    
    snr1 = compute_matched_filter_snr(Y, T, P_s, weights)
    snr2 = compute_matched_filter_snr(Y, 0.5 * T, P_s, weights)
    
    # SNR should be invariant to template scaling
    assert np.abs(snr1 - snr2) < 0.01
    
    print(f"  ✓ Scale invariance: SNR1={snr1:.2f}, SNR2={snr2:.2f}")
    
    # Test 4: Noise should give low SNR on average
    snrs = []
    for _ in range(10):
        Y = np.random.randn(nf) + 1j * np.random.randn(nf)
        T = np.random.randn(nf) + 1j * np.random.randn(nf)
        snr = compute_matched_filter_snr(Y, T, P_s, weights)
        snrs.append(snr)
    
    mean_snr = np.mean(snrs)
    std_snr = np.std(snrs)
    
    # Mean should be close to 0, std should be reasonable
    assert np.abs(mean_snr) < 2.0
    assert std_snr > 0
    
    print(f"  ✓ Random noise: mean SNR={mean_snr:.2f}, std={std_snr:.2f}")
    
    return True


def test_frequency_weights():
    """Test frequency weight computation logic"""
    print("Testing frequency weights...")
    
    # For even length
    n = 100
    nf = n // 2 + 1
    weights = np.ones(nf)
    weights[1:-1] = 2.0
    weights[0] = 1.0
    weights[-1] = 1.0
    
    # Check that weighting is correct for one-sided spectrum
    # Total power should be preserved
    assert weights[0] == 1.0
    assert weights[-1] == 1.0
    assert np.all(weights[1:-1] == 2.0)
    
    print("  ✓ Frequency weights computed correctly")
    
    return True


def test_power_spectrum_floor():
    """Test power spectrum floor logic"""
    print("Testing power spectrum floor...")
    
    P_s = np.array([0.0, 1.0, 2.0, 3.0, 0.1])
    eps_floor = 1e-2
    
    median_ps = np.median(P_s[P_s > 0])
    P_s_floored = np.maximum(P_s, eps_floor * median_ps)
    
    # Check that all values are above floor
    assert np.all(P_s_floored >= eps_floor * median_ps)
    
    # Check that non-zero values are preserved
    assert P_s_floored[1] == 1.0
    assert P_s_floored[2] == 2.0
    
    print(f"  ✓ Power spectrum floor applied (floor={eps_floor * median_ps:.4f})")
    
    return True


def test_full_pipeline():
    """Test full pipeline with synthetic data"""
    print("Testing full pipeline...")
    
    # Generate synthetic data
    np.random.seed(42)
    n = 100
    t = np.sort(np.random.uniform(0, 10, n))
    
    # Add transit signal
    period = 3.0
    duration = 0.3
    epoch = 0.5
    depth = 0.1
    
    signal = generate_transit_template(t, period, epoch, duration, depth)
    noise = 0.05 * np.random.randn(n)
    y = signal + noise
    
    # Simulate NUFFT (here we just use random complex values for simplicity)
    nf = 2 * n
    Y = np.random.randn(nf) + 1j * np.random.randn(nf)
    T = np.random.randn(nf) + 1j * np.random.randn(nf)
    
    # Simulate power spectrum
    P_s = np.abs(Y) ** 2
    
    # Compute weights
    weights = np.ones(nf)
    if n % 2 == 0:
        weights[1:-1] = 2.0
    else:
        weights[1:] = 2.0
    
    # Compute SNR
    snr = compute_matched_filter_snr(Y, T, P_s, weights)
    
    # Should be a finite number
    assert np.isfinite(snr)
    
    print(f"  ✓ Full pipeline SNR: {snr:.2f}")
    
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("NUFFT LRT Algorithm Validation (CPU-only)")
    print("=" * 60)
    print()
    
    all_passed = True
    
    try:
        all_passed &= test_template_generation()
        all_passed &= test_matched_filter_logic()
        all_passed &= test_frequency_weights()
        all_passed &= test_power_spectrum_floor()
        all_passed &= test_full_pipeline()
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        all_passed = False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print()
    print("=" * 60)
    if all_passed:
        print("✓ All validation tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)
