"""
Test pdot functionality in PDM
"""
import numpy as np
from numpy.testing import assert_allclose

# Test the CPU versions which don't require CUDA
from cuvarbase.pdm import pdm2_cpu, pdm2_single_freq, binned_pdm_model, var_binned
from cuvarbase.utils import weights

def generate_signal_with_pdot(t, freq0, pdot, amplitude=1.0, noise=0.1, seed=42):
    """Generate a sinusoidal signal with changing period"""
    np.random.seed(seed)
    # Phase with pdot: phi = freq0 * t + 0.5 * pdot * t^2
    phase = freq0 * t + 0.5 * pdot * t * t
    y = amplitude * np.sin(2 * np.pi * phase)
    y += noise * np.random.randn(len(t))
    return y

def test_pdm_no_pdot():
    """Test PDM without pdot (baseline)"""
    np.random.seed(42)
    
    ndata = 100
    t = np.sort(10 * np.random.rand(ndata))
    t -= np.mean(t)
    
    freq0 = 0.5
    y = generate_signal_with_pdot(t, freq0, pdot=0.0, noise=0.05)
    dy = 0.05 * np.ones_like(y)
    w = weights(dy)
    
    freqs = np.linspace(0.45, 0.55, 21)
    power = pdm2_cpu(t, y, w, freqs, nbins=20)
    
    # Best frequency should be close to true frequency
    best_freq = freqs[np.argmax(power)]
    assert_allclose(best_freq, freq0, atol=0.02)
    
    # Power should be reasonably high
    assert np.max(power) > 0.5, "PDM power should be significant for clear signal"

def test_pdm_with_pdot_improves_power():
    """Test that using correct pdot improves PDM power"""
    np.random.seed(42)
    
    ndata = 150
    t = np.sort(10 * np.random.rand(ndata))
    t -= np.mean(t)
    
    freq0 = 0.5
    pdot_true = 0.015
    y = generate_signal_with_pdot(t, freq0, pdot_true, noise=0.05)
    dy = 0.05 * np.ones_like(y)
    w = weights(dy)
    
    freqs = np.linspace(0.48, 0.52, 21)
    
    # Test without pdot
    power_no_pdot = pdm2_cpu(t, y, w, freqs, nbins=20)
    
    # Test with correct pdot
    pdots = pdot_true * np.ones_like(freqs)
    power_with_pdot = pdm2_cpu(t, y, w, freqs, nbins=20, pdots=pdots)
    
    # Power with correct pdot should be higher
    max_power_no_pdot = np.max(power_no_pdot)
    max_power_with_pdot = np.max(power_with_pdot)
    
    print(f"Max power without pdot: {max_power_no_pdot:.4f}")
    print(f"Max power with pdot: {max_power_with_pdot:.4f}")
    print(f"Improvement: {(max_power_with_pdot - max_power_no_pdot):.4f}")
    
    assert max_power_with_pdot > max_power_no_pdot, \
        "Power with correct pdot should be higher than without pdot"

def test_pdm_single_freq_pdot():
    """Test single frequency PDM with pdot"""
    np.random.seed(42)
    
    ndata = 100
    t = np.sort(10 * np.random.rand(ndata))
    t -= np.mean(t)
    
    freq0 = 0.5
    pdot_true = 0.01
    y = generate_signal_with_pdot(t, freq0, pdot_true, noise=0.05)
    dy = 0.05 * np.ones_like(y)
    w = weights(dy)
    
    # Test at true frequency with and without pdot
    power_no_pdot = pdm2_single_freq(t, y, w, freq0, nbins=20, pdot=0.0)
    power_with_pdot = pdm2_single_freq(t, y, w, freq0, nbins=20, pdot=pdot_true)
    
    assert power_with_pdot > power_no_pdot, \
        "Power with correct pdot should be higher"

def test_pdm_grid_search_pdot():
    """Test grid search over pdot values"""
    np.random.seed(42)
    
    ndata = 100
    t = np.sort(10 * np.random.rand(ndata))
    t -= np.mean(t)
    
    freq0 = 0.5
    pdot_true = 0.01
    y = generate_signal_with_pdot(t, freq0, pdot_true, noise=0.05)
    dy = 0.05 * np.ones_like(y)
    w = weights(dy)
    
    # Grid search over freq and pdot
    freqs = np.linspace(0.48, 0.52, 11)
    pdot_vals = np.linspace(-0.02, 0.02, 9)
    
    best_power = -np.inf
    best_freq = None
    best_pdot = None
    
    for pdot in pdot_vals:
        pdots = pdot * np.ones_like(freqs)
        powers = pdm2_cpu(t, y, w, freqs, nbins=20, pdots=pdots)
        max_power = np.max(powers)
        if max_power > best_power:
            best_power = max_power
            best_freq = freqs[np.argmax(powers)]
            best_pdot = pdot
    
    print(f"True freq: {freq0:.4f}, True pdot: {pdot_true:.4f}")
    print(f"Best freq: {best_freq:.4f}, Best pdot: {best_pdot:.4f}")
    print(f"Best power: {best_power:.4f}")
    
    # Check that we recover something close to the true values
    assert_allclose(best_freq, freq0, atol=0.02)
    # Pdot recovery might be less precise
    assert abs(best_pdot - pdot_true) < 0.01, \
        "Should recover pdot within reasonable tolerance"

def test_pdm_zero_pdot_consistency():
    """Test that pdot=0 gives same result as no pdot"""
    np.random.seed(42)
    
    ndata = 50
    t = np.sort(10 * np.random.rand(ndata))
    t -= np.mean(t)
    
    freq0 = 0.5
    y = generate_signal_with_pdot(t, freq0, pdot=0.0, noise=0.05)
    dy = 0.05 * np.ones_like(y)
    w = weights(dy)
    
    freqs = np.linspace(0.4, 0.6, 11)
    
    # Test without pdot parameter
    power_no_param = pdm2_cpu(t, y, w, freqs, nbins=20)
    
    # Test with pdot=0
    pdots_zero = np.zeros_like(freqs)
    power_zero_pdot = pdm2_cpu(t, y, w, freqs, nbins=20, pdots=pdots_zero)
    
    # Should be identical
    assert_allclose(power_no_param, power_zero_pdot, rtol=1e-6)

if __name__ == '__main__':
    # Run tests
    print("Running PDM pdot tests...\n")
    
    print("Test 1: PDM without pdot (baseline)")
    test_pdm_no_pdot()
    print("✓ Passed\n")
    
    print("Test 2: PDM with pdot improves power")
    test_pdm_with_pdot_improves_power()
    print("✓ Passed\n")
    
    print("Test 3: Single frequency PDM with pdot")
    test_pdm_single_freq_pdot()
    print("✓ Passed\n")
    
    print("Test 4: Grid search over pdot")
    test_pdm_grid_search_pdot()
    print("✓ Passed\n")
    
    print("Test 5: Zero pdot consistency")
    test_pdm_zero_pdot_consistency()
    print("✓ Passed\n")
    
    print("All PDM pdot tests passed!")
