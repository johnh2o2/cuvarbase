"""
Test pdot functionality in BLS
"""
import numpy as np
from numpy.testing import assert_allclose

# Test the CPU versions which don't require CUDA
from cuvarbase.bls import single_bls, sparse_bls_cpu, eebls_transit

def generate_transit_with_pdot(t, freq0, pdot, q=0.05, depth=0.1, noise=0.02, seed=42):
    """Generate a transit signal with changing period"""
    np.random.seed(seed)
    # Phase with pdot: phi = freq0 * t + 0.5 * pdot * t^2
    phase = (freq0 * t + 0.5 * pdot * t * t) % 1.0
    
    # Create transit signal
    y = np.ones_like(t)
    in_transit = phase < q
    y[in_transit] -= depth
    
    # Add noise
    y += noise * np.random.randn(len(t))
    
    return y

def test_single_bls_with_pdot():
    """Test single BLS evaluation with pdot"""
    np.random.seed(42)
    
    ndata = 100
    t = np.sort(10 * np.random.rand(ndata))
    t -= np.mean(t)
    
    freq0 = 0.5
    pdot_true = 0.01
    q_true = 0.05
    y = generate_transit_with_pdot(t, freq0, pdot_true, q=q_true, depth=0.1, noise=0.02)
    dy = 0.02 * np.ones_like(y)
    
    # Compute BLS at true frequency with and without pdot
    bls_no_pdot = single_bls(t, y, dy, freq0, q_true, phi0=0.0, pdot=0.0)
    bls_with_pdot = single_bls(t, y, dy, freq0, q_true, phi0=0.0, pdot=pdot_true)
    
    print(f"BLS without pdot: {bls_no_pdot:.6f}")
    print(f"BLS with pdot: {bls_with_pdot:.6f}")
    
    # BLS with correct pdot should be higher
    assert bls_with_pdot > bls_no_pdot, \
        "BLS with correct pdot should be higher than without pdot"

def test_sparse_bls_with_pdot():
    """Test sparse BLS with pdot"""
    np.random.seed(42)
    
    ndata = 80
    t = np.sort(10 * np.random.rand(ndata))
    t -= np.mean(t)
    
    freq0 = 0.5
    pdot_true = 0.015
    y = generate_transit_with_pdot(t, freq0, pdot_true, q=0.05, depth=0.15, noise=0.02)
    dy = 0.02 * np.ones_like(y)
    
    freqs = np.linspace(0.48, 0.52, 21)
    
    # Test without pdot
    powers_no_pdot, sols_no_pdot = sparse_bls_cpu(t, y, dy, freqs)
    
    # Test with correct pdot
    pdots = pdot_true * np.ones_like(freqs)
    powers_with_pdot, sols_with_pdot = sparse_bls_cpu(t, y, dy, freqs, pdots=pdots)
    
    max_power_no_pdot = np.max(powers_no_pdot)
    max_power_with_pdot = np.max(powers_with_pdot)
    
    print(f"Max BLS power without pdot: {max_power_no_pdot:.6f}")
    print(f"Max BLS power with pdot: {max_power_with_pdot:.6f}")
    print(f"Improvement: {(max_power_with_pdot - max_power_no_pdot):.6f}")
    
    # Power with correct pdot should be higher
    assert max_power_with_pdot > max_power_no_pdot, \
        "BLS power with correct pdot should be higher"

def test_eebls_transit_with_pdot():
    """Test eebls_transit wrapper with pdot (uses sparse BLS)"""
    np.random.seed(42)
    
    ndata = 100
    t = np.sort(10 * np.random.rand(ndata))
    t -= np.mean(t)
    
    freq0 = 0.5
    pdot_true = 0.01
    y = generate_transit_with_pdot(t, freq0, pdot_true, q=0.05, depth=0.1, noise=0.02)
    dy = 0.02 * np.ones_like(y)
    
    freqs = np.linspace(0.48, 0.52, 21)
    
    # Test with use_sparse=True and pdot
    pdots = pdot_true * np.ones_like(freqs)
    freqs_out, powers, sols = eebls_transit(t, y, dy, freqs=freqs, 
                                            use_sparse=True, pdots=pdots)
    
    assert len(powers) == len(freqs)
    assert np.max(powers) > 0, "Should detect transit signal with pdot"

def test_bls_pdot_grid_search():
    """Test grid search over pdot values"""
    np.random.seed(42)
    
    ndata = 80
    t = np.sort(10 * np.random.rand(ndata))
    t -= np.mean(t)
    
    freq0 = 0.5
    pdot_true = 0.01
    y = generate_transit_with_pdot(t, freq0, pdot_true, q=0.05, depth=0.15, noise=0.02)
    dy = 0.02 * np.ones_like(y)
    
    # Grid search over freq and pdot
    freqs = np.linspace(0.48, 0.52, 11)
    pdot_vals = np.linspace(-0.02, 0.02, 9)
    
    best_power = -np.inf
    best_freq = None
    best_pdot = None
    
    for pdot in pdot_vals:
        pdots = pdot * np.ones_like(freqs)
        powers, sols = sparse_bls_cpu(t, y, dy, freqs, pdots=pdots)
        max_power = np.max(powers)
        if max_power > best_power:
            best_power = max_power
            best_freq = freqs[np.argmax(powers)]
            best_pdot = pdot
    
    print(f"True freq: {freq0:.4f}, True pdot: {pdot_true:.4f}")
    print(f"Best freq: {best_freq:.4f}, Best pdot: {best_pdot:.4f}")
    print(f"Best power: {best_power:.6f}")
    
    # Check that we recover reasonable values
    assert_allclose(best_freq, freq0, atol=0.02)
    # Pdot recovery might be less precise for transits
    assert abs(best_pdot - pdot_true) < 0.015, \
        "Should recover pdot within reasonable tolerance"

def test_bls_zero_pdot_consistency():
    """Test that pdot=0 gives same result as no pdot"""
    np.random.seed(42)
    
    ndata = 60
    t = np.sort(10 * np.random.rand(ndata))
    t -= np.mean(t)
    
    freq0 = 0.5
    y = generate_transit_with_pdot(t, freq0, pdot=0.0, q=0.05, depth=0.1, noise=0.02)
    dy = 0.02 * np.ones_like(y)
    
    freqs = np.linspace(0.45, 0.55, 11)
    
    # Test without pdot parameter
    powers_no_param, sols_no_param = sparse_bls_cpu(t, y, dy, freqs)
    
    # Test with pdot=0
    pdots_zero = np.zeros_like(freqs)
    powers_zero_pdot, sols_zero_pdot = sparse_bls_cpu(t, y, dy, freqs, pdots=pdots_zero)
    
    # Should be identical
    assert_allclose(powers_no_param, powers_zero_pdot, rtol=1e-6)

def test_eebls_transit_pdot_error():
    """Test that error is raised when pdot is used without sparse mode"""
    np.random.seed(42)
    
    ndata = 600  # Large enough that sparse mode won't be auto-selected
    t = np.sort(10 * np.random.rand(ndata))
    y = np.random.randn(ndata)
    dy = 0.1 * np.ones_like(y)
    
    freqs = np.linspace(0.1, 1.0, 10)
    pdots = np.zeros_like(freqs)
    
    # Should raise error when use_sparse=False and pdots is provided
    try:
        eebls_transit(t, y, dy, freqs=freqs, use_sparse=False, pdots=pdots)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "pdot parameter is only supported" in str(e)

if __name__ == '__main__':
    print("Running BLS pdot tests...\n")
    
    print("Test 1: Single BLS with pdot")
    test_single_bls_with_pdot()
    print("✓ Passed\n")
    
    print("Test 2: Sparse BLS with pdot")
    test_sparse_bls_with_pdot()
    print("✓ Passed\n")
    
    print("Test 3: eebls_transit wrapper with pdot")
    test_eebls_transit_with_pdot()
    print("✓ Passed\n")
    
    print("Test 4: BLS pdot grid search")
    test_bls_pdot_grid_search()
    print("✓ Passed\n")
    
    print("Test 5: Zero pdot consistency")
    test_bls_zero_pdot_consistency()
    print("✓ Passed\n")
    
    print("Test 6: Error when pdot used without sparse")
    test_eebls_transit_pdot_error()
    print("✓ Passed\n")
    
    print("All BLS pdot tests passed!")
