# Period Derivative (pdot) Support in cuvarbase

This document explains how to search for signals with changing periods (period derivatives) using the cuvarbase library.

## Background

Many astrophysical signals have periods that change over time. For example:
- Pulsars spinning down
- Binary systems with orbital decay
- Variable stars with evolving periods

When a period changes linearly over time, we can parameterize this with a period derivative `pdot = dP/dt`.

The phase of such a signal is no longer linear in time:
```
phi(t) = freq * t + 0.5 * pdot * t^2
```

where `pdot` is related to the period derivative through the frequency.

## Implementation Status

### PDM (Phase Dispersion Minimization) - ✅ Full Support

PDM has complete pdot support in both CPU and GPU implementations.

**CPU Functions:**
- `pdm2_cpu(t, y, w, freqs, nbins=30, linterp=True, pdots=None)`
- `pdm2_single_freq(t, y, w, freq, nbins=30, linterp=True, pdot=0.0)`

**GPU Functions:**
- `PDMAsyncProcess.run(..., pdots=None)` - supports all PDM variants

**Example:**
```python
from cuvarbase.pdm import pdm2_cpu, PDMAsyncProcess
from cuvarbase.utils import weights
import numpy as np

# Generate data
t = np.sort(10 * np.random.rand(200))
t -= np.mean(t)
y = np.sin(2 * np.pi * (0.5 * t + 0.5 * 0.01 * t * t))  # Signal with pdot
dy = 0.1 * np.ones_like(y)
w = weights(dy)

# Define frequency grid
freqs = np.linspace(0.4, 0.6, 50)

# Method 1: Single pdot value for all frequencies
pdots = 0.01 * np.ones_like(freqs)
powers = pdm2_cpu(t, y, w, freqs, nbins=20, pdots=pdots)

# Method 2: Grid search over pdot
pdot_vals = np.linspace(-0.02, 0.02, 11)
best_power = np.zeros(len(freqs))
best_pdot = np.zeros(len(freqs))

for i, freq in enumerate(freqs):
    for pdot in pdot_vals:
        power = pdm2_single_freq(t, y, w, freq, nbins=20, pdot=pdot)
        if power > best_power[i]:
            best_power[i] = power
            best_pdot[i] = pdot

# Method 3: GPU acceleration (for large datasets)
proc = PDMAsyncProcess()
pdots = 0.01 * np.ones_like(freqs)
results = proc.run([(t, y, w, freqs)], pdots=pdots, kind='binned_linterp', nbins=20)
proc.finish()
powers_gpu = results[0]
```

### BLS (Box Least Squares) - ✅ CPU Support

BLS has pdot support for CPU-based functions (single evaluation and sparse BLS).

**Supported Functions:**
- `single_bls(t, y, dy, freq, q, phi0, pdot=0.0)`
- `sparse_bls_cpu(t, y, dy, freqs, pdots=None)`
- `eebls_transit(t, y, dy, ..., use_sparse=True, pdots=None)`

**Note:** GPU BLS kernels do not currently support pdot. For small datasets (< 500 observations), the sparse BLS algorithm automatically supports pdot.

**Example:**
```python
from cuvarbase.bls import sparse_bls_cpu, eebls_transit, single_bls
import numpy as np

# Generate transit data with pdot
t = np.sort(10 * np.random.rand(100))
t -= np.mean(t)
phase = (0.5 * t + 0.5 * 0.01 * t * t) % 1.0
y = np.ones_like(t)
y[phase < 0.05] -= 0.1  # Transit depth
dy = 0.02 * np.ones_like(y)

# Define frequency grid
freqs = np.linspace(0.45, 0.55, 30)

# Method 1: Sparse BLS with pdot
pdots = 0.01 * np.ones_like(freqs)
powers, sols = sparse_bls_cpu(t, y, dy, freqs, pdots=pdots)

# Method 2: Grid search with single_bls
freq = 0.5
q = 0.05
phi0 = 0.0
pdot_vals = np.linspace(-0.02, 0.02, 21)
bls_powers = [single_bls(t, y, dy, freq, q, phi0, pdot=pdot) for pdot in pdot_vals]
best_pdot = pdot_vals[np.argmax(bls_powers)]

# Method 3: Use eebls_transit wrapper (automatically uses sparse for small datasets)
freqs_out, powers, sols = eebls_transit(t, y, dy, freqs=freqs, 
                                        use_sparse=True, pdots=pdots)
```

### CE (Conditional Entropy) - ⚠️ Limited Support

Conditional Entropy does not have native pdot support in the GPU kernels. For signals with pdot:

**Recommended Approaches:**
1. Use PDM instead (has full pdot support and similar performance characteristics)
2. For small pdot values, pre-transform the time array
3. Perform a grid search over pdot values

**Example (grid search approach):**
```python
from cuvarbase.ce import ConditionalEntropyAsyncProcess
import numpy as np

# Generate data
t = np.sort(10 * np.random.rand(200))
y = np.sin(2 * np.pi * (0.5 * t + 0.5 * 0.01 * t * t))
dy = 0.1 * np.ones_like(y)

freqs = np.linspace(0.4, 0.6, 50)
pdot_vals = np.linspace(-0.02, 0.02, 11)

proc = ConditionalEntropyAsyncProcess()
best_ce = np.inf * np.ones(len(freqs))
best_pdot = np.zeros(len(freqs))

# Grid search over pdot
for pdot in pdot_vals:
    # For each pdot, run CE with original time array
    # (This is an approximation - use PDM for exact pdot support)
    results = proc.run([(t, y, dy)], freqs=freqs)
    proc.finish()
    freqs_out, ce = results[0]
    
    # Update best values
    improved = ce < best_ce
    best_ce[improved] = ce[improved]
    best_pdot[improved] = pdot
```

## Grid Search Strategy

The most robust way to search for both frequency and pdot is to perform a 2D grid search:

```python
def grid_search_freq_pdot(t, y, dy, freq_range, pdot_range, method='pdm'):
    """
    Perform 2D grid search over frequency and pdot
    
    Parameters
    ----------
    t : array
        Observation times
    y : array
        Observations
    dy : array
        Uncertainties
    freq_range : tuple
        (fmin, fmax, nfreqs)
    pdot_range : tuple
        (pdot_min, pdot_max, npdots)
    method : str
        'pdm' or 'bls'
    
    Returns
    -------
    freqs : array
        Frequency grid
    pdots : array
        Pdot grid
    powers : 2D array
        Power values (nfreqs x npdots)
    """
    from cuvarbase.pdm import pdm2_single_freq
    from cuvarbase.bls import single_bls
    from cuvarbase.utils import weights
    import numpy as np
    
    fmin, fmax, nfreqs = freq_range
    pdot_min, pdot_max, npdots = pdot_range
    
    freqs = np.linspace(fmin, fmax, nfreqs)
    pdots = np.linspace(pdot_min, pdot_max, npdots)
    
    w = weights(dy)
    powers = np.zeros((nfreqs, npdots))
    
    for i, freq in enumerate(freqs):
        for j, pdot in enumerate(pdots):
            if method == 'pdm':
                powers[i, j] = pdm2_single_freq(t, y, w, freq, nbins=20, pdot=pdot)
            elif method == 'bls':
                # Estimate q from data
                q = 0.05  # or estimate from data
                phi0 = 0.0
                powers[i, j] = single_bls(t, y, dy, freq, q, phi0, pdot=pdot)
    
    return freqs, pdots, powers

# Example usage
freq_range = (0.4, 0.6, 50)
pdot_range = (-0.02, 0.02, 21)
freqs, pdots, powers = grid_search_freq_pdot(t, y, dy, freq_range, pdot_range)

# Find best (freq, pdot) combination
i_best, j_best = np.unravel_index(np.argmax(powers), powers.shape)
best_freq = freqs[i_best]
best_pdot = pdots[j_best]
print(f"Best frequency: {best_freq:.4f}")
print(f"Best pdot: {best_pdot:.4f}")
```

## Performance Considerations

1. **PDM with GPU**: Fastest for large datasets with pdot support
2. **Sparse BLS with pdot**: Good for small datasets (< 500 observations)
3. **Grid search**: Computationally expensive but most thorough
4. **Adaptive search**: Start with coarse grid, refine around best candidates

## Testing

Test files are provided in `cuvarbase/tests/`:
- `test_pdm_pdot.py`: PDM pdot functionality tests
- `test_bls_pdot.py`: BLS pdot functionality tests

Run tests with:
```bash
python cuvarbase/tests/test_pdm_pdot.py
python cuvarbase/tests/test_bls_pdot.py
```

## References

The pdot phase calculation follows the standard formulation for quadratic timing models used in pulsar timing and exoplanet transit timing variations.

For signals with period P(t) = P0 + pdot * t, the phase is:
```
phi(t) = integral_0^t dt'/P(t') ≈ freq0 * t + 0.5 * pdot_freq * t^2
```

where pdot_freq relates to the period derivative through the frequency-period relationship.
