# Pdot Implementation Summary

This document summarizes the implementation of period derivative (pdot) support in cuvarbase.

## What was implemented

### 1. PDM (Phase Dispersion Minimization) - COMPLETE ✅

**CUDA Kernels (`cuvarbase/kernels/pdm.cu`):**
- Added `PHASE_PDOT` macro for phase calculation with pdot
- Added `phase_diff_pdot` device function
- Implemented pdot versions of all variance functions:
  - `var_step_function_pdot`
  - `var_linear_interp_pdot`
  - `var_binless_tophat_pdot`
  - `var_binless_gauss_pdot`
- Added pdot kernel versions:
  - `pdm_binless_tophat_pdot`
  - `pdm_binless_gauss_pdot`
  - `pdm_binned_linterp_pdot`
  - `pdm_binned_step_pdot`

**Python API (`cuvarbase/pdm.py`):**
- Modified `pdm2_cpu()` to accept `pdots` parameter
- Modified `pdm2_single_freq()` to accept `pdot` parameter
- Updated `binned_pdm_model()` to support pdot
- Updated `var_binned()` to support pdot
- Modified `PDMAsyncProcess.allocate()` to handle pdots
- Modified `PDMAsyncProcess.run()` to support pdots
- Updated `_compile_and_prepare_functions()` to prepare pdot kernel versions
- Modified `pdm_async()` to conditionally call pdot kernels

### 2. BLS (Box Least Squares) - CPU SUPPORT ✅

**CUDA Kernels (`cuvarbase/kernels/bls.cu`):**
- Added `mod1_pdot` device function for phase calculation with pdot

**Python API (`cuvarbase/bls.py`):**
- Modified `single_bls()` to accept `pdot` parameter
- Modified `sparse_bls_cpu()` to accept `pdots` parameter (array of pdot values)
- Updated `eebls_transit()` to support `pdots` parameter when using sparse BLS
- Added validation to prevent pdot usage with GPU kernels (not yet implemented)

**Note:** GPU BLS kernels (`eebls_gpu`, `eebls_gpu_fast`) do not yet support pdot. This would require extensive modifications to the shared memory binning code. Users with small datasets (< 500 observations) can use `sparse_bls_cpu` with pdot support.

### 3. CE (Conditional Entropy) - DOCUMENTATION ✅

**CUDA Kernels (`cuvarbase/kernels/ce.cu`):**
- Added `mod1_pdot` device function
- Added `phase_ind_pdot` device function

**Python API (`cuvarbase/ce.py`):**
- Added documentation explaining pdot support approach
- Recommended using PDM for signals with pdot (has full support)

**Note:** CE GPU kernels do not have native pdot support. The kernels are complex and would require significant refactoring. Users should use PDM for pdot searches, or perform grid searches manually.

### 4. Tests

**Created test files:**
- `cuvarbase/tests/test_pdm_pdot.py` - Comprehensive PDM pdot tests
  - Test without pdot (baseline)
  - Test with pdot improves power
  - Single frequency test
  - Grid search over pdot
  - Zero pdot consistency
  
- `cuvarbase/tests/test_bls_pdot.py` - Comprehensive BLS pdot tests
  - Single BLS with pdot
  - Sparse BLS with pdot
  - eebls_transit wrapper test
  - Grid search over pdot
  - Zero pdot consistency
  - Error handling test

### 5. Documentation

**Created documentation files:**
- `PDOT_USAGE.md` - Complete user guide for pdot functionality
  - Background on period derivatives
  - Implementation status for each algorithm
  - Usage examples for PDM, BLS, and CE
  - Grid search strategies
  - Performance considerations

- `examples/pdot_example.py` - Demonstration script
  - Example 1: PDM with pdot
  - Example 2: 2D grid search over frequency and pdot
  - Visualization of results

## Key Design Decisions

1. **Phase Calculation:** The pdot is incorporated as a quadratic term in the phase:
   ```
   phi(t) = freq * t + 0.5 * pdot * t^2
   ```

2. **API Design:** Pdot is passed as an array (one value per frequency) to allow:
   - Testing a single pdot value for all frequencies
   - Grid search over different pdot values
   - Frequency-dependent pdot values

3. **GPU vs CPU:** 
   - PDM: Full GPU support with pdot kernels
   - BLS: CPU support only (sparse BLS)
   - CE: No direct support, use PDM or manual grid search

4. **Backward Compatibility:** 
   - All functions maintain backward compatibility
   - `pdots=None` or `pdot=0.0` defaults to standard behavior
   - Existing code continues to work without modifications

## Usage Pattern

The typical workflow for pdot searches:

```python
# 1. Define grids
freqs = np.linspace(fmin, fmax, nfreq)
pdot_vals = np.linspace(pdot_min, pdot_max, npdot)

# 2. Grid search
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

# 3. Refine around best values (optional)
```

## Limitations and Future Work

1. **BLS GPU Kernels:** Could be extended to support pdot, but requires:
   - Modifying shared memory binning code
   - Updating all kernel variants
   - Testing with various q and phi values

2. **CE GPU Kernels:** Could be extended to support pdot, but requires:
   - Modifying histogram binning
   - Updating fast and faster CE variants
   - Testing with weighted and unweighted modes

3. **Optimization:** Grid search over pdot is computationally expensive. Could implement:
   - Adaptive grid refinement
   - Bayesian optimization
   - Machine learning-based prediction of pdot ranges

4. **Higher Order Terms:** Currently supports linear pdot. Could extend to:
   - Quadratic period changes (pdotdot)
   - Sinusoidal period modulation
   - Other timing models

## Testing

All functionality has been tested with synthetic data. The tests verify:
- Correct phase calculation with pdot
- Improved detection power when using correct pdot
- Grid search recovers true pdot values
- Backward compatibility (pdot=0 gives same results as no pdot)

## Files Modified/Created

**Modified:**
- `cuvarbase/kernels/pdm.cu`
- `cuvarbase/kernels/bls.cu`
- `cuvarbase/kernels/ce.cu`
- `cuvarbase/pdm.py`
- `cuvarbase/bls.py`
- `cuvarbase/ce.py`

**Created:**
- `cuvarbase/tests/test_pdm_pdot.py`
- `cuvarbase/tests/test_bls_pdot.py`
- `PDOT_USAGE.md`
- `examples/pdot_example.py`
- `PDOT_IMPLEMENTATION_SUMMARY.md` (this file)

## Conclusion

The pdot implementation provides users with the ability to search for and characterize signals with changing periods. PDM has full support, BLS has CPU support, and CE users should use PDM or manual approaches. The implementation maintains backward compatibility and follows the existing code patterns in cuvarbase.
