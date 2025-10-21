# NUFFT LRT Implementation Summary

## Overview

This document summarizes the implementation of NUFFT-based Likelihood Ratio Test (LRT) for transit detection in the cuvarbase library.

## What Was Implemented

### 1. CUDA Kernels (`cuvarbase/kernels/nufft_lrt.cu`)

Six CUDA kernels were implemented:

1. **`nufft_matched_filter`**: Core matched filter computation
   - Computes: `sum(Y * conj(T) * w / P_s) / sqrt(sum(|T|^2 * w / P_s))`
   - Uses shared memory reduction for efficient parallel computation
   - Handles both numerator and denominator in a single kernel

2. **`estimate_power_spectrum`**: Adaptive power spectrum estimation
   - Computes smoothed periodogram from NUFFT data
   - Uses boxcar smoothing with configurable window size
   - Provides adaptive noise estimation for the matched filter

3. **`compute_frequency_weights`**: One-sided spectrum weights
   - Converts two-sided spectrum to one-sided
   - Handles DC and Nyquist components correctly
   - Essential for proper power normalization

4. **`demean_data`**: Data preprocessing
   - Removes mean from data in-place on GPU
   - Preprocessing step for matched filter

5. **`compute_mean`**: Mean computation with reduction
   - Parallel reduction to compute data mean
   - Used for demeaning step

6. **`generate_transit_template`**: Transit template generation
   - Creates box transit model on GPU
   - Phase folds data at trial period
   - Generates template for matched filtering

### 2. Python Wrapper (`cuvarbase/nufft_lrt.py`)

Two main classes:

1. **`NUFFTLRTMemory`**: Memory management
   - Handles GPU memory allocation for LRT computations
   - Manages NUFFT results, power spectrum, weights, and results
   - Provides async transfer methods

2. **`NUFFTLRTAsyncProcess`**: Main computation class
   - Inherits from `GPUAsyncProcess` following cuvarbase patterns
   - Provides `run()` method for transit search
   - Integrates with existing `NFFTAsyncProcess` for NUFFT computation
   - Supports:
     - Multiple periods, durations, and epochs
     - Custom or estimated power spectrum
     - Single and double precision
     - Batch processing

### 3. Tests (`cuvarbase/tests/test_nufft_lrt.py`)

Nine comprehensive test functions:

1. `test_basic_initialization`: Tests class initialization
2. `test_template_generation`: Validates transit template creation
3. `test_nufft_computation`: Tests NUFFT integration
4. `test_matched_filter_snr_computation`: Validates SNR calculation
5. `test_detection_of_known_transit`: Tests transit detection
6. `test_white_noise_gives_low_snr`: Tests noise handling
7. `test_custom_psd`: Tests custom power spectrum
8. `test_double_precision`: Tests double precision mode
9. `test_multiple_epochs`: Tests epoch search

### 4. Documentation

Three documentation files:

1. **`NUFFT_LRT_README.md`**: Comprehensive documentation
   - Algorithm description
   - Usage examples
   - Parameter documentation
   - Comparison with BLS
   - Citations and references

2. **`examples/nufft_lrt_example.py`**: Example code
   - Basic usage demonstration
   - Shows how to generate synthetic data
   - Demonstrates period/duration search

3. **Updated `README.rst`**: Added NUFFT LRT to main README

### 5. Validation Scripts

Two validation scripts:

1. **`validation_nufft_lrt.py`**: CPU-only validation
   - Tests algorithm logic without GPU
   - Validates matched filter mathematics
   - Tests template generation
   - Verifies scale invariance

2. **`check_nufft_lrt.py`**: Import and structure check
   - Verifies module can be imported
   - Checks CUDA kernel structure
   - Validates test file
   - Checks documentation

## Algorithm Details

### Matched Filter Formula

The core matched filter statistic is:

```
SNR = Σ(Y_k * T_k* * w_k / P_s(k)) / √(Σ(|T_k|^2 * w_k / P_s(k)))
```

Where:
- `Y_k`: NUFFT of lightcurve at frequency k
- `T_k`: NUFFT of transit template at frequency k
- `P_s(k)`: Power spectrum at frequency k (noise estimate)
- `w_k`: Frequency weight (1 for DC/Nyquist, 2 for others)

### Key Features

1. **Amplitude Independence**: The normalized statistic is independent of transit depth
2. **Adaptive Noise**: Power spectrum estimation adapts to correlated noise
3. **Gappy Data**: NUFFT handles non-uniform sampling naturally
4. **Scale Invariance**: Template scaling doesn't affect detection ranking

### Advantages Over BLS

1. **Correlated Noise**: Handles red noise through PSD estimation
2. **Theoretical Foundation**: Based on optimal detection theory (LRT)
3. **Frequency Domain**: Efficient computation via FFT/NUFFT
4. **Flexible**: Can provide custom noise model via PSD

## Integration with cuvarbase

The implementation follows cuvarbase patterns:

1. **Inherits from `GPUAsyncProcess`**: Standard base class
2. **Uses existing NUFFT**: Leverages `NFFTAsyncProcess` for transforms
3. **Memory management**: Follows `NFFTMemory` pattern
4. **Async operations**: Uses CUDA streams for async execution
5. **Batch processing**: Supports `batched_run()` method
6. **Module structure**: Organized like other cuvarbase modules

## Files Added

```
cuvarbase/
├── kernels/
│   └── nufft_lrt.cu              # CUDA kernels (6 kernels)
├── tests/
│   └── test_nufft_lrt.py         # Unit tests (9 tests)
├── nufft_lrt.py                  # Main Python module (2 classes)
├── __init__.py                   # Updated with new imports
examples/
└── nufft_lrt_example.py          # Example usage
NUFFT_LRT_README.md               # Detailed documentation
README.rst                        # Updated main README
validation_nufft_lrt.py           # CPU validation
check_nufft_lrt.py                # Import check
```

## Testing Status

### CPU Validation
✓ All validation tests pass:
- Template generation
- Matched filter logic
- Frequency weights
- Power spectrum floor
- Full pipeline

### Import Check
✓ All checks pass:
- Module syntax valid
- 6 CUDA kernels present
- 9 test functions present
- Documentation complete

### GPU Testing
⚠ GPU tests require CUDA environment (not available in this environment)
- Tests are written and structured correctly
- Will run when CUDA is available
- Follow existing cuvarbase test patterns

## Reference Implementation

Based on: https://github.com/star-skelly/code_nova_exoghosts/blob/main/nufft_detector.py

Key differences from reference:
1. **GPU Acceleration**: Uses CUDA instead of CPU finufft
2. **Batch Processing**: Handles multiple trials efficiently
3. **Integration**: Works with cuvarbase ecosystem
4. **Memory Management**: Optimized for GPU memory usage

## Next Steps

For users:
1. Install cuvarbase with CUDA support
2. Run examples: `python examples/nufft_lrt_example.py`
3. Run tests: `pytest cuvarbase/tests/test_nufft_lrt.py`
4. See `NUFFT_LRT_README.md` for detailed usage

For developers:
1. Test with real CUDA environment
2. Benchmark performance vs BLS and reference implementation
3. Add more sophisticated templates (trapezoidal, etc.)
4. Add visualization utilities
5. Integrate with TESS/Kepler pipeline

## Acknowledgments

- Reference implementation: star-skelly/code_nova_exoghosts
- IEEE paper on matched filter detection in correlated noise
- cuvarbase framework by John Hoffman
- NUFFT implementation in cuvarbase
