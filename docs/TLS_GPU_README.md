# GPU-Accelerated Transit Least Squares (TLS)

## Overview

This is a GPU-accelerated implementation of the Transit Least Squares (TLS) algorithm for detecting periodic planetary transits in astronomical time series data. Unlike BLS (Box Least Squares), TLS uses a physically realistic limb-darkened transit template for fitting, improving sensitivity to small planets.

**Reference:** [Hippke & Heller (2019), A&A 623, A39](https://ui.adsabs.harvard.edu/abs/2019A%26A...623A..39H/abstract)

## Quick Start

### Standard Mode - Fixed Duration Range

```python
from cuvarbase import tls

results = tls.tls_search_gpu(
    t, y, dy,
    period_min=5.0,
    period_max=20.0,
    R_star=1.0,
    M_star=1.0
)

print(f"Period: {results['period']:.4f} days")
print(f"Depth: {results['depth']:.6f}")
print(f"SDE: {results['SDE']:.2f}")
```

### Keplerian Mode - Physically Motivated Duration Constraints

```python
results = tls.tls_transit(
    t, y, dy,
    R_star=1.0,      # Solar radii
    M_star=1.0,      # Solar masses
    R_planet=1.0,    # Earth radii (fiducial)
    qmin_fac=0.5,    # Search 0.5x to 2.0x Keplerian duration
    qmax_fac=2.0,
    n_durations=15,
    period_min=5.0,
    period_max=20.0
)
```

## Features

### 1. Limb-Darkened Transit Template

The key difference from BLS is the use of a physically realistic transit template
computed using the batman package (Kreidberg 2015). The template accounts for
stellar limb darkening, producing a rounded transit shape rather than a box.

The template is:
- Precomputed on the CPU with configurable limb darkening law and coefficients
- Transferred to GPU shared memory (4KB for 1000-point template)
- Interpolated via linear lookup during the chi-squared calculation
- Falls back to a trapezoidal shape if batman is not installed

### 2. Keplerian-Aware Duration Constraints

Just like BLS's `eebls_transit()`, TLS exploits Keplerian physics to focus the search on plausible transit durations:

```python
from cuvarbase import tls_grids

# Calculate expected fractional duration at each period
q_values = tls_grids.q_transit(periods, R_star=1.0, M_star=1.0, R_planet=1.0)

# Generate focused duration grid
durations, counts, q_vals = tls_grids.duration_grid_keplerian(
    periods, R_star=1.0, M_star=1.0, R_planet=1.0,
    qmin_fac=0.5, qmax_fac=2.0, n_durations=15
)
```

### 3. Optimal Period Grid Sampling

Implements Ofir (2014) frequency-to-cubic transformation for optimal period sampling:

```python
periods = tls_grids.period_grid_ofir(
    t,
    R_star=1.0,
    M_star=1.0,
    period_min=5.0,
    period_max=20.0,
    oversampling_factor=3,
    n_transits_min=2
)
```

**Reference:** Ofir (2014), "An optimized transit detection algorithm to search for periodic transits of small planets", A&A 561, A138

### 4. GPU Memory Management

Efficient GPU memory handling via `TLSMemory` class:
- Pre-allocates GPU arrays for t, y, dy, periods, template, results
- Supports both standard and Keplerian modes (qmin/qmax arrays)
- Memory pooling reduces allocation overhead

### 5. Optimized CUDA Kernels

Two optimized CUDA kernels in `cuvarbase/kernels/tls.cu`:

**`tls_search_kernel()`** - Standard search:
- Fixed duration range (0.5% to 15% of period)
- Limb-darkened transit template in shared memory
- Bitonic sort for phase-folding
- Warp shuffle reduction for finding minimum chi-squared

**`tls_search_kernel_keplerian()`** - Keplerian-aware:
- Per-period qmin/qmax arrays
- Focused search space
- Same core algorithm with template

Both kernels:
- Use shared memory for phase-folded data and transit template
- Minimize global memory accesses
- Support datasets up to ~100,000 points

## API Reference

### High-Level Functions

#### `tls_transit(t, y, dy, **kwargs)`

High-level wrapper with Keplerian duration constraints (analog of BLS's `eebls_transit()`).

**Parameters:**
- `t` (array): Time values
- `y` (array): Flux/magnitude values
- `dy` (array): Measurement uncertainties
- `R_star` (float): Stellar radius in solar radii (default: 1.0)
- `M_star` (float): Stellar mass in solar masses (default: 1.0)
- `R_planet` (float): Fiducial planet radius in Earth radii (default: 1.0)
- `qmin_fac` (float): Minimum duration factor (default: 0.5)
- `qmax_fac` (float): Maximum duration factor (default: 2.0)
- `n_durations` (int): Number of duration samples (default: 15)
- `period_min` (float): Minimum period in days
- `period_max` (float): Maximum period in days
- `n_transits_min` (int): Minimum transits required (default: 2)
- `oversampling_factor` (int): Period grid oversampling (default: 3)

**Returns:** Dictionary with keys:
- `period`: Best-fit period (days)
- `T0`: Best-fit transit epoch (days)
- `duration`: Best-fit transit duration (days)
- `depth`: Best-fit transit depth (fractional flux dip)
- `SDE`: Signal Detection Efficiency
- `chi2`: Chi-squared value
- `periods`: Array of trial periods
- `power`: Detrended power spectrum

#### `tls_search_gpu(t, y, dy, periods=None, **kwargs)`

Low-level GPU search function with custom period/duration grids.

**Additional Parameters:**
- `periods` (array): Custom period grid (if None, auto-generated)
- `qmin` (array): Per-period minimum fractional durations (Keplerian mode)
- `qmax` (array): Per-period maximum fractional durations (Keplerian mode)
- `n_durations` (int): Number of duration samples if using qmin/qmax
- `block_size` (int): CUDA block size (default: 128)

### Grid Generation Functions

#### `period_grid_ofir(t, R_star, M_star, **kwargs)`

Generate optimal period grid using Ofir (2014) frequency-to-cubic sampling.

#### `q_transit(period, R_star, M_star, R_planet)`

Calculate Keplerian fractional transit duration (q = duration/period).

#### `duration_grid_keplerian(periods, R_star, M_star, R_planet, **kwargs)`

Generate Keplerian-aware duration grid for each period.

## Algorithm Details

### Transit Template

The transit model uses a precomputed limb-darkened template:

```
model(t) = 1 - depth * template(transit_coord)
```

Where `transit_coord` maps the phase position within the transit window to [-1, 1],
and `template()` returns a value in [0, 1] via linear interpolation of the
precomputed template array. The template captures limb darkening effects, giving
a rounded bottom rather than the flat-bottomed box of BLS.

### Optimal Depth Fitting

For each trial (period, duration, T0), depth is solved via weighted least squares:
```
depth = sum[(1-y_i) * T(x_i) / sigma_i^2] / sum[T(x_i)^2 / sigma_i^2]
```
where T(x_i) is the template value at the transit coordinate of point i.

### Signal Detection Efficiency (SDE)

The SDE metric quantifies signal significance:
```
SDE = (max(SR) - mean(SR)) / std(SR)
```

Where SR (Signal Residue) = 1 - chi2 / chi2_null.

**SDE > 7** typically indicates a robust detection.

## Known Limitations

1. **Dataset Size**: Bitonic sort supports up to ~100,000 points
   - Designed for typical astronomical light curves (500-20,000 points)
   - For >100k points, consider binning or using CPU TLS
   - Performance is optimal for ndata < 20,000

2. **Memory**: Requires ~(3N + n_template + 4*block_size) floats of shared memory per block
   - 5,000 points: ~60 KB + 4 KB template
   - Should work on any GPU with >2GB VRAM

3. **Duration Grid**: Currently uniform in log-space
   - Could optimize further using Ofir-style adaptive sampling

4. **Single GPU**: No multi-GPU support yet
   - Trivial to parallelize across multiple light curves

## Related Work

**CETRA** (Smith et al. 2025) is a complementary GPU-accelerated transit detection
algorithm that uses a different approach (matched filtering with analytic templates).
CETRA may be preferable for survey-scale searches where computational throughput is
paramount. GPU TLS is valuable when standard TLS outputs (SDE, FAP, odd/even tests)
are needed for transit vetting pipelines, or when results must be directly comparable
to published CPU TLS results.

## Testing

### Pytest Suite

```bash
pytest cuvarbase/tests/test_tls_basic.py -v
```

Tests cover:
- Transit template generation (batman and trapezoidal fallback)
- Kernel compilation
- Memory allocation
- Period grid generation
- Statistics (SR, SDE, SNR)
- Signal recovery (synthetic transits)
- SDE > 0 regression test

## Implementation Files

### Core Implementation
- `cuvarbase/tls.py` - Main Python API
- `cuvarbase/tls_models.py` - Transit template generation
- `cuvarbase/tls_grids.py` - Grid generation utilities
- `cuvarbase/tls_stats.py` - Statistical calculations
- `cuvarbase/kernels/tls.cu` - CUDA kernels

### Testing
- `cuvarbase/tests/test_tls_basic.py` - Unit tests

### Documentation
- `docs/TLS_GPU_README.md` - This file

## References

1. **Hippke & Heller (2019)**: "Optimized transit detection algorithm to search for periodic transits of small planets", A&A 623, A39
   - Original TLS algorithm and SDE metric

2. **Kovacs et al. (2002)**: "A box-fitting algorithm in the search for periodic transits", A&A 391, 369
   - BLS algorithm (TLS is a refinement)

3. **Ofir (2014)**: "An optimized transit detection algorithm to search for periodic transits of small planets", A&A 561, A138
   - Optimal period grid sampling

4. **Smith et al. (2025)**: "CETRA: GPU-accelerated transit detection"
   - Complementary GPU transit detection approach

5. **Kreidberg (2015)**: "batman: BAsic Transit Model cAlculatioN in Python", PASP 127, 1161
   - Transit model package used for template generation

6. **transitleastsquares**: https://github.com/hippke/tls
   - Reference CPU implementation

## Citation

If you use this GPU TLS implementation, please cite both cuvarbase and the original TLS paper:

```bibtex
@MISC{2022ascl.soft10030H,
       author = {{Hoffman}, John},
        title = "{cuvarbase: GPU-Accelerated Variability Algorithms}",
 howpublished = {Astrophysics Source Code Library, record ascl:2210.030},
         year = 2022,
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022ascl.soft10030H}
}

@ARTICLE{2019A&A...623A..39H,
       author = {{Hippke}, Michael and {Heller}, Ren{\'e}},
        title = "{Optimized transit detection algorithm to search for periodic transits of small planets}",
      journal = {Astronomy & Astrophysics},
         year = 2019,
       volume = {623},
          eid = {A39},
          doi = {10.1051/0004-6361/201834672}
}
```
