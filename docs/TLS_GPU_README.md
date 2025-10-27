# GPU-Accelerated Transit Least Squares (TLS)

## Overview

This is a GPU-accelerated implementation of the Transit Least Squares (TLS) algorithm for detecting periodic planetary transits in astronomical time series data. The implementation achieves **35-202× speedup** over the CPU-based `transitleastsquares` package.

**Reference:** [Hippke & Heller (2019), A&A 623, A39](https://ui.adsabs.harvard.edu/abs/2019A%26A...623A..39H/abstract)

## Performance

Benchmarks comparing `cuvarbase.tls` (GPU) vs `transitleastsquares` v1.32 (CPU):

| Dataset Size | Baseline | GPU Time | CPU Time | Speedup |
|--------------|----------|----------|----------|---------|
| 500 points   | 50 days  | 0.24s    | 8.65s    | **35×** |
| 1000 points  | 100 days | 0.44s    | 26.7s    | **61×** |
| 2000 points  | 200 days | 0.88s    | 88.4s    | **100×** |
| 5000 points  | 500 days | 2.40s    | 485s     | **202×** |

*Hardware: NVIDIA RTX A4500 (20GB, 7,424 CUDA cores) vs Intel Xeon (8 cores)*

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
    qmin_fac=0.5,    # Search 0.5× to 2.0× Keplerian duration
    qmax_fac=2.0,
    n_durations=15,
    period_min=5.0,
    period_max=20.0
)
```

## Features

### 1. Keplerian-Aware Duration Constraints

Just like BLS's `eebls_transit()`, TLS now exploits Keplerian physics to focus the search on plausible transit durations:

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

**Why This Matters:**

For a circular orbit, the fractional transit duration q = duration/period depends on:
- **Period (P)**: Longer periods → longer durations
- **Stellar density (ρ = M/R³)**: Denser stars → shorter durations
- **Planet/star size ratio**: Larger planets → longer transits

By calculating the expected Keplerian duration and searching around it (0.5× to 2.0×), we achieve:
- **7-8× efficiency improvement** by avoiding unphysical durations
- **Better sensitivity** to small planets
- **Stellar-parameter aware** searches

**Comparison:**

| Period | Fixed Range | Keplerian Range | Efficiency Gain |
|--------|-------------|-----------------|-----------------|
| 5 days | q=0.005-0.15 (30×) | q=0.013-0.052 (4×) | **7.5×** |
| 10 days | q=0.005-0.15 (30×) | q=0.008-0.032 (4×) | **7.5×** |
| 20 days | q=0.005-0.15 (30×) | q=0.005-0.021 (4.2×) | **7.1×** |

### 2. Optimal Period Grid Sampling

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

This ensures no transit signals are missed due to aliasing in the period grid.

**Reference:** [Ofir (2014), ApJ 789, 145](https://ui.adsabs.harvard.edu/abs/2014ApJ...789..145O/abstract)

### 3. GPU Memory Management

Efficient GPU memory handling via `TLSMemory` class:
- Pre-allocates GPU arrays for t, y, dy, periods, results
- Supports both standard and Keplerian modes (qmin/qmax arrays)
- Memory pooling reduces allocation overhead
- Clean resource management with context manager support

### 4. Optimized CUDA Kernels

Two optimized CUDA kernels in `cuvarbase/kernels/tls.cu`:

**`tls_search_kernel()`** - Standard search:
- Fixed duration range (0.5% to 15% of period)
- Insertion sort for phase-folding
- Warp reduction for finding minimum chi-squared

**`tls_search_kernel_keplerian()`** - Keplerian-aware:
- Per-period qmin/qmax arrays
- Focused search space (7-8× more efficient)
- Same core algorithm

Both kernels:
- Use shared memory for phase-folded data
- Minimize global memory accesses
- Support datasets up to ~5000 points

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
- `power`: Chi-squared values for all periods

#### `tls_search_gpu(t, y, dy, periods=None, **kwargs)`

Low-level GPU search function with custom period/duration grids.

**Additional Parameters:**
- `periods` (array): Custom period grid (if None, auto-generated)
- `durations` (array): Custom duration grid (if None, auto-generated)
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

### Chi-Squared Calculation

The kernel calculates:
```
χ² = Σ [(y_i - model_i)² / σ_i²]
```

Where the model is a simple box:
```
model(t) = {
    1 - depth,  if in transit
    1,          otherwise
}
```

### Optimal Depth Fitting

For each trial (period, duration, T0), depth is solved via weighted least squares:
```
depth = Σ[(1-y_i) / σ_i²] / Σ[1 / σ_i²]  (in-transit points only)
```

This minimizes chi-squared for the given transit geometry.

### Signal Detection Efficiency (SDE)

The SDE metric quantifies signal significance:
```
SDE = (χ²_null - χ²_best) / σ_red
```

Where:
- `χ²_null`: Chi-squared assuming no transit
- `χ²_best`: Chi-squared for best-fit transit
- `σ_red`: Reduced chi-squared scatter

**SDE > 7** typically indicates a robust detection.

## Known Limitations

1. **Dataset Size**: Insertion sort limits data to ~5000 points
   - For larger datasets, consider binning or multiple searches
   - Future: Could implement radix/merge sort for scalability

2. **Memory**: Requires ~3×N floats of GPU memory per dataset
   - 5000 points: ~60 KB
   - Should work on any GPU with >1GB VRAM

3. **Duration Grid**: Currently uniform in log-space
   - Could optimize further using Ofir-style adaptive sampling

4. **Single GPU**: No multi-GPU support yet
   - Trivial to parallelize across multiple light curves
   - Harder to parallelize single search across GPUs

## Comparison to CPU TLS

### When to Use GPU TLS (`cuvarbase.tls`)

✓ Datasets with 500-5000 points (sweet spot)
✓ Bulk processing of many light curves
✓ Real-time transit searches
✓ When speed is critical (e.g., transient follow-up)
✓ **35-202× faster** for typical datasets

### When to Use CPU TLS (`transitleastsquares`)

✓ Very large datasets (>5000 points)
✓ Need for CPU-side features (limb darkening, eccentricity)
✓ Environments without CUDA-capable GPUs

## Testing

### Pytest Suite

```bash
pytest cuvarbase/tests/test_tls_basic.py -v
```

All 20 unit tests cover:
- Kernel compilation
- Memory allocation
- Period grid generation
- Signal recovery (synthetic transits)
- Edge cases

### End-to-End Validation

```bash
python test_tls_keplerian_api.py
```

Tests both standard and Keplerian modes on synthetic transit data.

### Performance Benchmarks

```bash
python scripts/benchmark_tls.py
```

Systematic comparison across dataset sizes (500-5000 points).

## Implementation Files

### Core Implementation
- `cuvarbase/tls.py` - Main Python API (1157 lines)
- `cuvarbase/tls_grids.py` - Grid generation utilities (312 lines)
- `cuvarbase/kernels/tls.cu` - CUDA kernels (372 lines)

### Testing
- `cuvarbase/tests/test_tls_basic.py` - Unit tests
- `analysis/test_tls_keplerian.py` - Keplerian grid demonstration
- `analysis/test_tls_keplerian_api.py` - End-to-end validation

### Documentation
- `docs/TLS_GPU_README.md` - This file
- `docs/TLS_GPU_IMPLEMENTATION_PLAN.md` - Detailed implementation plan

## References

1. **Hippke & Heller (2019)**: "Optimized transit detection algorithm to search for periodic transits of small planets", A&A 623, A39
   - Original TLS algorithm and SDE metric

2. **Kovács et al. (2002)**: "A box-fitting algorithm in the search for periodic transits", A&A 391, 369
   - BLS algorithm (TLS is a refinement)

3. **Ofir (2014)**: "An Analytic Theory for the Period-Radius Distribution", ApJ 789, 145
   - Optimal period grid sampling

4. **transitleastsquares**: https://github.com/hippke/tls
   - Reference CPU implementation (v1.32)

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
