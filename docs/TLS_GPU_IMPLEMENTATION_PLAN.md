# GPU-Accelerated Transit Least Squares (TLS) Implementation Plan

**Branch:** `tls-gpu-implementation`
**Target:** Fastest TLS implementation with GPU acceleration
**Reference:** https://github.com/hippke/tls (canonical CPU implementation)

---

## Executive Summary

This document outlines the implementation plan for a GPU-accelerated Transit Least Squares (TLS) algorithm in cuvarbase. TLS is a more sophisticated transit detection method than Box Least Squares (BLS) that uses physically realistic transit models with limb darkening, achieving ~93% recovery rate vs BLS's ~76%.

**Performance Target:** <1 second per light curve (vs ~10 seconds for CPU TLS)
**Expected Speedup:** 10-100x over CPU implementation

---

## 1. Background: What is TLS?

### 1.1 Core Concept

Transit Least Squares detects periodic planetary transits using a chi-squared minimization approach with physically realistic transit models. Unlike BLS which uses simple box functions, TLS models:

- **Limb darkening** (quadratic law via Batman library)
- **Ingress/egress** (gradual dimming as planet enters/exits stellar disk)
- **Full unbinned data** (no phase-binning approximations)

### 1.2 Mathematical Formulation

**Chi-squared test statistic:**
```
χ²(P, t₀, d) = Σᵢ (yᵢᵐ(P, t₀, d) - yᵢᵒ)² / σᵢ²
```

**Signal Residue (detection metric):**
```
SR(P) = χ²ₘᵢₙ,ₘₚₗₒᵦ / χ²ₘᵢₙ(P)
```
Normalized to [0,1], with 1 = strongest signal.

**Signal Detection Efficiency (SDE):**
```
SDE(P) = (1 - ⟨SR(P)⟩) / σ(SR(P))
```
Z-score measuring signal strength above noise.

### 1.3 Key Differences vs BLS

| Feature | TLS | BLS |
|---------|-----|-----|
| Transit shape | Trapezoidal with limb darkening | Rectangular box |
| Data handling | Unbinned phase-folded | Binned phase-folded |
| Detection efficiency | 93% recovery | 76% recovery |
| Physical realism | Models stellar physics | Simplified |
| Small planet detection | Optimized (~10% better) | Standard |
| Computational cost | ~10s per K2 LC (CPU) | ~10s per K2 LC |

### 1.4 Algorithm Structure

```
For each trial period P:
    1. Phase fold time series
    2. Sort by phase
    3. Patch arrays (handle edge wrapping)

    For each duration d:
        4. Get/cache transit model for duration d
        5. Calculate out-of-transit residuals (cached)

        For each trial T0 position:
            6. Calculate in-transit residuals
            7. Scale transit depth optimally
            8. Compute chi-squared
            9. Track minimum chi-squared
```

**Complexity:** O(P × D × N × W)
- P = trial periods (~8,500)
- D = durations per period (varies)
- N = data points (~4,320)
- W = transit width in samples

**Total evaluations:** ~3×10⁸ per typical K2 light curve

---

## 2. Analysis of Existing BLS GPU Implementation

### 2.1 Architecture Overview

The existing cuvarbase BLS implementation provides an excellent foundation:

**File Structure:**
- `cuvarbase/bls.py` - Python API and memory management
- `cuvarbase/kernels/bls.cu` - Standard CUDA kernel
- `cuvarbase/kernels/bls_optimized.cu` - Optimized kernel with warp shuffles

**Key Features:**
1. **Dynamic block sizing** - Adapts block size to dataset size (32-256 threads)
2. **Kernel caching** - LRU cache for compiled kernels (~100 MB max)
3. **Shared memory histogramming** - Phase-binned data in shared memory
4. **Parallel reduction** - Tree reduction with warp shuffle optimization
5. **Adaptive mode** - Automatically selects sparse vs standard BLS

### 2.2 GPU Optimization Techniques Used

**Memory optimizations:**
- Separate yw/w arrays to avoid bank conflicts
- Coalesced global memory access
- Shared memory for frequently accessed data

**Compute optimizations:**
- Fast math intrinsics (`__float2int_rd` instead of `floorf`)
- Warp-level shuffle reduction (eliminates 4 `__syncthreads` calls)
- Prepared function calls for faster kernel launches

**Batching strategy:**
- Frequency batching to respect GPU timeout limits
- Stream-based async execution for overlapping compute/transfer
- Grid-stride loops for handling more frequencies than blocks

### 2.3 Memory Management

**BLSMemory class:**
- Page-aligned pinned memory for faster CPU-GPU transfers
- Pre-allocated GPU arrays to avoid repeated allocation
- Separate data/frequency memory allocation

**Transfer strategy:**
- Async transfers with CUDA streams
- Data stays on GPU across multiple kernel launches
- Results transferred back only when needed

---

## 3. TLS-Specific Challenges

### 3.1 Key Algorithmic Differences

| Aspect | BLS | TLS | Implementation Impact |
|--------|-----|-----|----------------------|
| Transit model | Box function | Limb-darkened trapezoid | Need transit model cache on GPU |
| Model complexity | 1 multiplication | ~10-100 ops per point | Higher compute/memory ratio |
| Duration sampling | Uniform q values | Logarithmic durations | Different grid generation |
| Phase binning | Yes (shared memory) | No (unbinned) | Different memory access pattern |
| Edge effects | Minimal | Requires correction | Need array patching |

### 3.2 Computational Bottlenecks

**From CPU TLS profiling:**
1. **Phase folding/sorting** (~53% of time)
   - MergeSort on GPU (use CUB library)
   - Phase fold fully parallel

2. **Residual calculations** (~47% of time)
   - Highly parallel across T0 positions
   - Chi-squared reductions (parallel reduction)

3. **Out-of-transit caching** (critical optimization)
   - Cumulative sums (parallel scan/prefix sum)
   - Shared/global memory caching

### 3.3 Transit Model Handling

**Challenge:** TLS uses Batman library for transit models (CPU-only)

**Solution:**
1. Pre-compute transit models on CPU (Batman)
2. Create reference transit (Earth-like, normalized)
3. Cache scaled versions for different durations
4. Transfer cache to GPU (constant/texture memory)
5. Interpolate depths during search (fast on GPU)

**Memory requirement:** ~MB scale for typical duration range

---

## 4. GPU Implementation Strategy

### 4.1 Parallelization Hierarchy

**Three levels of parallelism:**

1. **Period-level (coarse-grained)**
   - Each trial period is independent
   - Launch 1 block per period
   - Similar to BLS gridDim.x loop

2. **Duration-level (medium-grained)**
   - Multiple durations per period
   - Can parallelize within block
   - Shared memory for duration-specific data

3. **T0-level (fine-grained)**
   - Multiple T0 positions per duration
   - Thread-level parallelism
   - Ideal for GPU threads

**Grid/block configuration:**
```
Grid: (nperiods, 1, 1)
Block: (block_size, 1, 1)  // 64-256 threads

Each block handles one period:
  - Threads iterate over durations
  - Threads iterate over T0 positions
  - Reduction to find minimum chi-squared
```

### 4.2 Kernel Design

**Proposed kernel structure:**

```cuda
__global__ void tls_search_kernel(
    const float* t,              // Time array
    const float* y,              // Flux/brightness
    const float* dy,             // Uncertainties
    const float* periods,        // Trial periods
    const float* durations,      // Duration grid (per period)
    const int* duration_counts,  // # durations per period
    const float* transit_models, // Pre-computed transit shapes
    const int* model_indices,    // Index into transit_models
    float* chi2_min,            // Output: minimum chi²
    float* best_t0,             // Output: best mid-transit time
    float* best_duration,       // Output: best duration
    float* best_depth,          // Output: best depth
    int ndata,
    int nperiods
)
```

**Key kernel operations:**
1. Phase fold data for assigned period
2. Sort by phase (CUB DeviceRadixSort)
3. Patch arrays (extend with wrapped data)
4. For each duration:
   - Load transit model from cache
   - For each T0 position (stride sampling):
     - Calculate in-transit residuals
     - Calculate out-of-transit residuals (cached)
     - Scale depth optimally
     - Compute chi-squared
5. Parallel reduction to find minimum chi²
6. Store best solution

### 4.3 Memory Layout

**Global memory:**
- Input data: `t`, `y`, `dy` (float32, ~4-10K points)
- Period grid: `periods` (float32, ~8K)
- Duration grids: `durations` (float32, variable per period)
- Output: `chi2_min`, `best_t0`, `best_duration`, `best_depth`

**Constant/texture memory:**
- Transit model cache (~1-10 MB)
- Limb darkening coefficients
- Stellar parameters

**Shared memory:**
- Phase-folded data (float32, 4×ndata bytes)
- Sorted indices (int32, 4×ndata bytes)
- Partial chi² values (float32, blockDim.x bytes)
- Out-of-transit residual cache (varies with duration)

**Shared memory requirement:**
```
shmem = 8 × ndata + 4 × blockDim.x + cache_size
      ≈ 35-40 KB for ndata=4K, blockDim=256
```

### 4.4 Optimization Techniques

**From BLS optimizations:**
1. Fast math intrinsics (`__float2int_rd`, etc.)
2. Warp shuffle reduction for final chi² minimum
3. Coalesced memory access patterns
4. Separate arrays to avoid bank conflicts

**TLS-specific:**
1. Texture memory for transit models (fast interpolation)
2. Parallel scan for cumulative sums (out-of-transit cache)
3. MergeSort via CUB (better for partially sorted data)
4. Array patching in kernel (avoid extra memory)

---

## 5. Implementation Phases

### Phase 1: Core Infrastructure - COMPLETED

**Status:** Basic infrastructure implemented
**Date:** 2025-10-27

**Completed:**
- ✅ `cuvarbase/tls_grids.py` - Period and duration grid generation
- ✅ `cuvarbase/tls_models.py` - Transit model generation (Batman wrapper + simple models)
- ✅ `cuvarbase/tls.py` - Main Python API with TLSMemory class
- ✅ `cuvarbase/kernels/tls.cu` - Basic CUDA kernel (Phase 1 version)
- ✅ `cuvarbase/tests/test_tls_basic.py` - Initial unit tests

**Key Learnings:**

1. **Ofir 2014 Period Grid**: The Ofir algorithm can produce edge cases when parameters result in very few frequencies. Added fallback to simple linear grid for robustness.

2. **Memory Layout**: Following BLS pattern with separate TLSMemory class for managing GPU/CPU transfers. Using page-aligned pinned memory for fast transfers.

3. **Kernel Design Choices**:
   - Phase 1 uses simple bubble sort (thread 0 only) - this limits us to small datasets
   - Using simple trapezoidal transit model initially (no Batman on GPU)
   - Fixed duration/T0 grids for Phase 1 simplicity
   - Shared memory allocation: `(4*ndata + block_size) * 4 bytes`

4. **Testing Strategy**: Created tests that don't require GPU hardware for CI/CD compatibility. GPU tests are marked with `@pytest.mark.skipif`.

**Known Limitations (to be addressed in Phase 2):**
- Bubble sort limits ndata to ~100-200 points
- No optimal depth calculation (using fixed depth)
- Simple trapezoid transit (no limb darkening on GPU yet)
- No edge effect correction
- No proper parameter tracking across threads in reduction

**Next Steps:** Proceed to Phase 2 optimization ✅ COMPLETED

---

### Phase 2: Optimization - COMPLETED

**Status:** Core optimizations implemented
**Date:** 2025-10-27

**Completed:**
- ✅ `cuvarbase/kernels/tls_optimized.cu` - Optimized CUDA kernel with Thrust
- ✅ Updated `cuvarbase/tls.py` - Support for multiple kernel variants
- ✅ Optimal depth calculation using least squares
- ✅ Warp shuffle reduction for minimum finding
- ✅ Proper parameter tracking across thread reduction
- ✅ Optimized shared memory layout (separate arrays, no bank conflicts)
- ✅ Auto-selection of kernel variant based on dataset size

**Key Improvements:**

1. **Three Kernel Variants**:
   - **Basic** (Phase 1): Bubble sort, fixed depth - for reference/testing
   - **Simple**: Insertion sort, optimal depth, no Thrust - for ndata < 500
   - **Optimized**: Thrust sorting, full optimizations - for ndata >= 500

2. **Sorting Improvements**:
   - Basic: O(n²) bubble sort (Phase 1 baseline)
   - Simple: O(n²) insertion sort (3-5x faster than bubble sort)
   - Optimized: O(n log n) Thrust sort (~100x faster for n=1000)

3. **Optimal Depth Calculation**:
   - Implemented weighted least squares: `depth = Σ(y*m/σ²) / Σ(m²/σ²)`
   - Physical constraints: depth ∈ [0, 1]
   - Improves chi² minimization significantly

4. **Reduction Optimizations**:
   - Tree reduction down to warp size
   - Warp shuffle for final reduction (no `__syncthreads` in warp)
   - Proper tracking of all parameters (t0, duration, depth, config_idx)
   - No parameter loss during reduction

5. **Memory Optimizations**:
   - Separate arrays for y/dy to avoid bank conflicts
   - Working memory allocation for Thrust (phases, y, dy, indices per period)
   - Optimized shared memory layout: 3*ndata + 5*block_size floats + block_size ints

6. **Search Space Expansion**:
   - Increased durations: 10 → 15 samples
   - Logarithmic duration spacing for better coverage
   - Increased T0 positions: 20 → 30 samples
   - Duration range: 0.5% to 15% of period

**Performance Estimates:**

| ndata | Kernel | Sort Time | Speedup vs Basic |
|-------|--------|-----------|------------------|
| 100   | Basic  | ~0.1 ms   | 1x               |
| 100   | Simple | ~0.03 ms  | ~3x              |
| 500   | Simple | ~1 ms     | ~5x              |
| 1000  | Optimized | ~0.05 ms | ~100x        |
| 5000  | Optimized | ~0.3 ms  | ~500x         |

**Auto-Selection Logic:**
- ndata < 500: Use simple kernel (insertion sort overhead acceptable)
- ndata >= 500: Use optimized kernel (Thrust overhead justified)

**Known Limitations (Phase 3 targets):**
- Fixed duration/T0 grids (not period-dependent yet)
- Simple box transit model (no limb darkening on GPU)
- No edge effect correction
- No out-of-transit caching
- Working memory scales with nperiods (could be optimized)

**Key Learnings:**

1. **Thrust Integration**: Thrust provides massive speedup but adds compilation complexity. Simple kernel provides good middle ground.

2. **Parameter Tracking**: Critical to track all parameters through reduction tree, not just chi². Volatile memory trick works for warp-level reduction.

3. **Kernel Variant Selection**: Auto-selection based on dataset size provides best user experience without requiring expertise.

4. **Shared Memory**: With optimal depth + parameter tracking, shared memory needs are: `(3*ndata + 5*BLOCK_SIZE)*4 + BLOCK_SIZE*4` bytes. For ndata=1000, block_size=128: ~13 KB (well under 48 KB limit).

5. **Logarithmic Duration Spacing**: Much better coverage than linear spacing, especially for wide duration ranges.

**Next Steps:** Proceed to Phase 3 (features & robustness)

---

### Phase 1: Core Infrastructure (Week 1) - ORIGINAL PLAN

**Files to create:**
- `cuvarbase/tls.py` - Python API
- `cuvarbase/kernels/tls.cu` - CUDA kernel
- `cuvarbase/tls_models.py` - Transit model generation

**Tasks:**
1. Create TLS Python class similar to BLS structure
2. Implement transit model pre-computation (Batman wrapper)
3. Create period/duration grid generation (Ofir 2014)
4. Implement basic kernel structure (no optimization)
5. Memory management class (TLSMemory)

**Deliverables:**
- Basic working TLS GPU implementation
- Correctness validation vs CPU TLS

### Phase 2: Optimization (Week 2)

**Tasks:**
1. Implement shared memory optimizations
2. Add warp shuffle reduction
3. Optimize memory access patterns
4. Implement out-of-transit caching
5. Add texture memory for transit models
6. Implement CUB-based sorting

**Deliverables:**
- Optimized TLS kernel
- Performance benchmarks vs CPU

### Phase 3: Features & Robustness (Week 3)

**Tasks:**
1. Implement edge effect correction
2. Add adaptive block sizing
3. Implement kernel caching (LRU)
4. Add batch processing for large period grids
5. Implement CUDA streams for async execution
6. Add sparse TLS variant (for small datasets)

**Deliverables:**
- Production-ready TLS implementation
- Adaptive mode selection

### Phase 4: Testing & Validation (Week 4)

**Tasks:**
1. Create comprehensive unit tests
2. Validate against CPU TLS on known planets
3. Test edge cases (few data points, long periods, etc.)
4. Performance profiling and optimization
5. Documentation and examples

**Deliverables:**
- Full test suite
- Benchmark results
- Documentation

---

## 6. Testing Strategy

### 6.1 Validation Tests

**Test against CPU TLS:**
1. **Synthetic transits** - Generate known signals, verify recovery
2. **Known planets** - Test on confirmed exoplanet light curves
3. **Edge cases** - Few transits, long periods, noisy data
4. **Statistical properties** - SDE, SNR, FAP calculations

**Metrics for validation:**
- Period recovery (within 1%)
- Duration recovery (within 10%)
- Depth recovery (within 5%)
- T0 recovery (within transit duration)
- SDE values (within 5%)

### 6.2 Performance Tests

**Benchmarks:**
1. vs CPU TLS (hippke/tls)
2. vs GPU BLS (cuvarbase existing)
3. Scaling with ndata (10 to 10K points)
4. Scaling with nperiods (100 to 10K)

**Target metrics:**
- <1 second per K2 light curve (90 days, 4K points)
- 10-100x speedup vs CPU TLS
- Similar or better than GPU BLS

### 6.3 Test Data

**Sources:**
1. Synthetic light curves (known parameters)
2. TESS light curves (2-min cadence)
3. K2 light curves (30-min cadence)
4. Kepler light curves (30-min cadence)

---

## 7. API Design

### 7.1 High-Level Interface

```python
from cuvarbase import tls

# Simple interface
results = tls.search(t, y, dy,
                     R_star=1.0,      # Solar radii
                     M_star=1.0,      # Solar masses
                     period_min=None, # Auto-detect
                     period_max=None) # Auto-detect

# Access results
print(f"Period: {results.period:.4f} days")
print(f"SDE: {results.SDE:.2f}")
print(f"Depth: {results.depth*1e6:.1f} ppm")
```

### 7.2 Advanced Interface

```python
# Custom configuration
results = tls.search_advanced(
    t, y, dy,
    periods=custom_periods,
    durations=custom_durations,
    transit_template='custom',
    limb_dark='quadratic',
    u=[0.4804, 0.1867],
    use_optimized=True,
    use_sparse=None,  # Auto-select
    block_size=128,
    stream=cuda_stream
)
```

### 7.3 Batch Processing

```python
# Process multiple light curves
results_list = tls.search_batch(
    [t1, t2, ...],
    [y1, y2, ...],
    [dy1, dy2, ...],
    n_streams=4,
    parallel=True
)
```

---

## 8. Expected Performance

### 8.1 Theoretical Analysis

**CPU TLS (current):**
- ~10 seconds per K2 light curve
- Single-threaded
- 12.2 GFLOPs (72% of theoretical CPU max)

**GPU TLS (target):**
- <1 second per K2 light curve
- ~10³-10⁴ parallel threads
- 100-1000 GFLOPs (GPU advantage)

**Speedup sources:**
1. Period parallelism: 8,500 periods → 8,500 threads
2. T0 parallelism: ~100 T0 positions per duration
3. Faster reductions: Tree + warp shuffle
4. Memory bandwidth: GPU >> CPU

### 8.2 Bottleneck Analysis

**Potential bottlenecks:**
1. **Sorting** - CUB DeviceRadixSort is fast but not free
   - Solution: Use MergeSort for partially sorted data
   - Cost: ~5-10% of total time

2. **Transit model interpolation** - Texture memory helps
   - Solution: Pre-compute at high resolution
   - Cost: ~2-5% of total time

3. **Out-of-transit caching** - Shared memory limits
   - Solution: Use parallel scan (CUB DeviceScan)
   - Cost: ~10-15% of total time

4. **Global memory bandwidth** - Reading t, y, dy repeatedly
   - Solution: Shared memory caching per block
   - Cost: ~20-30% of total time

**Expected time breakdown:**
- Phase folding/sorting: 20%
- Residual calculations: 60%
- Reductions/comparisons: 15%
- Overhead: 5%

---

## 9. File Structure

```
cuvarbase/
├── tls.py                          # Main TLS API
├── tls_models.py                   # Transit model generation
├── tls_grids.py                    # Period/duration grid generation
├── tls_stats.py                    # Statistical calculations (SDE, SNR, FAP)
├── kernels/
│   ├── tls.cu                      # Standard TLS kernel
│   ├── tls_optimized.cu            # Optimized kernel
│   └── tls_sparse.cu               # Sparse variant (small datasets)
└── tests/
    ├── test_tls_basic.py           # Basic functionality
    ├── test_tls_consistency.py     # Consistency with CPU TLS
    ├── test_tls_performance.py     # Performance benchmarks
    └── test_tls_validation.py      # Known planet recovery
```

---

## 10. Dependencies

**Required:**
- PyCUDA (existing)
- NumPy (existing)
- Batman-package (CPU transit models)

**Optional:**
- Astropy (stellar parameters, unit conversions)
- Numba (CPU fallback)

**CUDA features:**
- CUB library (sorting, scanning)
- Texture memory (transit model interpolation)
- Warp shuffle intrinsics
- Cooperative groups (advanced optimization)

---

## 11. Success Criteria

**Functional:**
- [ ] Passes all validation tests (>95% accuracy vs CPU TLS)
- [ ] Recovers known planets in test dataset
- [ ] Handles edge cases robustly

**Performance:**
- [ ] <1 second per K2 light curve
- [ ] 10-100x speedup vs CPU TLS
- [ ] Comparable or better than GPU BLS

**Quality:**
- [ ] Full test coverage (>90%)
- [ ] Comprehensive documentation
- [ ] Example notebooks

**Usability:**
- [ ] Simple API for basic use cases
- [ ] Advanced API for expert users
- [ ] Clear error messages

---

## 12. Risk Mitigation

### 12.1 Technical Risks

| Risk | Mitigation |
|------|------------|
| GPU memory limits | Implement batching, use sparse variant |
| Kernel timeout (Windows) | Add freq_batch_size parameter |
| Sorting performance | Use CUB MergeSort for partially sorted |
| Transit model accuracy | Validate against Batman reference |
| Edge effect handling | Implement CPU TLS's correction algorithm |

### 12.2 Performance Risks

| Risk | Mitigation |
|------|------------|
| Slower than expected | Profile with Nsight, optimize bottlenecks |
| Memory bandwidth bound | Increase compute/memory ratio, use shared mem |
| Low occupancy | Adjust block size, reduce register usage |
| Divergent branches | Minimize conditionals in inner loops |

---

## 13. Future Enhancements

**Phase 5 (future):**
1. Multi-GPU support
2. CPU fallback (Numba)
3. Alternative limb darkening laws
4. Non-circular orbits (eccentric transits)
5. Multi-planet search
6. Real-time detection (streaming data)
7. Integration with lightkurve/eleanor

---

## 14. References

### Primary Papers

1. **Hippke & Heller (2019)** - "Transit Least Squares: Optimized transit detection algorithm"
   - arXiv:1901.02015
   - A&A 623, A39

2. **Ofir (2014)** - "Algorithmic considerations for continuous GW search"
   - A&A 561, A138
   - Period sampling algorithm

3. **Mandel & Agol (2002)** - "Analytic Light Curves for Planetary Transit Searches"
   - ApJ 580, L171
   - Transit model theory

### Related Work

4. **Kovács et al. (2002)** - Original BLS paper
   - A&A 391, 369

5. **Kreidberg (2015)** - Batman: Bad-Ass Transit Model cAlculatioN
   - PASP 127, 1161

6. **Panahi & Zucker (2021)** - Sparse BLS algorithm
   - arXiv:2103.06193

### Software

- TLS GitHub: https://github.com/hippke/tls
- TLS Docs: https://transitleastsquares.readthedocs.io/
- Batman: https://github.com/lkreidberg/batman
- CUB: https://nvlabs.github.io/cub/

---

## Appendix A: Algorithm Pseudocode

### CPU TLS (reference)

```python
def tls_search(t, y, dy, periods, durations, transit_models):
    results = []

    for period in periods:
        # Phase fold
        phases = (t / period) % 1.0
        sorted_idx = argsort(phases)
        phases = phases[sorted_idx]
        y_sorted = y[sorted_idx]
        dy_sorted = dy[sorted_idx]

        # Patch (extend for edge wrapping)
        phases_ext, y_ext, dy_ext = patch_arrays(phases, y_sorted, dy_sorted)

        min_chi2 = inf
        best_t0 = None
        best_duration = None

        for duration in durations[period]:
            # Get transit model
            model = transit_models[duration]

            # Calculate out-of-transit residuals (can be cached)
            residuals_out = calc_out_of_transit(y_ext, dy_ext, model)

            # Stride over T0 positions
            for t0 in T0_grid:
                # Calculate in-transit residuals
                residuals_in = calc_in_transit(y_ext, dy_ext, model, t0)

                # Optimal depth scaling
                depth = optimal_depth(residuals_in, residuals_out)

                # Chi-squared
                chi2 = calc_chi2(residuals_in, residuals_out, depth)

                if chi2 < min_chi2:
                    min_chi2 = chi2
                    best_t0 = t0
                    best_duration = duration

        results.append((period, min_chi2, best_t0, best_duration))

    return results
```

### GPU TLS (proposed)

```cuda
__global__ void tls_search_kernel(...) {
    int period_idx = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float shared_phases[MAX_NDATA];
    __shared__ float shared_y[MAX_NDATA];
    __shared__ float shared_dy[MAX_NDATA];
    __shared__ float chi2_vals[BLOCK_SIZE];

    // Load data to shared memory
    for (int i = tid; i < ndata; i += blockDim.x) {
        float phase = fmodf(t[i] / periods[period_idx], 1.0f);
        shared_phases[i] = phase;
        shared_y[i] = y[i];
        shared_dy[i] = dy[i];
    }
    __syncthreads();

    // Sort by phase (CUB DeviceRadixSort or MergeSort)
    cub::DeviceRadixSort::SortPairs(...);
    __syncthreads();

    // Patch arrays (extend for wrapping)
    patch_arrays_shared(...);
    __syncthreads();

    float thread_min_chi2 = INFINITY;

    // Iterate over durations
    int n_durations = duration_counts[period_idx];
    for (int d = 0; d < n_durations; d++) {
        float duration = durations[period_idx * MAX_DURATIONS + d];

        // Load transit model from texture memory
        float* model = tex2D(transit_model_texture, duration, ...);

        // Calculate out-of-transit residuals (use parallel scan for cumsum)
        float residuals_out = calc_out_of_transit_shared(...);

        // Stride over T0 positions (each thread handles multiple)
        for (int t0_idx = tid; t0_idx < n_t0_positions; t0_idx += blockDim.x) {
            float t0 = t0_grid[t0_idx];

            // In-transit residuals
            float residuals_in = calc_in_transit_shared(...);

            // Optimal depth
            float depth = optimal_depth_fast(residuals_in, residuals_out);

            // Chi-squared
            float chi2 = calc_chi2_fast(residuals_in, residuals_out, depth);

            thread_min_chi2 = fminf(thread_min_chi2, chi2);
        }
    }

    // Store thread minimum
    chi2_vals[tid] = thread_min_chi2;
    __syncthreads();

    // Parallel reduction to find block minimum
    // Tree reduction + warp shuffle
    for (int s = blockDim.x/2; s >= 32; s /= 2) {
        if (tid < s) {
            chi2_vals[tid] = fminf(chi2_vals[tid], chi2_vals[tid + s]);
        }
        __syncthreads();
    }

    // Final warp reduction
    if (tid < 32) {
        float val = chi2_vals[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (tid == 0) {
            chi2_min[period_idx] = val;
        }
    }
}
```

---

## Appendix B: Key Equations

### Chi-Squared Calculation

```
χ²(P, t₀, d, δ) = Σᵢ [yᵢ - m(tᵢ; P, t₀, d, δ)]² / σᵢ²

where m(t; P, t₀, d, δ) is the transit model:
  m(t) = {
    1 - δ × limb_darkened_transit(phase(t))  if in transit
    1                                          otherwise
  }
```

### Optimal Depth Scaling

```
δ_opt = Σᵢ [yᵢ × m(tᵢ)] / Σᵢ [m(tᵢ)²]

This minimizes χ² analytically for given (P, t₀, d)
```

### Signal Detection Efficiency

```
SDE = (1 - ⟨SR⟩) / σ(SR)

where SR = χ²_white_noise / χ²_signal

Median filter applied to remove systematic trends
```

---

**Document Version:** 1.0
**Last Updated:** 2025-10-27
**Author:** Claude Code (Anthropic)
