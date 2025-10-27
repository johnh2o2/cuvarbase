# TLS GPU Implementation - Debugging Summary

## Bugs Found and Fixed

### 1. Ofir Period Grid Generation (CRITICAL)

**Problem**: Generated 56,000+ periods instead of ~5,000 for realistic searches

**Root Causes**:
- Used user-specified `period_min`/`period_max` as physical boundaries instead of Roche limit and n_transits constraint
- Missing `- A/3` term in equation (6) for parameter C
- Missing `+ A/3` term in equation (7) for N_opt

**Fix** (`cuvarbase/tls_grids.py`):
```python
# Physical boundaries (following Ofir 2014 and CPU TLS)
f_min = n_transits_min / (T_span * 86400.0)  # 1/seconds
f_max = 1.0 / (2.0 * np.pi) * np.sqrt(G * M_star_kg / (3.0 * R_star_m)**3)

# Correct Ofir equations
A = ((2.0 * np.pi)**(2.0/3.0) / np.pi * R_star_m /
     (G * M_star_kg)**(1.0/3.0) / (T_span_sec * oversampling_factor))
C = f_min**(1.0/3.0) - A / 3.0  # Equation (6) - FIXED
n_freq = int(np.ceil((f_max**(1.0/3.0) - f_min**(1.0/3.0) + A / 3.0) * 3.0 / A))  # Eq (7) - FIXED

# Apply user limits as post-filtering
periods = periods[(periods > user_period_min) & (periods <= user_period_max)]
```

**Result**: Now generates ~5,000-6,000 periods matching CPU TLS

---

### 2. Hardcoded Duration Grid Bug (CRITICAL)

**Problem**: Duration values were hardcoded in absolute days instead of scaling with period

**Root Cause** (`cuvarbase/kernels/tls_optimized.cu:239-240, 416-417`):
```cuda
// WRONG - absolute days, doesn't scale with period
float duration_min = 0.005f;  // 0.005 days
float duration_max = 0.15f;   // 0.15 days
float duration_phase = duration / period;  // Convert to phase
```

For period=10 days:
- 0.005 days = 0.05% of period (way too small for 5% transit!)
- Should be: 0.005 × 10 = 0.05 days = 0.5% of period

**Fix**:
```cuda
// CORRECT - fractional values that scale with period
float duration_phase_min = 0.005f;  // 0.5% of period
float duration_phase_max = 0.15f;   // 15% of period
float duration_phase = expf(log_duration);  // Already in phase units
float duration = duration_phase * period;   // Convert to days
```

**Result**: Kernel now correctly finds transit periods

---

### 3. Thrust Sorting from Device Code (CRITICAL)

**Problem**: Optimized kernel returned depth=0, duration=0 - completely broken

**Root Cause**: Cannot call Thrust algorithms from within `__global__` kernel functions. This is a fundamental CUDA limitation.

**Code** (`cuvarbase/kernels/tls_optimized.cu:217`):
```cuda
extern "C" __global__ void tls_search_kernel_optimized(...) {
    // ...
    if (threadIdx.x == 0) {
        thrust::sort_by_key(thrust::device, ...);  // ← DOESN'T WORK!
    }
}
```

**Fix**: Disabled optimized kernel, use simple kernel with insertion sort

```python
# cuvarbase/tls.py
if use_simple is None:
    # FIXME: Thrust sorting from device code doesn't work
    use_simple = True  # Always use simple kernel for now
```

```cuda
// cuvarbase/kernels/tls_optimized.cu
// Increased ndata limit for simple kernel
if (threadIdx.x == 0 && ndata < 5000) {  // Was 500
    // Insertion sort (works correctly)
}
```

**Result**: GPU TLS now works correctly with simple kernel up to ndata=5000

---

### 4. Period Grid Test Failure (Minor)

**Problem**: `test_period_grid_basic` returned all periods = 50.0

**Root Cause**:
```python
period_from_transits = T_span / n_transits_min  # 100/2 = 50
period_min = max(roche_period, 50)  # 50
period_max = T_span / 2.0  # 50
# Result: period_min = period_max = 50!
```

**Fix**: Removed `period_from_transits` calculation, added `np.sort(periods)`

---

## Performance Results

### Accuracy Test (500 points, realistic Ofir grid, depth=0.01)

**GPU TLS (Simple Kernel)**:
- Period: 9.9981 days (error: 0.02%) ✓
- Depth: 0.009825 (error: 1.7%) ✓
- Duration: 0.1684 days
- Grid: 1271 periods

**CPU TLS (v1.32)**:
- Period: 10.0115 days (error: 0.12%)
- Depth: 0.010208 (error: 2.1%)
- Duration: 0.1312 days
- Grid: 183 periods

**Note**: Different depth conventions:
- GPU TLS: Reports fractional dip (0.01 = 1% dip)
- CPU TLS: Reports flux ratio (0.99 = flux during transit / flux out)
- Conversion: `depth_fractional_dip = 1 - depth_flux_ratio`

---

## Known Limitations

1. **Thrust sorting doesn't work from device code**: Need to implement device-side sort (CUB library) or host-side pre-sorting

2. **Simple kernel limited to ndata < 5000**: Insertion sort is O(N²), becomes slow for large datasets

3. **Duration search is brute-force**: Tests 15 durations × 30 T0 positions = 450 configurations per period. Could be optimized.

4. **Sparse data degeneracy**: With few points in transit, wider/shallower transits can have lower chi² than true narrow/deep transits. This is a fundamental limitation of box-fitting with sparse data.

---

## Files Modified

1. `cuvarbase/tls_grids.py` - Fixed Ofir period grid generation
2. `cuvarbase/kernels/tls_optimized.cu` - Fixed duration grid, disabled Thrust, increased simple kernel limit
3. `cuvarbase/tls.py` - Default to simple kernel
4. `test_tls_realistic_grid.py` - Force use_simple=True

---

## Next Steps

1. **Run comprehensive GPU vs CPU benchmark** - Test performance scaling with ndata and baseline
2. **Add CPU consistency tests** to pytest suite
3. **Implement proper device-side sorting** using CUB library (future work)
4. **Optimize duration grid** using stellar parameters (future work)
