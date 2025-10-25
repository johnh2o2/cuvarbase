# Dynamic Block Size Design

## Problem Statement

Current BLS kernel uses fixed block size of 256 threads, leading to poor utilization for small ndata:
- ndata=10: 10/256 = 3.9% utilization
- ndata=100: 100/256 = 39% utilization
- ndata=1000: Uses multiple iterations, better utilization
- ndata=10000: Good utilization

## Strategy

### Block Size Selection

Choose block size based on ndata to maximize GPU utilization:

```
if ndata <= 32:
    block_size = 32   # Single warp
elif ndata <= 64:
    block_size = 64   # Two warps
elif ndata <= 128:
    block_size = 128  # Four warps
else:
    block_size = 256  # Default (8 warps)
```

### Thread Utilization Analysis

| ndata | Old Block | Old Util | New Block | New Util | Improvement |
|-------|-----------|----------|-----------|----------|-------------|
| 10    | 256       | 3.9%     | 32        | 31.3%    | 8x better   |
| 50    | 256       | 19.5%    | 64        | 78.1%    | 4x better   |
| 100   | 256       | 39.1%    | 128       | 78.1%    | 2x better   |
| 500   | 256       | 97.7%    | 256       | 97.7%    | Same        |
| 1000+ | 256       | 100%*    | 256       | 100%*    | Same        |

*Multiple iterations, full utilization

### Expected Performance Impact

**Small ndata (10-100)**:
- Current: Kernel launch overhead dominates (~0.17s)
- With dynamic sizing:
  - Fewer idle threads → less warp divergence
  - More frequencies per kernel launch → amortize overhead
  - **Expected: 2-5x speedup**

**Large ndata (>1000)**:
- Current: Good utilization already
- With dynamic sizing: No change (still use 256)
- **Expected: No regression**

## Implementation Plan

### Phase 1: Add block_size parameter support

Currently `compile_bls()` takes block_size but needs to be called for each size:
```python
def eebls_gpu_fast_adaptive(t, y, dy, freqs, qmin=1e-2, qmax=0.5, **kwargs):
    # Determine optimal block size
    ndata = len(t)
    if ndata <= 32:
        block_size = 32
    elif ndata <= 64:
        block_size = 64
    elif ndata <= 128:
        block_size = 128
    else:
        block_size = 256

    # Compile kernel with appropriate block size
    functions = compile_bls(block_size=block_size, use_optimized=True, **kwargs)

    # Call kernel
    return eebls_gpu_fast(t, y, dy, freqs, qmin=qmin, qmax=qmax,
                          functions=functions, **kwargs)
```

### Phase 2: Kernel caching

Avoid recompiling for same block size:
```python
_kernel_cache = {}  # (block_size, optimized) -> functions

def get_compiled_kernels(block_size, use_optimized=False):
    key = (block_size, use_optimized)
    if key not in _kernel_cache:
        _kernel_cache[key] = compile_bls(block_size=block_size,
                                         use_optimized=use_optimized)
    return _kernel_cache[key]
```

### Phase 3: Batch optimization for very small ndata

For ndata < 32, process multiple frequencies per block:
- 1 block handles multiple frequencies sequentially
- Reduces kernel launch overhead further
- **Expected: Additional 2x improvement for ndata < 32**

## Shared Memory Considerations

Shared memory usage scales with:
- Histogram bins: `2 * max_nbins * sizeof(float)`
- Reduction array: `block_size * sizeof(float)`
- Total: `(2 * max_nbins + block_size) * 4 bytes`

Smaller block sizes → more room for bins → can handle smaller qmin values!

Example (48KB shared memory limit):
- block_size=256: max_nbins = (48000 - 1024) / 8 = 5872 bins
- block_size=32:  max_nbins = (48000 - 128) / 8 = 5984 bins

Minimal difference, not a concern.

## Risks & Mitigations

### Risk 1: Kernel compilation overhead
**Mitigation**: Cache compiled kernels, compile on first use

### Risk 2: Different results with different block sizes
**Mitigation**: Atomic operations ensure same results regardless of thread count

### Risk 3: Warp shuffle assumes 32 threads
**Mitigation**: Current code already handles this correctly - final reduction always uses 32 threads

### Risk 4: Increased code complexity
**Mitigation**: Keep it simple - just choose block size, rest is unchanged

## Testing Strategy

1. **Correctness**: Run same test data with all block sizes (32, 64, 128, 256)
   - Verify results match within floating-point precision

2. **Performance**: Benchmark ndata=[10, 20, 50, 100, 200, 500, 1000, 5000, 10000]
   - Compare fixed 256 vs dynamic sizing

3. **Regression**: Ensure no slowdown for ndata > 1000

## Success Criteria

- ✓ No correctness issues (differences < 1e-6)
- ✓ 2x+ speedup for ndata < 100
- ✓ 5x+ speedup for ndata < 32
- ✓ No regression for ndata > 1000
