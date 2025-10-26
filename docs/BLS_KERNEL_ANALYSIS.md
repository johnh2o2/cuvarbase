# BLS Kernel Optimization Analysis

## Baseline Performance

**Hardware**: RTX 4000 Ada Generation
**Test**: ndata=[10, 100, 1000, 10000], nfreq=1000

| ndata | Time (s) | Throughput (M eval/s) |
|-------|----------|-----------------------|
| 10    | 0.146    | 0.07                  |
| 100   | 0.145    | 0.69                  |
| 1000  | 0.148    | 6.75                  |
| 10000 | 0.151    | 66.06                 |

**Key Observation**: Time is nearly constant (~0.15s) regardless of ndata! This suggests we're **kernel-launch or overhead bound**, not compute-bound.

## Current Implementation Analysis

### Main Kernel: `full_bls_no_sol`

**Architecture**:
- 1 block per frequency
- Each block processes all ndata points for its frequency
- Shared memory histogram (2 floats per bin)
- Reduction within block to find maximum BLS

**Current Parallelism Strategy**:
```cuda
// Line 207: One block per frequency
unsigned int i_freq = blockIdx.x;
while (i_freq < nfreq){
    // All threads in block work together
    ...
    i_freq += gridDim.x;
}
```

## Optimization Opportunities

### 1. **Memory Access Patterns** (HIGH IMPACT)

**Current**: Global memory reads in inner loop
```cuda
// Line 240-247: Each thread reads from global memory
for (unsigned int k = threadIdx.x; k < ndata; k += blockDim.x){
    phi = mod1(t[k] * f0);  // Read t[k] from global memory
    ...
    atomicAdd(&(block_bins[2 * b]), yw[k]);   // Read yw[k]
    atomicAdd(&(block_bins[2 * b + 1]), w[k]); // Read w[k]
}
```

**Opportunity**:
- All blocks read the same `t`, `yw`, `w` arrays
- Could use **texture memory** or **constant memory** for read-only data
- Or load data into **shared memory** first (already supported via `USE_LOG_BIN_SPACING`)

**Expected Impact**: 10-30% speedup from better memory coalescing

### 2. **Atomic Operations on Shared Memory** (MEDIUM IMPACT)

**Current**: Shared memory atomics in histogram
```cuda
// Line 246-247
atomicAdd(&(block_bins[2 * b]), yw[k]);
atomicAdd(&(block_bins[2 * b + 1]), w[k]);
```

**Issue**:
- Atomic operations serialize writes to the same bin
- With many threads and few bins, this creates contention

**Opportunity**:
- Use **warp-level primitives** (shuffle operations) to reduce atomics
- Each warp could accumulate locally, then one thread per warp writes
- Or use **private histograms** per warp, then merge

**Expected Impact**: 20-40% speedup for large ndata

### 3. **Bank Conflicts in Shared Memory** (MEDIUM IMPACT)

**Current**: Interleaved yw and w storage
```cuda
// Line 193: float *block_bins = sh;
// Stores: [yw0, w0, yw1, w1, yw2, w2, ...]
block_bins[2 * k]     = yw
block_bins[2 * k + 1] = w
```

**Issue**:
- When multiple threads access `block_bins[2*b]` where `b` varies
- Can cause bank conflicts (threads in same warp accessing same bank)

**Opportunity**:
- Separate arrays: `[yw0, yw1, ..., ywN, w0, w1, ..., wN]`
- Or pad arrays to avoid bank conflicts

**Expected Impact**: 5-15% speedup

### 4. **Reduction Algorithm** (LOW-MEDIUM IMPACT)

**Current**: Tree reduction for finding max
```cuda
// Line 308-316: Standard tree reduction
for(unsigned int k = (blockDim.x / 2); k > 0; k /= 2){
    if(threadIdx.x < k){
        ...
    }
    __syncthreads();
}
```

**Opportunity**:
- Use **warp shuffle instructions** for final warp (no sync needed)
- Reduces 5 synchronization points to 1 for 256-thread blocks

**Expected Impact**: 5-10% speedup

### 5. **Kernel Launch Overhead** (HIGH IMPACT for small ndata)

**Current**: Single kernel launch for all frequencies
- Grid size = nfreq (or max allowed)
- Block size = 256 threads

**Issue**:
- For ndata=10, each block has 256 threads but only 10 work items
- Thread utilization: 10/256 = 3.9%!

**Opportunity**:
- **Dynamic block size** based on ndata
- For small ndata: use smaller blocks, more blocks per freq
- Or **batch multiple frequencies per block**

**Expected Impact**: 2-5x speedup for ndata < 100

### 6. **Math Operations** (LOW IMPACT)

**Current**: Uses single precision floats
- `floorf`, `mod1`, etc.

**Opportunity**:
- Use fast math intrinsics (`__float2int_rd` instead of `floorf`)
- Already uses `--use_fast_math` in compilation

**Expected Impact**: 2-5% speedup

## Priority Ranking

1. **🔥 HIGH**: Kernel launch overhead (5x potential for small ndata)
2. **🔥 HIGH**: Memory access patterns (30% potential)
3. **🟡 MEDIUM**: Atomic operation reduction (40% potential)
4. **🟡 MEDIUM**: Bank conflicts (15% potential)
5. **🟢 LOW**: Reduction algorithm (10% potential)
6. **🟢 LOW**: Math intrinsics (5% potential)

## Implementation Strategy

### Phase 1: Quick Wins (Target: 20-30% improvement)
1. Add texture memory for read-only data (`t`, `yw`, `w`)
2. Fix bank conflicts (separate yw/w arrays)
3. Use fast math intrinsics explicitly

### Phase 2: Atomic Reduction (Target: additional 20-40%)
1. Implement warp-level reduction for atomics
2. Private histograms per warp

### Phase 3: Dynamic Block Sizing (Target: 2-5x for small ndata)
1. Choose block size based on ndata
2. Or batch multiple frequencies per block for small ndata

## Baseline vs Target Performance

| ndata  | Baseline (s) | Target (s) | Speedup |
|--------|--------------|------------|---------|
| 10     | 0.146        | 0.03       | 5x      |
| 100    | 0.145        | 0.10       | 1.5x    |
| 1000   | 0.148        | 0.08       | 1.8x    |
| 10000  | 0.151        | 0.08       | 1.9x    |

**Total potential**: 50-70% speedup for typical cases, 5x for small ndata

## Next Steps

1. Implement Phase 1 optimizations
2. Benchmark and verify
3. Iterate with Phase 2
4. Profile with nsys/nvprof to validate assumptions
