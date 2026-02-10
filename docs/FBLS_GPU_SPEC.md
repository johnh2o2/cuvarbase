# Spec: GPU-Accelerated Fast Folding BLS (fBLS)

## 1. Motivation

cuvarbase's current BLS kernel (`full_bls_no_sol` in `kernels/bls.cu`) does this for each trial frequency:

1. **Bin** all N observations into m phase bins via `atomicAdd` to shared memory — O(N) per frequency
2. **Scan** across (bin_start, bin_width) combinations to find max SR — O(m × n_widths) per frequency

Step 1 costs O(N × N_f) total. GPU parallelism across frequencies makes this fast in wall-clock time, but every data point is re-binned for every trial frequency. The Fast Folding Algorithm (FFA) eliminates this redundancy: it generates all folded profiles simultaneously in O(N_p × m × log N_p) total, where N_p is the number of trial periods and m is the number of phase bins.

For Kepler-class data (N=65K, N_p=131K), the theoretical speedup for the folding step is N/log₂(N_p) ≈ 65000/17 ≈ 3800x. Even accounting for the scoring step (which is the same for both methods), a GPU fBLS could be substantially faster than the current GPU BLS.

**Key property: fBLS produces identical output to the current binned BLS.** The same Signal Residue statistic, the same periodogram shape, the same detected periods. Zero accuracy sacrifice.

## 2. Algorithm Overview

### Standard BLS (current)

```
For each frequency f:                          O(N_f) iterations
    phase_i = frac(t_i × f) for all i         O(N)
    Bin phases into m bins                     O(N) with atomics
    Scan box across bins → max SR              O(m × n_widths)
```

Total: O(N_f × (N + m × n_widths))

### FFA-BLS (proposed)

```
Choose base section length m (= number of phase bins)
Divide time series into N_p = 2^n sections     O(N)

Level 0 — Initialize:
    For each section pair:                     N_p/2 pairs
        Bin section's observations into m bins O(N/N_p) per section
        Two shift variants (0, 1)              × 2
                                               = O(N) total

Levels 1 through n-1 — Butterfly:
    For each level l:                          log₂(N_p) levels
        For each combine:                      N_p combines
            Add two m-bin profiles w/ shift    O(m)
                                               = O(N_p × m) per level
                                               = O(N_p × m × log N_p) total

Scoring:
    For each of N_p folds:                     N_p iterations
        Scan box across m bins → max SR        O(m × n_widths)
                                               = O(N_p × m × n_widths) total
```

Total: O(N + N_p × m × (log N_p + n_widths))

The N_p × m × n_widths scoring term is common to both algorithms. The win is replacing O(N_f × N) folding with O(N + N_p × m × log N_p). Since m ≪ N, this is a large improvement.

## 3. Period Grid Structure

### How the FFA defines its period grid

The FFA with section length m (in cadence units) and N_p = 2^n sections produces N_p trial periods:

```
P(i) = (m + i / (N_p - 1)) × dt,    i = 0, 1, ..., N_p - 1
```

where dt is the cadence. These are **uniformly spaced in period** within the octave [m × dt, (m+1) × dt].

Period resolution: δP = dt / (N_p - 1) ≈ P² / (T × m), comparable to the Rayleigh resolution.

### Covering a broad period range

Each value of m covers one period octave of width dt. To search from P_min to P_max:

```
m_min = floor(P_min / dt)
m_max = ceil(P_max / dt)
```

Run the FFA independently for each m in [m_min, m_max]. Each octave is independent and can run in parallel.

Number of octaves: (P_max - P_min) / dt. For P=[0.5, 100]d with 2-minute cadence: ~72,000 octaves. This sounds like a lot, but each octave's butterfly operates on just m-element arrays and is very cheap.

### Keplerian grid compatibility

The Keplerian frequency grid (non-uniform spacing) doesn't map directly onto the FFA's period grid. Two approaches:

**Option A — Use the FFA's native period grid.** Accept the FFA's arithmetic-within-octave spacing. This is slightly denser than a Keplerian grid at short periods (where Keplerian spacing is coarser) and slightly sparser at long periods. For a first implementation, this is simplest.

**Option B — Keplerian octave selection.** Run the FFA only for octaves that contain Keplerian grid frequencies. Skip octaves that fall between Keplerian grid points. This recovers most of the Keplerian grid's frequency reduction without modifying the FFA internals. The Keplerian grid already implies which periods to search — just translate those periods to octaves.

**Recommendation**: Start with Option A. Benchmark against current BLS with Keplerian grid to see if the FFA's algorithmic advantage outweighs the extra frequencies from not using Keplerian spacing.

## 4. Detailed Algorithm for Irregular Sampling

Astronomical data is irregularly sampled. The first FFA level must handle this.

### Preprocessing (CPU, one-time)

```python
# Sort observations by time
order = np.argsort(t)
t_sorted, yw_sorted, w_sorted = t[order], yw[order], w[order]

# For a given section length m (in bins) and cadence dt:
P0 = m * dt  # base period for this octave
N_p = next_power_of_2(T_total / P0)  # number of sections

# Compute section boundaries
section_starts = np.searchsorted(t_sorted, np.arange(N_p) * P0)
section_ends = np.searchsorted(t_sorted, np.arange(1, N_p + 1) * P0)
```

Transfer `t_sorted`, `yw_sorted`, `w_sorted`, `section_starts`, `section_ends` to GPU.

### Level 0: Brute-Force Binning (GPU kernel)

For each pair of adjacent sections (s, s+1), bin observations into m phase bins at two drift values (0 and 1):

```
Kernel: ffa_init_kernel
Grid: (N_p / 2) blocks
Block: 128 threads (or adaptive based on section size)

For each pair (2*blockIdx.x, 2*blockIdx.x + 1):
    // Bin section 2*blockIdx.x
    for each obs k in section 2*blockIdx.x:  (threads cooperate)
        phase = frac(t[k] / P0)
        bin = floor(m * phase)
        atomicAdd(&yw_bins[pair][0][bin], yw[k])  // drift=0
        atomicAdd(&w_bins[pair][0][bin], w[k])

    // Bin section 2*blockIdx.x + 1 at drift=0 AND drift=1
    for each obs k in section 2*blockIdx.x + 1:
        phase = frac(t[k] / P0)
        bin0 = floor(m * phase)
        bin1 = (bin0 + 1) % m   // shifted by 1 bin

        atomicAdd(&yw_bins[pair][0][bin0], yw[k])  // drift=0: add unshifted
        atomicAdd(&w_bins[pair][0][bin0], w[k])
        // Store shifted version separately for drift=1 combine
        atomicAdd(&yw_bins[pair][1][bin1], yw[k])  // drift=1: add shifted
        atomicAdd(&w_bins[pair][1][bin1], w[k])
```

Wait — this isn't quite right. Let me reconsider the data structure.

At level 0, we need to produce N_p/2 pair-folds, each with 2 drift variants (0, 1). Each fold is an m-element array of (yw, w). The drift=0 fold sums both sections without shift. The drift=1 fold sums section[s] without shift + section[s+1] with a 1-bin circular shift.

More precisely:

```
pair_fold[p][drift=0][bin] = section_bins[2p][bin] + section_bins[2p+1][bin]
pair_fold[p][drift=1][bin] = section_bins[2p][bin] + section_bins[2p+1][(bin-1) % m]
```

So we first need to bin each section independently, then combine. This suggests two sub-kernels for level 0:

**Sub-kernel 0a: Bin observations into per-section profiles**

```
Grid: N_p blocks (one per section)
For each obs in this section:
    phase = frac(t[k] / P0)
    bin = floor(m * phase)
    atomicAdd(&section_yw[blockIdx.x][bin], yw[k])
    atomicAdd(&section_w[blockIdx.x][bin], w[k])
```

Memory: N_p × m × 2 floats for section profiles.

**Sub-kernel 0b: Combine pairs with 0/1 shift**

```
Grid: (N_p / 2) blocks
For each bin b (threads cooperate):
    pair_fold[blockIdx.x][0][b] = section[2*blockIdx.x][b] + section[2*blockIdx.x + 1][b]
    pair_fold[blockIdx.x][1][b] = section[2*blockIdx.x][b] + section[2*blockIdx.x + 1][(b - 1) % m]
```

This is clean and separates the irregular-sampling complexity (0a) from the FFA logic (0b). After level 0, the butterfly can proceed on the regular pair_fold arrays.

### Levels 1 through n-1: Butterfly (GPU kernel)

At level l, we have N_p/2^l groups, each containing 2^l folds. We combine pairs of groups to produce N_p/2^(l+1) groups, each containing 2^(l+1) folds.

The combine rule:

```
For group g, output fold index s (0 <= s < 2^(l+1)):
    s_left = s mod 2^l        // fold index in left half-group
    s_right = s / 2^l mod 2^l // fold index in right half-group  (*)
    extra_shift = S_{l+1}[s]  // cumulative shift from shift vector

    output[g][s][bin] = left[2g][s_left][bin] + right[2g+1][s_right][(bin - extra_shift) % m]
```

(*) The exact indexing into the shift vector follows the recurrence from Shahaf et al.:
```
S_1 = (0, 1)
S_{l+1} = concat(S_l, S_l + 2^(l-1))
```

**GPU kernel for one butterfly level:**

```
Kernel: ffa_butterfly_kernel
Grid: (N_p / 2^(l+1)) × 2^(l+1) = N_p blocks  (one per output fold)
Block: min(m, 256) threads  (threads process bins in parallel)

group = blockIdx.x / (2^(l+1))
s = blockIdx.x % (2^(l+1))
s_left = decompose(s, l)      // left half-group fold index
s_right = decompose(s, l)     // right half-group fold index
shift = shift_vector[l+1][s]

for bin b (threads cooperate):
    yw_out[group][s][b] = yw_in[2*group][s_left][b]
                        + yw_in[2*group + 1][s_right][(b - shift) % m]
    w_out[group][s][b]  = w_in[2*group][s_left][b]
                        + w_in[2*group + 1][s_right][(b - shift) % m]
```

Each butterfly level is one kernel launch. There are log₂(N_p) - 1 levels. All N_p output folds within a level are independent and execute in parallel.

**In-place vs out-of-place:** The butterfly can be done with two buffers (ping-pong), like FFT implementations. At each level, read from buffer A, write to buffer B, swap.

### Scoring: Box Scan (GPU kernel)

After the butterfly, we have N_p folded profiles, each m bins. Run the standard BLS box scan on each:

```
Kernel: ffa_score_kernel
Grid: N_p blocks (one per fold = one per trial period)
Block: 128 threads

// Same as current BLS kernel's scoring loop:
For each (bin_start, bin_width) combination:
    sum yw and w over the bin range
    compute SR = yw² / (w × (1 - w))
    track max SR

// Warp reduction to find block-max SR
// Write max SR and best (bin_start, bin_width) to output
```

This is essentially the second half of the existing `full_bls_no_sol` kernel, extracted into a standalone kernel that operates on pre-folded profiles rather than raw observations.

## 5. Memory Layout

### Per-octave memory

For section length m and N_p = 2^n sections:

| Array | Shape | Size | Description |
|-------|-------|------|-------------|
| `section_yw` | [N_p, m] | N_p × m × 4 B | Per-section binned weighted flux |
| `section_w` | [N_p, m] | N_p × m × 4 B | Per-section binned weights |
| `folds_yw_A` | [N_p, m] | N_p × m × 4 B | Butterfly buffer A (yw) |
| `folds_w_A` | [N_p, m] | N_p × m × 4 B | Butterfly buffer A (w) |
| `folds_yw_B` | [N_p, m] | N_p × m × 4 B | Butterfly buffer B (yw) |
| `folds_w_B` | [N_p, m] | N_p × m × 4 B | Butterfly buffer B (w) |
| `sr_out` | [N_p] | N_p × 4 B | Output SR per period |
| `shift_vectors` | [n, 2^n] | ~N_p × n × 4 B | Pre-computed shift vectors |

Total: ~6 × N_p × m × 4 bytes.

**Example sizes:**

| Octave | m | N_p | Memory |
|--------|---|-----|--------|
| P~1d, dt=2min | 720 | 2^11=2048 | 35 MB |
| P~10d, dt=2min | 7200 | 2^8=256 | 44 MB |
| P~100d, dt=2min | 72000 | 2^5=32 | 55 MB |

These fit comfortably in GPU memory. For small octaves (small m), we can batch many octaves into one allocation.

### Optimization: Shared memory for small m

When m ≤ ~4096 (fits in 48 KB shared memory as 2 × m × 4 bytes), the butterfly combine can operate entirely in shared memory. Load the two input folds into shared memory, compute the shifted sum, write to global memory. This avoids the latency of global memory reads for the shift operation.

## 6. Integration with cuvarbase

### New files

```
cuvarbase/kernels/ffa_bls.cu     — CUDA kernels (init, butterfly, score)
cuvarbase/ffa_bls.py             — Python wrapper
cuvarbase/memory/ffa_memory.py   — GPU memory management (FFABLSMemory class)
```

### Python API

```python
def eebls_ffa_gpu(t, y, dy, period_min, period_max, m_bins=None,
                  qmin=0.01, qmax=0.15, dlogq=0.2,
                  ignore_negative_delta_sols=True):
    """
    BLS periodogram using Fast Folding Algorithm on GPU.

    Parameters
    ----------
    t, y, dy : array-like
        Time, flux, flux uncertainty (same as eebls_gpu_fast_adaptive)
    period_min, period_max : float
        Period search range in same units as t
    m_bins : int, optional
        Number of phase bins. If None, auto-select based on qmin.
        Typical: ceil(1/qmin) (same as current BLS nbinsf).
    qmin, qmax : float
        Min/max transit duty cycle (same as current BLS)
    dlogq : float
        Logarithmic spacing of trial transit widths (same as current BLS)

    Returns
    -------
    periods : ndarray
        Trial periods (FFA native grid)
    power : ndarray
        BLS Signal Residue at each trial period
    """
```

### Relationship to existing BLS

The FFA-BLS is a **separate function**, not a replacement for `eebls_gpu_fast_adaptive`. The existing function supports arbitrary frequency grids (including Keplerian). The FFA-BLS uses its own period grid. Users choose based on their needs:

- `eebls_gpu_fast_adaptive`: Arbitrary frequency grid, Keplerian-compatible. Best when N_freq is small (Keplerian grid) or when a specific frequency grid is required.
- `eebls_ffa_gpu`: FFA native period grid, arithmetic spacing. Best when searching a broad period range at full resolution, especially for long-baseline / high-N surveys where the FFA's O(N_p log N_p) scaling dominates.

## 7. Handling Multiple Octaves

### Octave iteration strategy

For a broad period range, iterate over octaves:

```python
all_periods = []
all_sr = []

for m in range(m_min, m_max + 1):
    P0 = m * dt
    N_p = next_power_of_2(T_total / P0)

    if N_p < 4:
        continue  # too few sections, use direct BLS

    periods_m, sr_m = ffa_single_octave(t, yw, w, m, N_p, qmin, qmax, dlogq)
    all_periods.append(periods_m)
    all_sr.append(sr_m)

periods = np.concatenate(all_periods)
sr = np.concatenate(all_sr)
```

### Batching small octaves

For large m (long periods), N_p is small and the FFA is cheap. For small m (short periods), N_p is large and the FFA has more work. To avoid underutilizing the GPU on large-m octaves, batch several consecutive octaves together:

- Group octaves by similar N_p (e.g., all octaves with N_p = 2^k for the same k)
- Allocate memory for the largest group
- Process each group as a batch

### Skipping unnecessary octaves (Keplerian-inspired)

Even without using the full Keplerian grid, we can skip octaves where the period resolution is finer than needed. At short periods, the FFA gives many trial periods per octave (large N_p), but the Keplerian criterion says we need fewer frequencies. We can subsample the FFA output at short periods by taking every k-th period from each octave. This doesn't save FFA compute (the butterfly runs on all N_p), but it saves scoring compute.

Alternatively, for short periods where N_p is large, we could truncate N_p to match the Keplerian density. Since the FFA butterfly cost is O(N_p × m × log N_p), reducing N_p directly reduces cost. The tradeoff: the FFA's N_p must be a power of 2, so this gives coarse control.

## 8. Edge Cases and Challenges

### Gaps in the data

Empty sections (no observations due to gaps) produce zero-valued folds. The FFA handles this correctly — summing with a zero fold is a no-op. However, the SR scoring must account for bins with zero weight (w=0 means no data), which the existing `bls_value()` function already handles (returns 0 when w < 1e-10).

### Very sparse sections

When sections contain very few observations (e.g., 1-2 points), the binned profile is dominated by shot noise. This is inherent to the BLS approach — fBLS doesn't make it worse. The signal builds up across sections during the butterfly.

### Non-power-of-2 section counts

The number of sections T_total / P0 may not be a power of 2. Options:
1. Pad with empty sections (zero-valued folds) up to the next power of 2
2. Use a mixed-radix FFA (more complex, probably not worth it for v1)

Padding is simple and doesn't affect correctness — empty sections contribute nothing to the fold.

### Cadence estimation

The FFA assumes a reference cadence dt for defining section boundaries. For irregularly sampled data, use the **median cadence** as dt. The actual observation times within each section are used for exact phase computation, so the cadence is only used for section boundary placement, not for phase binning.

### Transit straddling section boundaries

A transit that spans a section boundary will be split between two sections. The FFA handles this correctly as long as the transit duration is shorter than the section length (i.e., q < 1, which is always true for transits). After folding, the transit signal from both sections will land in the same phase bins and add coherently.

## 9. Benchmark Plan

### Correctness tests

1. **Exact match with current BLS**: For a set of test lightcurves, verify that `eebls_ffa_gpu` and `eebls_gpu_fast_adaptive` produce the same SR values (within floating-point tolerance) at overlapping periods. Use m_bins = nbinsf from the current BLS to ensure identical binning.

2. **Transit injection-recovery**: Inject transits at known periods into synthetic lightcurves. Verify that fBLS recovers the correct period across all survey profiles (ZTF, HAT-Net, TESS, Kepler).

3. **Edge cases**: Empty sections (large gaps), single-observation sections, very short and very long periods.

### Performance benchmarks

Compare against `eebls_gpu_fast_adaptive` (with Keplerian grid) across survey profiles:

| Survey | N_obs | Baseline | Period range | Current BLS (Keplerian) | fBLS (native grid) |
|--------|-------|----------|-------------|------------------------|---------------------|
| ZTF | 150 | 730d | 0.5-100d | 60K freqs, ~5ms | ? |
| HAT-Net | 6,000 | 3,650d | 0.5-100d | 301K freqs, ~41ms | ? |
| TESS | 20,000 | 27d | 0.5-13.5d | 1.8K freqs, ~5ms | ? |
| Kepler | 65,000 | 1,460d | 0.5-500d | 131K freqs, ~179ms | ? |

Key metrics:
- Wall-clock time per lightcurve (single LC)
- Throughput (LC/s) for survey-scale batched processing
- Memory usage
- Correctness (SR correlation with current BLS)

### Scaling tests

- Fix N_obs=10K, vary N_p from 2^10 to 2^20: measure FFA time, verify O(N_p log N_p) scaling
- Fix N_p=2^16, vary N_obs from 100 to 100K: measure Level 0 time, verify O(N_obs) scaling
- Fix N_obs and N_p, vary m from 32 to 4096: measure butterfly time, verify O(m) scaling

## 10. Implementation Order

### Phase 1: Core FFA engine

1. **`ffa_bls.cu`**: Write three CUDA kernels:
   - `ffa_init_kernel`: Bin observations into per-section profiles
   - `ffa_butterfly_kernel`: One butterfly level (combine pairs with shift)
   - `ffa_score_kernel`: Box scan on folded profiles → max SR

2. **`ffa_bls.py`**: Python wrapper that:
   - Pre-computes section boundaries and shift vectors
   - Orchestrates kernel launches (init → butterfly levels → score)
   - Returns periods and SR array

3. **Correctness tests**: Compare against `eebls_gpu_fast_adaptive` on synthetic data.

### Phase 2: Optimization

4. **Shared memory butterfly**: For m ≤ 4096, load folds into shared memory for the butterfly combine.

5. **Octave batching**: Batch multiple small-N_p octaves into single kernel launches.

6. **Keplerian-inspired octave skipping**: Skip octaves at short periods where period resolution exceeds what's needed.

### Phase 3: Integration and benchmarking

7. **Batch API**: `eebls_ffa_gpu_batch()` for survey-scale processing (analogous to `eebls_gpu_batch()`).

8. **Full benchmark suite**: Run `scripts/benchmark_new_features.py` with fBLS added.

## 11. References

- Shahaf, S., Zackay, B., Mazeh, T., Faigler, S., & Ivashtenko, O. (2022). fBLS — a fast-folding BLS algorithm. MNRAS, 513, 2732. [arXiv:2204.02398](https://arxiv.org/abs/2204.02398)
- Kovacs, G., Zucker, S., & Mazeh, T. (2002). A box-fitting algorithm in the search for periodic transits. A&A, 391, 369.
- Staelin, D. H. (1969). Fast folding algorithm for detection of periodic pulse trains. Proc. IEEE, 57, 724. (Original FFA)
- Kondratiev, V. I. et al. (2009). A survey for pulsars in the LMC with the Parkes telescope. ApJ, 702, 692. (Modern FFA formulation)
