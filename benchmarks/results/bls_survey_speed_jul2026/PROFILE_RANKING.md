# BLS survey-scale profiling: measured bottleneck ranking (Jul 4 2026)

Pod: RunPod RTX A5000 ($0.27/hr), CUDA 12.4, driver 570.211.01,
commit base 89d5481 (branch feature/bls-survey-speed).
Baseline gates: full GPU suite 752 passed / 7 skipped; release gate 14/14.

## Profiling method

`ncu` is installed on the pod (`/usr/local/cuda-12.4/bin/ncu`) but GPU
performance counters are **blocked by the host driver**
(`ERR_NVGPUCTRPERM`; we are root in the container, no `modprobe`, and
`/proc/driver/nvidia/params` is not exposed — RunPod restriction, real
attempts documented in the session). Fallback per plan:

1. `nsys` timeline (kernel/memcpy/API rows + host gap analysis),
2. CUDA-event / wall-clock decomposition (bench_bls_survey.py variants:
   fast_naive / fast_reuse / kernel / kernel_1pass / pieces / batch),
3. one-axis parameter sweeps (sweep_bls_attrib.py: ndata, bin count,
   noverlap, block size) to attribute kernel time without counters,
4. targeted hypothesis tests (data-order shuffle; cgroup throttle
   counters).

Raw evidence: `benchmarks/results/bls_survey_speed_jul2026/raw/`
(bench_baseline.json = default env, bench_base_envfix.json = pinned
threadpools, attrib_*.json, *.log, parity/*.npz).

## Survey scales (Keplerian grids, qmin=0.5 q_kep, qmax=2 q_kep, oversampling=2)

| survey  | ndata | nfreq   | nbf range (fine bins/freq) |
|---------|-------|---------|-----------------------------|
| ZTF     | 150   | 60,121  | 16–571                      |
| HAT-Net | 6,000 | 300,592 | 16–571                      |
| TESS    | 20,000| 1,788   | 16–150                      |
| Kepler  | 65,000| 130,597 | 16–1,665                    |

## Finding 0 (host environment, affects every real deployment on shared pods)

**OpenBLAS threadpool vs container CPU quota.** The pod exposes 96 CPUs
but the cgroup quota is 7.65 cores (cfs_quota 765000/100000). numpy's
`np.dot` in the per-LC host path (BLSMemory.setdata / set_lightcurve /
_chi2_null) triggers OpenBLAS with 96 threads; the burst exhausts the
CFS quota and the process **freezes ~90 ms per 100 ms period** (nsys
timeline: recurring 88–93 ms host gaps with the GPU idle; cgroup
nr_throttled +5 per 8-LC loop, +23 s cumulative thread-throttle time).

- TESS fast_reuse: 52 ms/lc (default env) -> **6.4 ms/lc** with
  `OPENBLAS_NUM_THREADS=1` (zero throttle events).
- CU_CTX_SCHED_SPIN / ctx-sync placement: no effect (ruled out
  sync-wait oversleep).

All subsequent numbers use pinned threadpools (bench_base_envfix).
Library-side fix (avoid BLAS calls in the per-LC path) is queued as
part of the host-overhead change.

## Env-fixed baseline (warm medians, >=5 runs; ms per lightcurve)

| survey  | fast_naive | fast_reuse | kernel(nov=2) | kernel 1pass | batch | best today |
|---------|-----------|------------|---------------|--------------|-------|------------|
| ZTF     | 9.20      | 6.09       | 5.63          | 3.11         | **1.45** | batch |
| HAT-Net | 94.52     | 76.92      | 76.26         | 42.40        | **57.92** | batch |
| TESS    | 6.16      | 4.90       | 4.49          | 2.34         | 4.63  | ~tie  |
| Kepler  | 369.4     | 365.6      | 367.4         | 185.4        | **351.9** | batch |

pieces (host-side, per call): setdata+H2D 0.08–1.1 ms; D2H+norm
0.03–0.51 ms; fresh BLSMemory alloc 0.25–3.0 ms (pinned cuMemHostAlloc
dominates at HAT-Net's nfreq=301K).

## Kernel-time attribution (one-axis sweeps)

Marginal histogram cost from the ndata slope (9600->38400), per point
per frequency per pass ~ **18–22 ps** on ZTF/HAT/Kepler; TESS ~80 ps
(conflict-inflated, see below). Extrapolated shares at the real ndata:

| survey  | histogram (fold+2x shared atomicAdd) | per-freq fixed cost (bin init + 4 syncs + scan + 256-thread reduction) | noverlap scaling |
|---------|--------------------------------------|--------------------------------------------------------------------|------------------|
| ZTF     | ~10–15%                              | **~85%** (bin-scale sweep FLAT: 5.71/6.02/6.71 ms at 1x/2x/4x bins; ndata slope tiny at 150 pts) | 3.11/5.73/8.28 ms (linear) |
| HAT-Net | **~55–60%**                          | ~40%                                                               | linear           |
| TESS    | **~95%, conflict-bound**             | ~5%                                                                | 2.37/4.65/6.92 (linear) |
| Kepler  | **~92%**                             | ~8% (bin 2x flat; 4x +32% = shmem occupancy 3->2 blocks/SM)         | 187/371/554 (linear) |

**Atomic-conflict smoking guns (TESS):**
- bin-scale sweep INVERSE: 4.82 -> 3.27 -> 2.24 ms at 1x/2x/4x bins
  (more bins = fewer same-bin conflicts = faster, occupancy unchanged);
- **host-side data shuffle: 4.91 -> 1.59 ms (3.09x)** with a random
  permutation of (t,y,dy) — sorted 2-min cadence puts warp-adjacent
  points into the same phase bin at nearly every trial frequency
  (32-way shared-atomic serialization). Kepler 1.16x, HAT-Net 1.00x
  (sparse/random sampling already de-clusters phases).

Block size (D sweep): 256 default is right for HAT/TESS/Kepler; ZTF
slightly prefers 128 (5.54 vs 5.78 ms); 64 and 512 are 2x worse.

## Ranked bottlenecks -> action plan

1. **noverlap=2 re-fold+re-histogram (all surveys; 1.8–2.0x pass
   scaling measured).** Fuse into ONE kernel launch that histograms
   once at noverlap-times-finer phase resolution and derives every
   pass's box sums from the fine histogram: atomics and folds drop
   ~2x, per-freq fixed costs paid once instead of twice.
   Candidate (a). Expected ~1.5–2x kernel time on HAT/TESS/Kepler,
   ~1.5x ZTF. **DO FIRST.**
2. **Shared-atomic conflicts from time-ordered data (TESS 3.09x,
   Kepler 1.16x).** Host-side deterministic scatter permutation of
   (t, yw, w) before upload — bin sums are order-independent (up to
   float32 associativity already nondeterministic via atomics).
   Zero kernel change. **DO SECOND.**
3. **Host per-call overhead (candidate d).**
   (i) BLAS threadpool fix in-library (replace np.dot with
   numpy-core reductions in setdata/set_lightcurve/_chi2_null) —
   removes the 8x cgroup-throttle cliff for library users;
   (ii) naive-path allocation costs: fresh pinned buffers + 7
   gpuarray allocs per call (naive-reuse delta: ZTF 3.1 ms/lc, HAT-Net
   17.6 ms/lc — cuMemHostAlloc of 3x301K-float pinned arrays);
   provide/document a reusable-plan path for LC loops (memory= reuse
   exists; make the batch path reuse BLSBatchMemory across calls).
4. **ZTF-class per-freq fixed cost (~85% of ZTF kernel).** Mostly
   idle threads: median nbf~100 << block 256; the 256-wide tree
   reduction + 4 __syncthreads dominate. The batch path already
   mitigates the utilization half of this (1.45 ms/lc vs 5.63
   single-LC: GPU underfilled by one small LC). The fused kernel
   removes one full pass of it. Warp-per-frequency scan is a
   follow-up if time permits (higher risk).

## Candidates SKIPPED, with the profile evidence

- **(b) shared-memory frequency tiling (reuse LC across freqs per
  block):** global loads are L2-resident (t/yw/w = 0.78 MB max, L2 =
  6 MB; ndata-slope cost is atomic-issue-bound, not load-bound).
  Tiling reduces global loads only -> no headroom. SKIP.
- **(c) per-frequency bin sizing from Keplerian q:** already
  implemented in-kernel (nbinsf[i_freq] read per frequency; scan work
  scales with the per-freq nbf). The only global effect is the
  shared-memory RESERVATION (sized by batch max nbf): bin-scale
  sweeps show it's noise on A5000 until nbf_max ~3300 (Kepler 4x row:
  +32% when blocks/SM drop 3->2). Freq-chunked launches (existing
  freq_batch_size) recover it if ever needed. SKIP as a standalone
  change; revisit only if fused-kernel shmem doubling regresses
  Kepler.
- **(e) H2D/compute overlap:** setdata+H2D <= 1.1 ms vs kernels
  4.5–367 ms (<= 8% everywhere after fixes 1–3; ZTF 1.4%). Streamed
  double-buffering adds complexity for <10%; reassess after 1–3.
  DEFERRED (not dead — TESS post-fix could see ~10–15%).
- **(f) FFA-BLS phase 2:** stretch only; untouched this session.

## Correctness-gate protocol for every change

1. Full GPU suite on pod (752/7 expected) + scripts/check_release_gate.py (14/14).
2. Parity vs base_envfix dumps: corr > 0.999 AND identical argmax
   (bit-identical where math untouched), on fast, fast_bjd
   (t += 2455197.5), and batch arrays for all four surveys
   (benchmarks/compare_parity.py).
3. Warm-median before/after benchmark in the same pod session.
