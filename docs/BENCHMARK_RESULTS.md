# Benchmark Results: Survey-Scale Performance

> **Benchmark correction, September 2026.** The transit timing/sensitivity and cost claims below describe historical protocols. Use the [new transit benchmark](TRANSIT_BENCHMARKS.md) for current release claims. Equal scalar SDE did not establish equal sensitivity; some old BLS comparisons used different duration searches; warm GTLS compilation was not the dominant measured bottleneck. Historical values are retained for provenance, not as qualified performance promises.

Measured on NVIDIA RTX A5000 (24 GB), February 2026, except where noted. Source data in `benchmarks/results/benchmark_results_new_features.json`, scripts in `scripts/benchmark_new_features.py`. The multi-GPU comparison in Section 3 has its own per-architecture source data in `benchmarks/results/by_gpu/`.

## The Big Picture

cuvarbase makes GPU-accelerated period finding practical for entire astronomical surveys. The key results:

- **BLS/TLS:** current speed, independent recovery and cost measurements are in [one transit benchmark figure](TRANSIT_BENCHMARKS.md).
- **Lomb-Scargle**: At realistic survey frequency counts (100K-1.8M), GPU is **1.5-12.6x faster** than nifty-ls (the fastest CPU LS) in head-to-head measurements; at ZTF/HAT-Net scales nifty-ls cannot complete within the 120s timeout (lower bounds >27x and >15x). At small problem sizes (10K obs, 5K freqs, single LCs) nifty-ls on CPU is faster than the GPU implementation
- **Keplerian frequency grid**: Exploits the physics of Keplerian orbits to search 4-37x fewer frequencies in the historical grid examples below; these frequency counts alone do not establish unchanged detection sensitivity

## 1. Lomb-Scargle: GPU vs nifty-ls at Survey Scale

The question that matters for LS isn't "how fast is a single periodogram" — it's "how fast can I process my entire survey." This requires realistic frequency grids derived from actual survey parameters.

### How many frequencies does a real survey need?

For irregularly sampled data, there is no Nyquist limit (VanderPlas 2018). The number of independent frequencies is:

```
Nf = (1/Pmin - 1/Pmax) * oversampling * baseline
```

LS searches for all variability types (eclipsing binaries, RR Lyrae, delta Scuti, Cepheids, etc.), so the period range is broad: P_min ~ 0.01 days (short-period delta Scuti), P_max ~ baseline (LS can detect variability even without multiple complete cycles, unlike BLS).

| Survey | Baseline | P range | Nf (5x oversample) |
|--------|----------|---------|--------------------:|
| ZTF | 730 d (2 yr) | 0.01 - 730 d | **365,000** |
| HAT-Net | 3,650 d (10 yr) | 0.01 - 3,650 d | **1,825,000** |
| TESS (1 sector) | 27 d | 0.01 - 27 d | **13,500** |
| Kepler | 1,460 d (4 yr) | 0.01 - 1,460 d | **730,000** |

These are 10-350x larger than the toy benchmarks (5K-50K) that dominate the literature.

### Survey-scale throughput

All measurements use `batched_run_const_nfreq()` which pre-allocates GPU memory once and reuses it across lightcurves. No FAP computation (which would add ~70% CPU overhead unfairly to GPU timings).

| Survey | N_obs | N_freq | GPU (ms/LC) | nifty-ls (ms/LC) | GPU speedup |
|--------|------:|-------:|------------:|------------------:|------------:|
| ZTF | 150 | 365K | **4.4** | TIMEOUT (>120s/batch) | **>27x** |
| HAT-Net | 6,000 | 1.825M | **19.2** | TIMEOUT (>120s/batch) | **>15x** |
| TESS | 20,000 | 13.5K | **3.3** | 4.9 | **1.5x** |
| Kepler | 65,000 | 730K | **19.8** | 250.0 | **12.6x** |

**Takeaway**: At the frequency counts that real variability surveys require (>100K), GPU dominates. nifty-ls is only competitive for short-baseline surveys like TESS where N_freq is small.

### Why is nifty-ls fast at small N_freq but slow at large N_freq?

nifty-ls uses FINUFFT (CPU) with FFTW + AVX/SSE vectorization + multi-threading. It's extremely well-optimized for single-call execution. But for survey processing, each lightcurve requires a separate `nifty_ls.lombscargle()` call that creates a new FINUFFT plan, and plan creation has significant overhead (~50ms). At small N_freq, the FFT itself is fast enough that plan creation is a small fraction. At large N_freq, the overhead compounds across thousands of lightcurves.

cuvarbase's GPU LS avoids this by JIT-compiling CUDA kernels once and reusing them across all lightcurves with pre-allocated GPU memory.

## 2. cuFINUFFT vs Custom NFFT Kernel

cuvarbase now supports [cuFINUFFT](https://github.com/flatironinstitute/finufft) as an alternative GPU NFFT backend (via `use_cufinufft=True`). This uses the same library that powers nifty-ls's GPU mode.

### Single-LC steady-state performance (compilation excluded)

| N_obs | N_freq | Custom NFFT | cuFINUFFT | Ratio |
|------:|-------:|------------:|----------:|------:|
| 1,000 | 5K | 3.5 ms | 5.1 ms | 0.67x |
| 1,000 | 50K | 7.1 ms | 10.4 ms | 0.68x |
| 10,000 | 5K | 5.0 ms | 7.2 ms | 0.70x |
| 10,000 | 50K | 8.7 ms | 11.7 ms | 0.74x |
| 50,000 | 5K | 12.6 ms | 15.1 ms | 0.84x |
| 50,000 | 50K | 12.6 ms | 19.9 ms | 0.63x |

**cuFINUFFT is consistently 20-40% slower than the custom NFFT kernel.** The custom kernel wins because:

1. It's JIT-compiled by PyCUDA with parameters (N_obs, grid size, oversampling) baked into the kernel at compile time
2. No per-call plan creation overhead — the compiled kernel is cached and reused
3. The spreading kernel uses Gaussian gridding optimized for our specific use case

cuFINUFFT's exponential-of-semicircle spreading function and shared-memory bin-sorting are algorithmically superior, but the overhead of creating a new cuFFT plan on every call negates the improvement. A persistent-plan cuFINUFFT integration would likely close the gap.

**Recommendation**: Use the default custom NFFT backend. cuFINUFFT is available as a correctness cross-check but offers no performance benefit.

## 3. BLS: current comparisons

Use the [current transit benchmark](TRANSIT_BENCHMARKS.md) for actual PyPI 0.2.5, v1, Astropy and periodfind comparisons on ZTF/TESS cadences. periodfind provides both CPU and GPU BLS; cuvarbase is not the only GPU BLS implementation. fBLS was screened, with failed/time-limited pilots retained and excluded from speed denominators. The former 257–354× Astropy headline used unequal duration searches.

## 4. TLS: current comparisons

Use the [current transit benchmark](TRANSIT_BENCHMARKS.md) and [implementation comparison](GTLS_COMPARISON.md). Equal SDE did not establish equal sensitivity in the July measurements, and warm GTLS module compilation was not the dominant measured bottleneck. The original BLS/TLS tables remain in the [preserved benchmark document](../analysis/transit-recovery-20260908/sources/claims-before/docs/BENCHMARK_RESULTS.md) and the [provenance audit](../analysis/benchmark-audit-20260906/README.md).

## 5. Keplerian Frequency Grid

### What problem does it solve?

Standard BLS uses a uniform frequency grid (constant df). But transit signals have a fixed duration in time, not in frequency. At high frequencies (short periods), the transit occupies a larger fraction of the period, so the transit signal is broader in frequency space and doesn't need as fine a frequency grid to resolve. At low frequencies (long periods), the transit is a tiny fraction of the period, requiring finer frequency resolution.

The Keplerian frequency grid spaces trial frequencies proportionally to the expected transit duration at each period, which follows Kepler's third law: duration ~ P^(1/3). This means:

- **Short periods** (high frequency): coarser spacing → fewer frequencies needed
- **Long periods** (low frequency): finer spacing → same resolution as uniform grid

### Impact

| Survey | Baseline | Uniform N_freq | Keplerian N_freq | Reduction | BLS speedup |
|--------|----------|---------------:|-----------------:|----------:|------------:|
| ZTF | 730 d | 827,392 | 60,121 | **13.8x** | **14.3x** |
| HAT-Net | 3,650 d | 4,136,958 | 300,592 | **13.8x** | **14.4x** |
| TESS | 27 d | 7,792 | 1,788 | **4.4x** | **1.5x** |
| Kepler | 1,460 d | 4,858,154 | 130,597 | **37.2x** | **24.1x** |

The frequency reduction translates almost directly to BLS speedup because BLS is O(N_obs x N_freq). For long-baseline surveys (Kepler, HAT-Net), the Keplerian grid eliminates millions of redundant frequency evaluations. Correctness tests confirm that transit signals are detected identically with both grids.

### When does it matter most?

The Keplerian grid helps most when the ratio of maximum to minimum period is large. For Kepler (P_max/P_min = 1000), this yields 37x fewer frequencies. For TESS 1-sector (P_max/P_min = 27), only 4.4x. Long-baseline ground-based surveys benefit enormously.

## 6. Search-cost projections

Current [transit cost estimates](TLS_COST_ANALYSIS.md) derive from measured A40 batch search throughput. They exclude full-pipeline work. Earlier whole-survey dollar totals are preserved in the historical document linked above and should not be advertised as measured complete survey costs.

## Reproducibility

```bash
# Run on a GPU machine with cuvarbase installed
pip install -e .[cufinufft]
pip install nifty-ls astropy

# All correctness tests + benchmarks
python scripts/benchmark_new_features.py

# Benchmarks only (skip correctness tests)
python scripts/benchmark_new_features.py --bench-only

# Correctness tests only
python scripts/benchmark_new_features.py --tests-only
```

Results are saved to `benchmarks/results/benchmark_results_new_features.json`. The other harnesses (the multi-GPU BLS sweep, the survey-speed campaign, the TLS survey and GTLS comparisons, the 0.2.6 head-to-head) and the RunPod workflow are described in `scripts/README.md`.

## References

- Kovacs, G., Zucker, S., & Mazeh, T. (2002). A box-fitting algorithm in the search for periodic transits. A&A, 391, 369.
- VanderPlas, J. T. (2018). Understanding the Lomb-Scargle Periodogram. ApJS, 236, 16.
- Kunimoto, M. et al. (2023). TESS Quick-Look Pipeline GPU Transit Search. RNAAS, 7, 28.
- Wang, K. et al. (2024). GPU Phase Folding and Convolutional Neural Network. MNRAS, 528, 4053.
- Smith, L. C. et al. (2025). CETRA: Cambridge Exoplanet Transit Recovery Algorithm. MNRAS, 539, 297.
- Shahaf, S. et al. (2022). fBLS: A fast-folding BLS algorithm. MNRAS, 513, 2732.
- Garrison, L. H., Foreman-Mackey, D., Shih, Y.-H., & Barnett, A. (2024). nifty-ls: Fast and Accurate Lomb-Scargle Periodograms Using a Non-Uniform FFT. arXiv:2409.08090.
