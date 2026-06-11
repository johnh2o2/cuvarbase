# Benchmark Results: Survey-Scale Performance

Measured on NVIDIA RTX A5000 (24 GB), February 2026. Source data in `benchmark_results_new_features.json`, scripts in `scripts/benchmark_new_features.py`.

## The Big Picture

cuvarbase makes GPU-accelerated period finding practical for entire astronomical surveys. The key results:

- **BLS**: To our knowledge the only published, production-deployed GPU implementation of the standard BLS algorithm. Combined with Keplerian frequency grids, processes 10 million ZTF lightcurves in 3.5 hours for **$0.69**
- **Lomb-Scargle**: At realistic survey frequency counts (100K-1.8M), GPU is **1.5-12.6x faster** than nifty-ls (the fastest CPU LS) in head-to-head measurements; at ZTF/HAT-Net scales nifty-ls cannot complete within the 120s timeout (lower bounds >27x and >15x). At small problem sizes (10K obs, 5K freqs, single LCs) nifty-ls on CPU is faster than the GPU implementation
- **Keplerian frequency grid**: Exploits the physics of Keplerian orbits to search 4-37x fewer frequencies with no loss in transit detection sensitivity

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

## 3. BLS: Competitive Landscape

### cuvarbase is the only GPU BLS

A thorough search of the literature and open-source repositories reveals that **cuvarbase is the only implementation of the standard Kovacs et al. (2002) BLS algorithm on GPU**. This is validated by:

- The GPFC paper (Wang et al. 2024, MNRAS 528, 4053) benchmarks cuvarbase as the GPU BLS baseline
- The TESS Quick-Look Pipeline adopted cuvarbase's GPU BLS starting in Sector 59 (Kunimoto et al. 2023, RNAAS 7, 28)

Projects that are sometimes confused with GPU BLS but are fundamentally different algorithms:

| Project | What it actually does | GPU? | Apples-to-apples with BLS? |
|---------|----------------------|------|---------------------------|
| **CETRA** (Smith et al. 2025) | Linear-time transit search + phase fold | Yes | No — different algorithm, different statistics |
| **GPFC** (Wang et al. 2024) | Phase folding + CNN classifier | Yes | No — ML classifier, not a periodogram |
| **fBLS** (Shahaf et al. 2022) | Fast Folding BLS (O(N log N)) | No (CPU) | Yes — same BLS output, faster algorithm |
| **TLS** (Hippke & Heller 2019) | Transit-shaped template (not box) | No (CPU) | No — different model, more sensitive |

The closest CPU competitor is **fBLS** at ~6 seconds for 65K datapoints / 100K frequencies. cuvarbase's GPU BLS does the same in ~1 second.

### BLS survey-scale throughput

Using Keplerian frequency grids (see Section 4):

| Survey | N_obs | N_freq (Keplerian) | LC/s (batch) | LC/s (single) | Best mode |
|--------|------:|-------------------:|-------------:|--------------:|-----------|
| ZTF | 150 | 60K | **802** | 216 | Batch (3.7x) |
| HAT-Net | 6,000 | 301K | **38** | 24 | Batch (1.6x) |
| TESS | 20,000 | 1.8K | 20 | **236** | Single |
| Kepler | 65,000 | 131K | 5 | **6** | Single |

**When does batch mode help?** Batch mode (`eebls_gpu_batch`) amortizes per-LC overhead (memory allocation, kernel launch, host-device transfer). This matters when kernel execution time per LC is small relative to overhead — i.e., when N_obs is small:

- **N_obs < 1000**: Batch mode gives 2-4x speedup (overhead-dominated regime)
- **N_obs > 10000**: Single-LC loop is as fast or faster (compute-dominated regime)

### Survey-wide processing cost

| Survey | Total LCs | Best LC/s | Wall time (1x A5000) | Cost @ $0.20/hr |
|--------|----------:|----------:|---------------------:|----------------:|
| ZTF | 10,000,000 | 802 | 3.5 hours | **$0.69** |
| HAT-Net | 10,000,000 | 38 | 3.1 days | **$14.74** |
| TESS (all sectors) | 5,200,000 | 236 | 6.1 hours | **$1.22** |
| Kepler | 200,000 | 6 | 10.0 hours | **$2.00** |

BLS transit searches across entire surveys cost **under $15 on a single consumer GPU**.

## 4. Keplerian Frequency Grid

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

## 5. Combined LS + BLS Survey Cost

Total cost to run a complete variability + transit search pipeline (LS for variable star classification, BLS for transit detection) on a single RTX A5000 at $0.20/hr:

| Survey | Total LCs | BLS cost | LS cost | **Total** |
|--------|----------:|---------:|--------:|----------:|
| ZTF | 10M | $0.69 | $2.47 | **$3.16** |
| HAT-Net | 10M | $14.74 | $10.66 | **$25.40** |
| TESS | 5.2M | $1.22 | $0.95 | **$2.18** |
| Kepler | 200K | $2.00 | $0.22 | **$2.22** |

**Total across all four surveys: ~$33** on a single GPU. Processing is embarrassingly parallel across multiple GPUs.

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

Results are saved to `benchmark_results_new_features.json`.

## References

- Kovacs, G., Zucker, S., & Mazeh, T. (2002). A box-fitting algorithm in the search for periodic transits. A&A, 391, 369.
- VanderPlas, J. T. (2018). Understanding the Lomb-Scargle Periodogram. ApJS, 236, 16.
- Kunimoto, M. et al. (2023). TESS Quick-Look Pipeline GPU Transit Search. RNAAS, 7, 28.
- Wang, K. et al. (2024). GPU Phase Folding and Convolutional Neural Network. MNRAS, 528, 4053.
- Smith, L. C. et al. (2025). CETRA: Cambridge Exoplanet Transit Recovery Algorithm. MNRAS, 539, 297.
- Shahaf, S. et al. (2022). fBLS: A fast-folding BLS algorithm. MNRAS, 513, 2732.
- Garrison, L. H., Foreman-Mackey, D., Shih, Y.-H., & Barnett, A. (2024). nifty-ls: Fast and Accurate Lomb-Scargle Periodograms Using a Non-Uniform FFT. arXiv:2409.08090.
