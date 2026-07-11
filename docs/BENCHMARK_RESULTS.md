# Benchmark Results: Survey-Scale Performance

Measured on NVIDIA RTX A5000 (24 GB), February 2026, except where noted. Source data in `benchmarks/results/benchmark_results_new_features.json`, scripts in `scripts/benchmark_new_features.py`. The multi-GPU comparison in Section 3 has its own per-architecture source data in `benchmarks/results/by_gpu/`.

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
| **TLS** (Hippke & Heller 2019 reference package) | Transit-shaped template (not box) | No (CPU) — **cuvarbase 1.0 ships a GPU TLS; see section 4** | No — different model, more sensitive |

> **Comparison-version pin:** all astropy Lomb-Scargle and BoxLeastSquares comparisons in this document were measured against **astropy 7.2.0** (the latest release as of June 2026). astropy 8.0 is expected to ship an LRA-NUFFT default for Lomb-Scargle that may change the comparison; re-run before citing these numbers against astropy >= 8.

The closest CPU competitor is **fBLS** at ~6 seconds for 65K datapoints / 100K frequencies (Shahaf et al. 2022, their table 1). cuvarbase's single-LC GPU BLS measured ~0.17 s/LC at the same scale (Kepler row of the batch-vs-single table below: 6 LC/s, 65K points, 131K Keplerian frequencies, RTX A5000; `benchmarks/results/benchmark_results_new_features.json`).

### Standard BLS across 7 GPU architectures

Measured February 2026 with `scripts/benchmark_algorithms.py` (driven across pods by `scripts/benchmark_all_gpus.sh`; 10K observations, 5K frequencies, batches of 10 lightcurves; astropy `BoxLeastSquares` on the host CPU as the reference). Per-GPU source data: `benchmarks/results/by_gpu/benchmark_<GPU>.json`.

| GPU | BLS time/LC (ms) | vs astropy | $/hr (RunPod, Feb 2026) | $ per 1M LCs |
|-----|-----------------:|-----------:|------------------------:|-------------:|
| NVIDIA L40 | 2.62 | **354x** | $0.69 | $0.50 |
| NVIDIA H200 | 2.34 | 306x | $3.59 | $2.33 |
| Tesla V100-SXM2-16GB | 6.03 | 305x | $0.19 | $0.32 |
| NVIDIA GeForce RTX 4090 | 2.99 | 290x | $0.34 | $0.28 |
| NVIDIA RTX 4000 Ada | 2.57 | 284x | $0.20 | **$0.14** |
| NVIDIA H100 80GB HBM3 | 2.26 | 268x | $2.69 | $1.69 |
| NVIDIA A100-SXM4-80GB | 3.70 | **257x** | $1.19 | $1.22 |

The speedup over astropy is remarkably consistent — **257-354x across every architecture from Volta (2017) to Hopper (2024)** — because both the GPU kernel and astropy scale linearly in N x N_freq at this problem size. The cheapest way to process a million lightcurves is a workstation card (RTX 4000 Ada at **$0.14/M**), not a data-center flagship.

> An earlier revision of this table carried a "vs pre-v1.0 kernel" column (21-390x). Those numbers are **retracted**: the baseline paid per-call CUDA compilation, so the ratio measured pod-host compile speed, not GPU throughput. The measured comparison against the previous release is below.

### Versus the previous cuvarbase release (v0.2.6 tag; measured July 2026, RTX A5000)

Identical inputs both sides; v1.0 at `noverlap=1` for apples-to-apples (the old fast path silently ignored `noverlap`). Raw JSON + scripts: `benchmarks/results/v026_head_to_head_jul2026/`.

| Measurement | 0.2.6 | 1.0.0 | Change |
|---|---:|---:|---|
| BLS kernel-only, 20K obs x 13.5K freqs | 9.8 ms | 9.8 ms | 1.00x — kernel throughput unchanged |
| BLS per-call in a lightcurve loop (steady state) | 261 ms | 7.6 ms | **34x** (kernel cached vs recompiled every call) |
| BLS 100-lightcurve run incl. first compile | 28.4 s | 2.8 s | **10x** |
| Lomb-Scargle, 3K obs x 100K freqs | 33.3 ms | 11.7 ms | **2.85x** |
| BLS on BJD-scale timestamps | signal lost (peak 0.30 → 0.089, wrong freq) | identical to near-zero timestamps | correctness |

We claim **no raw-kernel speedup** over the previous release — the wins are architectural (compile-once caching, batching, Keplerian grids) plus correctness. The 0.2.6 baseline also required numpy < 1.24 and a 2022-era pycuda to run its LS/PDM paths at all (segfaults on pycuda >= 2025.1).

### BLS survey-scale throughput

Using Keplerian frequency grids (see Section 5):

| Survey | N_obs | N_freq (Keplerian) | LC/s (batch) | LC/s (single) | Best mode |
|--------|------:|-------------------:|-------------:|--------------:|-----------|
| ZTF | 150 | 60K | **802** | 216 | Batch (3.7x) |
| HAT-Net | 6,000 | 301K | **38** | 24 | Batch (1.6x) |
| TESS | 20,000 | 1.8K | 20 | **236** | Single |
| Kepler | 65,000 | 131K | 5 | **6** | Single |

> **Stale batch columns:** this table was measured February 2026, when `eebls_gpu_batch` recompiled its kernel on every call. That defect was fixed in July 2026, after which **batch beats the single-LC loop at every measured scale** (~10x at N_obs=200, ~5x at N_obs=20,000, 2.2x for 2-LC batches; warm cache, RTX A5000). The ZTF/HAT-Net batch rows above are therefore conservative and the TESS/Kepler "Best mode: Single" recommendations are obsolete — prefer `eebls_gpu_batch` when processing many lightcurves at any size.

**When does batch mode help?** Batch mode (`eebls_gpu_batch`) amortizes per-LC overhead (kernel launch, memory allocation, host-device transfer) and, since the July 2026 fix, shares one cached kernel across the whole collection. With a warm cache it outperformed the single-LC loop at every scale measured (N_obs 200 to 20,000).

### Survey-wide processing cost

| Survey | Total LCs | Best LC/s | Wall time (1x A5000) | Cost @ $0.20/hr |
|--------|----------:|----------:|---------------------:|----------------:|
| ZTF | 10,000,000 | 802 | 3.5 hours | **$0.69** |
| HAT-Net | 10,000,000 | 38 | 3.1 days | **$14.74** |
| TESS (all sectors) | 5,200,000 | 236 | 6.1 hours | **$1.22** |
| Kepler | 200,000 | 6 | 10.0 hours | **$2.00** |

BLS transit searches across entire surveys cost **under $15 on a single consumer GPU**.

## 4. Transit Least Squares (TLS): survey-scale GPU engine

cuvarbase 1.0's fast TLS path (`tls_search_batch()`: batch-native phase-binned
kernel + exact top-K refinement) measured end-to-end, 100% injected-transit
recovery in every regime (raw JSON in `benchmarks/results/tls_survey_jul2026/`,
provenance notes in that directory's README):

| Regime | RTX A5000 (sm86) | RTX 4000 Ada (sm89) | V100 (sm70) |
|---|---:|---:|---:|
| TESS FFI sector (1k pts, 8.5k periods) | **1.25 ms/LC** (~800 LC/s) | 3.05 ms | 1.42 ms |
| K2 90-d | 3.1 ms | 6.4 ms | 3.9 ms |
| TESS 2-min sector (20k pts) | 2.8 ms | 5.1 ms | 3.1 ms |
| 1-yr / 30-min cadence | 18.4 ms | 27.3 ms | 16.5 ms |
| Kepler 4-yr (65k pts, 172k periods) | **168 ms/LC** | 198 ms | 146 ms |

**Versus GTLS** (arXiv:2607.00348, the only other GPU TLS, CuPy-based): measured
head-to-head on the *same* RTX A5000 with an identical Ofir period grid, matched
per-period duration windows, matched epoch density, and the SDE recomputed with
one identical statistic on both methods' chi2 spectra — cuvarbase-TLS is
**30–171× faster over 200–2000-day baselines** (30× at 200 d growing to 171× at
2000 d) at 1–3% SDE parity and 100% recovery, and beats GTLS's own published
RTX-4090 numbers by 23–40× from the slower A5000. Cold single-shot (one star,
fresh process, compile included) still favors cuvarbase by 2.6–34× over the same
baselines. Full methodology: `analysis/GTLS_COMPARISON.md`.

**Versus the reference CPU `transitleastsquares`** (all cores of the same pod,
same light curves and grid): thousands of times faster — ~1,000–3,000× at
reference-matched epoch density (`t0_oversample=33`), ~10,000×+ at the default
grid; the exact multiple is CPU-dependent (archived references for one config
vary 2.7× between pods). Detection significance is preserved: SDE within 1–3%
of the reference at the default grid, within 1% at matched density (~5–15×
cost), with the exact refinement pass restoring full parameter precision either
way. Fidelity data: `benchmarks/results/tls_survey_jul2026/fidelity_raw_a5000.txt`
and `analysis/TLS_COST_ANALYSIS.md`.

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

## 6. Combined LS + BLS Survey Cost

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

Results are saved to `benchmarks/results/benchmark_results_new_features.json`.

## References

- Kovacs, G., Zucker, S., & Mazeh, T. (2002). A box-fitting algorithm in the search for periodic transits. A&A, 391, 369.
- VanderPlas, J. T. (2018). Understanding the Lomb-Scargle Periodogram. ApJS, 236, 16.
- Kunimoto, M. et al. (2023). TESS Quick-Look Pipeline GPU Transit Search. RNAAS, 7, 28.
- Wang, K. et al. (2024). GPU Phase Folding and Convolutional Neural Network. MNRAS, 528, 4053.
- Smith, L. C. et al. (2025). CETRA: Cambridge Exoplanet Transit Recovery Algorithm. MNRAS, 539, 297.
- Shahaf, S. et al. (2022). fBLS: A fast-folding BLS algorithm. MNRAS, 513, 2732.
- Garrison, L. H., Foreman-Mackey, D., Shih, Y.-H., & Barnett, A. (2024). nifty-ls: Fast and Accurate Lomb-Scargle Periodograms Using a Non-Uniform FFT. arXiv:2409.08090.
