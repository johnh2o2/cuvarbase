# cuvarbase

**GPU-accelerated time series analysis tools for astronomy** — period-finding and transit-detection algorithms (BLS, TLS, Lomb-Scargle, PDM, CE) built on [PyCUDA](https://mathema.tician.de/software/pycuda/) and [CuPy](https://docs.cupy.dev/en/v13.6.0/install.html). Created by John Hoffman, (c) 2017.

**Faster transit searches for TESS and ZTF.** v1 BLS is **1.8–4.3× faster than PyPI 0.2.5** in the measured batches; separated TESS supports the recovery comparison. **TLS is 3.6–4.6× faster than GTLS for one lightcurve and 1.5–2.4× faster in batches**, using its observation-level numerical search and full refinement, without phase binning.

![BLS and TLS search times on TESS and ZTF cadences](https://raw.githubusercontent.com/johnh2o2/cuvarbase/v1.0-fixes/docs/figures/transit_benchmarks_20260910.png)

Hollow markers: one lightcurve. Filled markers: time per lightcurve in a 16-source batch. Each comparison uses the same inputs and device: A40 for BLS, RTX A6000 for TLS. TLS batches compare one cuvarbase worker with the fastest eligible GTLS pool of 1, 2 or 4 workers. Times include each API's normal output work. The original campaign gate failed when four-worker GTLS ran out of memory during the separated-TESS warmup. The figure uses a separately audited, post hoc report of the completed configurations. The report preserves that failure and separates search time from GTLS's additional diagnostics. [Results and methodology](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/docs/TRANSIT_BENCHMARKS.md) · [PDF figure](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/docs/figures/transit_benchmarks_20260910.pdf)

## Transit-search performance

cuvarbase is built for processing millions of lightcurves. **TESS's Quick-Look Pipeline adopted cuvarbase's GPU BLS starting in Sector 59** ([Kunimoto et al. 2023](https://arxiv.org/abs/2302.01293)).

**BLS does less repeated work.** For each trial period, v1 reuses folded phase histograms across multiple phase offsets. Disabling this fusion made diagnostic API calls 1.35–1.57× slower. Vectorized host scans and Keplerian-grid construction remove Python loops over large grids; grid construction alone was 11–17× faster. The batch API amortizes allocation and dispatch across lightcurves. Both releases receive warmed kernels and reusable memory in these comparisons.

Against external BLS implementations, measured batch searches were **19–57× faster than the strongest tested CPU settings** (Astropy or periodfind) and **1.5–11.9× faster than periodfind GPU**. The report identifies comparisons whose recovery and false-positive results support the stated 5-point criterion.

**TLS preserves the search while removing repeated work.** The standard engine evaluates GTLS's sample windows and transit templates, using its depth estimates and full refinement. Fused kernels reuse residual calculations and reduce winning trials without storing the full residual tensor. Reusable CUDA graphs replay the native cumulative-sum operations with fewer Python dispatches. Smaller GPU workspaces do not narrow the duration search. Invalid candidates are excluded before ranking, correcting a GTLS mask-handling defect documented in the comparison.

**Thin transits use the same default.** There is no phase-bin cap or separate narrow-transit accuracy preset. All **184 independent injection and noise-only cases** matched corrected GTLS's numerical searches exactly, including ordinary, high-impact, eccentric and dense-M-dwarf regimes. The [numerical validation](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/docs/TLS_NUMERICS.md) also records extreme stress tests and shared floating-point limits. Both engines still need observed transits, an appropriate period domain and enough signal. The earlier phase-binned engine remains available explicitly as `method='binned'`; its much larger historical speed ratios do not describe the new default.

For a concrete QLP-oriented upgrade result, BLS on separated TESS sectors was **2.73× faster in batches**, or **10.18× faster including a fresh grid**, with the same **89/128** detected injections as PyPI. Paired confidence bounds support less than a 5-percentage-point recovery loss and less than a 5-point false-positive increase on this test population. Other PyPI comparisons remain inconclusive under that criterion.

The [benchmark report](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/docs/TRANSIT_BENCHMARKS.md) separates full API time, search computation and output diagnostics. [Cost projections](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/docs/TLS_COST_ANALYSIS.md) cover GPU search rental; preprocessing, I/O and candidate vetting are additional work.

## Features

- **Box Least Squares ([BLS](https://adsabs.harvard.edu/abs/2002A%26A...391..369K))** — the production-validated transit search behind the TESS QLP: standard, adaptive, and batched multi-lightcurve GPU paths, plus sparse BLS ([Panahi & Zucker 2021](https://arxiv.org/abs/2103.06193)) for small datasets (< 500 observations, GPU and CPU)
- **Transit Least Squares ([TLS](https://ui.adsabs.harvard.edu/abs/2019A%26A...623A..39H/abstract))** — limb-darkened transit templates, Ofir (2014) period grids, and a GTLS-compatible observation-level default with full refinement, plus a survey wrapper (`tls_search_batch`) and an explicit approximate binned option
- **Generalized [Lomb-Scargle](https://arxiv.org/abs/0901.2573) periodogram** — NFFT-accelerated, with multiharmonic support and Baluev false-alarm probabilities
- **Phase Dispersion Minimization ([PDM](https://www.stellingwerf.com/rfs-bin/index.cgi?action=PageView&id=29))** — binned and binless variants with fast shared-memory kernels; to our knowledge the only GPU PDM in existence
- **Conditional Entropy period finder ([CE](https://adsabs.harvard.edu/abs/2013MNRAS.434.2629G))** — maintenance mode: it works and will keep working, but for an actively developed GPU CE/AOV search we recommend [periodfind](https://github.com/scope-ml/periodfind)
- **Non-equispaced fast Fourier transform ([NFFT](http://epubs.siam.org/doi/abs/10.1137/0914081))** — the adjoint operation that powers the fast Lomb-Scargle

**Experimental** (emits a `UserWarning` at first construction; outside the 1.x stability promise): the NUFFT-based likelihood-ratio transit search `cuvarbase.nufft_lrt`, contributed by **Jamila Taaki** ([@xiaziyna](https://github.com/xiaziyna)) — a frequency-domain matched filter for box transits in correlated noise, with marginalized and sequential systematics-aware detectors. Its Sep-2026 fixes were re-validated by an injection-recovery campaign (the default path is correct on BJD-scale times; the systematics-aware detectors recover 98% of 1.6%-deep transits where basis-free BLS and the then-current binned TLS recover 2% or less; PSD whitening itself gave no gain over a flat PSD, and BLS/TLS were more complete in white noise) — see the [NUFFT-LRT page](https://johnh2o2.github.io/cuvarbase/nufft_lrt.html). It stays experimental because its defaults and `run()` conventions may still change; it is importable as `cuvarbase.nufft_lrt` but deliberately not exported from the top-level namespace.

## Installation

Requirements: an NVIDIA GPU, the CUDA Toolkit (1.0 is validated against CUDA 12.4; `nvcc` on your `PATH`), and Python 3.9-3.14.

The benchmarks describe the v1 candidate on `v1.0-fixes`. [PyPI](https://pypi.org/project/cuvarbase/) still provides 0.2.5 as of 10 September 2026. To install this candidate:

```bash
pip install 'cuvarbase @ git+https://github.com/johnh2o2/cuvarbase@v1.0-fixes'
```

For a development checkout, clone the `v1.0-fixes` branch and `pip install -e '.[test,tls]'` (Python 3.9–3.13 for the TLS extra). PyCUDA builds against your CUDA toolkit during installation, so a CUDA-less machine needs the `--no-deps` path described in INSTALL.rst (see the link below).

Notes:

- `import cuvarbase` does **not** create a CUDA context or require a GPU (or even pycuda) — the context is created lazily on first GPU use. The pure helpers in `cuvarbase.utils`, `cuvarbase.bls_frequencies`, `cuvarbase.tls_grids`, `cuvarbase.tls_models` and `cuvarbase.tls_stats` work without pycuda; the method modules (`cuvarbase.bls` with `sparse_bls_cpu`/`single_bls`, `cuvarbase.lombscargle` with `fap_baluev`, ...) import `pycuda.driver` at module top, so they need the pycuda package installed but touch no device until the first GPU call. See [INSTALL.rst](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/INSTALL.rst) for the `--no-deps` install path on CUDA-less machines.
- Device selection follows the `CUDA_DEVICE` environment variable, read at first GPU use (e.g. `CUDA_DEVICE=1 python script.py`; for multiple GPUs, split jobs across processes).
- `pip install 'cuvarbase[tls] @ git+https://github.com/johnh2o2/cuvarbase@v1.0-fixes'` includes CuPy 13 for CUDA 12 and [batman-package](https://github.com/lkreidberg/batman), required by the standard TLS engine (Python 3.9–3.13). See INSTALL.rst for other CUDA runtimes; the `cufinufft` extra enables the alternative cuFINUFFT Lomb-Scargle backend.

## Quick Start

```python
import numpy as np
from cuvarbase import bls

# Generate some sample time series data
t = np.sort(np.random.uniform(0, 10, 1000)).astype(np.float32)
y = np.sin(2 * np.pi * t / 2.5) + np.random.normal(0, 0.1, len(t))
dy = np.ones_like(y) * 0.1  # uncertainties

# Define frequency grid
freqs = np.linspace(0.1, 2.0, 5000).astype(np.float32)

# Standard BLS (returns power array and best (q, phi) solutions per frequency)
power, solutions = bls.eebls_gpu(t, y, dy, freqs)
best_freq = freqs[np.argmax(power)]
print(f"Best period: {1/best_freq:.2f} (expected: 2.5)")
```

Full documentation — including Lomb-Scargle, TLS, CE, and PDM walkthroughs — is at **https://johnh2o2.github.io/cuvarbase/**; two runnable notebooks (Lomb-Scargle and PDM) are in [notebooks/](https://github.com/johnh2o2/cuvarbase/tree/v1.0.0/notebooks/).

## What's New in v1.0

v1.0 is a major modernization — the first release since the `0.2.x` line on PyPI — with faster transit searches and Keplerian grid construction ([measured results](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/docs/TRANSIT_BENCHMARKS.md)), the new observation-level TLS engine, correct results on absolute BJD-scale timestamps (silently wrong before), sparse BLS, batched BLS, Keplerian frequency grids, multiharmonic GPU Lomb-Scargle, a PDM/CE overhaul contributed by [@astrobatty](https://github.com/astrobatty) (PRs #57-#62, #65), and Python 3.9-3.14 + numpy 2.x support without scikit-cuda. The standard TLS extra supports Python 3.9-3.13.

The new TLS implementation passed **265 GPU tests with zero failures or skips** on an NVIDIA A40 on 10 September 2026. The [validation receipts](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/docs/validation/README.md) record exact tested sources and retain the earlier full release suite's 1,785 passes and one expected failure separately.

The complete list: [CHANGELOG.rst](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/CHANGELOG.rst), with release notes in [docs/RELEASE_NOTES_v1.0.0.md](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/docs/RELEASE_NOTES_v1.0.0.md) and measured performance in [docs/BENCHMARK_RESULTS.md](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/docs/BENCHMARK_RESULTS.md).

## Testing

```bash
pytest
```

The test suite runs **on CPU**: `cuvarbase/tests/conftest.py` stubs `pycuda`, so a bare `pytest` (or `pytest --pyargs cuvarbase` from an installed wheel) runs the pure-CPU tests anywhere and the GPU-dependent tests skip (this is what CI does on Python 3.9-3.14). A CUDA-capable GPU is needed only to exercise the GPU kernels themselves, which are validated on-device before releases.

## Contributing

Contributions are very welcome — see the [Contributing Guide](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/CONTRIBUTING.md) for development setup, code standards, testing requirements, and the PR process, and the [issue tracker](https://github.com/johnh2o2/cuvarbase/issues) for bug reports and feature requests.

## Citation

If you use cuvarbase in your research, please cite [Hoffman (2022), ASCL record ascl:2210.030](https://ui.adsabs.harvard.edu/abs/2022ascl.soft10030H/abstract):

```bibtex
@MISC{2022ascl.soft10030H,
       author = {{Hoffman}, John},
        title = "{cuvarbase: GPU-Accelerated Variability Algorithms}",
     keywords = {Software},
 howpublished = {Astrophysics Source Code Library, record ascl:2210.030},
         year = 2022,
        month = oct,
          eid = {ascl:2210.030},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022ascl.soft10030H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

If you use the sparse BLS method, please also cite [Panahi & Zucker (2021)](https://arxiv.org/abs/2103.06193).

## A Personal Note

This project was created as part of a PhD thesis, intended mainly for myself and against the very wise advice of two advisors trying to help me stay on track. Joel Hartman -- legendary author of `vartools` -- and Gaspar Bakos both showed me an incredible amount of patience. I had promised Gaspar a catalog of variable stars from HAT telescopes, something that should have taken maybe a month but instead took years due to an irrational and irresponsible level of perfectionism, and even at the end wasn't comprehensive or useful, and which I never published. To both of you: thank you.

Much to my absolute delight this repository has -- organically! -- become useful to several people in the astro community; an ADS search in late 2025 found roughly two dozen papers (~430 citations) using cuvarbase in some shape or form. The biggest source of pride was seeing the Quick Look Pipeline adopt cuvarbase for TESS ([Kunimoto et al. 2023](https://ui.adsabs.harvard.edu/abs/2023RNAAS...7...28K/abstract)).

Though usage is modest, to put this in personal context it is by far the most useful product of my PhD, and the fact that, amidst a lot of bumbling about for 5 years accomplishing very little, something productive somehow found its way into my thesis has given me a lot of relief and happiness.

I want to personally thank people who have given their time and support to this project, including Kevin Burdge, Attila Bodi, Jamila Taaki, and to everyone in the community that has used this tool.

## Future Plans and Call for Contributors

In the years since 2017, I moved away from astrophysics and life has gone on. With coding agents finally good enough that a limited time investment can bring a lot of return, I would really like to encourage interested people to become official **contributors** so that I can pass the torch onto the larger community. With the world awash in GPUs and time-series datasets orders of magnitude larger than a decade ago, something like `cuvarbase` seems even more relevant today than when it started — and where others have built better tools for a given method (e.g. [periodfind](https://github.com/scope-ml/periodfind) for conditional entropy), we would rather point you to them than duplicate the effort.

**If you're interested in contributing, please see our [Contributing Guide](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/CONTRIBUTING.md)!**

## License & Acknowledgments

Licensed under GPLv3 — see [LICENSE.txt](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/LICENSE.txt).

The observation-level TLS engine adapts [GTLS](https://github.com/Farthing-0/GTLS/tree/74e449c325792a763dde4fbffab98039c5e8c111), with the original MIT notices crediting Michael Hippke and Quanquan Hu retained in the source. The [comparison](https://github.com/johnh2o2/cuvarbase/blob/v1.0-fixes/docs/GTLS_COMPARISON.md) documents the shared algorithm and cuvarbase's execution changes.

Special thanks to Joel Hartman (author of the original `vartools`), Gaspar Bakos, Kevin Burdge, Attila Bódi ([@astrobatty](https://github.com/astrobatty) — PDM, CE, Lomb-Scargle, and BLS contributions throughout v1.0), and **Jamila Taaki** ([@xiaziyna](https://github.com/xiaziyna) — the NUFFT likelihood-ratio transit search; see Taaki, Kamalabadi & Kemball 2020, *Bayesian Methods for Joint Exoplanet Transit Detection and Systematic Noise Characterization*, and the [reference implementation](https://github.com/star-skelly/code_nova_exoghosts)) — and to all users and contributors who have made cuvarbase useful to the astronomy community.

For questions, issues, or contributions: https://github.com/johnh2o2/cuvarbase/issues
