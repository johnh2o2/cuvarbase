# cuvarbase

**GPU-accelerated time series analysis tools for astronomy** — period-finding and transit-detection algorithms (BLS, TLS, Lomb-Scargle, PDM, CE) built on [PyCUDA](https://mathema.tician.de/software/pycuda/). Created by John Hoffman, (c) 2017.

> **Note:** the current PyPI release (`0.2.5`) predates this v1.0 rewrite. Until v1.0.0 is published to PyPI, install from source (see [Installation](#installation)).

## Performance at Survey Scale

cuvarbase is built for processing millions of lightcurves, and it is proven in production: **NASA's TESS Quick-Look Pipeline has run cuvarbase's GPU BLS on every TESS sector since Sector 59** ([Kunimoto et al. 2023](https://ui.adsabs.harvard.edu/abs/2023RNAAS...7...28K/abstract)).

The headline numbers, all traceable to archived benchmark data in this repository:

- **Standard BLS is 257-354x faster than astropy's `BoxLeastSquares`**, measured consistently across all 7 GPU architectures tested (V100 through H200)
- **Transit Least Squares is 30-171x faster than GTLS** — the only other GPU TLS — on the same GPU at matched search settings and equal (1-3%) detection significance, and thousands of times faster than the reference CPU `transitleastsquares` (methodology and the reproduced GTLS-paper figure: [analysis/GTLS_COMPARISON.md](analysis/GTLS_COMPARISON.md))
- **Survey-scale Lomb-Scargle beats [nifty-ls](https://github.com/flatironinstitute/nifty-ls)**, the fastest CPU implementation, by 1.5-12.6x per lightcurve at realistic survey frequency grids (>15x where nifty-ls exceeded the benchmark timeout). Honest caveat: for one-off small searches (< ~100K frequencies), nifty-ls on CPU is the better tool
- **Keplerian frequency grids search 4-37x fewer frequencies** than uniform grids at survey baselines by exploiting the orbital-mechanics link between period and transit duration
- **All four major surveys for ~$33 of GPU time**: Lomb-Scargle + BLS over ZTF + HAT-Net + TESS + Kepler scale collections, on a rented RTX A5000 at $0.20/hr

Full tables, per-survey costs, and methodology: [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md).

## Features

- **Box Least Squares ([BLS](https://adsabs.harvard.edu/abs/2002A%26A...391..369K))** — the production-validated transit search behind the TESS QLP: standard, adaptive, and batched multi-lightcurve GPU paths, plus sparse BLS ([Panahi & Zucker 2021](https://arxiv.org/abs/2103.06193)) for small datasets (< 500 observations, GPU and CPU)
- **Transit Least Squares ([TLS](https://ui.adsabs.harvard.edu/abs/2019A%26A...623A..39H/abstract))** — limb-darkened transit templates, Ofir (2014) period grids, and a survey-scale batch engine (`tls_search_batch`) with no lightcurve-length cap; golden-tested against the reference `transitleastsquares` package
- **Generalized [Lomb-Scargle](https://arxiv.org/abs/0901.2573) periodogram** — NFFT-accelerated, with multiharmonic support and Baluev false-alarm probabilities
- **Phase Dispersion Minimization ([PDM](https://www.stellingwerf.com/rfs-bin/index.cgi?action=PageView&id=29))** — binned and binless variants with fast shared-memory kernels; to our knowledge the only GPU PDM in existence
- **Conditional Entropy period finder ([CE](https://adsabs.harvard.edu/abs/2013MNRAS.434.2629G))** — maintenance mode: it works and will keep working, but for an actively developed GPU CE/AOV search we recommend [periodfind](https://github.com/scope-ml/periodfind)
- **Non-equispaced fast Fourier transform ([NFFT](http://epubs.siam.org/doi/abs/10.1137/0914081))** — the adjoint operation that powers the fast Lomb-Scargle

**Experimental** (emits a `UserWarning` at first construction; not yet validated for science use; outside the 1.x stability promise, with its injection-recovery re-validation pending): the NUFFT-based likelihood-ratio transit search `cuvarbase.nufft_lrt`, contributed by **Jamila Taaki** ([@xiaziyna](https://github.com/xiaziyna)) — a frequency-domain matched filter for box transits in correlated noise, with marginalized and sequential systematics-aware detectors. It is importable as `cuvarbase.nufft_lrt` but deliberately not exported from the top-level namespace.

## Installation

Requirements: an NVIDIA GPU, the CUDA Toolkit (11.x or 12.x recommended), and Python 3.9+.

Until v1.0.0 is published to PyPI (the current PyPI release is the older `0.2.5`), install the v1.0 line from GitHub:

```bash
pip install "git+https://github.com/johnh2o2/cuvarbase.git@v1.0"
```

or clone the repository and `pip install -e .` for a development checkout.

Notes:

- `import cuvarbase` does **not** create a CUDA context or require a GPU (or even pycuda) — the context is created lazily on first GPU use. The pure helpers in `cuvarbase.utils`, `cuvarbase.bls_frequencies`, `cuvarbase.tls_grids`, `cuvarbase.tls_models` and `cuvarbase.tls_stats` work without pycuda; the method modules (`cuvarbase.bls` with `sparse_bls_cpu`/`single_bls`, `cuvarbase.lombscargle` with `fap_baluev`, ...) import `pycuda.driver` at module top, so they need the pycuda package installed but touch no device until the first GPU call. See [INSTALL.rst](INSTALL.rst) for the `--no-deps` install path on CUDA-less machines.
- Device selection follows the `CUDA_DEVICE` environment variable, read at first GPU use (e.g. `CUDA_DEVICE=1 python script.py`; for multiple GPUs, split jobs across processes).
- Optional extras: [batman-package](https://github.com/lkreidberg/batman) enables limb-darkened TLS templates; `cuvarbase[cufinufft]` enables the alternative cuFINUFFT Lomb-Scargle backend.

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

Full documentation — including Lomb-Scargle, TLS, CE, and PDM walkthroughs — is at **https://johnh2o2.github.io/cuvarbase/**; two runnable notebooks (Lomb-Scargle and PDM) are in [notebooks/](notebooks/).

## What's New in v1.0

v1.0 is a major modernization — the first release since the `0.2.x` line on PyPI — with large architectural speedups (an LRU kernel cache alone makes per-lightcurve loops **34x faster**; survey-speed BLS kernels add **2.0-12.7x end-to-end**), the new survey-scale TLS engine, correct results on absolute BJD-scale timestamps (silently wrong before), sparse BLS, batched BLS, Keplerian frequency grids, multiharmonic GPU Lomb-Scargle, a PDM/CE overhaul contributed by [@astrobatty](https://github.com/astrobatty) (PRs #57-#62, #65), Python 3.9-3.14 + numpy 2.x support without scikit-cuda, and a GPU-validated test suite of 1,582 tests (0 skips on-device, September 2026).

The complete list: [CHANGELOG.rst](https://github.com/johnh2o2/cuvarbase/blob/master/CHANGELOG.rst), with release notes in [docs/RELEASE_NOTES_v1.0.0.md](docs/RELEASE_NOTES_v1.0.0.md) and measured performance in [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md).

## Testing

```bash
pytest
```

The test suite runs **on CPU**: `cuvarbase/tests/conftest.py` stubs `pycuda`, so a bare `pytest` (or `pytest --pyargs cuvarbase` from an installed wheel) runs the pure-CPU tests anywhere and the GPU-dependent tests skip (this is what CI does on Python 3.9-3.14). A CUDA-capable GPU is needed only to exercise the GPU kernels themselves, which are validated on-device before releases.

## Contributing

Contributions are very welcome — see the [Contributing Guide](CONTRIBUTING.md) for development setup, code standards, testing requirements, and the PR process, and the [issue tracker](https://github.com/johnh2o2/cuvarbase/issues) for bug reports and feature requests.

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

**If you're interested in contributing, please see our [Contributing Guide](CONTRIBUTING.md)!**

## License & Acknowledgments

Licensed under GPLv3 — see [LICENSE.txt](LICENSE.txt).

Special thanks to Joel Hartman (author of the original `vartools`), Gaspar Bakos, Kevin Burdge, Attila Bódi ([@astrobatty](https://github.com/astrobatty) — PDM, CE, Lomb-Scargle, and BLS contributions throughout v1.0), and **Jamila Taaki** ([@xiaziyna](https://github.com/xiaziyna) — the NUFFT likelihood-ratio transit search; see Taaki, Kamalabadi & Kemball 2020, *Bayesian Methods for Joint Exoplanet Transit Detection and Systematic Noise Characterization*, and the [reference implementation](https://github.com/star-skelly/code_nova_exoghosts)) — and to all users and contributors who have made cuvarbase useful to the astronomy community.

For questions, issues, or contributions: https://github.com/johnh2o2/cuvarbase/issues
