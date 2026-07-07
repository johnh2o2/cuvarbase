# cuvarbase

**GPU-accelerated time series analysis tools for astronomy**

> **Note:** the current PyPI release (`0.2.5`) predates this v1.0 rewrite. Until v1.0.0 is published to PyPI, install from source (see [Installation](#installation)).

## Performance at Survey Scale

cuvarbase is built for processing millions of lightcurves, and it is proven in production: **NASA's TESS Quick-Look Pipeline has run cuvarbase's GPU BLS on every TESS sector since Sector 59** ([Kunimoto et al. 2023](https://ui.adsabs.harvard.edu/abs/2023RNAAS...7...28K/abstract)).

The headline numbers, all traceable to benchmark data in this repository:

- **Standard BLS is 257-354x faster than astropy's `BoxLeastSquares`**, measured consistently across all 7 GPU architectures tested (V100 through H200)
- **Transit Least Squares is 30-170x faster than GTLS** — the only other GPU TLS — on the same GPU at matched search settings and equal (1-3%) detection significance, and thousands of times faster than the reference CPU `transitleastsquares` ([details](#transit-least-squares-tls))
- **Keplerian frequency grids search 4-37x fewer frequencies** than uniform grids at survey baselines by exploiting the orbital-mechanics link between period and transit duration
- **All four major surveys for ~$33 of GPU time**: running both Lomb-Scargle and BLS over ZTF + HAT-Net + TESS + Kepler scale lightcurve collections costs roughly $33 total on a rented RTX A5000 at $0.20/hr (tables below)

### BLS Transit Search

cuvarbase provides a production-validated GPU implementation of the standard BLS algorithm ([Kovacs et al. 2002](https://adsabs.harvard.edu/abs/2002A%26A...391..369K)) — the implementation behind the TESS QLP transit search. Combined with Keplerian frequency grids:

| Survey | Lightcurves | N_freq (Keplerian) | Throughput | Total cost |
|--------|------------:|-------------------:|-----------:|-----------:|
| ZTF | 10,000,000 | 60K | 802 LC/s | **$0.69** |
| HAT-Net | 10,000,000 | 301K | 38 LC/s | **$14.74** |
| TESS (all sectors) | 5,200,000 | 1.8K | 236 LC/s | **$1.22** |
| Kepler | 200,000 | 131K | 6 LC/s | **$2.00** |

### Lomb-Scargle Periodogram

At the frequency counts real variability surveys require (100K-1.8M), GPU LS is **1.5-12.6x faster** than [nifty-ls](https://github.com/flatironinstitute/nifty-ls), the fastest CPU implementation, in head-to-head measurements — and **>15x** where nifty-ls could not finish within the 120s timeout:

| Survey | N_freq | GPU (ms/LC) | nifty-ls (ms/LC) | Speedup |
|--------|-------:|------------:|------------------:|--------:|
| ZTF | 365K | 4.4 | timeout | >27x |
| HAT-Net | 1.825M | 19.2 | timeout | >15x |
| TESS | 13.5K | 3.3 | 4.9 | 1.5x |
| Kepler | 730K | 19.8 | 250.0 | 12.6x |

**Honest caveat**: at small problem sizes (e.g. 10K observations x 5K
frequencies, single lightcurves), nifty-ls on CPU is faster than cuvarbase's
GPU LS — the GPU advantage appears at survey-scale frequency grids (>~100K
frequencies) and batched workloads. Use nifty-ls for one-off small searches.

### Transit Least Squares (TLS)

cuvarbase's survey-scale TLS ([Hippke & Heller 2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...623A..39H/abstract)) is, to our knowledge, the fastest GPU TLS available. Reproducing the benchmark from the GTLS paper ([arXiv:2607.00348](https://arxiv.org/abs/2607.00348)) apples-to-apples on one RTX A5000 — identical Ofir period grid, matched per-period duration window, matched epoch density, one injected transit — cuvarbase-TLS is **30–170x faster than GTLS** over 200–2000 day baselines (the gap grows with baseline), at **1–3% detection-significance (SDE) parity** and 100% recovery:

| Baseline | GTLS | cuvarbase TLS | Speedup |
|--------|-------:|-------------:|--------:|
| 200 d | 4.1 s | 0.14 s | **30x** |
| 1000 d | 75.8 s | 0.88 s | **86x** |
| 2000 d | 348 s | 2.0 s | **171x** |

It also beats GTLS's *own* published RTX-4090 numbers by 23–40x from a slower A5000, and runs thousands of times faster than the reference CPU `transitleastsquares`. Full methodology and the reproduced figure: [analysis/GTLS_COMPARISON.md](analysis/GTLS_COMPARISON.md).

See [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) for methodology, competitive analysis, and cost projections.

## About

`cuvarbase` is a Python library that uses [PyCUDA](https://mathema.tician.de/software/pycuda/) to implement several time series analysis tools used in astronomy on GPUs. It provides GPU-accelerated implementations of period-finding and variability analysis algorithms for astronomical time series data.

Created by John Hoffman, (c) 2017

## Features

Currently includes implementations of:

- **Generalized [Lomb-Scargle](https://arxiv.org/abs/0901.2573) periodogram** - Fast period finding for unevenly sampled data
- **Box Least Squares ([BLS](https://adsabs.harvard.edu/abs/2002A%26A...391..369K))** - Transit detection algorithm
  - **Adaptive GPU version** with automatic block-size tuning (`eebls_gpu_fast_adaptive()`)
  - Standard GPU-accelerated version (`eebls_gpu_fast()`)
  - Sparse BLS ([Panahi & Zucker 2021](https://arxiv.org/abs/2103.06193)) for small datasets (< 500 observations)
    - GPU implementation: `sparse_bls_gpu()` (default)
    - CPU implementation: `sparse_bls_cpu()` (per-call alternative;
      the `pycuda` package must still be installed/importable —
      `cuvarbase.bls` imports `pycuda.driver` at module top — but no
      GPU or CUDA context is created until a GPU search actually runs)
- **Non-equispaced fast Fourier transform (NFFT)** - Adjoint operation ([paper](http://epubs.siam.org/doi/abs/10.1137/0914081))
- **Conditional Entropy period finder ([CE](https://adsabs.harvard.edu/abs/2013MNRAS.434.2629G))** - Non-parametric period finding
  - **Maintenance mode**: CE works and will keep working, but no further development is planned here. For new projects that want an actively developed GPU conditional entropy (or AOV) search, we recommend [periodfind](https://github.com/scope-ml/periodfind) from the ZTF/SCoPe team
- **Phase Dispersion Minimization ([PDM](http://www.stellingwerf.com/rfs-bin/index.cgi?action=PageView&id=29))** - Statistical period finding
  - Binned (step and linear-interpolation) and binless (tophat and Gaussian kernel) variants, each with fast shared-memory kernels
  - To our knowledge the only GPU PDM implementation in existence

### Experimental Features

This module ships in this release but has **known correctness issues** and
is not recommended for science use yet. It emits a `UserWarning` on import.

- **Transit Least Squares ([TLS](https://ui.adsabs.harvard.edu/abs/2019A%26A...623A..39H/abstract))** (`cuvarbase.tls`) - GPU transit
  detection with a limb-darkened template, optimal depth fitting, and
  Ofir (2014) period grids. The survey-scale fast path
  (`tls_search_batch`, default) folds each light curve once into phase
  bins and refines the top candidates exactly, handling **arbitrary
  light-curve length** (the legacy per-point kernel is still available
  and caps at ~3,500 points). Detection significance now matches both
  the reference `transitleastsquares` and the GTLS package to **1–3%**
  with 100% injected-transit recovery in our tests (see
  [Performance](#transit-least-squares-tls)), but a full
  injection-recovery completeness campaign is still outstanding — so it
  remains flagged experimental and emits a `UserWarning` on import.

- **NUFFT-based Likelihood Ratio Test** (`cuvarbase.nufft_lrt`,
  contributed by **Jamila Taaki** / [@xiaziyna](https://github.com/xiaziyna)) -
  a frequency-domain matched-filter / likelihood-ratio test for box
  transits in correlated noise. The data and templates are transformed
  with the GPU adjoint NFFT, which handles gappy / multi-season sampling
  over the full baseline (the earlier CPU-rfft and grid-truncation issues
  are fixed). The matched-filter combination runs on the host; the method
  has not yet had a full injection-recovery validation.

### Planned Features

Future developments may include:

- (Weighted) wavelet transforms
- Spectrograms (for PDM and GLS)

## Installation

### Prerequisites

- CUDA-capable GPU (NVIDIA)
- CUDA Toolkit (11.x or 12.x recommended)
- Python 3.9 or later

Note: `import cuvarbase` does **not** create a CUDA context or require a
GPU — the primary context is retained lazily on first GPU use (compiling
a kernel, constructing a periodogram process, or calling a GPU search
function). So `import cuvarbase` and the CPU-only helpers (e.g.
`sparse_bls_cpu`, `single_bls`, `fap_baluev`) run on a GPU-less machine.
The GPU modules still `import pycuda.driver` at module top, so the
`pycuda` package must be installed to use them, but importing them
allocates no context. Device selection follows the `CUDA_DEVICE`
environment variable, read at first GPU use (not at import) — set it
before the first GPU call to select a device other than 0, and prefer
spawning fresh processes over forking when using multiple GPUs.

### Dependencies

**Essential:**
- [PyCUDA](https://mathema.tician.de/software/pycuda/) - Python interface to CUDA

**Optional (for additional features and testing):**
- [matplotlib](https://matplotlib.org/) - For plotting utilities
- [nfft](https://github.com/jakevdp/nfft) - For unit testing
- [astropy](http://www.astropy.org/) - For unit testing
- [cufinufft](https://github.com/flatironinstitute/cufinufft) - Optional alternative NFFT backend for Lomb-Scargle (`use_cufinufft=True`)

### Install from source

Until v1.0.0 is published to PyPI (the current PyPI release is the older
`0.2.5`), install the v1.0 line directly from GitHub:

```bash
pip install "git+https://github.com/johnh2o2/cuvarbase.git@v1.0"
```

Or for a development checkout:

```bash
git clone https://github.com/johnh2o2/cuvarbase.git
cd cuvarbase
pip install -e .
```

### Docker Installation

For easier setup with CUDA 11.8:

```bash
docker build -t cuvarbase .
docker run -it --gpus all cuvarbase
```

## Documentation

Full documentation is available at: https://johnh2o2.github.io/cuvarbase/

## Quick Start

### Box Least Squares (BLS) - Transit Detection

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

# Or use adaptive BLS for automatic block-size tuning
power_adaptive = bls.eebls_gpu_fast_adaptive(t, y, dy, freqs)
```

For more advanced usage including Lomb-Scargle, Conditional Entropy, and PDM walkthroughs, see the [full documentation](https://johnh2o2.github.io/cuvarbase/) and the runnable notebooks in [notebooks/](notebooks/). (The [examples/](examples/) directory currently holds only the TLS example.)

## Using Multiple GPUs

If you have more than one GPU, you can choose which one to use in a given script by setting the `CUDA_DEVICE` environment variable:

```bash
CUDA_DEVICE=1 python script.py
```

If anyone is interested in implementing a multi-device load-balancing solution, they are encouraged to do so! At some point this may become important, but for the time being manually splitting up the jobs to different GPUs will have to suffice.

## What's New in v1.0

v1.0 is a major modernization of cuvarbase — the first major release since the `0.2.x` line on PyPI. Highlights:

### ⚡ Performance Improvements (Major Update)

**Faster BLS transit search** — **257-354x faster** than astropy `BoxLeastSquares`, consistent across all 7 GPU architectures tested (V100 through H200). Relative to the last release (0.2.6), whose BLS *kernel* v1.0 inherits essentially unchanged:

- **Survey-speed kernels** (fused-noverlap, conflict-scatter, occupancy-aware
  chunking) make the per-frequency kernel **2.9-9.2x faster** and end-to-end
  survey searches **2.0-12.7x faster** than the pre-optimization v1.0 path
- **Batched multi-lightcurve search** (`eebls_gpu_batch`) is new — 0.2.6 offered
  only single-lightcurve calls, which recompiled the kernel on *every* call;
  v1.0's LRU kernel cache alone makes a naive per-lightcurve loop **34x faster**
- **Adaptive block sizing** (`eebls_gpu_fast_adaptive()`) auto-tunes the CUDA
  block size from the dataset (~1.3x over the fixed-block kernel on realistic
  Keplerian grids)
- Best cost-efficiency: RTX 4000 Ada at **$0.14 per million lightcurves**;
  see [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) for full results across GPUs

This makes large-scale BLS searches practical and efficient for all-sky surveys.

### Breaking Changes
- **Dropped Python 2.7 support** - now requires Python 3.9+
- Removed `future` package dependency and all Python 2 compatibility code
- Updated minimum dependency versions: numpy>=1.17, scipy>=1.3

### New Features

**Community contributions** (PRs #57-#62, with particular thanks to [@astrobatty](https://github.com/astrobatty)):
- **PDM overhaul**: fast shared-memory CUDA kernels for all four PDM variants, a backward-compatible `(t, y, err)` input API for `PDMAsyncProcess.run()` with automatic frequency grids, unit tests, and new [documentation](https://johnh2o2.github.io/cuvarbase/) — PDM is now a tested, documented, first-class method (and to our knowledge still the only GPU PDM available anywhere)
- **Conditional entropy**: optional log-probability periodogram (`compute_log_prob=True`), input normalization before processing, a 32-bit overflow guard for large `nfreq x ndata` runs, and a clear error for the unsupported `use_fast` + `weighted` combination
- **Lomb-Scargle**: improved GPU memory estimation (now accounts for cuFFT work areas and per-batch buffers) and lightcurve normalization for numerical stability

**Sparse BLS implementation** for efficient transit detection on small datasets:
- Based on algorithm from [Panahi & Zucker (2021)](https://arxiv.org/abs/2103.06193)
- **Both GPU (`sparse_bls_gpu`) and CPU (`sparse_bls_cpu`) implementations available**
- Optimized for datasets with < 500 observations
- Avoids binning and grid searching - directly tests all observation pairs as transit boundaries
- New `eebls_transit` wrapper automatically selects between sparse and standard BLS
  - **Default: GPU sparse BLS** for small datasets (use_gpu=True)
  - `use_gpu=False` runs the search itself on the CPU (`sparse_bls_cpu`).
    Since v1.0 `import cuvarbase` no longer creates a CUDA context, so the
    CPU helpers run on GPU-less machines (the `pycuda` package must still
    be installed, but no GPU is touched until a GPU search runs)
- Particularly useful for ground-based surveys with limited phase coverage

**Citation for Sparse BLS**: If you use this method, please cite:
- Panahi, A., & Zucker, S. (2021). *Sparse BLS: A sparse-modeling approach to the Box-fitting Least Squares periodogram.* [arXiv:2103.06193](https://arxiv.org/abs/2103.06193)

**Refactored codebase organization**:
- Cleaner module structure: `base/` and `memory/`
- Better maintainability and extensibility

### Improvements
- Modern Python packaging with `pyproject.toml`
- Docker support for easier installation with CUDA 11.8
- GitHub Actions CI: CPU test suite (GPU tests stubbed/skipped) on Python 3.9-3.12, plus a build-wheel-install-import packaging check; GPU kernels validated manually before releases
- Cleaner, more maintainable codebase (89 lines of compatibility code removed)
- Updated documentation and contributing guidelines

### Additional Documentation
- [Benchmark Results](docs/BENCHMARK_RESULTS.md) - Survey-scale performance, competitive analysis, and cost projections
- [Benchmarking Guide](docs/BENCHMARKING.md) - Performance testing methodology
- [RunPod Development](docs/RUNPOD_DEVELOPMENT.md) - Cloud GPU development setup
- [BLS Optimization History](docs/BLS_OPTIMIZATION.md) - Thread-safety, memory management, and GPU optimizations

For a complete list of changes, see [CHANGELOG.rst](https://github.com/johnh2o2/cuvarbase/blob/master/CHANGELOG.rst).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Development setup and prerequisites
- Code standards and conventions
- Testing requirements
- Pull request process
- Performance considerations for GPU code

### How to Contribute

1. **Bug Reports**: Open an issue with a clear description and minimal reproduction case
2. **Feature Requests**: Open an issue describing the feature and its use case
3. **Code Contributions**: 
   - Fork the repository
   - Create a feature branch
   - Make your changes following our coding standards
   - Add tests for new functionality
   - Submit a pull request with a clear description

### Best Practices for Issues and PRs

**Opening Issues:**
- Search existing issues first to avoid duplicates
- Provide a clear, descriptive title
- Include version information (cuvarbase, Python, CUDA, GPU model)
- For bugs: include minimal code to reproduce the issue
- For features: explain the use case and expected behavior

**Opening Pull Requests:**
- Reference related issues in the PR description
- Provide a clear description of changes and motivation
- Ensure all tests pass
- Add new tests for new functionality
- Follow the existing code style and conventions
- Keep PRs focused - one feature/fix per PR when possible

## Testing

Run tests with:

```bash
pytest cuvarbase/tests/
```

The test suite runs **on CPU**: the root `conftest.py` stubs `pycuda`/`scikit-cuda`, so the pure-CPU tests run anywhere and the GPU-dependent tests skip (this is what CI does on Python 3.9-3.12). A CUDA-capable GPU is needed only to exercise the GPU kernels themselves, which are validated manually before releases.

## Citation

If you use cuvarbase in your research, please cite:

**Hoffman, J. (2022). cuvarbase: GPU-Accelerated Variability Algorithms. Astrophysics Source Code Library, record ascl:2210.030.**

Available at: https://ui.adsabs.harvard.edu/abs/2022ascl.soft10030H/abstract

BibTeX:
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

## A Personal Note

This project was created as part of a PhD thesis, intended mainly for myself and against the very wise advice of two advisors trying to help me stay on track. Joel Hartman -- legendary author of `vartools` -- and Gaspar Bakos both showed me an incredible amount of patience. I had promised Gaspar a catalog of variable stars from HAT telescopes, something that should have taken maybe a month but instead took years due to an irrational and irresponsible level of perfectionism, and even at the end wasn't comprehensive or useful, and which I never published. To both of you: thank you.

Much to my absolute delight this repository has -- organically! -- become useful to several people in the astro community; an ADS search in late 2025 found roughly two dozen papers (~430 citations) using cuvarbase in some shape or form. The biggest source of pride was seeing the Quick Look Pipeline adopt cuvarbase for TESS ([Kunimoto et al. 2023](https://ui.adsabs.harvard.edu/abs/2023RNAAS...7...28K/abstract)).

Though usage is modest, to put this in personal context it is by far the most useful product of my PhD, and the fact that, amidst a lot of bumbling about for 5 years accomplishing very little, something productive somehow found its way into my thesis has given me a lot of relief and happiness.

I want to personally thank people who have given their time and support to this project, including Kevin Burdge, Attila Bodi, Jamila Taaki, and to everyone in the community that has used this tool.

## Future Plans and Call for Contributors

In the years since 2017, I moved away from astrophysics and life has gone on. I have regrettably had very little time to update this repository. The code quality -- abstractions, documentation, etc -- are reflective of my level of skill back then, which was quite rudimentary.

In 2025, for the first time, coding agents like `copilot` are finally at a level of quality that even a limited time investment in updating this repository can bring a lot of return. I would really like to encourage people interested to become official **contributors** so that I can pass the torch onto the larger community.

It would be nice to incorporate additional capabilities and algorithms, and improve robustness and portability, to make this library a much more professional and easy-to-use tool. Especially nowadays, with the world awash in GPUs and with the scale of time-series data becoming many orders of magnitude larger than it was 10 years ago, something like `cuvarbase` seems even more relevant today than it was back then. (Where others have built better tools for a given method — e.g. [periodfind](https://github.com/scope-ml/periodfind) for conditional entropy — we would rather point you to them than duplicate the effort.)

**If you're interested in contributing, please see our [Contributing Guide](CONTRIBUTING.md)!**

## License

See [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgments

This project has benefited from contributions and support from many people in the astronomy community. Special thanks to:

- Joel Hartman (author of the original `vartools`)
- Gaspar Bakos
- Kevin Burdge
- Attila Bodi
- **Jamila Taaki** ([@xiaziyna](https://github.com/xiaziyna)) - for contributing the NUFFT-based Likelihood Ratio Test (`cuvarbase.nufft_lrt`) for transit detection with correlated noise (reinstated in v1.0 with the GPU adjoint-NFFT rewire). See her papers:
  - Taaki, J. S., Kamalabadi, F., & Kemball, A. (2020). *Bayesian Methods for Joint Exoplanet Transit Detection and Systematic Noise Characterization.*
  - Reference implementation: https://github.com/star-skelly/code_nova_exoghosts
- All users and contributors who have helped make cuvarbase useful to the astronomy community

## Contact

For questions, issues, or contributions, please use the GitHub issue tracker:
https://github.com/johnh2o2/cuvarbase/issues
