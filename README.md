# cuvarbase

[![PyPI version](https://badge.fury.io/py/cuvarbase.svg)](https://badge.fury.io/py/cuvarbase)

**GPU-accelerated time series analysis tools for astronomy**

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

## About

`cuvarbase` is a Python library that uses [PyCUDA](https://mathema.tician.de/software/pycuda/) to implement several time series analysis tools used in astronomy on GPUs. It provides GPU-accelerated implementations of period-finding and variability analysis algorithms for astronomical time series data.

Created by John Hoffman, (c) 2017

### A Personal Note

This project was created as part of a PhD thesis, intended mainly for myself and against the very wise advice of two advisors trying to help me stay on track (including Joel Hartman -- legendary author of `vartools`, and Gaspar Bakos, who I promised to provide a catalog of variable stars from HAT telescopes -- something that should have taken maybe a month but instead took years due to an irrational and irresponsible level of perfectionism, and even at the end wasn't comprehensive or useful, and which I never published. To both of you, thank you for an incredible amount of patience.).

Much to my absolute delight this repository has -- organically! -- become useful to several people in the astro community; an ADS search reveals 23 papers with ~430 citations as of October 2025 using cuvarbase in some shape or form. The biggest source of pride was seeing the Quick Look Pipeline adopt cuvarbase for TESS ([Kunimoto et al. 2023](https://ui.adsabs.harvard.edu/abs/2023RNAAS...7...28K/abstract)).

Though usage is modest, to put this in personal context it is by far the most useful product of my PhD, and the fact that, amidst a lot of bumbling about for 5 years accomplishing very little, something productive somehow found its way into my thesis has given me a lot of relief and happiness.

I want to personally thank people who have given their time and support to this project, including Kevin Burdge, Attila Bodi, Jamila Taaki, and to everyone in the community that has used this tool.

### Future Plans and Call for Contributors

In the years since 2017, I moved away from astrophysics and life has gone on. I have regrettably had very little time to update this repository. The code quality -- abstractions, documentation, etc -- are reflective of my level of skill back then, which was quite rudimentary.

In 2025, for the first time, coding agents like `copilot` are finally at a level of quality that even a limited time investment in updating this repository can bring a lot of return. I would really like to encourage people interested to become official **contributors** so that I can pass the torch onto the larger community.

It would be nice to incorporate additional capabilities and algorithms (e.g. [Katz et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.2665K/abstract) greatly improved on the inefficient conditional entropy implementation in this repository), and improve robustness and portability, to make this library a much more professional and easy-to-use tool. Especially nowadays, with the world awash in GPUs and with the scale of time-series data becoming many orders of magnitude larger than it was 10 years ago, something like `cuvarbase` seems even more relevant today than it was back then.

**If you're interested in contributing, please see our [Contributing Guide](CONTRIBUTING.md)!**

## What's New in v1.0

This represents a major modernization effort compared to the `master` branch:

### ⚡ Performance Improvements (Major Update)

**Dramatically Faster BLS Transit Detection** - Up to **90x speedup** for sparse datasets:
- Adaptive block sizing automatically optimizes GPU utilization based on dataset size
- **5-90x faster** depending on number of observations (most dramatic for ndata < 500)
- Particularly beneficial for ground-based surveys and sparse time series
- Thread-safe kernel caching with LRU eviction for production environments
- **New function**: `eebls_gpu_fast_adaptive()` - drop-in replacement with automatic optimization
- See [docs/ADAPTIVE_BLS_RESULTS.md](docs/ADAPTIVE_BLS_RESULTS.md) for detailed benchmarks

This optimization makes large-scale BLS searches practical and efficient for all-sky surveys.

### Breaking Changes
- **Dropped Python 2.7 support** - now requires Python 3.7+
- Removed `future` package dependency and all Python 2 compatibility code
- Updated minimum dependency versions: numpy>=1.17, scipy>=1.3

### New Features

**NUFFT Likelihood Ratio Test (LRT)** for transit detection with correlated noise:
- Contributed by **Jamila Taaki** ([@xiaziyna](https://github.com/xiaziyna))
- GPU-accelerated matched filter in frequency domain with adaptive noise estimation
- Particularly effective for gappy data with red/correlated noise
- Naturally handles correlated (non-white) noise through power spectrum estimation
- More robust than traditional BLS under stellar activity and systematic noise
- See [docs/NUFFT_LRT_README.md](docs/NUFFT_LRT_README.md) for complete documentation

**Citation for NUFFT-LRT**: If you use this method, please cite:
- Taaki, J. S., Kamalabadi, F., & Kemball, A. (2020). *Bayesian Methods for Joint Exoplanet Transit Detection and Systematic Noise Characterization.*
- Reference implementation: https://github.com/star-skelly/code_nova_exoghosts

**Sparse BLS implementation** for efficient transit detection on small datasets:
- Based on algorithm from [Panahi & Zucker (2021)](https://arxiv.org/abs/2103.06193)
- **Both GPU (`sparse_bls_gpu`) and CPU (`sparse_bls_cpu`) implementations available**
- Optimized for datasets with < 500 observations
- Avoids binning and grid searching - directly tests all observation pairs as transit boundaries
- New `eebls_transit` wrapper automatically selects between sparse and standard BLS
  - **Default: GPU sparse BLS** for small datasets (use_gpu=True)
  - CPU fallback available (use_gpu=False)
- Particularly useful for ground-based surveys with limited phase coverage

**Citation for Sparse BLS**: If you use this method, please cite:
- Panahi, A., & Zucker, S. (2021). *Sparse BLS: A sparse-modeling approach to the Box-fitting Least Squares periodogram.* [arXiv:2103.06193](https://arxiv.org/abs/2103.06193)

**Refactored codebase organization**:
- Cleaner module structure: `base/`, `memory/`, and `periodograms/`
- Better maintainability and extensibility

### Improvements
- Modern Python packaging with `pyproject.toml`
- Docker support for easier installation with CUDA 11.8
- GitHub Actions CI/CD for automated testing across Python 3.7-3.12
- Cleaner, more maintainable codebase (89 lines of compatibility code removed)
- Updated documentation and contributing guidelines

### Additional Documentation
- [Benchmarking Guide](docs/BENCHMARKING.md) - Performance testing methodology
- [RunPod Development](docs/RUNPOD_DEVELOPMENT.md) - Cloud GPU development setup
- [Code Quality Fixes](docs/CODE_QUALITY_FIXES.md) - Thread-safety and memory management

For a complete list of changes, see [CHANGELOG.rst](CHANGELOG.rst).

## Features

Currently includes implementations of:

- **Generalized [Lomb-Scargle](https://arxiv.org/abs/0901.2573) periodogram** - Fast period finding for unevenly sampled data
- **Box Least Squares ([BLS](http://adsabs.harvard.edu/abs/2002A%26A...391..369K))** - Transit detection algorithm
  - **Adaptive GPU version** with 5-90x speedup (`eebls_gpu_fast_adaptive()`)
  - Standard GPU-accelerated version (`eebls_gpu_fast()`)
  - Sparse BLS ([Panahi & Zucker 2021](https://arxiv.org/abs/2103.06193)) for small datasets (< 500 observations)
    - GPU implementation: `sparse_bls_gpu()` (default)
    - CPU implementation: `sparse_bls_cpu()` (fallback)
- **Transit Least Squares ([TLS](https://ui.adsabs.harvard.edu/abs/2019A%26A...623A..39H/abstract))** - GPU-accelerated transit detection with optimal depth fitting
  - **35-202× faster** than CPU TLS (transitleastsquares package)
  - Keplerian-aware duration constraints (`tls_transit()`) - searches physically plausible transit durations
  - Standard mode (`tls_search_gpu()`) for custom period/duration grids
  - Optimal period grid sampling (Ofir 2014)
  - Designed for datasets with 500-5000 observations
- **Non-equispaced fast Fourier transform (NFFT)** - Adjoint operation ([paper](http://epubs.siam.org/doi/abs/10.1137/0914081))
- **NUFFT-based Likelihood Ratio Test (LRT)** - Transit detection with correlated noise (contributed by Jamila Taaki)
  - Matched filter in frequency domain with adaptive noise estimation
  - Particularly effective for gappy data with red/correlated noise
  - See [docs/NUFFT_LRT_README.md](docs/NUFFT_LRT_README.md) for details
- **Conditional Entropy period finder ([CE](http://adsabs.harvard.edu/abs/2013MNRAS.434.2629G))** - Non-parametric period finding
- **Phase Dispersion Minimization ([PDM2](http://www.stellingwerf.com/rfs-bin/index.cgi?action=PageView&id=29))** - Statistical period finding method
  - Currently operational but minimal unit testing or documentation

### Planned Features

Future developments may include:

- (Weighted) wavelet transforms
- Spectrograms (for PDM and GLS)
- Multiharmonic extensions for GLS
- Improved conditional entropy implementation (e.g., Katz et al. 2021)

## Installation

### Prerequisites

- CUDA-capable GPU (NVIDIA)
- CUDA Toolkit (11.x or 12.x recommended)
- Python 3.7 or later

### Dependencies

**Essential:**
- [PyCUDA](https://mathema.tician.de/software/pycuda/) - Python interface to CUDA
- [scikit-cuda](https://scikit-cuda.readthedocs.io/en/latest/) - Used for access to the CUDA FFT runtime library

**Optional (for additional features and testing):**
- [matplotlib](https://matplotlib.org/) - For plotting utilities
- [nfft](https://github.com/jakevdp/nfft) - For unit testing
- [astropy](http://www.astropy.org/) - For unit testing

### Install from PyPI

```bash
pip install cuvarbase
```

### Install from source

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

# Standard BLS
power = bls.eebls_gpu(t, y, dy, freqs)
best_freq = freqs[np.argmax(power)]
print(f"Best period: {1/best_freq:.2f} (expected: 2.5)")

# Or use adaptive BLS for automatic optimization (5-90x faster!)
power_adaptive = bls.eebls_gpu_fast_adaptive(t, y, dy, freqs)
```

### Transit Least Squares (TLS) - Advanced Transit Detection

```python
from cuvarbase import tls

# Generate transit data
t = np.sort(np.random.uniform(0, 50, 500)).astype(np.float32)
y = np.ones(len(t), dtype=np.float32)
dy = np.ones(len(t), dtype=np.float32) * 0.001

# Add 1% transit at 10-day period
phase = (t % 10.0) / 10.0
in_transit = (phase < 0.01) | (phase > 0.99)
y[in_transit] -= 0.01
y += np.random.normal(0, 0.001, len(t)).astype(np.float32)

# TLS with Keplerian duration constraints (35-202x faster than CPU TLS!)
results = tls.tls_transit(
    t, y, dy,
    R_star=1.0,      # Solar radii
    M_star=1.0,      # Solar masses
    period_min=5.0,
    period_max=20.0
)

print(f"Best period: {results['period']:.2f} days")
print(f"Transit depth: {results['depth']:.4f}")
print(f"SDE: {results['SDE']:.1f}")
```

For more advanced usage including Lomb-Scargle and Conditional Entropy, see the [full documentation](https://johnh2o2.github.io/cuvarbase/) and [examples/](examples/).

## Using Multiple GPUs

If you have more than one GPU, you can choose which one to use in a given script by setting the `CUDA_DEVICE` environment variable:

```bash
CUDA_DEVICE=1 python script.py
```

If anyone is interested in implementing a multi-device load-balancing solution, they are encouraged to do so! At some point this may become important, but for the time being manually splitting up the jobs to different GPUs will have to suffice.

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

Note: Tests require a CUDA-capable GPU and may take several minutes to complete.

## License

See [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgments

This project has benefited from contributions and support from many people in the astronomy community. Special thanks to:

- Joel Hartmann (author of the original `varbase`)
- Gaspar Bakos
- Kevin Burdge
- Attila Bodi
- **Jamila Taaki** - for contributing the NUFFT-based Likelihood Ratio Test (LRT) implementation for transit detection with correlated noise. Her work on adaptive matched filtering in the frequency domain has significantly expanded cuvarbase's capabilities for handling realistic astrophysical noise. See [docs/NUFFT_LRT_README.md](docs/NUFFT_LRT_README.md) and her papers:
  - Taaki, J. S., Kamalabadi, F., & Kemball, A. (2020). *Bayesian Methods for Joint Exoplanet Transit Detection and Systematic Noise Characterization.*
  - Reference implementation: https://github.com/star-skelly/code_nova_exoghosts
- All users and contributors who have helped make cuvarbase useful to the astronomy community

## Contact

For questions, issues, or contributions, please use the GitHub issue tracker:
https://github.com/johnh2o2/cuvarbase/issues
