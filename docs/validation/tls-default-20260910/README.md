# Standard TLS implementation checks — 10 September 2026

All **265 TLS tests passed on an NVIDIA A40**, with zero failures or skips,
against the frozen production source used for the independent
[GTLS comparison](../../../benchmarks/results/tls_reference_2026-09-10/README.md).
These are correctness tests, separate from recovery and speed measurements.

- [GPU receipt](receipt.json): exact command, source hashes before and after,
  dependency versions, device identity, timestamps and artifact hashes.
- [GPU test output](tests.log) and [JUnit results](tests.xml).
- [Distribution receipt](distribution.json): wheel/sdist hashes, TLS extra
  dependencies and byte-for-byte checks of the new installed modules and
  kernels against the GPU-tested source.
- [Installed-wheel tests](wheel-tests.log): 87 passing host-side TLS math,
  frontend and kernel-inventory tests, run outside the source checkout.
- [Current CPU suite](cpu-suite.json) and [output](cpu-suite.log): 1,215 passed,
  851 environment-dependent skips and one expected failure, including the
  TLS benchmark harness tests. Runtime and GPU TLS test sources are unchanged;
  one README consistency test now reflects the candidate installation.
- [Documentation build](docs-build.json): HTML builds successfully; the five
  expected GPU plot warnings on this CPU host are retained in the
  [build log](docs-build.log) and [warning log](docs-warnings.log).
- [BLS source continuity](bls-source-continuity.json): the measured BLS code
  and its local dependencies are unchanged, apart from one documentation link.
  The nine September 8 timing records therefore describe the same BLS
  implementation; their original workload and hardware qualifications remain.

The GPU environment used Python 3.11.10, CUDA 12.4, CuPy 13.6.0,
PyCUDA 2025.1.2, NumPy 2.2.6, SciPy 1.15.3 and batman-package 2.5.3.
The installed-wheel host tests used the separately recorded
[CPU environment](host-environment.json).
Wheel and sdist hashes identify local build artifacts, not a PyPI publication.

The earlier [full release suite](../README.md) is dated evidence for its
recorded source. Its test count is not combined with these results to claim
that the expanded current full suite ran on a GPU.
