# Developer checks

Run these commands from the repository root. Install the package and test dependencies in the environment being checked.

- `python -m pytest`: CPU tests run without CUDA; device tests require a CUDA GPU.
- `python tools/check_release_gate.py`: additional numerical release checks, run on a CUDA device after the full suite.
- `python tools/ci_wheel_smoke.py`: installed-package smoke check used by CI. Run it in a fresh environment containing the built wheel or sdist; it removes the working directory from the import path.

GPU checks can run on any suitable local or rented CUDA device. Resource provisioning and personal SSH configuration are outside these tools. The [release validation record](../docs/validation/README.md) contains the measured checks for the frozen v1 source. Reproducible performance experiments live under [benchmarks/](../benchmarks/README.md).
