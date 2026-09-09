# Installation

## Requirements

- Python >= 3.8
- NumPy
- One or both backends:
    - **GPU backend**: CUDA toolkit with `nvcc` on `$PATH` (or `$CUDA_HOME` set), plus Cython
    - **CPU backend**: Rust toolchain and [maturin](https://github.com/PyO3/maturin)

## GPU Backend (CUDA)

Install Cython and NumPy first, then install periodfind:

```bash
pip install cython numpy
pip install -e .
```

This compiles the CUDA kernels for Conditional Entropy, Analysis of Variance,
Lomb-Scargle, and FPW. Requires `nvcc` to be available.

### Supported Compute Capabilities

The build includes GPU code for compute capabilities:
5.0, 5.2, 6.0, 6.1, 7.0, 7.5, 8.0, 8.6, 8.9, 9.0.

## CPU Backend (Rust)

Install maturin, then build the Rust extension:

```bash
pip install maturin
cd rust && maturin develop --release
```

This builds the `periodfind_cpu` native module using
[Rayon](https://github.com/rayon-rs/rayon) for multithreaded parallelism.
No GPU required.

## Both Backends

Install both to enable automatic device detection:

```bash
# GPU
pip install cython numpy
pip install -e .

# CPU
pip install maturin
cd rust && maturin develop --release
```

## Verifying Installation

```python
import periodfind
print(periodfind.get_device())  # 'gpu' if CUDA available, otherwise 'cpu'
```

## C++ API

For standalone C++ usage (without Python), build with CMake:

```bash
mkdir build && cd build
cmake ..
make
sudo make install
```

This installs the library to `/usr/local/lib/` and headers to
`/usr/local/include/periodfind/`. Requires CMake >= 3.8.
