# periodfind CPU runtime for the OSG / SkyPortal analysis service.
#
# A generic periodfind CPU runtime: the plugin ships periodfind_wrapper.py +
# periodfind_bridge.py per-job, so no plugin code is baked in. GPU is a separate
# image (Dockerfile.gpu — needs nvcc + the CUDA Cython extensions); this one is
# Rust-CPU only, which the device API falls back to when no GPU is present.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain for the periodfind CPU (Rust/pyo3) backend.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# periodfind build tooling + the bridge's runtime deps.
RUN pip install --no-cache-dir numpy cython maturin astropy matplotlib requests

# Build+install the CPU backend (periodfind_cpu) and keep the pure-python
# periodfind package importable from source. The top-level `pip install .` is
# deliberately skipped: it compiles the CUDA extensions (needs nvcc), which the
# CPU image doesn't have — `periodfind.set_device('cpu')` uses periodfind_cpu.
COPY . /opt/periodfind
RUN pip install --no-cache-dir /opt/periodfind/rust
ENV PYTHONPATH=/opt/periodfind

# Fail the build if the CPU period-finder can't import and run.
RUN python -c "import numpy as np, periodfind; periodfind.set_device('cpu'); \
ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10); \
t = [np.linspace(0, 10, 200).astype('float32')]; \
m = [np.sin(2*np.pi*t[0]/1.3).astype('float32')]; \
p = np.linspace(0.1, 2.0, 500).astype('float32'); \
peaks = ce.calc(t, m, p, np.array([0.0], 'float32'), output='peaks', n_peaks=3); \
print('periodfind CPU OK:', len(peaks[0]), 'peaks')"

ENTRYPOINT ["python"]
