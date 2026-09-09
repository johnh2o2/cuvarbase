# periodfind

**GPU-accelerated period finding for astronomical light curves.**

periodfind provides CUDA-accelerated and CPU (Rust) implementations of
periodicity detection and feature extraction algorithms, with a unified
PyTorch-style device API.

### Period-Finding

| Algorithm | Class |
|-----------|-------|
| Conditional Entropy | `periodfind.ConditionalEntropy` |
| Analysis of Variance | `periodfind.AOV` |
| Lomb-Scargle | `periodfind.LombScargle` |
| Fast Phase-folding Weighted | `periodfind.FPW` |

### Feature Extraction

| Algorithm | Class |
|-----------|-------|
| Fourier Decomposition | `periodfind.FourierDecomposition` |
| dm-dt Histograms | `periodfind.DmDt` |
| Basic Statistics | `periodfind.BasicStats` |

## Key Features

- **Dual backends** — CUDA GPU acceleration and multithreaded Rust CPU fallback
- **Device dispatch** — `set_device('cpu')` / `set_device('gpu')` with auto-detection
- **Batched processing** — analyze thousands of light curves in a single call
- **Chunked peak finding** — memory-efficient `output='peaks'` mode
- **NumPy integration** — all inputs and outputs use NumPy arrays (`float32`)

## Quick Example

```python
import numpy as np
import periodfind

ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10)

times = [np.array([...], dtype=np.float32)]
mags  = [np.array([...], dtype=np.float32)]
periods    = np.linspace(0.1, 10.0, 1000, dtype=np.float32)
period_dts = np.array([0.0], dtype=np.float32)

results = ce.calc(times, mags, periods, period_dts)
print(f"Best period: {results[0].params[0]:.4f}")
```

## License

BSD 3-Clause License. Copyright (c) 2020, California Institute of Technology.

Funding provided by the Larson Scholar Fellowship as part of the SURF program.
