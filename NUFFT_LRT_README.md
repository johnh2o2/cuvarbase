# NUFFT-based Likelihood Ratio Test (LRT) for Transit Detection

## Overview

This implementation integrates a concept and reference prototype originally developed by
**Jamila Taaki** ([@xiaziyna](https://github.com/xiaziyna), [website](https://xiazina.github.io)),
It provides a **GPU-accelerated, non-uniform matched filter** (NUFFT-LRT) for transit/template detection under correlated noise.

The key advantage of this approach is that it naturally handles correlated (non-white) noise through adaptive power spectrum estimation, making it more robust than traditional Box Least Squares (BLS) methods when dealing with red noise.

## Algorithm

The matched filter statistic is computed as:

```
SNR = sum(Y_k * T_k* * w_k / P_s(k)) / sqrt(sum(|T_k|^2 * w_k / P_s(k)))
```

where:
- `Y_k` is the Non-Uniform FFT (NUFFT) of the lightcurve
- `T_k` is the NUFFT of the transit template
- `P_s(k)` is the power spectrum (adaptively estimated from data or provided)
- `w_k` are frequency weights for one-sided spectrum conversion
- The sum is over all frequency bins

For gappy (non-uniformly sampled) data, NUFFT is used instead of standard FFT.

## Key Features

1. **Handles Gappy Data**: Uses NUFFT for non-uniformly sampled time series
2. **Correlated Noise**: Adapts to noise properties via power spectrum estimation
3. **GPU Accelerated**: Leverages CUDA for fast computation
4. **Normalized Statistic**: Amplitude-independent, only searches period/duration/epoch
5. **Flexible**: Can provide custom power spectrum or estimate from data

## Usage

```python
import numpy as np
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess

# Lightcurve data
t = np.array([...], dtype=float)   # observation times
y = np.array([...], dtype=float)   # flux measurements

# Initialize
proc = NUFFTLRTAsyncProcess()

# 1) Period+duration search (no epoch axis)
periods = np.linspace(1.0, 10.0, 100)
durations = np.linspace(0.1, 1.0, 20)
snr_pd = proc.run(t, y, periods, durations=durations)
# snr_pd.shape == (len(periods), len(durations))
best_idx = np.unravel_index(np.argmax(snr_pd), snr_pd.shape)
best_period = periods[best_idx[0]]
best_duration = durations[best_idx[1]]

# 2) Epoch search (adds an epoch axis)
# For a single candidate period, search epochs in [0, P]
P = 3.0
dur = 0.2
epochs = np.linspace(0.0, P, 50)
snr_pde = proc.run(t, y, np.array([P]), durations=np.array([dur]), epochs=epochs)
# snr_pde.shape == (1, 1, len(epochs))
best_epoch = epochs[np.argmax(snr_pde[0, 0, :])]
```

## Comparison with BLS

| Feature | NUFFT LRT | BLS |
|---------|-----------|-----|
| Noise Model | Correlated (adaptive PSD) | White noise assumption |
| Data Sampling | Handles gaps naturally | Works with gaps |
| Computation | O(N log N) per trial | O(N) per trial |
| Best For | Red noise, stellar activity | White noise, many transits |

## Parameters

### NUFFTLRTAsyncProcess

- `sigma` (float, default=2.0): Oversampling factor for NFFT
- `m` (int, optional): NFFT truncation parameter (auto-estimated if None)
- `use_double` (bool, default=False): Use double precision
- `use_fast_math` (bool, default=True): Enable CUDA fast math
- `block_size` (int, default=256): CUDA block size
- `autoset_m` (bool, default=True): Auto-estimate m parameter

### run() method

- `t` (array): Observation times
- `y` (array): Flux measurements
- `periods` (array): Trial periods to search
- `durations` (array, optional): Trial transit durations
- `epochs` (array, optional): Trial epochs. If provided, an extra axis of
  length `len(epochs)` is appended to the output. For multi-period searches,
  supply a common epoch grid (or run separate calls per period).
- `depth` (float, default=1.0): Template depth (normalized out in statistic)
- `nf` (int, optional): Number of frequency samples (default: `2*len(t)`).
- Returns
  - If `epochs` is None: array of shape `(len(periods), len(durations))`.
  - If `epochs` is given: array of shape `(len(periods), len(durations), len(epochs))`.
- `estimate_psd` (bool, default=True): Estimate power spectrum from data
- `psd` (array, optional): Custom power spectrum
- `smooth_window` (int, default=5): Smoothing window for PSD estimation
- `eps_floor` (float, default=1e-12): Floor for PSD to avoid division by zero

## Reference Implementation

This implementation is based on the prototype at:
https://github.com/star-skelly/code_nova_exoghosts/blob/main/nufft_detector.py

## Citation

If you use this implementation, please cite:

1. **cuvarbase** – Hoffman *et al.* (see cuvarbase main README for canonical citation).
2. **Taaki, J. S., Kamalabadi, F., & Kemball, A. (2020)** – *Bayesian Methods for Joint Exoplanet Transit Detection and Systematic Noise Characterization.*
3. **Reference prototype** — Taaki (@xiaziyna / @hexajonal), `star-skelly`, `tab-h`, `TsigeA`: https://github.com/star-skelly/code_nova_exoghosts
4. **Kay, S. M. (2002)** – *Adaptive Detection for Unknown Noise Power Spectral Densities.* S. Kay IEEE Trans. Signal Processing.


## Notes

- The method requires sufficient frequency resolution to resolve the transit signal
- Power spectrum estimation quality improves with more data points
- For very gappy data (< 50% coverage), consider increasing `nf` parameter
- The normalized statistic is independent of transit amplitude, so depth parameter doesn't affect ranking

## Example

See `examples/nufft_lrt_example.py` for a complete working example.
