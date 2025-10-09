# NUFFT-based Likelihood Ratio Test (LRT) for Transit Detection

## Overview

This module implements a GPU-accelerated matched filter approach for detecting periodic transit signals in gappy time-series data. The method is based on the likelihood ratio test described in:

> "Wavelet-based matched filter for detection of known up to parameters signals in unknown correlated Gaussian noise" (IEEE paper)

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

# Generate or load your lightcurve data
t = np.array([...])  # observation times
y = np.array([...])  # flux measurements

# Initialize processor
proc = NUFFTLRTAsyncProcess()

# Define search grid
periods = np.linspace(1.0, 10.0, 100)
durations = np.linspace(0.1, 1.0, 20)

# Run search
snr = proc.run(t, y, periods, durations=durations)

# Find best match
best_idx = np.unravel_index(np.argmax(snr), snr.shape)
best_period = periods[best_idx[0]]
best_duration = durations[best_idx[1]]
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
- `epochs` (array, optional): Trial epochs
- `depth` (float, default=1.0): Template depth (normalized out in statistic)
- `nf` (int, optional): Number of frequency samples (default: 2*len(t))
- `estimate_psd` (bool, default=True): Estimate power spectrum from data
- `psd` (array, optional): Custom power spectrum
- `smooth_window` (int, default=5): Smoothing window for PSD estimation
- `eps_floor` (float, default=1e-12): Floor for PSD to avoid division by zero

## Reference Implementation

This implementation is based on the prototype at:
https://github.com/star-skelly/code_nova_exoghosts/blob/main/nufft_detector.py

## Citation

If you use this implementation, please cite:
1. The original IEEE paper on the matched filter method
2. The cuvarbase package: Hoffman et al. (see main README)
3. The reference implementation repository (if applicable)

## Notes

- The method requires sufficient frequency resolution to resolve the transit signal
- Power spectrum estimation quality improves with more data points
- For very gappy data (< 50% coverage), consider increasing `nf` parameter
- The normalized statistic is independent of transit amplitude, so depth parameter doesn't affect ranking

## Example

See `examples/nufft_lrt_example.py` for a complete working example.
