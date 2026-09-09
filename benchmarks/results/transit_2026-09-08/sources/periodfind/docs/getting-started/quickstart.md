# Quick Start

This guide shows how to find the period of a light curve using periodfind.

## Basic Usage

```python
import numpy as np
import periodfind

# Create an algorithm instance (auto-detects GPU/CPU)
ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10)

# Prepare light curve data (must be float32)
times = [np.array([1.0, 2.0, 3.0, ...], dtype=np.float32)]
mags  = [np.array([15.1, 14.8, 15.3, ...], dtype=np.float32)]

# Define trial periods
periods    = np.linspace(0.1, 10.0, 1000, dtype=np.float32)
period_dts = np.array([0.0], dtype=np.float32)

# Find the best period
results = ce.calc(times, mags, periods, period_dts)
best = results[0]
print(f"Best period: {best.params[0]:.4f}, significance: {best.significance:.2f}")
```

## Device Selection

periodfind uses a PyTorch-style device API:

```python
import periodfind

# Global default
periodfind.set_device('gpu')
ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10)  # uses GPU

# Per-call override
ce_cpu = periodfind.ConditionalEntropy(n_phase=10, n_mag=10, device='cpu')

# Check current device
print(periodfind.get_device())
```

If no device is set, periodfind auto-detects: it tries importing the CUDA
extensions and running `nvidia-smi`, falling back to CPU.

## Output Modes

Each period-finding algorithm's `.calc()` method supports three output modes.

### Statistics (default)

Returns the best period(s) with significance metrics:

```python
results = ce.calc(times, mags, periods, period_dts,
                  output='stats', n_stats=3)

# n_stats > 1 returns a list per light curve
for stat in results[0]:
    print(f"Period: {stat.params[0]:.4f}, value: {stat.value:.6f}")
```

### Periodogram

Returns the full periodogram for further analysis:

```python
results = ce.calc(times, mags, periods, period_dts,
                  output='periodogram')

pgram = results[0]
best = pgram.best_params(n=1)
print(f"Best period: {best.params[0]:.4f}")
print(f"Periodogram shape: {pgram.data.shape}")
```

### Peaks

Returns the top peaks without materialising the full periodogram, saving
memory on large grids:

```python
results = ce.calc(times, mags, periods, period_dts,
                  output='peaks', n_peaks=32, min_distance=1)

# results[i] is a list of Statistics for the i-th light curve
for peak in results[0]:
    print(f"Period: {peak.params[0]:.4f}, value: {peak.value:.6f}")
```

## Batched Processing

All algorithms accept lists of light curves for batch processing:

```python
# Process 100 light curves at once
times_list = [np.array([...], dtype=np.float32) for _ in range(100)]
mags_list  = [np.array([...], dtype=np.float32) for _ in range(100)]

results = ce.calc(times_list, mags_list, periods, period_dts)
# results[i] = Statistics for the i-th light curve
```

## Comparing Algorithms

```python
import periodfind

ce  = periodfind.ConditionalEntropy(n_phase=10, n_mag=10)
aov = periodfind.AOV(n_phase=15)
ls  = periodfind.LombScargle()
fpw = periodfind.FPW(n_bins=10)

for name, algo in [("CE", ce), ("AOV", aov), ("LS", ls), ("FPW", fpw)]:
    result = algo.calc(times, mags, periods, period_dts)
    print(f"{name}: best period = {result[0].params[0]:.4f}")
```

## Feature Extraction

### Fourier Decomposition

Given pre-determined periods, compute Fourier features (14 per curve):

```python
fd = periodfind.FourierDecomposition()

# periods: one period per curve (float32)
features = fd.calc(times, mags, errs, periods)
# features.shape == (n_curves, 14)
# [power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]
```

### dm-dt Histograms

```python
dmdt = periodfind.DmDt()

dt_edges = np.array([...], dtype=np.float32)
dm_edges = np.array([...], dtype=np.float32)
hists = dmdt.calc(times, mags, dt_edges, dm_edges)
# hists.shape == (n_curves, n_dm_bins, n_dt_bins)
```

### Basic Statistics

Compute 22 summary statistics per light curve:

```python
bs = periodfind.BasicStats()
stats = bs.calc(times, mags, errs)
# stats.shape == (n_curves, 22)
```

### High-Cadence Removal

```python
filtered = periodfind.remove_high_cadence(times, mags, errs, cadence_minutes=30.0)
# filtered[i] = (times_i, mags_i, errs_i) with high-cadence points removed
```
