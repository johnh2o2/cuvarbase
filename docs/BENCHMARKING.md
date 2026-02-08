# cuvarbase Benchmarking Guide

Benchmark cuvarbase GPU algorithms against CPU baselines, measure cost-per-lightcurve on cloud GPUs, and compare across hardware.

## Quick Start

```bash
# Run all algorithms (requires GPU + pycuda)
python scripts/benchmark_algorithms.py

# Specific algorithms only
python scripts/benchmark_algorithms.py --algorithms bls_standard ls ce

# Custom parameters (TESS-like: 20k obs, 2yr baseline)
python scripts/benchmark_algorithms.py --ndata 20000 --baseline 730

# Tag with GPU model for cost calculation
python scripts/benchmark_algorithms.py --gpu-model H100_SXM

# Visualize results
python scripts/visualize_benchmarks.py benchmark_results.json
```

## What Gets Benchmarked

| Algorithm | cuvarbase GPU | CPU Baselines | Complexity |
|-----------|--------------|---------------|------------|
| Standard BLS (binned) | `eebls_gpu_fast_adaptive` | astropy `BoxLeastSquares` | O(N × Nfreq) |
| Sparse BLS | `sparse_bls_gpu` | `sparse_bls_cpu` | O(N² × Nfreq) |
| Lomb-Scargle | `LombScargleAsyncProcess` | astropy `LombScargle`, nifty-ls | O(N + Nf log Nf) |
| PDM | `PDMAsyncProcess` | `pdm2_cpu`, PyAstronomy | O(N × Nfreq) |
| Conditional Entropy | `ConditionalEntropyAsyncProcess` | numpy reference | O(N × Nfreq) |
| TLS | `tls_transit` | `transitleastsquares` | O(N × Np × Nd) |

For standard BLS, the benchmark also compares cuvarbase v1.0 (`eebls_gpu_fast_adaptive`) against the pre-optimization kernel (`eebls_gpu_fast`) to quantify the v1.0 improvements.

## Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ndata` | 10,000 | Observations per lightcurve |
| `--nbatch` | 100 | Lightcurves in batch |
| `--nfreq` | 10,000 | Frequency grid points |
| `--baseline` | 3652.5 | Observation baseline (days, = 10 years) |

## Timing Methodology

- **GPU**: CUDA event timing (`pycuda.driver.Event`) — measures actual GPU execution time, excluding Python overhead and host-device transfer setup
- **CPU**: `time.perf_counter()` — wall-clock time
- **Iterations**: 1 warmup + 3 timed runs; median reported
- **Batch**: Total time for all `nbatch` lightcurves; per-lightcurve time = total / nbatch

## Cost-per-Lightcurve

The benchmark computes cost using RunPod on-demand pricing:

```
cost_per_lc = (gpu_seconds_per_lc) × ($/hr) / 3600
```

### RunPod GPU Pricing (community cloud, on-demand)

| GPU | $/hr | VRAM | Architecture |
|-----|------|------|-------------|
| RTX 4000 Ada | $0.20 | 20 GB | Ada Lovelace |
| RTX 4090 | $0.34 | 24 GB | Ada Lovelace |
| V100 | $0.19 | 16 GB | Volta |
| L40 | $0.69 | 48 GB | Ada Lovelace |
| A100 PCIe | $0.79 | 80 GB | Ampere |
| A100 SXM | $1.19 | 80 GB | Ampere |
| H100 PCIe | $1.99 | 80 GB | Hopper |
| H100 SXM | $2.69 | 80 GB | Hopper |
| H200 SXM | $3.59 | 141 GB | Hopper |

*Prices as of 2025-Q4. Check [runpod.io/gpu-pricing](https://www.runpod.io/gpu-pricing) for current rates.*

### Interpreting Cost Results

The cost table shows projected cost-per-lightcurve for each GPU model. For the GPU actually used in the benchmark, the number is exact. For other GPUs, the time is held constant (same seconds/lc) and only the hourly rate changes — **actual performance varies by architecture**. To get accurate numbers for a specific GPU, run the benchmark on that hardware.

The most cost-efficient GPU is not necessarily the fastest — a cheap slow GPU can beat an expensive fast GPU on $/lc. The cost table helps identify the optimal price-performance point.

## Running on RunPod

```bash
# 1. Create a pod (see scripts/runpod-create.sh)
# 2. Sync code
bash scripts/sync-to-runpod.sh

# 3. SSH in and run
ssh runpod
cd /workspace/cuvarbase
pip install -e .
pip install astropy nifty-ls transitleastsquares PyAstronomy

# 4. Run benchmarks
python scripts/benchmark_algorithms.py --gpu-model H100_SXM

# 5. Visualize
python scripts/visualize_benchmarks.py benchmark_results.json \
    --output-prefix examples/benchmark_results/benchmark \
    --report examples/benchmark_results/report.md
```

See [RUNPOD_DEVELOPMENT.md](RUNPOD_DEVELOPMENT.md) for pod setup details.

## Output Format

### JSON (`benchmark_results.json`)

```json
{
  "system": {
    "gpu_name": "NVIDIA H100 80GB HBM3",
    "gpu_total_memory_mb": 81559,
    "platform": "Linux-...",
    ...
  },
  "results": [
    {
      "algorithm": "bls_standard",
      "display_name": "Standard BLS (binned)",
      "ndata": 10000,
      "nbatch": 100,
      "nfreq": 10000,
      "gpu": {
        "cuvarbase_v1": {"total_time": 1.23, "time_per_lc": 0.0123},
        "cuvarbase_preopt": {"total_time": 2.34, "time_per_lc": 0.0234}
      },
      "cpu": {
        "astropy": {"total_time": 45.6, "time_per_lc": 0.456}
      },
      "speedups": {"gpu_vs_astropy": 37.1, "v1_vs_preopt": 1.9},
      "cost": {"cuvarbase_v1": {"cost_per_lc": 0.0000092, ...}}
    }
  ],
  "runpod_pricing": {...}
}
```

### Plots

- `benchmark_speedups.png` — GPU speedup vs each CPU baseline
- `benchmark_time_per_lc.png` — Time per lightcurve across all implementations
- `benchmark_cost.png` — Cost per million lightcurves across GPU models

### Markdown Report

`benchmark_report.md` — Summary tables, per-algorithm details, and cost comparison.

## Adding a New Algorithm

1. Write a benchmark function in `scripts/benchmark_algorithms.py`:

```python
def bench_myalgo_gpu(ndata, nbatch, nfreq, baseline):
    batch = generate_batch(ndata, nbatch, baseline)
    freqs = make_freq_grid(nfreq)

    def run():
        for t, y, dy in batch:
            my_gpu_function(t, y, dy, freqs)

    med, times = time_function(run, n_iter=3, warmup=1, use_cuda=True)
    return med, {'variant': 'my_gpu_function', 'times': times}
```

2. Register it in the `ALGORITHMS` dict:

```python
ALGORITHMS['myalgo'] = {
    'display_name': 'My Algorithm',
    'complexity': 'O(N * Nfreq)',
    'gpu_func': bench_myalgo_gpu,
    'cpu_funcs': OrderedDict([('baseline', bench_myalgo_cpu)]),
    'gpu_old_func': None,
}
```

3. Add complexity to `ALGORITHM_COMPLEXITY` if you need extrapolation support.

## See Also

- [Main README](../README.md) — Installation and basic usage
- [RunPod Development Guide](RUNPOD_DEVELOPMENT.md) — Remote GPU testing
- [API Documentation](https://johnh2o2.github.io/cuvarbase/) — Algorithm details
