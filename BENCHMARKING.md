# cuvarbase Benchmarking Guide

This guide explains how to run comprehensive performance benchmarks for cuvarbase algorithms and interpret the results.

## Quick Start

```bash
# Run benchmarks for sparse BLS (default)
python scripts/benchmark_algorithms.py

# Run benchmarks for multiple algorithms
python scripts/benchmark_algorithms.py --algorithms sparse_bls bls_gpu_fast

# Generate visualizations
python scripts/visualize_benchmarks.py benchmark_results.json

# View the report
cat benchmark_report.md
```

## Benchmark Configuration

The benchmark suite tests algorithms across a grid of problem sizes:

- **ndata (observations per lightcurve)**: 10, 100, 1000
- **nbatch (number of lightcurves)**: 1, 10, 100, 1000
- **nfreq (frequency grid points)**: 100 (default)

This creates 12 experiments per algorithm (3 × 4 grid).

### Data Generation

All lightcurves are generated with:
- **Baseline**: 5 years (1826.25 days)
- **Sampling**: Uniform random over baseline
- **Signal**: Simple sinusoid (100-day period) + Gaussian noise
- **SNR**: Moderate (amplitude = 2× noise level)

## Scaling Laws and Extrapolation

For experiments that would take too long on CPU (> 5 minutes by default), the benchmark extrapolates using algorithm-specific scaling laws:

### Algorithm Complexities

| Algorithm | Complexity | Scaling |
|-----------|-----------|---------|
| `sparse_bls` | O(N² × Nf) | Quadratic in ndata |
| `bls_gpu_fast` | O(N² × Nf) | Quadratic in ndata |
| `lombscargle` | O(N × Nf) | Linear in ndata |
| `pdm` | O(N × Nf) | Linear in ndata |

Where:
- N = ndata (observations per lightcurve)
- Nf = nfreq (frequency grid points)

### Extrapolation Method

For a target configuration `(ndata_target, nbatch_target, nfreq_target)`:

1. Find closest measured reference: `(ndata_ref, nbatch_ref, nfreq_ref)`
2. Compute scaling factors based on algorithm complexity
3. Estimate: `time_target = time_ref × (ndata_target/ndata_ref)^α × (nbatch_target/nbatch_ref) × (nfreq_target/nfreq_ref)`

Where α is the complexity exponent (1 for linear, 2 for quadratic).

Extrapolated values are marked with `*` in output.

## GPU Architecture Performance

Expected relative performance across GPU generations (normalized to RTX A5000 = 1.0x):

| GPU | Architecture | Year | Memory | Bandwidth | Expected Speedup |
|-----|-------------|------|--------|-----------|------------------|
| RTX A5000 | Ampere | 2021 | 24 GB | 768 GB/s | 1.0x (baseline) |
| L40 | Ada Lovelace | 2023 | 48 GB | 864 GB/s | 1.5-2.0x |
| A100 | Ampere | 2020 | 40/80 GB | 1.5-2.0 TB/s | 1.5-2.5x |
| H100 | Hopper | 2022 | 80 GB | ~3 TB/s | 3.0-4.0x |
| H200 | Hopper | 2024 | 141 GB | 4.8 TB/s | 3.5-4.5x |
| B200 | Blackwell | 2025 | 192 GB | ~8 TB/s | 5.0-7.0x |

### Why Memory Bandwidth Matters

cuvarbase algorithms are primarily **memory-bound**, not compute-bound:

1. **BLS algorithms** iterate over data arrays repeatedly
2. **Memory access patterns** dominate runtime (not FLOPs)
3. **Bandwidth improvements** translate directly to speedup
4. **Large VRAM** enables bigger batches without CPU transfers

### Architecture-Specific Notes

**Ampere (A5000, A100)**:
- Good baseline for FP32 workloads
- A100 has 2x bandwidth of A5000 → up to 2x faster

**Ada Lovelace (L40)**:
- Improved FP32 throughput
- Better power efficiency
- Good for production deployments

**Hopper (H100, H200)**:
- Massive bandwidth improvements (3-5 TB/s)
- 3-4x faster than A5000 for memory-bound code
- H200 adds 75% more VRAM (141 GB vs 80 GB)
- Best for large-scale surveys

**Blackwell (B200)**:
- Designed for AI workloads but benefits scientific computing
- ~8 TB/s bandwidth (10x A5000!)
- 192 GB VRAM enables massive batches
- Expected 5-7x speedup vs A5000 for our workloads
- Most gains from bandwidth, not new tensor features

## Advanced Usage

### Custom Timeouts

```bash
# Allow up to 10 minutes CPU time before extrapolation
python scripts/benchmark_algorithms.py --max-cpu-time 600

# Allow up to 2 minutes GPU time before extrapolation
python scripts/benchmark_algorithms.py --max-gpu-time 120
```

### Custom Output

```bash
# Save results to custom file
python scripts/benchmark_algorithms.py --output my_results.json

# Generate plots with custom prefix
python scripts/visualize_benchmarks.py my_results.json --output-prefix my_benchmark

# Custom report filename
python scripts/visualize_benchmarks.py my_results.json --report my_report.md
```

### Adding New Algorithms

To benchmark a new algorithm:

1. Add complexity to `ALGORITHM_COMPLEXITY` dict in `benchmark_algorithms.py`
2. Implement benchmark function following this signature:

```python
def benchmark_my_algorithm(ndata: int, nbatch: int, nfreq: int,
                          backend: str = 'gpu') -> float:
    """
    Run algorithm benchmark.

    Returns
    -------
    runtime : float
        Total runtime in seconds
    """
    # Generate data
    lightcurves = generate_batch(ndata, nbatch)
    freqs = np.linspace(0.005, 0.02, nfreq)

    # Run algorithm
    start = time.time()
    for t, y, dy in lightcurves:
        if backend == 'gpu':
            result = my_gpu_function(t, y, dy, freqs)
        else:
            result = my_cpu_function(t, y, dy, freqs)

    return time.time() - start
```

3. Add to main benchmarking loop:

```python
if 'my_algorithm' in args.algorithms:
    runner.benchmark_algorithm('my_algorithm', benchmark_my_algorithm,
                              ndata_values, nbatch_values, nfreq)
```

## Interpreting Results

### Performance Metrics

**Speedup**: Ratio of CPU time to GPU time
- < 1x: GPU slower (rare, usually small problems)
- 1-10x: Good for small/medium problems
- 10-100x: Excellent for medium/large problems
- 100x+: Outstanding for large-scale problems

**Scaling Behavior**:
- **Strong scaling**: Speedup vs problem size (fixed batch)
- **Weak scaling**: Performance vs batch size (fixed ndata)

### Expected Patterns

**Small problems (ndata < 100, nbatch < 10)**:
- GPU overhead dominates
- CPU may be faster
- Kernel launch latency matters

**Medium problems (ndata 100-1000, nbatch 10-100)**:
- GPU starts to excel
- 10-50x speedups common
- Sweet spot for most algorithms

**Large problems (ndata > 1000, nbatch > 100)**:
- Massive GPU advantages
- 100-1000x speedups possible
- Limited by GPU memory

## Troubleshooting

### Out of Memory Errors

Reduce batch size or ndata:
```bash
python scripts/benchmark_algorithms.py --algorithms sparse_bls
# If OOM, reduce manually by editing script
```

### Slow Benchmarks

Reduce timeout thresholds:
```bash
python scripts/benchmark_algorithms.py --max-cpu-time 60 --max-gpu-time 30
```

### Missing GPU Support

CPU-only benchmarks will still work:
```bash
# Will skip GPU benchmarks but run CPU
python scripts/benchmark_algorithms.py
```

## Citation

If you use these benchmarks in published work, please cite:

```bibtex
@software{cuvarbase,
  author = {Hoffman, John},
  title = {cuvarbase: GPU-accelerated time series analysis},
  url = {https://github.com/johnh2o2/cuvarbase},
  year = {2025}
}
```

## See Also

- [Main README](README.md) - Installation and basic usage
- [RunPod Development Guide](RUNPOD_DEVELOPMENT.md) - Remote GPU testing
- [API Documentation](https://johnh2o2.github.io/cuvarbase/) - Algorithm details
