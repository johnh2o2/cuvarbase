# TESS Catalog BLS Cost Analysis

**Status: NEEDS REAL BENCHMARKS**

This document previously contained cost projections based on extrapolated and fabricated benchmark numbers. Those have been removed pending real GPU measurements.

## Key Algorithmic Finding (still valid)

**Sparse BLS is the wrong algorithm for TESS-scale data.** The O(N^2) complexity of sparse BLS (Panahi & Zucker 2021) makes it impractical for lightcurves with ~20,000 observations. For TESS transit searches, use:

- **Standard binned BLS** (cuvarbase `eebls_gpu_fast` or astropy `BoxLeastSquares`) — O(N) per frequency
- **Sparse BLS** is designed for small datasets (< 500 observations), e.g., ground-based surveys

## Generating Real Cost Estimates

```bash
# On a RunPod GPU (e.g. H100):
python scripts/benchmark_algorithms.py --algorithms bls_standard bls_sparse \
    --ndata 20000 --baseline 730 --gpu-model H100_SXM

# Visualize
python scripts/visualize_benchmarks.py benchmark_results.json
```

See [docs/BENCHMARKING.md](../docs/BENCHMARKING.md) for full instructions.
