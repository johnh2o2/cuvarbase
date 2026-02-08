# cuvarbase Algorithm Benchmarks

**Status: NEEDS REAL BENCHMARKS**

Previous benchmark results in this directory used incorrect extrapolation
(linear instead of quadratic scaling for sparse BLS due to a bug in
`scripts/benchmark_algorithms.py`). The bug has been fixed.

To generate new results, run on a GPU:

```bash
python scripts/benchmark_algorithms.py --algorithms sparse_bls bls_gpu_fast
python scripts/visualize_benchmarks.py benchmark_results.json
```

See [docs/BENCHMARKING.md](../../docs/BENCHMARKING.md) for full instructions.
