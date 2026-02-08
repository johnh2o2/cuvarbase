# TESS Catalog: Standard BLS Cost Analysis

**Status: NEEDS REAL BENCHMARKS**

This document previously contained cost projections based on extrapolated benchmark numbers that were internally inconsistent. Those have been removed pending real GPU measurements.

## Expected Result (based on algorithmic analysis)

Standard (binned) BLS should show excellent GPU acceleration for TESS-scale data (ndata ~20,000) because:
- O(N) complexity per frequency — computation scales well
- Large ndata means kernel overhead is negligible relative to computation
- Batch processing of multiple lightcurves amortizes GPU setup cost

## TODO

To produce real cost estimates:
1. Run `scripts/benchmark_standard_bls.py` on RunPod GPUs (V100 through H200)
2. Measure actual GPU time per lightcurve at TESS-scale parameters
3. Multiply by on-demand cloud GPU pricing to get cost per lightcurve
4. See [docs/BENCHMARKING.md](../docs/BENCHMARKING.md) for methodology
