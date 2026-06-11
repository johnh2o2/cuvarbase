# Benchmarking cuvarbase

Two benchmark entry points (both require a CUDA GPU):

## `benchmark_algorithms.py` — cross-algorithm / cross-GPU comparison

Benchmarks each algorithm against its CPU baseline (astropy where
available) at a fixed problem size.

```bash
python3 scripts/benchmark_algorithms.py \
    --algorithms bls_standard ls \
    --ndata 10000 --nbatch 100 --nfreq 10000 \
    --gpu-model H100_SXM \
    --output benchmark_results.json \
    --max-cpu-time 120
```

Registered algorithm keys: see the `ALGORITHMS` dict in the script
(`bls_standard`, `bls_sparse`, `ls`, ...). `--gpu-model` only labels the
output JSON (pricing lookup); detect your GPU with `nvidia-smi`.

`benchmark_all_gpus.sh` wraps this for RunPod sweeps;
`combine_gpu_benchmarks.py` merges per-GPU JSONs into comparison tables.

## `benchmark_new_features.py` — v1.0 feature benchmarks + GPU correctness checks

Covers batch BLS, the Keplerian frequency grid, the cuFINUFFT LS backend,
and survey-scale LS vs nifty-ls, with correctness cross-checks
(`--tests-only` runs just the checks; `--bench-only` just the timings).

```bash
python3 scripts/benchmark_new_features.py --output benchmarks/results/benchmark_results_new_features.json
```

## Results

Published results live in `benchmarks/results/` (single-GPU feature
benchmarks and the 7-GPU sweep in `by_gpu/`) and are summarized with
methodology notes in [docs/BENCHMARK_RESULTS.md](../docs/BENCHMARK_RESULTS.md).
General methodology guidance: [docs/BENCHMARKING.md](../docs/BENCHMARKING.md).

Remote execution helpers for RunPod (pod lifecycle, sync, remote runs)
are documented in [docs/RUNPOD_DEVELOPMENT.md](../docs/RUNPOD_DEVELOPMENT.md).
