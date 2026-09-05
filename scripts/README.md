# cuvarbase scripts: benchmarks and the RunPod GPU workflow

Everything in this directory needs a CUDA GPU except the result mergers
and plotters. There is no GPU in CI; the maintained way to run any of it
is the RunPod workflow in the second half of this page.

## Benchmark entry points

### `benchmark_algorithms.py` — cross-algorithm / cross-GPU comparison

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

`benchmark_all_gpus.sh` wraps this for RunPod sweeps (creates one pod per
GPU type, runs, terminates); `combine_gpu_benchmarks.py` merges the
per-GPU JSONs into comparison tables and `visualize_benchmarks.py` plots
them.

### `benchmark_new_features.py` — v1.0 feature benchmarks + GPU correctness checks

Covers batch BLS, the Keplerian frequency grid, the cuFINUFFT LS backend,
and survey-scale LS vs nifty-ls, with correctness cross-checks
(`--tests-only` runs just the checks; `--bench-only` just the timings).

```bash
python3 scripts/benchmark_new_features.py --output benchmarks/results/benchmark_results_new_features.json
```

### Campaign harnesses (each is the named producer of a tracked result)

| script | result it produced |
|---|---|
| `benchmark_pdm.py` | `benchmarks/results/pdm_a5000.json` |
| `benchmark_block_size.py` | `benchmarks/results/block_size_a5000.json` |
| `benchmark_adaptive_bls.py` | `benchmarks/results/bls_adaptive_keplerian_benchmark_rtxa5000_jun2026.json` |
| `benchmark_tls_survey.py`, `tls_fidelity_experiment.py`, `tls_matched_timing.py` | `benchmarks/results/tls_survey_jul2026/` |
| `gtls_benchmark/` (see its README) | `benchmarks/results/gtls_comparison_jul2026/`, [docs/GTLS_COMPARISON.md](../docs/GTLS_COMPARISON.md) |
| `bench_v026_head_to_head.py`, `decomp_v026_head_to_head.py`, `summarize_v026_head_to_head.py` | `benchmarks/results/v026_head_to_head_jul2026/` |
| `../benchmarks/bench_bls_survey.py`, `profile_bls_survey.py`, `sweep_bls_attrib.py`, `compare_parity.py` | `benchmarks/results/bls_survey_speed_jul2026/` |

`nufft_lrt_validation.py` / `summarize_lrt_validation.py` belong to the
experimental NUFFT-LRT detector and have no committed run yet.

### Release tooling

- `check_release_gate.py` — the GPU release-gate checks that go beyond the
  pytest suite. It imports `cuvarbase`, so run it from a pod where the
  package is `pip install -e .`-installed, or from the repo root with
  `PYTHONPATH=. python scripts/check_release_gate.py`.
- `ci_wheel_smoke.py` — packaging smoke test run by CI against the built
  wheel (no GPU).

## Results

Published results live in `benchmarks/results/` (single-GPU feature
benchmarks, the campaign folders above, and the 7-GPU sweep in `by_gpu/`)
and are summarized with methodology notes in
[docs/BENCHMARK_RESULTS.md](../docs/BENCHMARK_RESULTS.md). The GTLS
comparison behind the README's TLS claim is
[docs/GTLS_COMPARISON.md](../docs/GTLS_COMPARISON.md).

## RunPod GPU workflow

cuvarbase needs a CUDA GPU, so the development loop is: edit locally,
sync to a RunPod pod, run tests/benchmarks there, stream the output back.
Every script below reads `.runpod.env` in the repo root (gitignored) and
must be run from the repo root.

### Configuration: `.runpod.env`

```bash
cp .runpod.env.template .runpod.env
```

| key | meaning |
|---|---|
| `RUNPOD_SSH_HOST`, `RUNPOD_SSH_PORT`, `RUNPOD_SSH_USER` | direct-SSH endpoint of the pod (`root@<ip>:<port>`); written by `runpod-create.sh`, or copied from the pod's "Connect" button |
| `RUNPOD_SSH_KEY` | optional path to the private key passed as `ssh -i` (`runpod-create.sh` authorizes `~/.ssh/id_ed25519.pub` on the pod) |
| `RUNPOD_REMOTE_DIR` | where the source tree is synced on the pod (default `/workspace/cuvarbase`; `/workspace` is the pod's persistent volume) |
| `RUNPOD_API_KEY` | RunPod GraphQL key from https://www.runpod.io/console/user/settings; needed by `runpod-create.sh`, `runpod-stop.sh`, `gpu-test.sh`, `benchmark_all_gpus.sh` |
| `RUNPOD_POD_ID` | id of the pod created by `runpod-create.sh` (auto-populated); the only pod `runpod-stop.sh` will ever touch |

### Lifecycle scripts

| script | what it does |
|---|---|
| `runpod-create.sh [GPU type ...]` | creates an on-demand pod (image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 20 GB volume at `/workspace`), trying each GPU type in order until one deploys (default `"NVIDIA RTX A4000"`; e.g. `./scripts/runpod-create.sh "NVIDIA RTX A5000" "NVIDIA A40"`), waits for it, starts `sshd` through the RunPod proxy, authorizes your key, then rewrites `RUNPOD_SSH_HOST/PORT/USER` and `RUNPOD_POD_ID` in `.runpod.env` |
| `setup-remote.sh` | syncs the tree, `pip install --break-system-packages -e .[test]` on the pod, prints the GPU and verifies `import cuvarbase` + `pycuda` |
| `sync-to-runpod.sh` | `rsync` of the working tree to `RUNPOD_REMOTE_DIR` (excludes `.git`, build products, `.runpod.env`, images other than the docs logo) |
| `run-remote.sh "<command>"` | sync, then run an arbitrary shell command in `RUNPOD_REMOTE_DIR` with the CUDA toolkit auto-detected (`ls -d /usr/local/cuda-*`, newest wins) and exported |
| `test-remote.sh [path] [pytest args]` | sync, then `pytest <path> <args> -v` on the pod (default path `cuvarbase/tests/`) |
| `gpu-test.sh [--keep] [pytest args]` | one shot: reuse a RUNNING pod or create one, set up, run tests, stop the pod unless `--keep` |
| `runpod-stop.sh [--terminate]` | stops the pod in `RUNPOD_POD_ID` (resumable, keeps the volume); `--terminate` deletes it and its volume |

Typical session:

```bash
./scripts/runpod-create.sh "NVIDIA RTX A5000"
source .runpod.env && ssh -i ~/.ssh/id_ed25519 -p $RUNPOD_SSH_PORT root@$RUNPOD_SSH_HOST \
    "apt-get update -qq && apt-get install -y -qq rsync"          # see gotchas
./scripts/setup-remote.sh
./scripts/test-remote.sh cuvarbase/tests/test_bls.py -k fast -v
./scripts/run-remote.sh "PYTHONPATH=. python scripts/check_release_gate.py"
./scripts/run-remote.sh "python scripts/benchmark_new_features.py --tests-only"
./scripts/runpod-stop.sh --terminate
```

Direct SSH, when you need a shell on the pod:

```bash
source .runpod.env
ssh -i ${RUNPOD_SSH_KEY:-~/.ssh/id_ed25519} -p ${RUNPOD_SSH_PORT} ${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}
```

### Known gotchas

- **The pod image has no `rsync`.** `sync-to-runpod.sh` (and therefore
  `setup-remote.sh`, `test-remote.sh`, `run-remote.sh`) fails until you
  install it over direct SSH (every sync-based script is unusable until
  then): `apt-get update -qq && apt-get install -y -qq rsync` on the
  pod, before the first `setup-remote.sh`.
- **`nvcc` is not on `PATH` in a bare SSH session.** `run-remote.sh`
  exports `PATH=$CUDA_DIR/bin:$PATH`, `CUDA_HOME` and `LD_LIBRARY_PATH`
  for you (the CUDA version varies by pod: 12.4, 12.8, ...). In an
  interactive shell do the same by hand:
  `export CUDA_DIR=$(ls -d /usr/local/cuda-* | sort -V | tail -1); export PATH=$CUDA_DIR/bin:$PATH CUDA_HOME=$CUDA_DIR LD_LIBRARY_PATH=$CUDA_DIR/lib64:$LD_LIBRARY_PATH`,
  otherwise pycuda's compile step reports `nvcc not found`.
- **`scripts/check_release_gate.py` needs `PYTHONPATH=.`** when cuvarbase
  is not pip-installed in the pod's interpreter (`python scripts/...`
  puts `scripts/` on `sys.path`, not the repo root).
- **Editable install vs. `/workspace`.** Running python from `/workspace`
  rather than the source dir imports cuvarbase through the PEP-660
  editable finder; the package's `__file__`-relative kernel lookup
  handles that, but keep `RUNPOD_REMOTE_DIR` as the cwd for scripts.
- **`cuInit failed: initialization error`** with `nvidia-smi` healthy
  is a container GPU-passthrough fault: restart the pod from the RunPod
  dashboard, or terminate and create a new one.
- `runpod-stop.sh` only ever acts on `RUNPOD_POD_ID`; if you share an
  account with other pods, do not edit that key by hand.
- `.runpod.env` holds the API key: it is gitignored and excluded from the
  sync. Never commit it.
