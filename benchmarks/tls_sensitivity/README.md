# Independent TLS sensitivity study

These tools compare three frozen cuvarbase TLS configurations with public GTLS on identical synthetic lightcurves and trial periods. Observing times, relative uncertainties and exposures come from the earlier TESS and ZTF cadence examples. The [numerical explanation](../../docs/TLS_NUMERICS.md) describes phase binning and its limitations.

The experiment separates null calibration, independent recovery evaluation and exclusive GPU timing. Distributed recovery-run durations are diagnostic and never supply the published speed ratios. No tool here creates cloud resources or needs cloud credentials.

| Tool | Purpose |
|---|---|
| `generate.py` | Generate a reproducible 128-case cohort shard from a frozen cadence archive |
| `run.py` | Search the shard through the shared [public-API adapter](../transit/worker.py), retaining all scalar outcomes and output hashes |
| `analyze.py` | Freeze null thresholds, then evaluate the nine predeclared recovery/false-positive comparisons |
| `timing.py`, `analyze_timings.py` | Measure and verify five repetitions of isolated single-source and 16-source searches |
| `bls_control.py` | Analyze the secondary box-search control on the same period-restricted inputs |
| `archive.py` | Verify input/sample-spectrum bytes and export lossless compact scalar evidence |
| `binning.py`, `plot_binning.py` | Isolate the SNR cost of phase compression at known ephemerides and illustrate the template |
| `hatpi_cost.py` | Price a fully synthetic HATPI-like season and five-minute time averages; this is a cost pilot, not HATPI sensitivity evidence |

Run commands from the repository root. Analysis of compact evidence needs NumPy and SciPy; plotting also needs Matplotlib. Cohort generation and the binning diagnostic need batman-package and Numba. GPU workers additionally need the measured CUDA/software environment and pinned numerical packages. The library's general Python support range is separate from the experiment's Python 3.11 environment.

Given a result directory containing `evidence/`, `cadences/`, `configs/`, and the published analysis files, recompute into a scratch directory:

```bash
python benchmarks/tls_sensitivity/analyze.py \
  --root RESULT/evidence --configs RESULT/configs \
  --thresholds /tmp/tls-thresholds.json --calibrate
python benchmarks/tls_sensitivity/analyze.py \
  --root RESULT/evidence --configs RESULT/configs \
  --thresholds /tmp/tls-thresholds.json --out /tmp/tls-recovery.json
```

Replace `RESULT` with the experiment directory. The analyzer refuses to overwrite a differing frozen threshold or analysis file. It validates case counts and IDs, paired input hashes, configuration identity and installed numerical-source identity. The same interface accepts full per-shard measurement directories instead of compact evidence.

Generate and search a new shard in an environment with the pinned packages:

```bash
python benchmarks/tls_sensitivity/generate.py \
  --cadence RESULT/cadences/tess_200s.npz --profile tess_200s \
  --split calibration --start 0 --count 128 --out /tmp/tess-input.npz
python benchmarks/tls_sensitivity/run.py \
  --input /tmp/tess-input.npz --config RESULT/configs/v1_original.json \
  --out /tmp/tess-search
```

Seeds include profile, split and global case index, so changing the shard size or machine assignment preserves the case definition. A fresh numerical environment can introduce floating-point differences; compare recorded per-array hashes before treating regenerated observations as byte-identical. `v1_resolved` is the intermediate configuration's file identifier, not a claim that its resolution is exact.

Recreate the explanatory figure from the pinned template implementation:

```bash
python benchmarks/tls_sensitivity/plot_binning.py \
  --template-source cuvarbase/tls_models.py --out /tmp/tls-phase-binning
```

Timing requires an otherwise idle GPU. `timing.py` uses the earlier experiment's `*_heldout.npz` files solely as fixed timing inputs, with eight injections and eight nulls. It isolates each configuration in a process, warms the workload, randomizes configuration order and records process intervals. Initial API calls are reported separately; caches may already be populated on disk. GTLS single-source latency uses one worker and is contextual when the sensitivity study uses a concurrent batch policy.

Verify timing arithmetic, source identity and the actual committed input arrays:

```bash
python benchmarks/tls_sensitivity/analyze_timings.py \
  --timing RESULT/timing \
  --inputs benchmarks/results/transit_2026-09-08/inputs \
  --cadences RESULT/cadences --configs RESULT/configs \
  --thresholds RESULT/thresholds.json \
  --recovery RESULT/recovery_analysis.json --out /tmp/tls-timing.json
```

`archive.py` preserves all scalar results, input/source/spectrum hashes and original summary hashes, while deduplicating common truth and source maps. It verifies all input arrays and the retained first-four-case spectra in each shard before export. The compact archive does not contain the full prepared observations or sampled periodograms: its receipt records the original verification, and its hashes allow later checks against the larger measurement archive. Re-running analysis from compact evidence verifies the summaries, not those omitted bytes.
