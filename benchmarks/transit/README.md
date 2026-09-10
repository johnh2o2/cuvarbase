# Transit benchmark tools

These tools analyze and reproduce parts of the [2026-09-08 transit experiment](../results/transit_2026-09-08/README.md). Its TLS arm uses the earlier **binned** engine, now retained as `method='binned'`. The [current report](../../docs/TRANSIT_BENCHMARKS.md) distinguishes those historical TLS results from the standard observation-level search and identifies the BLS measurements used for release claims. Run commands from the repository root. Plotting and analysis require Python, NumPy, SciPy and Matplotlib; backend searches additionally require the pinned scientific packages and a CUDA device for GPU methods.

Regenerate the current timing figure without a GPU:

```bash
python benchmarks/transit/plot_main.py \
  --root benchmarks/results/transit_2026-09-08 \
  --tls-reference benchmarks/results/tls_reference_2026-09-10 \
  --output-dir /tmp/cuvarbase-figure
```

| Tool | Purpose |
|---|---|
| `plot_main.py` | Six-panel timing figure: BLS and TLS across three cadences; accepts current or historical TLS evidence |
| `analyze.py`, `recovery_statistics.py` | Independent null calibration, injection recovery, false positives and paired confidence bounds |
| `analyze_timings.py`, `analyze_runtime_cohorts.py` | Timing medians, repetition ranges, cost projections and cohort checks |
| `analyze_components.py` | Component tables and ablations |
| `worker.py`, `cpu_batch.py`, `grid_and_search.py` | Backend searches and timing jobs; configuration is supplied explicitly |
| `components.py`, `components_tls.py` | Diagnostic BLS and TLS component measurements |
| `generate.py` | Construct seeded synthetic flux/noise on the retained observed cadences |

The committed [inputs](../results/transit_2026-09-08/inputs) and [selection record](../results/transit_2026-09-08/selection.json) define the measured experiment. Use each worker's `--help` for arguments; `worker.py --config` takes a JSON configuration from the selected method records. Install the selected backend in its own environment, including fBLS on the import path when selecting that backend. The original cloud controller and environment setup are retained in the pinned Git archive described below; no cloud resources are started by the analysis or plotting tools.

The command above combines this experiment's BLS measurements with the current observation-level TLS study. For the historical binned comparison, replace `--tls-reference` with `--tls-study benchmarks/results/tls_sensitivity_2026-09-09`; omit both options to recreate the initial September 8 figure. Those older TLS figures do not describe the new default engine.

Analysis scripts write into `--root`. Use a scratch copy to recompute tables. Without `--verify-arrays`, recovery and timing analysis checks committed per-job summaries and inputs; it does not re-verify the omitted periodograms, and records that distinction in its output. `analyze_components.py` and full-array recovery/timing validation require restoring the periodogram archive. Do not overwrite the published verification receipts with a summary-only rerun.

The [archive notes](../results/transit_2026-09-08/ARCHIVE.md) explain exactly which evidence is present, how to retrieve the frozen original harness, and which files require the larger local archive. The moved workers retain the numerical search implementation; module lookup paths have been made independent of the original pod. A fresh run measures its own hardware and environment and must record new provenance.
