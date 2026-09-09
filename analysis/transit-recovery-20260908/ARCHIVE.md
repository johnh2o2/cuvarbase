The Git checkout contains the benchmark figure, reports, analysis tables, all nine frozen input datasets, per-job JSON results and logs, selected configurations, source snapshots, environment records, and verification receipts. These are enough to inspect the timing arithmetic, recovery decisions, configuration selection, and reported provenance, and to regenerate the main figure from the committed analysis JSON.

The full returned periodograms and transport archives remain in the local experiment archive. They are excluded from Git because the periodograms alone occupy about 20 GB. Their filenames and SHA256 values are retained in the per-job JSON records and transfer manifests. A verification receipt records checks performed against that complete archive; it does not mean a fresh Git checkout contains every verified array.

To regenerate the figure from the verified analysis records, run from the repository root with NumPy and Matplotlib installed:

```bash
python scripts/benchmark_transit_recovery/plot_main.py --root analysis/transit-recovery-20260908
```

Recomputing full-array validation with `analyze.py --verify-arrays`, `analyze_timings.py --verify-arrays`, or `verify_provenance.py` requires restoring the complete local archive at its recorded paths. The inputs and frozen worker/controller needed for a new GPU run are included, together with setup scripts and pinned environments. A new run is a new measurement; its timing and numerical output may vary.

The [publication manifest](publication-manifest.json) inventories the files included with this benchmark update. The [rental ledger](rental-ledger.json) records the terminated resources and experiment cost. No additional cloud resources are needed to view or regenerate the published figure.
