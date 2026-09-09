# Transit evidence archive

This directory contains the BLS and initial TLS report, timing figure, analysis tables, all nine frozen input datasets, per-job JSON results and logs, selected configurations, upstream source snapshots and original verification receipts. These support inspection of the timing arithmetic, recovery decisions and configuration selection. The public analysis tools are in [benchmarks/transit](../../transit/README.md).

The files were originally published under `analysis/transit-recovery-20260908` at commit `f0dc981`. The reorganization preserves the input, result and analysis-record bytes. The report and timing figure have been updated for readability; removing recovery panels does not change the underlying recovery results.

The full returned periodograms are about 20 GB and are not in Git. Their filenames and SHA256 values remain in the per-job records. Full-array verification receipts attest to checks against the original complete archive; a checkout alone cannot repeat those checks. Summary-based analysis and figure generation need only the committed files.

The [original publication inventory and full harness](https://github.com/johnh2o2/cuvarbase/tree/f0dc981/analysis/transit-recovery-20260908) remain in Git history. Original manifests, source-hash records and verification receipts refer to that layout and its original document/figure bytes. Cloud coordination records, transport helpers, duplicate harness copies and intermediate publication receipts were retired from the working tree. For an exact historical file:

```bash
git show f0dc981:scripts/benchmark_transit_recovery/worker.py
```

To regenerate the current figure without a GPU, see the [tool instructions](../../transit/README.md). Recomputing full-array validation requires restoring the omitted periodograms at the paths expected by the selected historical harness. A new GPU run is a new measurement and must record its own source, environment and timing provenance.

The [rental ledger](rental-ledger.json) records the experiment cost and terminated resources. Viewing, checking summaries or plotting these records incurs no cloud expense.
