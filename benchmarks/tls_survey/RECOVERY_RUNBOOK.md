# Rendering the sealed recovery report

Run this CPU-only formatting step after the original scientific analysis,
complete immutable-baseline qualification, and held-out SNR diagnostic finish:

```sh
/workspace/tls-survey/modern/bin/python \
  /workspace/tls-survey/candidate/benchmarks/tls_survey/report_recovery.py \
  --recovery /workspace/tls-survey/final-campaign/detection-results.json \
  --seal /workspace/tls-survey/evidence/seal-final.json \
  --exactness /workspace/tls-survey/evidence/exactness-final.json \
  --snr /workspace/tls-survey/evidence/heldout-snr-final.json \
  --output /workspace/tls-survey/final-campaign/report
```

The renderer imports only the Python standard library. It checks the original
seal and threshold identities, planned regime/method/FPR rows and denominators,
paired-contrast counts, subgroup membership, original execution receipt hashes,
and every planned baseline comparison. It reports scientific execution failures
and numerical mismatches; they are not removed from denominators or replaced
by diagnostic reruns. A complete exactness execution can validly produce a
report that withholds aggregate exactness. An incomplete or inconsistent input
fails before rendering.

Outputs are `RECOVERY.md`, `recovery_fpr.csv`, `paired_contrasts.csv`,
`thresholds.csv`, `subgroups.csv`, `exactness.csv`, `exactness_mismatches.csv`,
`snr_descriptive.csv`, `snr_cases.csv`, and `provenance.json`. The provenance
retains source JSON hashes, the renderer hash, validated counts, and output
hashes. Existing inference intervals are copied unchanged into CSVs. Markdown
percentages are rounded only for readability. Native TLS/BLS detection evidence
and baseline/optimized TLS exactness have separate sections.

SNR subgroup levels 6/8/10/12 are preassigned latent targets. An unsampled
injection can realize SNR zero and remains in its target group. The
`subgroups.csv` rows with `kind=snr` preserve those assignments; the separate
held-out diagnostic computes the realized centered signal norm.

`--snr` is optional. When supplied, it must be the complete held-out diagnostic
for the original injection manifest and case identities. Medians and observed
ranges describe its native-family/ideal-box white/OU SNR values by regime and
original TLS detected/missed groups. White responses are the enumerated
known-period family ceilings; OU values evaluate those same white-selected
filters using the OU covariance, rather than independently optimizing its
objective. These diagnostics are not package SNR, actual blind-search gain, new inference, or new
approximation allowances. Undefined ratios and empty groups remain visible.

Rerendering the same inputs is allowed. A previous report directory cannot be
reused for different inputs or a changed fixture mode. Synthetic smoke fixtures
must carry `synthetic_fixture: true` in every input and use `--synthetic`, which
places a prominent non-scientific watermark on the report. The external smoke
fixture is `/tmp/cuvarbase-recovery-report-SYNTHETIC-v2/`; unit tests construct
their own explicitly synthetic temporary inputs.
