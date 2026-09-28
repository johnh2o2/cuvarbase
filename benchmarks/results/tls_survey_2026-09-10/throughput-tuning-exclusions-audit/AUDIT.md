# Frozen throughput tuning exclusions

Read-only audit of the completed development pilot. No sources, gates, selections or GPU jobs were changed. These are tuning results, not final sustained-throughput or calibrated recovery results.

The campaign completed **16 configurations: 12 eligible, 4 excluded**. Frozen selections are baseline **4 workers / batch 8**, candidate **4 / 4**, and public GTLS **2 / 1**. BLS has no qualifying setting. Its first configuration failed, so the predeclared rule stopped further BLS tuning.

| Excluded configuration | Evidence and classification |
| --- | --- |
| GTLS 4 workers / batch 1 | Six GPU out-of-memory API failures across five distinct lightcurves, followed by three worker membership errors. Qualification stopped before any measured queue. |
| GTLS 2 / 4 | Post-queue repeatability failure for gapped-TESS development case 0003, worker 1: power and chi2 changed; SDE changed from 12.744205474853516 to 12.748966217041016. Period stayed 18.95005062135495 days; period arrays and finite masks matched. Its completed timing repetition is excluded. |
| GTLS 2 / 8 | Pre-queue out-of-memory failure requesting 1,837,246,464 bytes. The failed task contained gapped-TESS cases 0000–0007; the receipt does not identify the triggering member. |
| BLS 1 / 1 | Selected likelihood score changed from 138.0975799560547 to 138.09754943847656 (−0.000030517578125) on gapped-TESS case 0006 during measured task 22. Period stayed 12.847274301670177 days. API and membership checks passed, but the frozen exact selected-score gate failed. |

All four excluded configurations passed GPU ownership checks with no foreign GPU processes. The score/spectrum changes are numerical qualification failures; **no calibrated threshold crossing or recovery change is established by this audit**. The BLS comparison in the scientific recovery campaign remains distinct from these throughput exclusions.

The called public GTLS implementation exposes no period-batch or memory-fraction keyword. Its active core.py:620–631 chooses period groups from instantaneous free GPU memory, a fixed safety factor, and a cap of one-thirtieth of the period grid. Our queue batch size groups sequential lightcurve calls within each worker; it leaves this internal heuristic unchanged. This is the declared conditional optimum over queue batches and worker counts, not an optimum over modified GTLS allocation algorithms.

Measurement has not yet run. Source inspection confirms that the absent BLS selection produces explicit missing panels for TESS solar, gapped TESS, ZTF solar and varied sampling; the other selected backends continue. The renderer marks missing results and excludes failed measurements from speed denominators. It separately requires the full held-out exactness receipt before presenting the global exactness status.

The five original campaign/result files were downloaded and their bytes verified against remote SHA256 values. Four local timing/protocol source hashes also match the deployed sources; the three active public GTLS source hashes are recorded. All identities, exact cases, counts, deltas, and remote/local paths are in [audit.json](../../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/throughput-tuning-exclusions-audit/audit.json") and [transfer-and-source-hashes.json](../../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/throughput-tuning-exclusions-audit/transfer-and-source-hashes.json").

Original receipts:

- [Completed campaign](../../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/throughput-tuning-exclusions-audit/originals/campaign.json")
- [GTLS 4 / 1](../../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/throughput-tuning-exclusions-audit/originals/gtls-mixed-w4-b1__result.json")
- [GTLS 2 / 4](../../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/throughput-tuning-exclusions-audit/originals/gtls-mixed-w2-b4__result.json")
- [GTLS 2 / 8](../../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/throughput-tuning-exclusions-audit/originals/gtls-mixed-w2-b8__result.json")
- [BLS 1 / 1](../../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/throughput-tuning-exclusions-audit/originals/bls-mixed-w1-b1__result.json")
