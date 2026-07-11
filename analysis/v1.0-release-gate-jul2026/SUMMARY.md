# v1.0.0 Release Gate — July 10, 2026

First GPU validation of the **merged** `v1.0-fixes` tip (the union of PRs
#66 + #67 + #68, which had each been pod-validated only on their own
branches), plus the July 10 audit/docs/release-content commits.

**Code state**: working tree at commit `c13dbe3` + the July 10 release-prep
deltas committed immediately after this run (docs/claims-content edits, one
docstring blank-line fix in `utils.py`, sync-script logo include — no
runtime-behavior changes; the suite and gate results are unaffected by
them).

**Pod**: RunPod RTX A5000 (24 GB), driver 580.126.09, CUDA 12.4 image,
Python 3.11.10, pycuda 2026.1, numpy 2.4.6, scipy 1.17.1, with
batman-package, cufinufft, and `transitleastsquares` (reference package)
installed. Pod `6nzaafwb96j38r`, created and terminated the same session
(termination API-verified).

## Results

| Check | Result | Log |
|---|---|---|
| Full pytest suite | **796 passed, 0 skipped, 0 failed** (17:14) | `suite_final.log` |
| Release-gate script (`scripts/check_release_gate.py`) | **14/14 ALL CHECKS PASSED** | `release_gate.log` |
| TLS matched-fidelity timing (re-archival of the audit's missing raws) | tess-yr 12.8× (25.3→325.2 ms/LC), kepler-4yr 8.1× (188.3→1520.5 ms/LC), 100% recovery at both fidelities | `matched_timing.log` (copied to `benchmarks/results/tls_survey_jul2026/matched_timing_a5000_jul2026.txt`) |
| Docs build with GPU figures | build succeeded, **zero warnings/errors**, all 14 plot-directive figures rendered + logo | staged for gh-pages |

Notes:

- An earlier same-session suite run recorded 794 passed / 2 skipped
  (`suite_full.log`): the 2 skips were the TLS golden tests
  (`test_tls_golden.py`) skipping because the reference
  `transitleastsquares` package was not yet installed on the pod. After
  installing it, the full zero-skip rerun (`suite_final.log`) is the gate
  record — the golden tests execute and pass.
- The suite count includes the new audit-driven tests from
  `analysis/tls-audit-jul2026.md` (banded-vs-single-band TLS parity, SDE
  kernel-size behavior, batch-preprocess validation).
- Kernel-cache gate check measured first/second call 3551 ms → 7 ms
  (standard) and 906 ms → 7 ms (optimized), consistent with the published
  caching claims.
