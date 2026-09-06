# analysis/

Release records and audits that shipped documentation or docstrings cite.
Everything else that used to live here (planning punchlists, workflow
JSON dumps, cost projections, superseded GPU gate/batch records, one-off
probe scripts) was pruned before 1.0 and is preserved, unchanged, at the
annotated tag `archive/pre-1.0-process`:
https://github.com/johnh2o2/cuvarbase/tree/archive/pre-1.0-process
(`git show archive/pre-1.0-process:<path>` for any single file).

What remains and why:

- `RELEASE_RUNBOOK_v1.0.0.md`, `release-staging-v1.0.0/` — the 1.0 release
  mechanics and the drafts it posts (deleted in the post-release commit).
- `audit-sep2026/` — the 1.0 audit of record (readiness + algorithm audits,
  the execution plan, findings, campaign JSON and `repro/` scripts).
- `tls-audit-jul2026.md`, `claims-trace-jul2026.md`,
  `nufft-lrt-audit-jul2026.md` — the July 2026 audits of record, cited by
  the runbook and the NUFFT-LRT documentation.
- `v1.0-release-gate-jul2026/` — the GPU release-gate record cited by the
  release notes (superseded by the final 1.0 gate record when it lands).
- `v1.0-gpu-batch3-jul2026/{SUMMARY,A3_DIAGNOSIS,E1_E2_DIAGNOSIS}.md` and
  `kernel-hygiene-jul2026/PI_HYGIENE.md` — diagnoses referenced from
  shipped docstrings (`cuvarbase/bls.py`, `cuvarbase/lombscargle.py`) and
  the changelog.

Benchmark methodology writeups moved to `docs/` (`GTLS_COMPARISON.md`,
`TLS_COST_ANALYSIS.md`); raw benchmark data lives under `benchmarks/results/`.
