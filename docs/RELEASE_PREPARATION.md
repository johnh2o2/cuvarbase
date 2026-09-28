# Release preparation: 1.0.1

The reviewed candidate is prepared as **1.0.1**. The completed work on
`v1.0-fixes` is integrated with `master` on **`release/v1.0.1`**, the source
branch for the release pull request. The owner authorized pushing these source
branches and the annotated `v1.0.1` tag on 28 September 2026. Creating a GitHub
release, uploading to PyPI and deploying documentation remain deferred.
The existing `v1.0.0` tag retains June commit `5553248`; it is not moved or replaced.

The merge of `master` retains its normalization fixes, which were already in
the reviewed implementation. Its complete tree matches the prepared candidate
at `fcfee0e`; four normalization regression tests also pass. Subsequent handoff
documentation updates do not change the prepared package or its build inputs.

[Release notes](RELEASE_NOTES_v1.0.1.md) ·
[Benchmark and retained qualifications](TRANSIT_BENCHMARKS.md) ·
[GPU validation](../benchmarks/results/tls_survey_2026-09-10/release-gate-20260927/README.md) ·
[Final package verification](validation/release-prepared-20260927/package-verification.json) ·
[Preparation checks](validation/release-prepared-20260927/checks.json).

The expanded GPU suite passed 2,091 tests with one expected notebook failure
and zero skips. The separately corrected installed-wheel gate passed 14
numerical/runtime checks and six dependency preflights. Those runs used the
preserved 1.0.0 candidate wheel. For 1.0.1, 85 package files remain byte-identical;
the sole package-file change is the `__version__` string in `__init__.py`.
The comparison checks that replacement exactly, as well as the wheel and
source-distribution inventories. Distribution metadata and documentation
reflect the new version.

Scientific conclusions are unchanged: baseline TLS remains the default,
experimental TLS remains opt-in after its failed aggregate exactness gate,
and all five unavailable timing panels stay unavailable. BLS execution rates
do not gain numerical qualification. No new benchmark or GPU rental is
required for this version and documentation preparation.

The local delivery directory is
`/Users/johnhoffman/Documents/cuvarbase-release-prepared-20260927/`.
It contains `dist/`, artifact checksums, build and verification logs, a Git
bundle, the prepared GitHub release text and a publication runbook. The
committed source and that delivery are backed up in the private R2 bucket;
the local completion receipt records the exact object prefix and read-back.
That directory preserves the original local preparation snapshot. The source
push, pull request, final tag and CI receipts are recorded separately in
`/Users/johnhoffman/Documents/cuvarbase-release-pr-20260928/`.

To inspect the prepared state without publishing:

```sh
git status --short
git show --no-patch v1.0.1
git diff v1.0.1 -- cuvarbase pyproject.toml README.md CHANGELOG.rst
```

Before publication, review the pull request into `master` and its CI checks.
The later delivery directory contains the current `PUBLISH.md`; it supersedes
the original runbook's deferred branch/tag push steps. When publication is
authorized, verify the recorded commit, artifact checksums and current remote
state before creating a GitHub release and uploading the two distributions.
Neither publication nor PR creation moves the existing `v1.0.0` tag.
