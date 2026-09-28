# Development history cleanup — 28 September 2026

The owner requested external storage for bulk benchmark output and removal of
that output from the development history before merging PR #69. The cleanup
keeps the release's numerical implementation, failures and qualification intact.

The affected remote refs are `release/v1.0.1`, `v1.0-fixes` and the unpublished
`v1.0.1` source tag. Their old release head was
`403c75d7425e92b8a3d112672a04177189117a61`; the old annotated tag object was
`1e2537c2fb77028fc2872f7a2ccbbb67303caa2f`. The tag is updated as part of the
owner-authorized history cleanup before any GitHub release or PyPI upload.
`master`, all previously published version tags, June `v1.0.0`, other feature
branches and the deployed `gh-pages` branch are unchanged.

## Evidence and source identity

All 86 package files, all tracked sdist build inputs and the prepared wheel and
sdist remain byte-identical. No GPU experiment was repeated. The original nine
TLS exactness mismatches, five unavailable timing panels and BLS repeatability
failures are preserved in the full archives and still reported in the summaries.

Ten [evidence archives](../BENCHMARK_ARCHIVES.md) contain every original member
of the removed study directories, including original versions of retained
reports. Full cloud downloads matched their SHA256 values. The restore helper
also recovered all 506 members of the TLS survey archive downloaded from R2,
with every original member hash verified.

The complete original histories are in private R2 at
`history-cleanup-20260928/before/`:

| File | SHA256 |
| --- | --- |
| `remote-before.bundle` | `4264610a4028a79c76399d2c038b74937c17612e0fc90ab0be2f92a83ced93b1` |
| `local-before.bundle` | `a3331873a95d7b963870c0dc3137848eacc8b2d2ea7f4becbd9623e6dc72ad55` |

The accompanying inventories preserve all original branch/tag values. Both
bundles record complete history and were checked with `git bundle verify`;
their full R2 read-backs matched these hashes. The local bundle also preserves
local-only development branches and the frozen validation worktree's commit.

[commit-map.txt](commit-map.txt) maps original development commits to filtered
commits. Changes after filtering are ordinary commits on top. Original commit
IDs inside scientific receipts remain original IDs, resolvable in the archived
bundle; they are not silently replaced by current source identities.

## What was removed from development history

The filter removes bulk `benchmarks/results/` and `docs/validation/` artifacts,
retaining the explicitly selected reports, figures and summaries. Historical
`analysis/` snapshots are retained in the original Git bundles. The rewrite
excludes `master` and its ancestors, preserving the PR's upstream ancestry.

Fresh clones of all ordinary heads and tags were measured before and after
filtering: Git object storage fell from approximately **296 MiB to 59 MiB**,
about **80%**. Exact final sizes and ref checks are recorded in the local
cleanup delivery. Most remaining bulk belongs to the untouched 2017 docs
branch, including its old dependency cache. This cleanup does not deploy docs.

GitHub may retain old pull-request refs and cached objects independently of
the updated branches. These are not fetched by an ordinary clone. The measured
reduction describes reachable clone contents, not immediate server garbage
collection or destruction of historical evidence.

## Continuing development

Use a fresh clone after the rewritten refs are pushed. Preserve local work as
patches and apply those to the new history; do not merge an old development
branch into the cleaned branches, which would reintroduce the removed objects.
Original local-only refs remain recoverable from the local history bundle.
Use the [archive helper](../BENCHMARK_ARCHIVES.md#access-and-restoration) for raw
study evidence. Normal package and tooling tests need no archive credentials.

Release publication and merging PR #69 remain deferred.
