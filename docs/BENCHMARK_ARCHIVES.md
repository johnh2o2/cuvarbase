# Benchmark evidence archives

The repository keeps benchmark runners, protocols, readable reports, small
result tables and selected figures. Complete results, inputs, logs, archived
source copies and build artifacts live in immutable Cloudflare R2 archives.
Every original file, including failed and partial outcomes, is preserved.
Moving evidence does not change any numerical qualification or release claim.

The [archive inventories](../benchmarks/archives/) identify each original path,
size and SHA256, the exact archive checksum, its object key, and the original
source commit. Each inventory row records whether the file remains in Git.
All archives were downloaded from R2 and checked against their complete SHA256;
every extracted member was also compared with the original source tree.

## Access and restoration

The bucket is private. A maintainer can supply the archive file or read-only R2
access. No write credentials are needed to restore evidence. Normal package
and tooling tests run without archive access; reproducing the large studies
requires their original inputs and evidence.

List studies and sizes:

```sh
python tools/benchmark_archive.py list
```

Restore a downloaded archive into its original, ignored locations:

```sh
python tools/benchmark_archive.py restore tls_survey_2026-09-10 --archive /path/to/tls_survey_2026-09-10.tar.gz
```

With a configured `rclone` remote named `archive` and bucket `cuvarbase`, the
same command can download, verify and restore in one step:

```sh
python tools/benchmark_archive.py restore tls_survey_2026-09-10 --remote archive:cuvarbase
```

Set `CUVARBASE_ARCHIVE_REMOTE` to use that remote by default. Downloads are
cached under ignored `.benchmark-archives/downloads/`. The helper verifies the
complete archive and every member before installing any missing files. It
rejects traversal, symlinks, corrupt content and conflicting existing files.
It preserves current tracked reports and never overwrites existing evidence.

To recover the complete original tree, including original report text, use
`--full --destination /path/to/empty-directory`. Archived reports preserve
their original relative links and scientific identities there.

Some benchmark runners deliberately use their original input paths. Restore
the relevant studies before using those runners. In particular, TLS population
generation uses the `tls_sensitivity_2026-09-09` cadence archive; transit
generation uses `transit_2026-09-08`; follow-up reporting also uses the original
`tls_survey_2026-09-10` exactness receipt. Historical analysis must use its
recorded source and environment, not silently substitute a new experiment.

## Studies

The sections below are stable targets for links to externally stored evidence.
Hovering an archive link in a report exposes its original path; restore the
named study to inspect that file.

### nufft_lrt_validation_2026-09-06

[Inventory](../benchmarks/archives/nufft_lrt_validation_2026-09-06.json): 16 files; 0.28 MiB compressed.

Object: `r2://cuvarbase/benchmark-evidence/20260928/nufft_lrt_validation_2026-09-06.tar.gz`.

### tls_accuracy_2026-09-09

[Inventory](../benchmarks/archives/tls_accuracy_2026-09-09.json): 59 files; 1.16 MiB compressed.

Object: `r2://cuvarbase/benchmark-evidence/20260928/tls_accuracy_2026-09-09.tar.gz`.

### tls_profile_2026-09-08

[Inventory](../benchmarks/archives/tls_profile_2026-09-08.json): 112 files; 1.01 MiB compressed.

Object: `r2://cuvarbase/benchmark-evidence/20260928/tls_profile_2026-09-08.tar.gz`.

### tls_reference_2026-09-10

[Inventory](../benchmarks/archives/tls_reference_2026-09-10.json): 159 files; 27.76 MiB compressed.

Object: `r2://cuvarbase/benchmark-evidence/20260928/tls_reference_2026-09-10.tar.gz`.

### tls_sensitivity_2026-09-09

[Inventory](../benchmarks/archives/tls_sensitivity_2026-09-09.json): 87 files; 45.21 MiB compressed.

Object: `r2://cuvarbase/benchmark-evidence/20260928/tls_sensitivity_2026-09-09.tar.gz`.

### tls_survey_2026-09-10

[Inventory](../benchmarks/archives/tls_survey_2026-09-10.json): 506 files; 7.99 MiB compressed.

Object: `r2://cuvarbase/benchmark-evidence/20260928/tls_survey_2026-09-10.tar.gz`.

### transit_2026-09-08

[Inventory](../benchmarks/archives/transit_2026-09-08.json): 1,249 files; 120.08 MiB compressed.

Object: `r2://cuvarbase/benchmark-evidence/20260928/transit_2026-09-08.tar.gz`.

### validation-release-prepared-20260927

[Inventory](../benchmarks/archives/validation-release-prepared-20260927.json): 9 files; 0.04 MiB compressed.

Object: `r2://cuvarbase/benchmark-evidence/20260928/validation-release-prepared-20260927.tar.gz`.

### validation-tls-default-20260910

[Inventory](../benchmarks/archives/validation-tls-default-20260910.json): 14 files; 0.02 MiB compressed.

Object: `r2://cuvarbase/benchmark-evidence/20260928/validation-tls-default-20260910.tar.gz`.

### validation-v1.0.0

[Inventory](../benchmarks/archives/validation-v1.0.0.json): 13 files; 0.06 MiB compressed.

Object: `r2://cuvarbase/benchmark-evidence/20260928/validation-v1.0.0.tar.gz`.


## Original Git history

The complete pre-cleanup local and GitHub histories are preserved in the private
R2 prefix `history-cleanup-20260928/before/`, with original refs and checksums.
`remote-before.bundle` is a self-contained Git bundle. Original commit IDs in
scientific receipts and historical reports refer to that preserved history.
The cleanup does not edit those identities or reclassify failed experiments.

To inspect an original source commit, download and verify the bundle against
its archived manifest, then clone it into a separate directory:

```sh
git clone --no-checkout /path/to/remote-before.bundle cuvarbase-original
git -C cuvarbase-original checkout ORIGINAL_COMMIT_ID
```

Do not merge an old clone back into the cleaned development branches: that
would restore the removed archive history. Use a fresh clone for development
and carry any local source changes across as patches.

## Repository policy

New runs write to an ignored workspace or object storage. Commit the runner,
protocol, concise results and failure summaries, plus a checksum inventory.
Keep bulk arrays, per-case JSON, logs and copied dependencies in the archive.
`tools/check_repository_artifacts.py` enforces the selected evidence files and
a 1 MiB per-file limit in CI. Updating an inventory must accompany a verified,
immutable archive; a green test suite never replaces scientific qualification.
