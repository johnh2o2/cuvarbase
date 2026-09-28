# Study storage

## September 28 repository cleanup

The benchmark and release checks are complete, with failed scientific
qualifications retained. Bulk evidence is now kept in private R2 archives;
Git retains reports, selected figures, small summaries and checksum inventories.
See [benchmark archive access and restoration](BENCHMARK_ARCHIVES.md).
The sections below preserve the earlier storage decisions and their dates.

## September 24 storage pause

The first cloud archive transfer is complete and verified; local archive copies are still retained. The benchmark follow-up remains paused for the storage decision. The user selected an existing Cloudflare R2 `cuvarbase` bucket, whose public endpoints are disabled. Its new A40 rental was terminated after setup, before any benchmark searches; provider absence and supervisor exit were verified. Setup evidence was downloaded and all member hashes checked. Estimated compute was $0.086, with a separate $0.50 storage reserve retained in the conservative ledger. The [follow-up checkpoint](/Users/johnhoffman/Documents/cuvarbase-tls-throughput-20260924/PROGRESS.json) records how to resume.

The data volume had about **25 GiB free** on September 24. Related cuvarbase workspaces occupied about **40 GiB** in allocated file blocks. The broader disk review is recorded in the local migration plan. These are filesystem usage measurements, not promises of space reclaimed: APFS sharing and snapshots can affect that result.

The first cloud transfer contains the existing **489 compressed archives (12.974 GB)** plus **1,537 restore-kit files (0.129 GB)** and two inventory files: **2,028 objects, 13.104 GB in total**. Every remote object passed full SHA256 read-back. All 489 archives decoded directly from R2 to their complete original tar lengths and hashes. Eight NPZ samples were recovered with valid ZIP CRCs, array loading, modes and modification times; one also exercised a hardlink pair. A separate archive was restored using the preserved helper, with mode, mtime, uid/gid and xattrs verified. These tests used scratch paths; the complete historical NPZ restoration was not run.

The [completion receipt](BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/storage-r2-archive-20260924/summary.json") records the result and evidence hashes. A 31-file recovery and receipt bundle was also uploaded and verified under `archive-20260924/_transfer-receipts/first-batch-v1/`. The [migration plan](/Users/johnhoffman/Documents/cuvarbase-storage-plan-20260924/PLAN.md) and [restore instructions](/Users/johnhoffman/Documents/cuvarbase-storage-plan-20260924/R2_RESTORE.md) describe the remaining local storage decision. No local study data was removed. The proposed removal list contains about **12.1 GiB** of archive file blocks; it does not include every file in the old study workspaces.

## Completed local reclamation

On September 12, 2026, storage reclamation for the inactive September 8 and 9 studies completed in two stages. First, removing verified archive-backed extracted NPZ copies reclaimed **26.148 GB of unique file data** (26,147,639,730 bytes). Then exact-byte compression of all **489 retained tar archives** reduced their 26.448 GB of raw bytes to **12.974 GB**, saving another **13.474 GB** of file bytes. Together the two stages reduced the retained file footprint by **39.622 GB** (39,621,672,056 bytes). The independent postcheck passed for all 489 compressed files and their original-matching decode receipts. No cloud storage was purchased or created.

| Completed operation | Unique file bytes removed or saved |
| --- | ---: |
| September 9: 5,060 extracted NPZ files | 7.846 GB |
| September 8: 11,701 result files, each with two hardlink names | 18.183 GB |
| September 8: nine extracted input files | 0.119 GB |
| September 9: exact compression of 480 retained tar archives | 2.608 GB |
| September 8: exact compression of nine retained tar archives | 10.866 GB |

Each removed NPZ matched a complete member in an original archive whose SHA256 matched the retained transfer evidence. Both hardlink names were accounted for before removing a group. Removing only one name would have freed no file data. The original tars were unchanged during this first stage.

The second stage retained a sibling `.tar.zst` for each original tar. Before removing an original, the migration verified the complete compressed stream, independently decoded it to the original SHA256 and byte length, rehashed the original, and rechecked its recorded metadata. It preserved exact raw tar bytes, including padding and retained prefix/tail bytes; it did not reconstruct archives from their members. Durable receipts and removal intents preceded each unlink. Archive compression preserves the existing NPZ restoration plans and member offsets. The actual 489-file collection shrank by **50.95%**: September 9 archives by 32.19% and September 8 archives by 59.23%. The migration took 84.81 seconds; its largest sampled parent-plus-codec RSS was 236 MB. This is sampled resource evidence, not an instantaneous OS-enforced memory bound.

The first stage's execution windows recorded a combined 26.15 GB increase in free space. A later, separate increase of about 40 GB was unattributed and is excluded. The new compression figure is the difference between verified original and compressed file byte lengths, rather than an attribution of all concurrent filesystem changes. APFS sharing, snapshots and unrelated writes can affect observed free space. The [NPZ reclamation receipts](BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/storage-reclamation/summary.json") and [archive compression receipts](BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/storage-archive-compression/summary.json") contain exact counts, hashes and original evidence locations.

## Restoring removed data

**Restore the original tar files first** using the shared [archive recovery kit](/Users/johnhoffman/Documents/CUVARBASE_ARCHIVE_RESTORE_20260912/RESTORE_AFTER_COMPLETION.md) and its [recovery notes](/Users/johnhoffman/Documents/CUVARBASE_ARCHIVE_RESTORE_20260912/RECOVERY_NOTES.md). It contains the pinned helper, plan, metadata, proof backups and transaction receipts. Automated restoration requires the pinned installed Zstandard 1.5.7 executable. If that environment later changes, a compatible generic decoder can recover the raw tar bytes, subject to complete original SHA/length verification and a separately reviewed metadata/publication procedure.

After all tars required by an NPZ plan exist at their original paths, use the unchanged NPZ kits beside the old studies:

- [September 9 NPZ instructions](/Users/johnhoffman/Documents/cuvarbase-tls-study-20260909/STORAGE_RESTORE_20260912/README.md)
- [September 8 NPZ instructions](/Users/johnhoffman/Documents/cuvarbase-work-archive-20260908/STORAGE_RESTORE_20260912/README.md)

Keep the shared archive kit, compressed files and original NPZ kits together as one recoverable collection. The NPZ kits retain their exact tools, plans, metadata, execution journals and verification manifests. Their original 22 single-link and 31 hardlink tests passed, with both suites independently replayed. The archive helper added 36 synthetic tests, including actual exact tar restoration followed by both unchanged NPZ helpers, array verification, hardlink topology, xattrs, corruption, interrupted operations and destination conflicts. The actual old study arrays were not restored after cleanup, and the original tars were not materialized after compression. The independent postcheck rehashed compressed files and validated the complete decode receipts; it did not add a new decode.

Replaying a restore requires the original absolute target/archive layout. The archive helper currently materializes its complete 489-entry plan, requiring at least 26.448 GB plus margin in addition to retained compressed files. Restoring the NPZ data then requires another 26.148 GB of unique file data. A partially completed migration or restore remains a partial result; follow its receipts before continuing. The September 8 hardlink transaction must run **outside the archived study, on the same filesystem**; its runbook explains staging the kit there before starting. Ordinary tar extraction elsewhere is distinct from restoring the recorded path and hardlink layout. Mode, modification time, ownership and xattrs are preserved; inode numbers and filesystem creation/change times are not reproduced.

## Long-term storage choice

**Cloudflare R2 is the selected destination for this study.** Backblaze B2 remains a lower storage-cost alternative. Keep active inputs and small reports locally; upload completed archives and their restore kits to private object storage. Current official list prices, checked September 24, 2026:

| Service | Storage for 100 GB/month, before free allowances | Best fit |
| --- | ---: | --- |
| [Backblaze B2](https://www.backblaze.com/cloud-storage/pricing) | About $0.70 | Infrequently retrieved archives; free egress up to three times average monthly stored data, then $0.01/GB |
| [Cloudflare R2 Standard](https://developers.cloudflare.com/r2/pricing/) | $1.50 | Frequent retrieval; internet egress is free |
| [RunPod network volume](https://docs.runpod.io/storage/network-volumes) | $7.00 | Files needed directly by GPU jobs; charged on allocated capacity and persists after compute ends |

B2 and R2 each offer an initial 10 GB storage allowance. B2 currently lists $6.95/TB/month, with Class A/B/C API calls free; R2 includes monthly request allowances and charges for excess requests. Any applicable taxes and transfer overages are additional. At these rates, 500 GB of B2 storage is about $3.41/month after the free allowance. The selected R2 bucket now holds the verified archive copy. Storage charges are tracked separately from GPU usage.

Before removing the **last locally recoverable copy** of an archive, upload that retained representation and its restore kit, verify a full read-back against its recorded SHA256, and exercise a restore from the destination. For a compressed representation, also verify that a complete decode reproduces the original tar SHA256 and byte length. Preserve a local inventory and receipt; a multipart ETag alone is not an archive SHA256. A separately verified local lossless compressed copy, as used above, remains a local recoverable copy and does not require a cloud upload merely to remove its redundant uncompressed representation.

## Avoiding future growth

NPZ arrays are already compressed, so small within-file gzip samples are a poor estimate of whole-archive savings. Earlier 4 MiB samples gained only about 0.4–4.3%, which did not test repeated compressed streams across files. A complete 1.644 GB original archive saved **13.32%** with default-window Zstandard and **59.31%** with `-3 --long=27 --single-thread`; both full decoded streams matched the original SHA and length. The later 489-file migration provides the actual collection-wide total reported above. Keep one verified archival representation of each finished artifact and extract only what the next analysis needs.

The September 10 TLS survey also uses a verified numerical input bank: roughly 14.6 GB of repeated raw NPZ inputs reduce to about 1.04 GB of unique arrays. This preserves the arrays and their identities, not the original ZIP-container bytes. Keep original manifests and verification receipts; regenerated NPZ hashes must not replace historical hashes. Its [final collection](BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/collection/primary-archive-receipt.json") completed on September 12, preserving the bank and original verification evidence in a 2.512 GB archive; the original GPU rental was then terminated.
