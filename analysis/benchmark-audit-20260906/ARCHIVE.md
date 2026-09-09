The Git checkout includes this historical-claim audit, measurement tables, figures, per-job JSON records and logs, source snapshots, runners, and verification receipts.

The original input/output NPZ arrays remain in the local experiment archive. Their hashes are retained in the input, result, and transfer records. Commands that validate full arrays require restoration of that archive; the committed verification receipt records checks already performed against the original files.

The later [transit recovery benchmark](../transit-recovery-20260908/README.md) supersedes these exploratory BLS/TLS measurements for current release claims. Its [archive note](../transit-recovery-20260908/ARCHIVE.md) and publication manifest describe the material included in Git.
