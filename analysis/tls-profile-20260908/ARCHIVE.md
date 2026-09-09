The Git checkout includes the component report and figure, timing and phase tables, per-job JSON records and logs, diagnostic code, source snapshots, and verification receipts.

Input/output NPZ arrays and the transport tar files remain in the local experiment archive. Their hashes are retained in `transfer-sha256.json` and the per-job records. Repeating full-array verification or rerunning the component analysis requires that complete archive. The committed receipt records the verification already performed on the original files.

The later [transit recovery benchmark](../transit-recovery-20260908/README.md) provides the main release comparison. Its [archive note](../transit-recovery-20260908/ARCHIVE.md) and publication manifest describe the material included in Git.
