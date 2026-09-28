# Independent collected release-wiring audit

The local audit passed against the pinned final collection. It verified 410 selected collected members, including all 247 package sources (79 immutable baseline, 82 frozen precursor, 86 release), the reviewed runner and protocol, the unchanged 80-case development manifest, and the 11 specified original input files. It independently checked all 58 stored result JSON/NPZ pairs and their 820 typed arrays, rebuilt the normalized fingerprints from those original bytes, and reproduced all 24 prespecified exact comparisons. The 56 worker call records produce 58 results because each release worker includes a two-lightcurve TESS batch.

The paired checks are 12 release-baseline versus immutable-baseline comparisons and 12 release-experimental versus frozen-precursor comparisons: eleven full searches and one fast search per pair. Only the two predeclared execution metadata fields are removed before comparison. All other typed fields, array bytes, dtypes, shapes, masks, NaN words and signed zero remain included. The convenience and batch/FAP outputs were checked for stored-byte integrity, selector metadata and selected-engine traces; those extra outputs do not create additional prespecified paired comparisons.

The exact 86-node device XML inventory passed without skips, errors or failures. The baseline workers recorded no experimental-backend imports; experimental execution recorded both the short prefix kernel and long-row graph fallback. All five recorded stages exited normally with an empty GPU, and the recorded campaign finished within its 900-second cap in 177.825134370476 seconds. These are checks of immutable recorded execution evidence, not a new live GPU or provider observation.

This establishes finite development-case release wiring, not universal equivalence or renewed sensitivity qualification. The original 79/80 development and 5,111/5,120 held-out exactness results and all failed zero-tolerance gates remain unchanged. This audit does not claim experimental execution equals baseline execution, requalify throughput, or independently repeat the original recovery study. The transport archive SHA was already verified by root; this audit did not rehash the entire transport archive.

The first external checker draft reached its final stage-receipt check and rejected a schema assumption: campaign stage rows add the role field, while their standalone execution receipts omit it. The pinned runner explicitly constructs this addition. The correction checks that exact construction, preserving every other field. The initial checker and correction record are retained in checker-review-history-v1; no collected output, scientific definition or gate changed.

Reproduce with Python and NumPy using the collected directory, which contains extracted/ and collection-verification.json. The output must be a new path; no overwrite is supported:

```sh
python -B audit_wiring.py --collection /path/to/release-integration-collected --output /path/to/new-audit-receipt.json
```

No cuvarbase import, test execution, GPU use, process action or provider request occurs. The source records the original artifact pins, while the receipt records the command, local Python/NumPy versions and every checked member hash.
