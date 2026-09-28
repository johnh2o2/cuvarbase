# Verified partial capacity checkpoint — 2026-09-12

The checkpoint was **secured locally at 06:57:26 UTC**. Its complete input
backup contains all **10,240 frozen cases** across calibration, injections and
test nulls: **71,680 array uses and 30,714 unique arrays**, verified using the
unchanged, captured [exporter](helpers/inputs.py). The [promotion receipt](actual/bank-promotion.json)
binds the published archive and local bank to the [numerical verification](actual/local-verification.json).
This is an actual checkpoint record, separate from the unchanged
[prospective workflow](plans/CHECKPOINT_BANK_WORKFLOW.md).

| Split | Frozen input cases secured | Outcomes in the Stage1 snapshot | Snapshot status |
|---|---:|---:|---|
| Calibration | 5,120 | 10,240 valid | Four completed shards |
| Injections | 2,560 | 5,120 valid | Four completed shards |
| Test nulls | 2,560 | 3,360 valid | Four running shards; partial |

Stage1 captured individual files at approximately **06:36:13 UTC**, with no
claim of a simultaneous snapshot across shards. Its [local receipt](actual/stage1-local-verification.json)
binds the eight completed calibration/injection shards and thresholds to the
earlier completion audits. The original raw null-completion audit remains
required after all null searches finish. This checkpoint does **not** establish
final recovery, false-positive rates, baseline/candidate numerical qualification,
sustained throughput, final collection completion or provider teardown. Both
`complete_campaign` and `scientific_qualification` remain false.

The original NPZ containers are **not preserved by this checkpoint**. The bank
preserves every original numerical array's dtype, shape and values, the original
manifest bytes and NPZ hashes, metadata, and exporter hashes. A later exact-array
restoration can reconstruct numerical inputs; it need not reproduce the original
compressed NPZ bytes. No signals or inputs were regenerated here.

## Actual execution and retained locations

The export ran once with one CPU thread at nice 19, overlapping the ongoing
science search, and completed in **234.565 s** with exit 0 and unchanged source
pins. The [raw launch/execution receipts](actual/provenance/bank-export-execution.json)
are retained without alteration. The export wrapper used its exact Popen handle
and `wait()`; it did not record /proc start ticks.

Packaging used an actual **180 s** command limit and completed in **5.575 s**;
the prospective workflow's 600 s limit was not used. Transfer took **47.737 s**;
local archive/exact-array verification took **6.728 s**. These elapsed times
include their recorded command boundaries and are checkpoint operations, not
search-throughput benchmarks. The [package execution](actual/bank-package-execution.json),
[transfer](actual/bank-transfer.json), [local execution](actual/bank-local-verification-execution.json)
and [promotion](actual/bank-promotion.json) retain the actual commands. Root
promoted the verified tar from its `.partial` download name; the original local
verification receipt still correctly records the earlier transport path.

The bulk files remain outside git under
`/Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/capacity-checkpoint-20260912/`:

| Product | Bytes | SHA256 |
|---|---:|---|
| `stage1.tar` | 65,628,160 | `6eeb65a620a323f8ee01f17a24d2b044c96de4acd8484b56cd0eab3d3bed4d3f` |
| `bank-only.tar` | 1,038,684,160 | `a1b0b1d0d3f379c5dbf00dcae83c9d1ef8a88bfe8c2761b56334cf33a6c41e56` |

The extracted original sources/receipts are in `stage1/`; the verified bank is
`bank-recovered/input-bank/`. Its `bank.json` SHA256 is
`2dfcda710cd4c6ab925fec1b43b72c10dc163cff1eeef58c1f441746d05be67f`;
its `arrays.npz` SHA256 is
`350569a2f867c2282e1065761e214992dcb6a373ebf8ed955ebd57f45f6a50b3`.
These archive/bank identities come from the retained actual verification chain;
assembling this compact directory did not re-read the large numerical files.

## Scope of these compact copies

[INVENTORY.json](INVENTORY.json) records each selected original small file's
source path, destination, byte count and SHA256. It includes the reviewed designs,
helpers, prospective commands, original raw export receipts, actual operations,
and synthetic checks. Full archives, arrays ZIPs, large input manifests and
search-result shards are intentionally kept at the verified external locations.
Their original membership and hashes are in the retained
[Stage1 receipt](actual/provenance/stage1-receipt.json) and
[bank package inventory](actual/checkpoint.json).

Validation history is retained as history. The Stage1 helper's initial small
check preceded capture, but the retained [synthetic driver and repeat receipt](validation/checkpoint-stage1-synthetic-repeat-receipt-v1.json)
were created **after the actual Stage1 capture**; they do not backdate the earlier
inline check. Both bank test iterations remain unchanged; the retained bank
driver corresponds to the final v2 receipt. An independent reviewer incorrectly
reported a JSON newline defect, then retracted it after checking character values.
The [correction](validation/checkpoint-bank-review-correction-v1.json) is retained;
ordinary strict JSON parsing was used throughout, with no normalization exception.
No new tests, remote operations or scientific changes were performed to assemble
this documentation directory.
