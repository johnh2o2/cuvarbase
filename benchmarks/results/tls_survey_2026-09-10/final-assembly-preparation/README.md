# Final delivery copy preparation

This additive helper was prepared during the frozen injection search on
2026-09-11. These files are **synthetic helper validation, not scientific or
throughput measurements**. No live result, source, setting, threshold or
controller was changed.

After the actual archives have been verified, the supplement extracted and
the combined figure rendered as described in [the handoff](../FINAL_HANDOFF.md),
run:

```sh
python3 assemble_final_delivery.py --work-root /path/to/collected-study-work \
  --destination /path/to/fresh-compact-delivery
```

The normal complete collection, both figure sets, reviewed design identities
and all required compact products must exist. Partial/rescue collections need
separate review; this helper refuses them. It verifies archive, inventory and
product hashes, preserves relative timing layouts, and writes source/destination
hashes and sizes in `ASSEMBLY.json`. Failed numerical qualifications remain
false. Failed fresh qualification receipts listed only under `unavailable`
are retained; an originally absent reference is explicitly recorded. Bulk
arrays and journals remain in the referenced external archives.

The root reviewer checked the helper against the actual frozen collector,
controller, campaign and renderer schemas. This exposed and corrected omission
of unavailable-only qualification references. The original agent check receipts
are retained as `agent-check-v1.json` and `agent-check-v2.json`; v1 precedes
that correction. The root independently verified 48 copied files with the
corrected helper.

The 51-file fixture is only 15,369 uncompressed bytes. Its archive and figure
files contain mock bytes for testing the copy contract; it cannot validate
archive-member verification, scientific inference or figure rendering. Those
remain the responsibilities of the existing collection and rendering checks.
Replay the standalone copy check with Python 3.9 or later:

```sh
python3 replay_synthetic_check.py
```

The replay verifies all 48 copies, retained failure labels and the missing
original reference, then checks refusal of an existing destination, a tampered
reference, a missing expected reference and incomplete collection. It uses a
temporary directory and never reads the real campaign. `root-replay.json`
records its result. `artifact-manifest.json` inventories this preparation.
