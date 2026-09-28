# Independent injection completion audit

The corrected structural audit passed at 2026-09-12 01:35:58 UTC with no campaign discrepancy. The original four workers (PIDs 62060–62063, starttime ticks 209815123) were observed running with the expected commands and working directory before the audit. Their stage completed at 01:32:21 UTC with four exit codes zero. At 01:32:46 UTC all four original process handles were absent and every injection receipt was complete. A matching zombie would not have satisfied the audit's exit gate.

The audit verified:

- Exactly 2,560 unique injection inputs and 5,120 valid method outcomes: 256 inputs in each of ten regimes, paired between TLS and its development-selected BLS comparator. Each of four shards contains exactly its planned 640 inputs and 1,280 outcomes, assigned by manifest index modulo four.
- All 2,560 complete original NPZ file hashes (3,646,206,730 bytes), their stored metadata, array inventories, and period-array hashes match the manifest. All 5,120 returned period-grid hashes match the corresponding original input grid. Grid and search settings agree with the frozen regime declarations; denser grid oversampling remains distinct from search-statistic oversampling.
- All 15 scientific sources and the complete 82-file production source inventory match the scientific seal. Runner, manifest, campaign, threshold, and method/ranker bindings match. Threshold bytes were hashed without reading or applying their values.
- Every selected score is finite; every selected period is finite and positive or represents the frozen valid zero-score/no-candidate contract. Unselected diagnostic rankers do not redefine selected-detector validity. No API errors occur.
- Both methods retain every sparse-sampling case: 13 unsampled, 34 one-event, 248 two-event, and 22 cases with one to four in-transit observations. These categories overlap. Each regime retains 64 inputs at each target/latent SNR label (6, 8, 10, 12). Labels and event counts are checked against original metadata, not independently regenerated physics.

`audit-v2.stdout.json` is the final receipt. Its SHA is `4c526604ba7422dade7312a3774d26d2ff12aabf1e0eb2a5b3712431c47fe63f`. The injection manifest SHA is `7f06664e99aea638c8dd04897413c4717b3e15918c44a1d460691939a4637d3a`. Selected score/period values, recovery flags, alias flags, and recovery contrasts are not reported or analyzed. No independent test-null outcome was read.

## Retained checker correction

The first audit used the search-receipt array hash when checking the manifest's per-array period identity. Those formats differ in the pre-existing sealed code:

- `generate.py` uses `tls_reference/cases.py`, which imports `tls_reference/validate.py::array_hash`: JSON-encoded dtype string (or structured dtype description), JSON shape, then contiguous bytes.
- `run.py` uses `tls_survey/common.py::array_hash`: `str(dtype)`, `str(shape)`, then contiguous bytes.

Consequently, that checker reported 2,560 manifest-period-hash mismatches even though all complete NPZ hashes and all returned grid hashes passed. This was an external checker error, not a scientific gate change or campaign-data mismatch. The correction adds the original manifest hash convention and uses it only for the manifest comparison; it also labels both conventions in the receipt. `checker-correction.diff` gives the complete source change. The repeated audit uses identical scientific criteria and original artifacts. No scientific/operational source, input, setting, threshold, or controller was changed.

The initial failed result remains in `audit.stdout.json`, with empty `audit.stderr.txt`, nonzero status in `execution.json`, and its executed source preserved as `audit_injections_v1.py` (matching the source SHA in that execution record). The corrected source is `audit_injections.py`; its execution record and empty stderr are `execution-v2.json` and `audit-v2.stderr.txt`. Both executions used one CPU thread for array decoding and hashing, no GPU calls, and no cuvarbase or campaign imports. They took about 35 and 40 seconds respectively while the controller independently continued its planned test-null stage.

## Reproduction and scope

The exact execution arguments, interpreter, source hashes, initial process-handle receipt hash, timestamps, and output hashes are retained in the execution records. **Replaying this exact original-file audit requires the original raw NPZ containers**, available from the live study, a raw-rescue archive that retains them, or separately retained originals. A normal completed `verified_banks` archive excludes original NPZ files covered by a verified input bank and is therefore insufficient by itself to rerun this audit's original-NPZ-hash checks.

If the original raw NPZs and the other required study files are available, use Python with NumPy, mount them read-only at the recorded Linux paths below, and run from a separate writable directory containing the audit script and `initial-worker-handles.json`:

```python
from pathlib import Path
import subprocess
import sys

with open('new-audit.json', 'x') as output:
    subprocess.run([
        sys.executable, '-B', 'audit_injections.py',
        '--repo', '/workspace/tls-survey/candidate',
        '--campaign', '/workspace/tls-survey/final-campaign',
        '--seal', '/workspace/tls-survey/evidence/seal-final.json',
        '--seal-sha256', '1b81c75bd1a498c0dbed607e3221da1f374fc05be765de6dd2670c8d2f2b0807',
        '--thresholds-sha256', 'caccda435f944e04682dd298a1b0fae659060f63e13ce281c8ae9cb850d373aa',
        '--initial-handles-json', Path('initial-worker-handles.json').read_text(),
    ], stdout=output, check=True)
```

A later replay with the original raw NPZs can verify original file identities and recorded exit statuses; it cannot independently re-observe historical process exits on the original host. The retained live handle observations and completion-stage receipt document those checks. Do not edit original receipts to accommodate a different directory layout.

For a normal completed bank archive, the supported reproduction route is **separate numerical-array verification and restoration**. `benchmarks/tls_reference/inputs.py::export_bank` preserves the exact numerical arrays, metadata, and original manifests. `verify_bank` checks every stored numerical identity. `restore_bank` writes new NPZ containers and a reproduction manifest with their recomputed container hashes plus the original NPZ hashes; it does not guarantee reproduction of the original container bytes. The historical original-file audit and export/verification receipts supply provenance for the originals; bank verification does not newly recheck unavailable original NPZ bytes.

For the usual final bank location, run these commands from a writable directory, with the archived study mounted read-only at `/workspace/tls-survey`:

```sh
python3 /workspace/tls-survey/candidate/benchmarks/tls_reference/inputs.py verify \
  --bank /workspace/tls-survey/final-campaign/input-bank
python3 /workspace/tls-survey/candidate/benchmarks/tls_reference/inputs.py restore \
  --bank /workspace/tls-survey/final-campaign/input-bank \
  --study injections \
  --manifest /workspace/tls-survey/final-campaign/input-bank/manifests/injections.json \
  --out restored-injections
```

If the archive instead uses the additional-input bank, select its location from the retained bank and archive receipts. Restoration writes into the new `restored-injections` directory outside the archive; these are the same frozen numerical inputs, not a newly independent population. Keep its reproduction manifest and `reproduction.json` distinct from the original manifests. Do not substitute restored containers into this exact original-file audit or replace the original hashes to make it pass.

The prior delivered README and inventory are retained under `review-copies/`. This clarification changes documentation and the unsealed inventory only; the audit code, executed receipts, original-file checks, scientific gates, and remote study are unchanged.

This audit establishes structural completeness, recorded input/source/grid bindings, and retention of stored sampling strata. Complete NPZ hashes protect all arrays; only period-array hashes are separately recomputed. Returned full-grid identities do not independently prove every internal template/trial was evaluated. Numerical equivalence belongs to the separate frozen exactness stage. This receipt makes no recovery, equal-FPR sensitivity, approximation-equivalence, or universal-coverage claim.
