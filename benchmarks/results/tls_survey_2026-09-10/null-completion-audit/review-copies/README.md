# Prepared independent test-null completion audit

**Prepared only; the completion audit has not been run.** No test-null outcome record or input manifest was opened during preparation. Root review and successful completion of the existing test-null search are required before executing the command below. This audit changes no frozen source, input, setting, threshold, sample count, or controller and launches no GPU work.

The preparation captured the original four live worker handles at 2026-09-12 03:20:39.701587 UTC. PIDs 88217–88220 were all running, each with starttime ticks 212311381, the expected command, working directory `/workspace/tls-survey/candidate`, and executable `/usr/bin/python3.11`. Their recorded `search-nulls` stage began at 2026-09-12 01:32:21.530751 UTC. `initial-worker-handles.json` preserves the complete original stage metadata, raw `/proc/PID/stat`, raw command-line bytes, parsed handles, and the campaign-state hash at capture. `capture_initial_handles.py` and `capture-execution.json` document this read-only metadata capture; it did not open outcome files.

`audit_nulls.py` is a minimal adaptation of the corrected injection audit (`../injection-completion-audit/audit_injections.py`, SHA256 `164c40803e6738155eb3e77a0166ecce69e3906247a434d5117e279fcee4fdd3`). The full difference is retained in `source-diff-from-injection-audit.patch`. The prepared null-audit source SHA256 is `4f8d93496421daab3f179e179a8bb2596c8dbbe452cff47a288f204c2de68eef`.

The differences are limited to the following:

- Stage and file names select `search-nulls`, `inputs-nulls/manifest.json`, and `nulls-search-{0..3}.json`. Split/cohort/null fields must identify test nulls. The frozen count is still 256 inputs per regime, ten regimes, and four shards: 2,560 paired inputs and 5,120 method outcomes.
- Null noise-scale labels are IID draws from the equal four-level mixture (6, 8, 10, 12); realized counts need not be balanced. The added pure helper checks allowed support and total population size. It imposes neither an empirical frequency test nor forced balance. All cases are retained.
- Sampling summaries are explicitly called **latent**. They describe the stored signal used to define sampling and noise scale; no transit is inserted into the null flux. They do not count detected events or imply that a sampled signal was present in the noise-only data.

All other structural checks retain the corrected injection audit's definitions. Before opening the input manifest or outcome receipts, the audit requires the matching original stage to be complete, all four recorded worker exit codes to be zero, and each original `/proc` identity to be absent or replaced by a different starttime. A same-identity zombie does not pass. The original commands and stage identity must remain unchanged.

The audit then checks exact manifest-index-modulo-four membership, unique paired names and input identities, selected BLS method/ranker settings, complete scientific and production source inventories, original seal and threshold byte hashes, every original NPZ container hash, stored metadata and array inventories, declared grids, both stored and returned period-array identities, valid finite selected outcomes, and retention of every latent sampling stratum. Threshold values and recovery/alias fields are not inspected or applied. Scores are checked only for validity and finiteness, never aggregated, reported, compared to truth, or compared to thresholds. No calibration or injection outcome is read.

The two existing hash conventions remain separate: manifest period arrays use JSON-encoded dtype and shape plus contiguous bytes (`tls_reference/validate.py`); returned grids use `str(dtype)` and `str(shape)` plus contiguous bytes (`tls_survey/common.py`). Complete original NPZ hashes protect all stored arrays; only period-array hashes are additionally recomputed. Other per-array hashes are not separately recomputed. Stored physical labels are checked for retention and consistency, not regenerated. Returned grid identities alone do not prove evaluation of every internal template or establish numerical equivalence.

## Preparation validation

`preparation-checks.json` records **31 passing static and synthetic checks** under Python 3.9.6 / NumPy 1.26.4. The validator parses and compiles the audit without executing its main function, compares unchanged helper ASTs with the corrected injection source, verifies split paths and the early worker-exit gate, excludes recovery/alias accesses and threshold-value parsing, and exercises only the changed mixture helper and retained pure strata/hash helpers on synthetic inputs. It never opens outcome records. This is preparation validation, not a completed campaign audit.

The executed local validation command was:

```sh
/Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/local-env/bin/python -B \
  /Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/evidence/null-completion-audit/check_preparation.py \
  --source /Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/evidence/null-completion-audit/audit_nulls.py \
  --injection-source /Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/evidence/injection-completion-audit/audit_injections.py
```

## Invocation after completion and root review

Use Python with NumPy on the original Linux host after all four original workers have exited normally. The example below is a concrete invocation from a separate writable directory containing the reviewed `audit_nulls.py` and captured `initial-worker-handles.json`. It creates new output files exclusively; it does not overwrite previous evidence. The script also caps numerical-library threads at one before importing NumPy.

```python
from pathlib import Path
import hashlib
import subprocess
import sys

assert hashlib.sha256(Path('audit_nulls.py').read_bytes()).hexdigest() == \
    '4f8d93496421daab3f179e179a8bb2596c8dbbe452cff47a288f204c2de68eef'
assert hashlib.sha256(Path('initial-worker-handles.json').read_bytes()).hexdigest() == \
    '4f3f360eba16035041ed7c191c74adbc91130699c16242925cf3f52c89520d29'
with open('audit.stdout.json', 'x') as output, open('audit.stderr.txt', 'x') as errors:
    subprocess.run([
        sys.executable, '-B', 'audit_nulls.py',
        '--repo', '/workspace/tls-survey/candidate',
        '--campaign', '/workspace/tls-survey/final-campaign',
        '--seal', '/workspace/tls-survey/evidence/seal-final.json',
        '--seal-sha256', '1b81c75bd1a498c0dbed607e3221da1f374fc05be765de6dd2670c8d2f2b0807',
        '--thresholds-sha256', 'caccda435f944e04682dd298a1b0fae659060f63e13ce281c8ae9cb850d373aa',
        '--initial-handles-json', Path('initial-worker-handles.json').read_text(),
    ], stdout=output, stderr=errors, check=True)
```

The eventual execution receipt should preserve the exact command, interpreter/environment, reviewed source and initial-handle hashes, timestamps, exit status, and output hashes, including any failed run. No completion-audit output currently exists in this preparation set.

## Original-container and archive reproduction limits

Replaying the exact original-file checks requires **original raw NPZ containers**, available from the live study, a raw-rescue archive retaining them, or separately retained originals. A normal completed `verified_banks` archive excludes original NPZ files covered by a verified input bank and cannot alone support a new check of their original container bytes. With original raw NPZs and the other study files available, mount a read-only copy at the recorded `/workspace/tls-survey` Linux paths and run from a separate writable directory as above. Literal recorded command/working-directory checks must not be bypassed by changing receipts. A later replay can check recorded exits but cannot independently re-observe historical process exits on the original host; the retained live capture and eventual completion observation provide that evidence.

For a normal bank archive, use **separate numerical-array verification and restoration**. `tls_reference/inputs.py::export_bank` preserves exact numerical arrays, metadata, and original manifests. `verify_bank` verifies the bank and numerical identities. `restore_bank` creates new NPZ containers and a reproduction manifest with recomputed container hashes plus original NPZ hashes; it does not guarantee original container bytes. Historical original-file audits and bank-export/verification receipts document the originals. Do not replace original hashes or substitute restored containers into this exact audit to make it pass.

For the usual final bank location, from a writable directory with the archived study mounted read-only at `/workspace/tls-survey`:

```sh
python3 /workspace/tls-survey/candidate/benchmarks/tls_reference/inputs.py verify \
  --bank /workspace/tls-survey/final-campaign/input-bank
python3 /workspace/tls-survey/candidate/benchmarks/tls_reference/inputs.py restore \
  --bank /workspace/tls-survey/final-campaign/input-bank \
  --study nulls \
  --manifest /workspace/tls-survey/final-campaign/input-bank/manifests/nulls.json \
  --out restored-nulls
```

If the archive instead uses an additional-input bank, select it from the retained bank/archive receipts. Restoration writes into the new directory outside the archive and reproduces the same numerical population, not a newly independent null sample. Keep the reproduction manifest and `reproduction.json` distinct from original manifests. This audit makes no false-positive-rate, recovery, sensitivity-equivalence, or universal-coverage claim.
