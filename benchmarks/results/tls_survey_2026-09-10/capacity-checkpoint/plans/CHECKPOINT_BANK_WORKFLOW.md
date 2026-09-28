# Prepared bank-only checkpoint workflow

Prepared on 2026-09-12; no real packaging, transfer or local array verification was run while preparing this workflow. Root must confirm the existing export completed with exit 0, unchanged pins and a published bank before executing. This is a partial evidence backup, never a replacement for final collection or scientific qualification. Stage1 is already locally verified separately.

The new helper is `capacity-contingency/checkpoint_bank.py`, SHA256 **b6cf36a4b6953f637719a2a562884a28737c8d510420e2b3295b667df1981ef7**. It fixes the reviewed science/auxiliary/exporter/three-manifest and Stage1 receipt/archive identities in source. The original exporter and all original sources, receipts and input banks remain unchanged. It preserves the original launch/execution/stdout/stderr bytes; no JSON normalization is used. Export completion is linked by the original wrapper's Popen handle followed by wait(), plus matching PID/start/command/pins. No /proc start-tick evidence was recorded, and none is inferred.

1. Root reviews this helper, the successful export execution receipt and current free space. Upload only this new helper to the already-existing sibling tool directory:

```sh
python3 /Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/ops/cloud.py put survey01 /Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/capacity-contingency/checkpoint_bank.py /workspace/tls-capacity-checkpoint-20260912-tools/checkpoint_bank.py
```

2. Run one CPU-only package command, retaining its exact stdout and exit status in a fresh local receipt. The helper validates every metadata gate before creating its fresh output. It writes an uncompressed tar, verifies its exact regular-file membership and byte inventory, then publishes the archive. A failure or timeout leaves any new partial output in place; do not reuse it automatically.

```sh
python3 /Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/ops/cloud.py ssh survey01 'env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 nice -n 19 timeout --signal=TERM --kill-after=30s 600 /workspace/tls-survey/modern/bin/python -B /workspace/tls-capacity-checkpoint-20260912-tools/checkpoint_bank.py package --checkpoint-root /workspace/tls-capacity-checkpoint-20260912 --output /workspace/tls-capacity-checkpoint-20260912/bank-package --helper-sha256 b6cf36a4b6953f637719a2a562884a28737c8d510420e2b3295b667df1981ef7'
```

Require exit 0 and `status=bank_archive_verified_remotely`. The actual archive SHA256 and size are future products of this command; preserve those values from its stdout independently of the later download. The published paths are `/workspace/tls-capacity-checkpoint-20260912/bank-package/bank-only.tar` and adjacent `package-receipt.json`. Its members are the exact seven bank files, four original export receipts, original Stage1 receipt, reviewed helper source and an outer `checkpoint.json` byte inventory. It does not copy full case NPZs or ongoing search files.

3. Download the small package receipt to the fresh local path below, compare its fields with the already-recorded successful remote stdout, and record its SHA. Confirm the two destination paths do not already exist before the SCP calls; the transport itself overwrites existing paths.

```sh
python3 /Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/ops/cloud.py get survey01 /workspace/tls-capacity-checkpoint-20260912/bank-package/package-receipt.json /Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/capacity-checkpoint-20260912/bank-package-receipt.json
python3 /Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/ops/cloud.py get survey01 /workspace/tls-capacity-checkpoint-20260912/bank-package/bank-only.tar /Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/capacity-checkpoint-20260912/bank-only.tar.partial
```

4. After successful transfer, use the following local invocation. The downloaded receipt must already match the independently retained remote stdout; this block does not establish that external comparison by itself. It reads the actual archive SHA from that reviewed receipt rather than inventing a future hash. It checks the complete external archive SHA, every safe member and original metadata pin, then calls `verify_bank` from the exact captured Stage1 exporter, checking all 10,240 cases / 71,680 array uses. It streams array verification without restoring case NPZs.

```python
import json, os, pathlib, subprocess
w = pathlib.Path('/Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910')
c = w / 'capacity-checkpoint-20260912'
r = json.loads((c / 'bank-package-receipt.json').read_bytes())
helper_sha = 'b6cf36a4b6953f637719a2a562884a28737c8d510420e2b3295b667df1981ef7'
assert r['status'] == 'bank_archive_verified_remotely'
assert r['helper_sha256'] == helper_sha
assert r['archive'] == '/workspace/tls-capacity-checkpoint-20260912/bank-package/bank-only.tar'
assert r['complete_campaign'] is False and r['scientific_qualification'] is False
assert (c / 'bank-only.tar.partial').stat().st_size == r['bytes']
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
subprocess.run([
    str(w / 'local-env/bin/python'), '-B',
    str(w / 'capacity-contingency/checkpoint_bank.py'), 'verify',
    '--archive', str(c / 'bank-only.tar.partial'), '--archive-sha256', r['sha256'],
    '--exporter', str(c / 'stage1/candidate/benchmarks/tls_reference/inputs.py'),
    '--output', str(c / 'bank-recovered'), '--helper-sha256', helper_sha,
], env=env, check=True)
```

Only exit 0 plus `bank-recovered/local-verification.json` with `status=partial_input_checkpoint_secured_locally` establishes local input backup. `bank-recovered/input-bank` is then usable by the existing bank verifier/restorer; this workflow does not restore it. The tar keeps its `.partial` transport name to avoid confusing download completion with scientific completion; its successful verification receipt gives its exact verified SHA. Failures retain `bank-recovered.partial` and must be reported, never promoted or retried over existing paths.

For a bank of B bytes, this adds approximately B remote archive bytes and 2B local bytes (download plus extracted bank), with metadata overhead; it does not allocate the approximately 14.6 GB original case containers again. Remote packaging performs several sequential byte reads of B; local verification performs archive reads plus the unchanged exporter's checksum and numerical-array checks. The 600-second package timeout is a bounded operational limit, not a measured runtime prediction. Root should use the actual exported byte count and current free space before transfer. No GPU work, search criteria, controllers, collection handback or provider lifecycle is changed.

Validation: ten tiny synthetic checks passed, including actual frozen-exporter export and verification, byte/status preservation, failed export refusal, external hash rejection, immutable Stage1 descendant refusal, and unsafe/duplicate tar refusal with retained partials. Driver: `capacity-contingency/test_checkpoint_bank_synthetic.py` (SHA fd756e9d6355e2fa5b65db91c66b27e358fe7fdfa425242e0e4a031b2a1460d4). Receipt: `capacity-contingency/checkpoint-bank-synthetic-receipt-v2.json` (SHA 57a43288232afebcddc2a641638c00007f5f925b207a0e7ccbb52d2a1f9d4f16). Its three-case pins are changed only in the imported module's memory, not the operational helper source or CLI.
