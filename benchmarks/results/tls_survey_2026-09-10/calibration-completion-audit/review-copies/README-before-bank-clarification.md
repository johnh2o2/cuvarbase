# Independent calibration completion audit

The read-only audit passed on 2026-09-11 at 18:26:50 UTC. It found no discrepancy in the completed calibration or its thresholds.

- Four complete shards contain exactly 10,240 unique method outcomes: 512 unique names in each of 20 regime/method groups. Actual shard membership exactly matches the manifest's index modulo four allocation.
- TLS and the frozen selected BLS method use identical input names and byte identities within every regime. All 5,120 original calibration NPZ files were hashed (7,292,425,446 bytes total) and match their manifest and search receipts.
- The pinned scientific seal, all 15 scientific sources, the complete 82-file production source inventory, each runner identity, and all four threshold source-receipt links match. Actual selected BLS methods and rankers match the seal. All outcomes are valid with finite selected scores; no API errors appear.
- All 40 persisted threshold dictionaries exactly match independent standard-library recomputation. At 512 calibration scores, the 5% point uses ascending rank 488 and the 1% point uses rank 508. Under the exchangeable-null design, their marginal bounds are respectively 25/513 (4.8733%) and 5/513 (0.97466%).
- Every cut has one score equal to the threshold, so there are 24 strict exceedances at the 5% point and four at the 1% point. No cut has additional boundary-tie conservatism. The actual frozen detection code uses `score > cut`; equality is excluded.
- The 38 zero-score TLS grazing nulls are valid successful-no-candidate outcomes and remain in calibration. They do not produce ties at either chosen cut.

The calibration bank is paired between TLS and BLS; method-specific thresholds are estimated separately from that shared bank. It is independent of development and the test-null bank. Calibration exceedance fractions are order-statistic properties, not measured independent-test false-positive rates. This audit makes no held-out recovery claim and does not establish a universal physical or noise model.

`audit_calibration.py` uses only Python's standard library, built-in sorting, and exact rational rank arithmetic. It never imports the campaign's threshold implementation or a GPU package. It reads only calibration products, the frozen seal, their source files, and calibration input bytes. It writes its receipt to stdout; the SSH caller stores stdout locally. The candidate's source files and completed calibration products were rehashed at the end to check stability. No remote files or controllers were changed and no held-out outcomes were read.

The full result is `audit.stdout.json`; `execution.json` records the exact arguments, source hash, transport, exit status, and stdout/stderr hashes. Empty stderr and exit status zero are retained. To reproduce on Linux, mount a read-only copy of the archived study at its recorded `/workspace/tls-survey` paths, including the candidate, final campaign, and evidence directories below. Run the command from a separate writable directory containing `audit_calibration.py`, so the new receipt is written outside the mounted archive:

```sh
python3 audit_calibration.py \
  --repo /workspace/tls-survey/candidate \
  --campaign /workspace/tls-survey/final-campaign \
  --seal /workspace/tls-survey/evidence/seal-final.json \
  --expected-seal-sha256 1b81c75bd1a498c0dbed607e3221da1f374fc05be765de6dd2670c8d2f2b0807 \
  --expected-thresholds-sha256 caccda435f944e04682dd298a1b0fae659060f63e13ce281c8ae9cb850d373aa \
  --expected-thresholds-bytes 59643 > new-audit.json
```

The script checks threshold receipt paths literally, so the read-only archive must be mounted at these original Linux paths. Relocation alone does not change input, score, threshold, or source hashes; do not edit original receipts to make a different layout pass.
