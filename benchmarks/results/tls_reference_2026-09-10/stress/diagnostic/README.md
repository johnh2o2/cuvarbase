# Cumulative-sum repeatability diagnostic

The original solar SNR-20 discrepancy was reproduced and isolated to the
**float32 flux cumulative sum used by both implementations**. Changing only
that intermediate array reproduces the residual and winning-window difference.
Using identical intermediate inputs, the native and fused scoring kernels
agree bitwise on every inspected window and packed winner. The original
[failed comparison](../README.md) remains failed.

This diagnostic repeats one existing development input on another RTX A6000:
77,888 observations, a 365.25-day signal, white-noise oracle SNR 20 and 97
selected periods. The source, input arrays and search settings are unchanged.
Three uncaptured full searches per implementation are followed by two
instrumented searches and 30 scan repetitions per selected row. Instrumented
searches copy intermediate buffers and can change execution scheduling; they
remain separate from the uncaptured runs. No independent sensitivity samples
or performance measurements are added.

## What changed, and what did not

All 97 captured rows agree in phases, sorted observation indices, flux,
inverse variance, edge corrections and error cumulative sums. The flux
cumulative sums differ in 27 rows. Even the phase ties at period 0.6 days have
identical sorted indices in these captures.

At period **1.05052417069 days**, two flux-prefix variants change the minimum
residual from `0.000522289890796` to `0.000522329763044`: the original absolute
difference `3.98722477257e-8`. The winning start changes from 76,728 to 72,520
and width from 1,113 to 1,482 samples. Twenty-five evaluated trial windows
cross the depth cutoff. Both native scans and graph scans produce two variants
across the 30 repetitions.

At period **0.704129632465 days**, a prefix variant changes a finite residual
`0.000522272545` into the `77888` no-valid-window sentinel. Thirty evaluated
depth-cutoff decisions change. Native full-search repeats consequently differ
in masks and final normalized spectra. This is not merely a harmless change
in the last displayed digit.

| Implementation | Reported SDE across the three uncaptured runs |
| --- | --- |
| Untouched GTLS | 3.24657798, 3.23005271, 3.23005271 |
| Corrected GTLS | 3.24343755, 3.23427696, 3.24343755 |
| cuvarbase | 3.24343755, 3.24343755, 3.24343755 |

All nine runs select **121.75 days**, the one-third alias, and have identical
shared final-fit fields. The injected 365.25-day period and selected alias
tie for the minimum residual in every run, both before and after refinement:
`0.000520125613548` and `0.000520052679349`, respectively. Both have rank one
when ties share a rank. This identifies a represented true-period peak and an
unresolved alias; it does not establish unique fundamental-period recovery.
The original uncollected truth-rank cells remain separate from these new
repeated-run measurements.

cuvarbase's three full runs repeat exactly here, but its standalone graph
scans can also vary. This result does not establish universally deterministic
cuvarbase output, harmless cumulative-sum rounding, or equivalent sensitivity
at every threshold. It identifies inherited scan/depth-cutoff sensitivity on
this long input. The independent 160-case population and separate 24-null
population retain their own completed gates.

## Evidence and reproduction

`runs.csv` gives each repeat's period, SDE, truth rank and residual tie.
`summary.json` records the prefix-only interventions and explicit limits.
`records.tar.gz` preserves all original compact JSON/log files, including
every small numerical vector as dtype, shape and hexadecimal raw bytes.
All 79 original inventory files were collected and hash-verified. The 17
larger or redundant NPZ files remain outside the repository; their identities
are retained, and the public compact vectors reproduce their small arrays
exactly. The whole diagnostic package is approximately 330 KB.

`execution-sources-v3.tar.gz` contains the exact frozen probe, expected source
identities, protocol and CPU tests. Its unchanged source inventory is
`source_manifest.json`; version 3 is the executed protocol. The two earlier
protocol versions were amended before execution and contain no measurements.
`postprocessing_sources.tar.gz` preserves the independent decoder and this
result packager. `environment.json` and `dependencies.txt` record the execution
environment. The reported elapsed time includes instrumentation and is not a
benchmark.

For a GPU rerun, first install the [recorded search environment](../../../../tls_reference/README.md#reproduce-the-numerical-comparison).
From the repository root, stage the frozen sources in a new workspace:

```sh
mkdir -p reproduced-prefix-diagnostic/candidate reproduced-prefix-diagnostic/frozen
tar -xzf benchmarks/results/tls_reference_2026-09-10/sources/production_sources.tar.gz \
  -C reproduced-prefix-diagnostic/candidate
tar -xzf benchmarks/results/tls_reference_2026-09-10/sources/scientific_sources.tar.gz \
  -C reproduced-prefix-diagnostic/frozen
cp -R reproduced-prefix-diagnostic/frozen/main reproduced-prefix-diagnostic/validation
tar -xzf benchmarks/results/tls_reference_2026-09-10/stress/diagnostic/execution-sources-v3.tar.gz \
  -C reproduced-prefix-diagnostic
python benchmarks/tls_reference/inputs.py restore \
  --bank benchmarks/results/tls_reference_2026-09-10/inputs --study stronger_controls \
  --manifest benchmarks/results/tls_reference_2026-09-10/stress/stronger_controls_inputs.json \
  --out reproduced-prefix-diagnostic/inputs
```

The restorer verifies every original numerical array. Its NPZ container
encoding can differ, so bind a **reproduction-only** expectation to that new
container while preserving all original code identities and the original
expectation file:

```sh
python - <<'PY'
import hashlib, json
from pathlib import Path

root = Path("reproduced-prefix-diagnostic")
original = root / "diagnostic/expected.json"
restored = root / "inputs/dense_long_solar_snr20.npz"
expected = json.loads(original.read_text())
expected["reproduction"] = {
    "original_expectation_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
    "original_input_container_sha256": expected["input_sha256"],
    "restoration_receipt_sha256": hashlib.sha256((root / "inputs/reproduction.json").read_bytes()).hexdigest(),
    "scope": "Same bank-verified arrays; another execution is not independent evidence",
}
expected["input_sha256"] = hashlib.sha256(restored.read_bytes()).hexdigest()
(root / "replay-expected.json").write_text(json.dumps(expected, indent=2) + "\n")
PY
python reproduced-prefix-diagnostic/diagnostic/long_control_probe.py \
  --root reproduced-prefix-diagnostic \
  --input reproduced-prefix-diagnostic/inputs/dense_long_solar_snr20.npz \
  --expected reproduced-prefix-diagnostic/replay-expected.json \
  --output reproduced-prefix-diagnostic/results --max-seconds 180
```

The rerun preserves the original failed gate. Different scan variants or
hardware can produce different diagnostic outcomes; each must remain visible.
