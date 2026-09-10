# Exact numerical inputs

This 20.8 MB bank stores the original observation, uncertainty, flux, injected
signal, exposure, band and period arrays. Repeated arrays are stored once;
restoration performs no physical-signal generation or random draws. NumPy and
the Python standard library are sufficient.

| Study label | Cases | Scope |
| --- | ---: | --- |
| `main` | 160 | Independent full-grid confirmation |
| `supplement` | 24 | Separately sealed null population |
| `selected_grid` | 21 | Development numerical stresses |
| `long_period` | 2 | Selected-grid long-period parents |
| `stronger_controls` | 2 | Same long-period signals with half the original noise |

`bank.json` identifies the lossless array archive, each unique dtype/shape/byte
identity, and every unchanged original manifest under `manifests/`. The
[restoration proof](../sources/input_restoration_proof.json) independently
checked all 209 cases and all 1,463 numerical arrays against their original
files. All arrays and metadata match. The two stronger controls have different
NPZ container encodings after restoration; both container hashes are recorded.
Container encoding is separate from the identity of the numerical inputs.

From the repository root, restore the primary population into a new directory:

```sh
python benchmarks/tls_reference/inputs.py restore \
  --bank benchmarks/results/tls_reference_2026-09-10/inputs --study main \
  --manifest benchmarks/results/tls_reference_2026-09-10/validation/input_manifest.json \
  --out reproduced-inputs
```

The output keeps `original_manifest.json` unchanged and creates a separately
labeled reproduction manifest. The [full comparison and timing workflow](../../../tls_reference/README.md)
uses that manifest without changing an original scientific seal. For the
supplement, use `--study supplement` and its original manifest in the
`supplement/` directory.

The three development labels have no independent-study seal. Their selected
grids contain truth and aliases; use them as numerical fixtures, not blind
recovery or throughput evidence. Their original manifests are under `stress/`
as `selected_grid_inputs.json`, `long_period_inputs.json` and
`stronger_controls_inputs.json`. The control manifest originally lacked
per-array hashes; its bank entries explicitly derive those hashes only after
verifying the original NPZ file hash and metadata. This does not invent a
new source identity or seal.

The earlier 21-case generator version was not retained. Its original identity
and numerical vectors remain available here. The maintained generator
separately recreated those vectors exactly on the original CPU environment;
the [source notes](../sources/README.md) distinguish that check from preserving
the historical generator bytes.
