# Release integration source transfer — prepared locally, unlaunched

`source-bundle.tar.gz` is **1,418,579 bytes** and contains only the three separately pinned package roots, the reviewed integration operations files, and small provenance/verification files. All 263 tar members are regular files with safe relative paths; there are no symlinks, hardlinks, `.git` files, environments or NPZ inputs. The 262 payload files contain baseline **79**, frozen optimized precursor **82**, and staged release **86** package files, including `.cu`/`.cuh` kernels and their package tests/conftest. The final runner and its setup command are unchanged.

The earlier 254.6 MB `baseline.tar` covers a full repository; only its 79 pinned package members were selected here. The previously tested release wheel is retained separately but does not include the other two controls or operations files. No existing combined bundle covered this complete source layout. The selective bundle is not an installed wheel and makes no new packaging or numerical qualification claim.

| Transfer item | SHA256 |
| --- | --- |
| `source-bundle.tar.gz` | `e1d0fe2ba33237665453b33b464c01e230317b66d39aa67fbd7e304fd4c6f40e` |
| `transfer-inventory.json` | `01f43057e7537fee4010aa1f12af0bdd4d814547523e873b79d3659106a8ce8e` |
| `verify_transfer.py` | `7f381583d6a4f80d072f1eb1d1fce9b5c391428136ded10f0c31d39d419d77ed` |
| Original operations inventory | `68b1146d045b147cd8a2f90a7365a48a25765ba16ed4b9c87075d6f07b05a12a` |
| Original runner | `6edeb1e7536300dd1d8e11781a6b3dc4d2fa0f7b55f4d7453b8101294a4d5f6e` |
| Original integration protocol | `1aeca13d85142a89822b5ee1b6b64dbdcd259e5d360981f6be4db05f6d77ac03` |

The fresh local `verified-layout/` was extracted only after verifying the complete archive hash, exact safe member set, every payload SHA/size, and the internal inventory bytes. The unchanged runner's CPU-only `protocol()` and `check_sources()` then confirmed all three maps. This performed no tests, CUDA imports, provider calls, setup, provisioning or remote writes. The original collector was still `status=phase=waiting`, and final collected development/freeze/install files were absent at preparation. Nothing here permits starting another rental before the original lifecycle closes.

## Payload layout

Extract directly into a fresh `/workspace/release-integration` on a later root-approved rental:

```text
baseline/cuvarbase/     79 exact baseline6ced75d files
precursor/cuvarbase/    82 frozen optimized study files
release/cuvarbase/      86 separately staged default-preserving release files
ops/                   final runner, setup code, protocol, fixed plan and test pins
provenance/            original staging validation and local-only rental review/runbook
transfer-tools/        this source-transfer verifier
transfer-inventory.json exact internal/external inventory bytes
```

No project installation, `setup.py` or worktree metadata is needed: the reviewed runner selects one complete package root with `PYTHONPATH` in each fresh subprocess and checks the actual import path. Local rental/provider controller code and credentials are deliberately excluded. The copied rental runbook is provenance; the original local `WORK/release-validation-rental/rental.py` remains the lifecycle owner for any later rental.

## Required later additions — not available in this bundle

Wait for normal primary → supplement → verified collection → provider termination. Use the original verified collection inventory to pin these bytes:

- The complete, unchanged **80-case** `WORK/collected/dev-final/manifest.json`, SHA256 `a1d18d6cf2d09fc4450a6f4ce9cf6f794e05685f65c755bbac5e7429c3116328`.
- Only the **11 original NPZs** named in `ops/development-validation-plan.json`, SHA256 `56dc97b664dfbebc5f02541563c2bc9ce8c7fc7ffacd81482a75d4bf4c9ca811`. Select their `cases[].file` paths from the original manifest, preserve those relative paths under `dev-final/`, and verify each file against its already frozen `input_sha256`. Do not rewrite the manifest to 11 rows. Do not substitute `WORK/development`, `development-v2`, regenerated ZIPs, calibration or held-out files.
- Original collected `evidence/freeze.txt` and `install.json`, plus `gpu.xml`, `cpu.txt`, `cpu-quota.json` for environment comparison. Their final verified inventory SHAs are still unknown here. The separate observed build-tool metadata in `ops/` does not replace them.
- The final original `WORK/collection-state.json`, copied byte-for-byte as `original-collector-final.json`. It must report complete/evidence-verified/provider-absence status for `okideq277lpb4a`; the current waiting receipt is not included.
- CPU-derived `setup-derived/{setup.json,requirements.txt,original-freeze.txt,original-install.json}` from the reviewed runner's `setup` command. Requirements must come from those original collected receipts, with the pinned observed build-tool supplement. No versions have been guessed or upgraded in this preparation.

Prepare those later inputs/evidence as a **separate verified supplement** so this source bundle remains immutable. Check the runtime environment against the original frozen packages and Python3.11.10 before binding. Setup/provisioning costs remain separate from the unchanged 900-second GPU integration cap and within the reviewed local rental cap. The full setup/lifecycle prerequisites are in the copied `ops/RUNBOOK.md` and original local rental runbook.

## Concrete later transfer instructions — none executed

Root first verifies original teardown, final spending, and the separate rental review plan. After root authorizes a new host, set the task-specific host/SSH port from that rental's verified status receipt. No current study SSH alias is assumed or reused here.

```sh
TLS_TRANSFER=/Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/release-validation-transfer-prep
: "${TLS_RELEASE_HOST:?Set the root-reviewed new host}"
: "${TLS_RELEASE_SSH_PORT:?Set its verified public SSH port}"
scp -P "$TLS_RELEASE_SSH_PORT" \
  "$TLS_TRANSFER/source-bundle.tar.gz" "$TLS_TRANSFER/transfer-inventory.json" \
  "$TLS_TRANSFER/verify_transfer.py" "root@$TLS_RELEASE_HOST:/workspace/"
```

On that later host, compare all three `sha256sum` values against the table above before running the verifier. Then extract to a new directory (before creating its venv); an existing destination is refused:

```sh
sha256sum /workspace/source-bundle.tar.gz /workspace/transfer-inventory.json /workspace/verify_transfer.py
/usr/bin/python3 /workspace/verify_transfer.py \
  --archive /workspace/source-bundle.tar.gz \
  --archive-sha256 e1d0fe2ba33237665453b33b464c01e230317b66d39aa67fbd7e304fd4c6f40e \
  --inventory /workspace/transfer-inventory.json \
  --inventory-sha256 01f43057e7537fee4010aa1f12af0bdd4d814547523e873b79d3659106a8ce8e \
  --destination /workspace/release-integration
```

Retain the receiver's verification output. After the separately verified original inputs/closure/setup supplement is placed in this layout and pinned software is installed, use the unchanged binding step on the GPU host:

```sh
TLS_INTEGRATION=/workspace/release-integration
: "${TLS_DERIVED_SETUP_SHA:?Set the reviewed derived setup SHA}"
: "${TLS_FINAL_COLLECTOR_SHA:?Set the verified final collector receipt SHA}"
"$TLS_INTEGRATION/venv/bin/python" "$TLS_INTEGRATION/ops/integrate.py" bind \
  --baseline-root "$TLS_INTEGRATION/baseline" --precursor-root "$TLS_INTEGRATION/precursor" \
  --release-root "$TLS_INTEGRATION/release" --manifest "$TLS_INTEGRATION/dev-final/manifest.json" \
  --inputs "$TLS_INTEGRATION/dev-final" --setup "$TLS_INTEGRATION/setup-derived/setup.json" \
  --setup-sha256 "$TLS_DERIVED_SETUP_SHA" --termination "$TLS_INTEGRATION/original-collector-final.json" \
  --termination-sha256 "$TLS_FINAL_COLLECTOR_SHA" --output "$TLS_INTEGRATION/binding.json"
```

Root reviews and pins that concrete binding before the separate `run` command described in the original runbook. This source-transfer check cannot qualify release wiring, experimental sensitivity, throughput, GPU behavior or provider termination. The historical development79/80 and held-out5111/5120 zero-mismatch failures remain unchanged.
