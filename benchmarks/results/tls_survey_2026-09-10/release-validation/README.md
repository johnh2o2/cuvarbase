# Release wiring validation — completed 12 September 2026

The release preserves baseline `6ced75d` execution by default and exposes the bundled optimizations through explicit `execution="experimental"`. The [branch application receipt](branch/application.json) records the 14 applied files, all 86 resulting package hashes, and the three restored default files checked against Git baseline bytes. The [copy plan](branch/integration-plan.json) preserves exact before/after maps; 72 other package files were unchanged.

The separate development-only GPU check passed **24 exact paired comparisons and all 86 device tests** in **177.825 seconds**, inside its predeclared 900-second cap. This validates release wiring on the fixed cases; it does **not** requalify experimental sensitivity or historical throughput. The original study remains **5,111/5,120 exact held-out comparisons**, with nine discrepancies and failed zero-mismatch qualification; its original development result remains 79/80. No tolerance, historical receipt or held-out population was changed.

## What was checked

| Check | Result and evidence |
|---|---|
| Default execution against immutable baseline | 12/12 exact pairs: eleven full-observation development cases plus one TESS fast-path case |
| Experimental execution against frozen optimized precursor | 12/12 exact pairs on the same inputs; this is not an experimental-versus-baseline equality claim |
| Device tests | 86/86 passed; exact node inventory, no failures or skips |
| Dispatch | Default avoided experimental imports; experimental short-prefix kernel and graph fallback were both observed |
| Public APIs | Convenience, batch and fixed-seed FAP output paths exercised for both release modes; these are additional API checks, not extra prespecified control pairs |
| Host and wheel | Broad host run: 872 passed, 18 skipped and one expected failure. Focused host and unpacked-wheel runs: 219 passed each |
| Cleanup | Every GPU stage exited successfully; GPU empty at completion. Both real rented pods are absent and all owned controls are closed |

The unchanged [protocol](gpu/ops/protocol.json), [fixed eleven-case plan](gpu/ops/development-validation-plan.json), [complete original 80-case manifest](gpu/dev-final/manifest.json), [binding](gpu/binding.json), [campaign](gpu/results-attempt-1/campaign.json) and [device XML](gpu/results-attempt-1/device-tests.xml) retain the exact trial grids, identities and execution records. Each of the four worker directories under `gpu/results-attempt-1/` contains its original receipt and complete result metadata JSON. Full binary arrays remain in the external archive below. Pair comparison removes only the two expected execution-selector metadata fields; scientific values, masks, array dtypes/shapes/bytes and scalar bit patterns retain exact comparison.

The [independent audit](independent-audit/receipt.json) recomputed all 24 pairs, checked 58 saved outputs and 820 arrays, and matched the exact device inventory. There are 56 call records and 58 output records because the two release-mode TESS batch calls each contain a duplicate input result. Its checker correction history is retained. The audit did not rerun GPU work. [Host validation](host/validation.json), [host XML](host/host-suite.xml), [wheel XML](host/wheel-tests.xml) and [wheel verification](host/wheel-verification.json) retain the earlier local evidence, including the failed initial wheel metadata build; their historical pending labels describe their original creation time.

Neither finite cases nor these passing pairs establish universal equivalence or bitwise determinism. The scientific failures that motivated opt-in execution remain relevant, including difficult thin/grazing regimes described in the main study.

## Environment and setup history

The integration ran on a **new NVIDIA A40** with a **20 GB container disk**, zero persistent volume, the original CUDA 12.4.1 image, Python 3.11.10, NVCC 12.4.131 and all **64 pinned package versions**. CPU quota remained **7.65 cores**, memory limit **49,999,998,976 bytes**, and reported GPU memory **46,068 MiB**. The Linux CPU description was byte-identical to the original receipt.

The host driver changed from **570.195.03** in the original study to **570.211.01**. The GPU UUID changed from `GPU-bd0c2d60-9d88-0e9c-91ca-bcf8b371d7fa` to `GPU-f62d0cdc-bf50-9862-e2b9-476e4c7d103e`. These differences are recorded in the [original GPU report](gpu/original-environment/gpu.xml), [validation GPU report](gpu/environment-v2/gpu.xml) and [root binding review](setup-ops/root-binding-review.json). This new allocation is not a repeated survey throughput measurement.

The first installation failed after 33.400 seconds because PyCUDA's build could not import NumPy. Its [execution receipt](gpu/setup-execution.json) and [complete install log](gpu/environment/install.log) are retained. The second installer first installed the already pinned NumPy 2.2.6 and completed the unchanged requirements in 126.493 seconds; see its [execution receipt](gpu/setup-execution-v2.json), [installer](setup-ops/install-v2.sh), [64-version environment receipt](gpu/environment-v2/receipt.json) and [installation report](gpu/environment-v2/install.json). No GPU integration attempt ran before setup succeeded. Setup time is separate from the 177.825-second integration clock, which includes context/JIT/canary work.

The original 80 GB rental request received an explicit capacity rejection. A separately reviewed 20 GB request succeeded; the [first rejection and control closure](lifecycle/rejected-80gb/capacity-rejection-closure.json) and [actual rental termination](lifecycle/rental-20gb/attempt/termination.json) are both retained. The first request is described as “no allocation observed,” not as a fabricated pod termination. Its full $1.50 uncertainty reserve remains in the conservative budget. The [final ledger](lifecycle/final-ledger.json) records approximately **$71.85225 observed elapsed-cost estimate** and **$73.75634 conservative total with reserves**, within the existing $100 authorization; these are estimates, not invoices.

## Full outputs and reproduction

This compact folder contains metadata, logs, XML and source maps only. [Provenance](provenance.json) maps every copied file to its exact original source path, SHA256, size and collected archive member where applicable; [artifact inventory](artifact-inventory.json) pins all published files. The original [archive receipt](collection/archive-receipt.json) and [local collection verification](collection/collection-verification.json) establish that all **448 original files** were downloaded and verified before teardown.

The retained local artifacts are:

| External artifact | Bytes | SHA256 |
|---|---:|---|
| [Complete outputs and all three source trees](</Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/release-integration-collected/bundle.tar.gz>) | 358,677,468 | `354ee1ade51d9f8579d063b1c548f206c239c2f049b7712047f6d7b4631b399e` |
| [Frozen source/runner transfer](</Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/release-validation-transfer-prep/source-bundle.tar.gz>) | 1,418,579 | `e1d0fe2ba33237665453b33b464c01e230317b66d39aa67fbd7e304fd4c6f40e` |
| [Original input/setup transfer](</Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/release-validation-data-transfer/data-setup-bundle.tar.gz>) | 15,111,320 | `634239ec084d487c932cdfecc8e5153bc53f0163e4cf00809752fddf3bca2183` |
| [Host-verified release wheel](</Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910/release-validation-evidence/wheels/cuvarbase-1.0.0-py3-none-any.whl>) | 542,446 | `05c58466dd1943c3fbac07f77a5d80a5692c1e7c07cf78db4d21b142c2689134` |

These are local workspace artifacts, not public downloads. Preserve their exact bytes when relocating them and update only the external location record. The complete archive contains all arrays needed to independently inspect the existing result; the compact [audit checker](independent-audit/audit_wiring.py) shows how the recorded comparisons were verified.

For a new run, verify the source and data archives against these SHAs, use the [source transfer instructions](transfer/sources/README.md) and [data transfer instructions](transfer/data/README.md) to assemble fresh roots, and retain the full 80-case manifest with only the eleven pinned NPZ inputs. Follow the unchanged [integration runbook](gpu/ops/RUNBOOK.md) and setup versions, create a **new** binding on the actual host, then run once into a fresh results directory with the fixed 900-second cap. Do not overwrite this binding or any result receipt. The retained v2 installer documents the pinned-NumPy prerequisite missing from the first setup attempt. The wheel checks were local; the GPU validation used isolated exact source trees through `PYTHONPATH`.

A future execution requires its own resource authorization within the remaining total budget. These archived instructions do not leave any rental or background job running.
