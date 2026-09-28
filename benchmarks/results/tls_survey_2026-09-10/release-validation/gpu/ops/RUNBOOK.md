# Prepared release integration — unlaunched

Nothing in this directory has run on a GPU, provisioned a resource, contacted a
provider, changed a scientific source, or modified the release checkout. Root
owns the original study lifecycle and any later short rental. This runner has
no cloud API, provisioning or provider-termination commands.

The complete fixed plan is copied byte-for-byte as
`development-validation-plan.json` (SHA256
`56dc97b664dfbebc5f02541563c2bc9ce8c7fc7ffacd81482a75d4bf4c9ca811`).
It includes eleven original development inputs, full original period grids,
zero numerical tolerance and two immutable-control/release pairs. No calibration
or held-out input is accepted. The earlier local `WORK/development` and
`WORK/development-v2` arrays have different hashes and must not be substituted.
Use **`WORK/collected/dev-final/` after normal collection**. The original
completion archive preserves those NPZs verbatim.

`protocol.json` fixes the original baseline79, optimized precursor82 and staged
release86 package-source hashes. The original default files remain exact
6ced75d bytes. Three separate full package roots are required. Neither the
scientific runner nor its thresholds are imported or edited here. Comparison
is of complete public numerical outputs, arrays, masks and metadata, excluding
only `search_configuration.execution` and
`search_configuration.experimental_execution`. Every original result is saved
as a typed JSON tree plus a compressed NPZ, preserving float words including
NaN payloads and signed zero. Comparisons record missing/failed results and the
changed paths. They do not retry or substitute any call.

## Setup evidence and prerequisites

Original survey evidence is not yet locally collected. Required files after
normal primary → supplement → collection → provider termination are:

- `WORK/collected/evidence/freeze.txt`
- `WORK/collected/evidence/install.json`
- `WORK/collected/evidence/gpu.xml`, `cpu.txt`, `cpu-quota.json` for comparison
- `WORK/collection-state.json`, the original collector’s actual state path; it must have `status=phase=complete`,
  `evidence_verified=true`, and recorded owned-provider absence for
  `okideq277lpb4a`.

Use the SHAs from the verified collection inventory, not freshly declared
unknown evidence. `setup` requires the original freeze AND install-report pins.
It retains both raw receipts, checks their common distribution versions, and
emits only exact `name==version` requirements. The local cuvarbase and gputls
project entries are explicitly excluded: cuvarbase is supplied through one of
the three pinned source roots, and native GTLS is not executed by this wiring
check. Other unportable or unpinned dependency lines fail preparation.

Original `pip freeze` may omit build tools. `build-tool-metadata.json` is the
root's separate read-only observation at 2026-09-12 15:40:35 UTC, SHA256
`e21581bf01b305f00b5c37610d40b383a952a1c2ec73f704e2544d2cb9b8e6cb`:
Python3.11.10, pip26.2.1, setuptools84.0.0, wheel0.48.0. Those exact observed
build-tool versions supplement missing freeze entries; contradictions fail.
This is not retroactively labeled an original installation receipt. The original
freeze/install are still mandatory. Do not use the local macOS host/build
environment freezes or an older study's requirements.

Root may provision **only after the original lifecycle and teardown finish**,
within the existing budget. Expected prerequisite is the same single A40/SM86,
Python3.11.10/CUDA12.4.1-devel image and NVCC12.4.131 toolchain described in the
prototype README. Record actual image/driver/GPU/CPU/memory identity and any
differences. Setup/provisioning/install time and cost are recorded separately
from the 900-second integration envelope. Do not perform an unpinned upgrade.
Create a fresh Python3.11 venv, bootstrap the three exact observed build-tool
pins, then install the derived exact requirements with a new pip report. Use
`--no-build-isolation` for builds after the pinned build prerequisites are
installed, so installation cannot silently select newer isolated build tools.
Save the final `pip freeze --all`. An installation problem is evidence and must
be resolved before the GPU envelope starts; no different numerical package
versions are silently accepted. The runtime verifies every derived package
version and the observed Python3.11.10 interpreter before starting a GPU subprocess.

No project installation is necessary: fresh processes select one complete
pinned source root via `PYTHONPATH`, and assert the actual imported package path.
Set CUDA PATH/library paths normally, and preserve compiler flags required by
the short-prefix guard. The runner fixes one numerical-library thread per
process, NUMBA_NUM_THREADS=4, CUPY_ACCELERATORS=cub, and PYTHONNOUSERSITE=1. It
requires one visible GPU and checks compute-context ownership between/during
stages. Missing build support or skipped device coverage cannot pass the full
integration check.

## Concrete command sequence for later reviewed operation

The following is a recipe, not a launched command. `WORK` below is a task-specific
variable, not a system environment variable. Transfer the reviewed ops directory,
three full source roots and exact development originals only when root authorizes
the new rental. On the GPU host, the paths passed to `bind` must be the actual
local copies there. Copies of the final termination/setup receipts accompany
that bundle. No SSH, upload, setup or rental is performed by this code.

```sh
WORK=/Users/johnhoffman/Documents/cuvarbase-tls-survey-20260910
cd "$WORK/release-integration-ops"
# CPU-only; fresh output directory, with original verified receipt SHAs.
../local-env/bin/python integrate.py setup \
  --freeze "$WORK/collected/evidence/freeze.txt" --freeze-sha256 ORIGINAL_FREEZE_SHA \
  --install-report "$WORK/collected/evidence/install.json" --install-report-sha256 ORIGINAL_INSTALL_SHA \
  --output setup-derived
```

After pinned setup and transfer to the later GPU host, run this CPU-only binding
step. `ROOT` here denotes a task-specific example directory containing the full
verified copies; replace it deliberately when preparing the final launch.

```sh
ROOT=/workspace/release-integration
"$ROOT/venv/bin/python" "$ROOT/ops/integrate.py" bind \
  --baseline-root "$ROOT/baseline" --precursor-root "$ROOT/precursor" \
  --release-root "$ROOT/release" --manifest "$ROOT/dev-final/manifest.json" \
  --inputs "$ROOT/dev-final" --setup "$ROOT/setup-derived/setup.json" \
  --setup-sha256 DERIVED_SETUP_SHA --termination "$ROOT/original-collector-final.json" \
  --termination-sha256 VERIFIED_FINAL_COLLECTOR_SHA --output "$ROOT/binding.json"
```

Root reviews and pins the concrete binding bytes. The eventual single launch is:

```sh
"$ROOT/venv/bin/python" "$ROOT/ops/integrate.py" run \
  --binding "$ROOT/binding.json" --binding-sha256 REVIEWED_BINDING_SHA \
  --output "$ROOT/results-attempt-1"
```

An existing output directory is refused. There is no resume/retry CLI. An
interruption preserves its original stage, logs, receipt and any complete
outputs. Further experiments would require separate root review and new output
paths; the tool never rewrites failed outcomes.

## Fixed execution and interpretation

A single wall-clock **and monotonic** 900-second deadline begins before the first
GPU ownership query/subprocess. Cold contexts, JIT, short-prefix compilation and
canary, input preparation/transfers, output compression, and focused device
tests are included. The last30seconds are reserved for owned-child termination
and ownership checks. No stage starts in the last60seconds. CUDA work can block
Python signal handling. Each stage is therefore wrapped by Linux GNU
`/usr/bin/timeout`, which remains an independent guardian if the Python parent
dies; it sends TERM at remaining35seconds and KILL5seconds later. The parent
also enforces both deadlines and stops scheduling on SIGINT/SIGTERM. It
terminates only its owned subprocess session, escalating to SIGKILL if needed.
GPU PID ownership requires a matching Linux session, inherited unique owner
marker and stable `/proc` start ticks; unsupported NVML/PID-namespace mappings
fail closed. A bounded release poll tolerates short NVML cleanup delays inside
the existing reserve. Final nonempty/unknown GPU ownership or either exceeded
deadline prevents a successful result.
Provider teardown remains root's separate mandatory responsibility. Driver-level
cleanup failure is retained and is not labeled successful GPU emptiness.

Four fresh sequential workers execute, in order: immutable6ced75d, release
implicit baseline default, frozen optimized precursor, release experimental.
Each runs all eleven full searches plus TESS-solar0000 with `full=False` to
exercise final winner fitting. All original arrays/grids/settings stay intact.
The two release workers additionally exercise `tls_transit` on TESS-solar0000
and batch/FAP calls on the fixed three cases with two null permutations,
seed20260912. TESS-solar uses two copies of its original lightcurve in a batch;
ZTF-solar and HATpi0001 each use one. This tests a multi-lightcurve batch while
keeping each case's original complete grid. It does not substitute a shared
cross-regime grid or create a new physical population. During fast and batch
smokes, lightweight wrappers record calls to the selected backend and require
all observed/FAP calls and final full fitting to remain there; wrappers are
restored immediately. No such instrumentation is added to the eleven primary
scalar comparisons.

The final fresh process runs the staged
`test_tls_reference_prefix.py` and `test_tls_reference_short_prefix.py` suites.
They include actual separate module/graph-buffer ownership, thread caches,
packed winner bits, literal/graph prefixes, guarded short scans, fallback,
compiler/canary failure and runtime-fault propagation. The host routing suite
was already validated locally and is not rerun for scientific evidence here.
Device-test errors/skips, missing short-module dispatch, or missing native graph
fallback mark coverage unavailable/failed. Exactly86 device node IDs were
collected locally with inert CuPy imports and no test execution;
`device-test-inventory.json` freezes those IDs. Actual JUnit names must match
that complete inventory, not merely report a positive count. A pinned pytest
configuration and disabled third-party plugin autoload replace inherited
PYTEST selection/plugin environment settings. Baseline workers must never import
experimental backend/short-prefix modules, even after their batch/FAP calls.

Planning proxy from the two pinned exploratory receipts: the eleven-case sums
were27.60164s literal and18.48342s prototype, suggesting about92.17s for two
passes of each. These literal timings used the then-current optimized precursor
with native prefix, **not** immutable6ced75d, and cannot establish the new
baseline's latency. Extra API calls, fresh contexts, hashing/compression and
focused device tests consume the remaining cap. This is a margin estimate,
not a guaranteed runtime or a throughput benchmark; exceeding it preserves
partial work and fails the attempt.

No result here requalifies experimental sensitivity or waives the frozen
zero-mismatch gate. Original development remains79/80; original held out
remains5111/5120 exact, with nine unresolved chi2/SDE differences. The new
release wiring has no historical benchmark result of its own. Control/new-path
variations, missing results and byte mismatches are retained separately without
causal attribution, retries or tolerance relaxation.

Offline preparation tests: `../local-env/bin/python -m pytest -q test_integrate.py`
passed31 small synthetic/mock checks. Those tests perform no CUDA imports or
cloud operations. `offline-tests.xml` is their retained receipt.

The pre-review drafts and their previous offline-test receipts are retained in
`pre-review-v1/`, `pre-review-v2/` and `pre-review-v3/`. Independent review identified and corrected
terminal GPU/deadline pass checks, device-suite deselection, interruption/parent
death protection and one editable-freeze parser edge. No draft was launched.

Cleanup also checks live marked processes in the owned Linux session after its
timeout leader exits, using fresh owner/session/start-tick evidence. It escalates
to KILL after5seconds for TERM-ignoring survivors even if the leader is already
gone. Two synthetic interruption cases and a mocked `/proc` fixture cover this
branch; no real process was signalled during preparation.
