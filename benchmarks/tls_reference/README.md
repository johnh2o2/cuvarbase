# Observation-level TLS validation and timing

These tools compare cuvarbase's standard TLS engine with the complete search
in [GTLS commit `74e449c`](https://github.com/Farthing-0/GTLS/tree/74e449c325792a763dde4fbffab98039c5e8c111).
The [results archive](../results/tls_reference_2026-09-10/README.md) supplies
the frozen inputs, source identities, outcomes and timing receipts behind the
[transit benchmark report](../../docs/TRANSIT_BENCHMARKS.md).

## What is tested

The primary comparison checks complete residual and power spectra, masks,
candidate and harmonic refinements, and final selections. It also checks the
actual cuvarbase public result contract. The independent full-grid population
contains ordinary, high-impact, eccentric and dense-M-dwarf transits on TESS
and ZTF cadences, plus noise-only controls. Selected-period stress fixtures
are separate numerical tests; they are not independent recovery trials.
Passband baselines are assumed already removed and the injected signals are
achromatic. Both search APIs receive only `t`, `y` and `dy`; the stored band
labels are provenance metadata, not a fitted multiband model.

Untouched GTLS outcomes are retained. Exact differential checks additionally
use one disclosed host correction: exclude masked/nonfinite candidates before
ranking. `corrected_reference.py` applies that correction temporarily in
memory and records the original and corrected function hashes. Native CUDA
source and templates are unchanged. A native result is reused for the corrected
comparison only when its complete selection trace proves the correction is a
no-op. See the [defect explanation](../../docs/GTLS_COMPARISON.md#invalid-candidate-correction).

| Tool | Purpose |
| --- | --- |
| `inputs.py` | Restore exact published observation, noise, signal and period arrays from the compact input bank |
| `cases.py` | Generate physical injections and nulls, or regenerate published numerical inputs with hash checks |
| `validate.py`, `comparison.py` | Instrument both searches, compare complete outputs and retain failures |
| `corrected_reference.py` | Auditable native host correction and sufficient checks for no-op reuse |
| `summarize.py` | Per-case and per-regime recovery, null and numerical-agreement summaries |
| `reproduce.py` | Rerun a published population without claiming new independent evidence |
| [timing/](timing/README.md) | Public API latency, 16-source throughput and component timings after numerical validation |
| `analyze_timing.py` | Validate timing provenance and produce the figure's normalized measurements |

## Reproduce the numerical comparison

Exact input restoration needs only NumPy and the Python standard library.
The compact bank stores the original numerical vectors and checks every
array's dtype, shape and bytes. It does not regenerate the signals or noise.
The original manifests and seals remain unchanged.

From the repository root, use a new output directory:

```sh
python benchmarks/tls_reference/inputs.py restore \
  --bank benchmarks/results/tls_reference_2026-09-10/inputs --study main \
  --manifest benchmarks/results/tls_reference_2026-09-10/validation/input_manifest.json \
  --out reproduced-inputs

python benchmarks/tls_reference/reproduce.py --repo-root . \
  --manifest reproduced-inputs/manifest.json \
  --seal benchmarks/results/tls_reference_2026-09-10/validation/seal.json \
  --out reproduced-results
```

The search requires the recorded CUDA environment, cuvarbase's `tls` extra,
and the pinned GTLS package. These scripts do not provision a GPU or install
dependencies. The original seal remains unchanged; reproduction receipts
record the executing source separately. Running the same frozen inputs again
does not create a new independent study.

The original Linux search environment used Python 3.11 and the dependencies in
[execution_environment.json](../results/tls_reference_2026-09-10/validation/execution_environment.json).
In a fresh Python 3.11 environment with a working CUDA compiler and driver,
the core installation can be reproduced from the repository root:

```sh
python -m pip install numpy==2.2.6 scipy==1.15.3 pycuda==2025.1.2 \
  cupy-cuda12x==13.6.0 batman-package==2.5.3 numba==0.67.0 \
  astropy==8.0.1 tqdm==4.70.0 pynvml==13.0.1 transitleastsquares==1.32
git clone https://github.com/Farthing-0/GTLS.git GTLS
git -C GTLS checkout 74e449c325792a763dde4fbffab98039c5e8c111
python -m pip install --no-deps -e . ./GTLS
python - <<'PY'
from importlib.util import find_spec
from pathlib import Path
import shutil

destination = Path(find_spec("gputls").origin).parent
for source in Path("GTLS/src/gputls").glob("*.cu"):
    shutil.copy2(source, destination / source.name)
PY
```

The final copy preserves the CUDA source files needed by this pinned native
package. Reproduction checks the installed GTLS source hashes against the
original seal. Put the CUDA compiler on `PATH` and set the loader's CUDA
library path as required by the local CUDA installation. Use the timing
runner's recorded thread settings for timing; the original main scientific
cohort used four Numba threads and one BLAS/OpenMP thread.

The archive preserves the original generator/protocol source identities.
These maintained runners include packaging and usability changes and therefore
need not have the same file hashes as the original runners. Production source
hashes and the exact input arrays are checked separately.

To audit physical input generation separately, use NumPy, SciPy and batman in
the [recorded generation environment](../results/tls_reference_2026-09-10/sources/README.md).
The original environment installs `batman-package==2.5.3`, whose module reports
version `2.5.1`. Regeneration checks every numerical hash and stops if values
differ; arbitrary platforms and dependency versions are not assumed to round
identically.

```sh
python benchmarks/tls_reference/cases.py --repo-root . \
  --replay-manifest benchmarks/results/tls_reference_2026-09-10/validation/input_manifest.json \
  --out regenerated-inputs
```

## Timing and interpretation

The [timing protocol](timing/README.md) specifies five single-source calls and
three 16-source batch repetitions per regime, using noise-only inputs to
represent searches in which transits are rare. It compares native worker
pools of 1, 2 and 4 with cuvarbase's complete public calls, and separately
measures search components and GTLS's extra output diagnostics. A configuration
must reproduce its frozen numerical outputs before its elapsed time can enter
a speed ratio.

Numerical equality establishes the same search decisions on tested inputs.
Small recovery cohorts do not measure a universal one- or two-percentage-point
population margin. A fixed SDE threshold is descriptive here; it is not a
common calibrated false-positive rate across cadences.

CPU tests for the comparison, timing accounting and publication checks:

```sh
python -m pytest -q benchmarks/tls_reference
```
