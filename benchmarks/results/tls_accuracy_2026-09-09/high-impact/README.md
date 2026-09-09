# Recovery of short, high-impact transits

**GTLS recovered signals that cuvarbase's default TLS search missed in this
targeted experiment.** GTLS recovered 112/256 injected transits; the baseline
v1 defaults recovered 61/256. Widening v1's duration search raised recovery to
116/256. These results identify a meaningful limitation of the defaults;
they do not establish equivalent sensitivity between the widened search and
GTLS, or measure a new headline speedup.

The injections are Earth-size planets crossing near the limb of a Sun-like
star: impact parameter `b = 0.94–0.96`, periods 2–6 days, on the frozen dense
TESS 200-second cadence. They are high-impact **but not geometrically
grazing**: the planet still passes fully inside the stellar disk. Their
durations are only **31.0–36.7%** of the central-transit estimate used by v1;
the default duration grid starts at 50%. All four configurations were frozen
before generating the pilot's light curves.

| Search | Recovered / 256 | Recovery, with 95% Wilson interval | SNR 8 / 128 | SNR 10 / 128 | False positives / 256 |
| --- | ---: | --- | ---: | ---: | ---: |
| v1 defaults | 61 | 23.8% [19.0%, 29.4%] | 9 | 52 | 8 (3.1%) |
| v1 finer sampling, same duration window | 78 | 30.5% [25.2%, 36.4%] | 14 | 64 | 8 (3.1%) |
| v1 wider duration window | 116 | 45.3% [39.3%, 51.4%] | 29 | 87 | 8 (3.1%) |
| Public GTLS, `fast=True` | 112 | 43.8% [37.8%, 49.9%] | 30 | 82 | 7 (2.7%) |

“Recovered” requires both the correct primary period and a native score above
that method's independently calibrated threshold. The false-positive Wilson
intervals are [1.6%, 6.0%] for each v1 configuration and [1.3%, 5.5%] for GTLS.
The nominal calibration target was 5%; these finite held-out samples do not
prove identical false-positive rates. No complete light-curve search failed.

The light curves are paired across methods. Counts below show exactly which
methods detected different injected signals; differences are in percentage
points (pp).

| Left minus right | Left only | Right only | Both | Neither | Recovery difference, conservative 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| GTLS − v1 defaults | 56 | 5 | 56 | 139 | +19.9 pp [+11.4, +27.7] |
| v1 wider − v1 defaults | 56 | 1 | 60 | 139 | +21.5 pp [+13.9, +28.2] |
| v1 finer − v1 defaults | 19 | 2 | 59 | 176 | +6.6 pp [+1.1, +11.9] |
| v1 wider − GTLS | 16 | 12 | 100 | 128 | +1.6 pp [−5.2, +8.3] |
| v1 finer − GTLS | 8 | 42 | 70 | 136 | −13.3 pp [−21.1, −5.0] |

The wider search's interval relative to GTLS includes differences exceeding
five percentage points in either direction. **This pilot has not passed a
5 pp equivalence test.** All intervals are descriptive for this targeted
population and the frozen thresholds; comparisons are not adjusted together
as a family. Per-SNR intervals and paired false-positive counts are retained
in [analysis.json](analysis.json).

## What the configurations test

All v1 calls use the same pre-optimization numerical source,
[`11317fb`](https://github.com/johnh2o2/cuvarbase/tree/11317fb0ff1b68af05ae3f67de5f298c9a90e46b),
with a fixed normalized `batman` template (reference radius ratio 0.1),
`R_planet=1` Earth radius **for the duration prior**, solar stellar parameters,
`refine_top_k=50`, and `refine_oversample=33`. The later kernel optimization
is absent from this experiment.

| Configuration | Phase/epoch oversampling | Durations | Window relative to central estimate | Phase bins |
| --- | ---: | ---: | --- | --- |
| `v1_defaults` | 3 | 15 | [0.5, 2] | Automatic: 256–512 |
| `v1_fine` | 16 | 32 | [0.5, 2] | 8,192 |
| `v1_wide` | 3 | 25 | [0.1857492861, 2] | Automatic: 256–2,048 |

The wider grid uses `qmin_fac = 0.5 * 4**(-10/14)`: it retains the original
15 logarithmic widths and prepends ten shorter widths. Automatic bin counts
also increase when the minimum duration decreases. Thus the improvement
cannot be attributed exclusively to the duration prior or exclusively to
binning. The finer configuration changes bin, epoch and duration sampling
while retaining the original duration window. Its partial improvement does
not show that increasing bins alone fixes an excluded duration. The default
automatic bins do not reach the 8,192-bin cap in this short-period pilot.

GTLS uses pinned
[`74e449c`](https://github.com/Farthing-0/GTLS/tree/74e449c325792a763dde4fbffab98039c5e8c111),
`fast=True`, one worker, `T0_fit_margin=0.125`, and `duration_grid_step=1.1`.
Its accepted stellar bounds are in [configs/gtls.json](configs/gtls.json),
but the pinned implementation also uses internal host and CUDA duration
limits. Every GTLS row records its actual template, integer duration cache
and nominal CUDA width envelope. Every injected signal has a nominally
eligible cache width within 4.9% of its geometric duration in sample units.
This checks width coverage; it does not equate the template, sampled epochs,
depth estimate or score. `fast=True` returns before GTLS's subsequent
candidate refinement.

The result answers whether this particular native GTLS configuration can
find signals missed by default v1. It does not isolate phase-bin information
loss, benchmark canonical CPU TLS, or characterize all transiting planets.
The separate [expected-SNR diagnostic](../accuracy/README.md) isolates
several approximation costs at the true period.

## Frozen protocol and retained evidence

[design.json](design.json) was frozen at **19:46:34 UTC on 2026-09-09**.
It specifies 256 calibration nulls, 256 injections and 256 independent test
nulls, with seed `2026090943`. The base cadence has 9,736 samples over
25.7568 days in one band. Each case randomly drops 0–3% of samples and retains
at least five in-transit samples across two events. Every search receives
the full 3,084-period grid spanning 0.600289–12.878375 days.

Periods are log-uniform from 2–6 days, epochs uniform over a period, and
impact parameters uniform from 0.94–0.96. The radius ratio is 0.0092, orbits
are circular, and both the injected models and cuvarbase templates use
quadratic limb darkening `[0.4804, 0.1867]`. The injected `batman` model
integrates each 200-second exposure with seven subsamples.

The noise scale sets the weighted-centered latent signal's **oracle
white-noise SNR** to 8 or 10, balanced within each split. This is an input
definition, not the algorithms' reported SNR or SDE. Nulls use the same
latent-signal noise-scale recipe. Independent Gaussian noise is supplemented
by an Ornstein–Uhlenbeck process with amplitude 0.25 times the median error
and correlation time 0.15 days. The generator supports unequal relative
errors, but this frozen cadence has constant relative error 3: white errors
are therefore equal within each light curve. Flux has a known unit baseline;
detrending and real stellar variability are outside this experiment.

For each method, calibration sorts 256 null scores and freezes the
zero-based order statistic 243, the “higher” 95th percentile. Detection
requires a score **strictly above** this threshold. Period recovery requires
`abs(P_found/P_true - 1) * observed_baseline <= 0.5 * true_duration`;
harmonics do not count. The frozen runner retains failed cases as injection
misses or null scores of minus infinity.

All calibrations completed by **20:05:16 UTC**. [thresholds.json](thresholds.json)
was frozen at **20:05:21 UTC**, and the first held-out run started at
**20:05:22 UTC**. The runner requires thresholds before opening held-out
arrays and refuses calibration if held-out results already exist. Every
held-out record contains the threshold-file hash. These recorded gates,
timestamps and hashes support the chronology; they are not an external
attestation of operator actions.

The conservative paired interval subtracts confidence limits for the two
discordant-cell probabilities. Four one-sided exact binomial bounds, each
with tail probability 0.0125, give at least 95% coverage by Bonferroni.
[validation.json](validation.json) independently checks every count and
recovery flag, Wilson intervals, paired intervals, numerical source pins,
threshold chronology, and the privately retained input-array hashes.

The full grid was supplied to every method, but native outputs can contain
masked or nonfinite trial entries. GTLS deliberately masks residuals above
100 times the median before forming its spectrum
([pinned core.py](https://github.com/Farthing-0/GTLS/blob/74e449c325792a763dde4fbffab98039c5e8c111/src/gputls/core.py#L890)).
Such entries occur in 36/256 GTLS injection spectra and 2/256 default-v1
injection spectra; every light curve still returns a valid result and stays
in the analysis. The validation receipt records all splits. Raw residuals
are unavailable here, so individual excluded trials cannot be diagnosed
from this compact archive. GTLS's native cleaner also drops the `t=0`
sample in 251/256 cases per split, a difference of one sample when present.
The sole retained warning concerns an unclosed baseline CUDA source file;
there were no template-fallback warnings.

The archive preserves the original scalar results under [results/](results/),
the [input manifest](inputs/manifest.json), four configurations, thresholds,
analysis, [generation receipt](generation-environment.json), and the exact
[runner snapshot](source_snapshots/high_impact.py). Generated light curves
(approximately 90 MB) remain outside the release repository and were
independently hash-checked before publication. Full periodograms were not
retained; their hashes are recorded. [provenance.json](provenance.json) and
[SHA256SUMS.json](SHA256SUMS.json) bind the public artifacts.

## Environment and timing scope

The [search environment receipt](search-environment.json) records one NVIDIA
A40, driver 570.211.01, with Python
3.11.10. Recorded package versions are NumPy 2.2.6, SciPy 1.15.3,
`batman-package` 2.5.3, PyCUDA 2025.1.2, and `cupy-cuda12x` 13.6.0.
Distribution metadata reports cuvarbase 1.0.0 and GTLS 0.5.1; the verified
source hashes and commits identify the numerical code actually used.
The CPU quota was 7.65 cores; OpenMP/OpenBLAS/MKL threads were one and
Numba threads four. Inputs were generated separately on macOS with Python
3.9.6, NumPy 1.26.4, SciPy 1.12.0 and `batman-package` 2.5.3.

`operational_api_seconds` in the analysis retains synchronized public-API
elapsed time: cuvarbase batches of 16, GTLS one light curve at a time. These
are single passes with first-call compilation included and no excluded
warmup. They exclude result serialization and input verification. They are
operational accounting, **not a controlled timing comparison or a new
equivalent-sensitivity speedup**.

## Reproduce the scalar analysis without a GPU

Run from the repository root with NumPy and SciPy installed. Use a fresh
scratch root because the frozen runner refuses to overwrite an existing
analysis if last-bit floating-point values differ across environments.
The exact snapshot is required: the design verifies its SHA-256.

```sh
python - <<'PY'
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

published = Path('benchmarks/results/tls_accuracy_2026-09-09/high-impact').resolve()
scratch = Path(tempfile.mkdtemp(prefix='cuvarbase-tls-reanalysis-'))
for name in ('design.json', 'thresholds.json'):
    shutil.copyfile(published / name, scratch / name)
for name in ('configs', 'inputs', 'results'):
    shutil.copytree(published / name, scratch / name)
subprocess.run([sys.executable, str(published / 'source_snapshots/high_impact.py'),
                'analyze', '--root', str(scratch)], check=True)

def compare(a, b):
    assert type(a) is type(b)
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y)
    elif isinstance(a, float):
        assert abs(a - b) <= 1e-12
    else:
        assert a == b

compare(json.loads((published / 'analysis.json').read_text()),
        json.loads((scratch / 'analysis.json').read_text()))
print('Verified; regenerated analysis:', scratch / 'analysis.json')
PY
```

[reanalysis-validation.json](reanalysis-validation.json) records a successful
CPU replay with Python 3.9.6, NumPy 1.26.4 and SciPy 1.12.0. All counts,
strings, flags and hashes match exactly; the largest floating-point
difference is below `7e-18`.

## Regenerate and rerun the experiment

Use the frozen generation versions above for the closest input reproduction.
Generate into an empty directory after copying only the published design
and configurations:

```sh
export TLS_PILOT_RELEASE="$PWD/benchmarks/results/tls_accuracy_2026-09-09/high-impact"
export TLS_PILOT_RUN="$(mktemp -d)"
cp "$TLS_PILOT_RELEASE/design.json" "$TLS_PILOT_RUN/design.json"
cp -R "$TLS_PILOT_RELEASE/configs" "$TLS_PILOT_RUN/configs"
python "$TLS_PILOT_RELEASE/source_snapshots/high_impact.py" generate \
  --root "$TLS_PILOT_RUN" \
  --cadence benchmarks/results/tls_sensitivity_2026-09-09/cadences/tess_200s.npz
```

Generation creates a new timestamped manifest, so its hash need not equal
the historical manifest even if the case arrays match. Regeneration on a
different numerical stack may also change model values. Keep the newly
generated inputs, manifest, thresholds and results together.

For searches, use a Linux CUDA host with the recorded Python 3.11 search
packages. The earlier [search dependency pins](../../tls_sensitivity_2026-09-09/requirements-search.txt)
provide the compatible stack. Install cuvarbase from a checkout of
`11317fb0ff1b68af05ae3f67de5f298c9a90e46b` and GTLS from the frozen
[source archive](../../tls_profile_2026-09-08/sources/gtls-head.tar).
The GTLS source installation may omit `.cu` resources; copy the unmodified
`src/gputls/*.cu` files into its installed package directory. The runner
checks 37 cuvarbase and 19 GTLS numerical source files and stops if any
differs. Set `TLS_PILOT_BASELINE` to the absolute baseline checkout path.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=4
export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUTF8=1
for method in v1_defaults v1_wide v1_fine gtls; do
  python "$TLS_PILOT_RELEASE/source_snapshots/high_impact.py" run \
    --root "$TLS_PILOT_RUN" --source-root "$TLS_PILOT_BASELINE" \
    --method "$method" --split calibration
done
python "$TLS_PILOT_RELEASE/source_snapshots/high_impact.py" calibrate \
  --root "$TLS_PILOT_RUN"
for method in v1_defaults v1_wide v1_fine gtls; do
  for split in injections nulls; do
    python "$TLS_PILOT_RELEASE/source_snapshots/high_impact.py" run \
      --root "$TLS_PILOT_RUN" --source-root "$TLS_PILOT_BASELINE" \
      --method "$method" --split "$split"
  done
done
python "$TLS_PILOT_RELEASE/source_snapshots/high_impact.py" analyze \
  --root "$TLS_PILOT_RUN"
```

GPU roundoff and GTLS's memory-dependent period chunks can affect individual
scores on another machine. Preserve reruns as new linked results with their
own frozen thresholds, keeping the published measurements intact.
