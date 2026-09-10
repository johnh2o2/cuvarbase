# Expected SNR retained by fast TLS

This 9 September 2026 diagnostic evaluates the earlier phase-binned TLS
engine, retained as `method='binned'`. “API defaults” below means its frozen
settings. See the [current transit report](../../../../docs/TRANSIT_BENCHMARKS.md)
for the observation-level default.

Phase binning has a small cost in many of these examples, but **1–2% is not a
universal upper bound**. This CPU diagnostic separates the fixed transit
template, its phase-bin approximation, and its epoch/duration grids. It
calculates expected white-noise SNR at the **true period**; it does not measure
detection recovery, false-positive rates, GTLS sensitivity, or speed.

The table uses the API defaults: automatic bins, `t0_oversample=3`, and 15
durations. “Bin loss” is the worst of 32 sampled phase offsets, relative to
the best unbinned fixed TLS template with the same fitted duration and epoch.
The box independently optimizes its duration and epoch; its loss is relative
to an oracle filter using the true physical signal.

| Earth-size planet; stellar mass/radius in solar units | Period | Impact parameter | Bins across transit | Bin cap reached | Below duration prior | Additional bin SNR loss | Optimized box SNR loss |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Sun | 10 d | 0 | 8.42 | No | No | 1.22% | 1.12% |
| Sun | 365.25 d | 0 | 6.12 | No | No | 2.07% | 1.12% |
| Sun | 365.25 d | 0.8 | 3.73 | No | No | 3.29% | 0.97% |
| Mass = radius = 0.1 | 365.25 d | 0 | 2.85 | Yes | No | 5.35% | 1.37% |
| Mass = radius = 0.1 | 365.25 d | 0.9 | 1.61 | Yes | No | 20.31% | 3.53% |
| Sun | 10 d | 0.95 | 2.84 | No | Yes | 5.16% | 1.13% |
| Sun, eccentricity 0.8 at periastron | 100 d | 0.5 | 2.10 | No | Yes | 10.24% | 1.10% |

These bin-loss values isolate binning; the full search can additionally lose
SNR through the duration prior, grid spacing, candidate selection and other
steps. A duration below the prior cannot be repaired by increasing the number
of bins alone. The long-period examples describe a sufficiently observed
signal; whether a survey samples it depends on its baseline and cadence.

All 19 regimes and three configurations are in [summary.csv](summary.csv).
The previous benchmark's `t0_oversample=4`, 16-duration configuration and its
8,192-bin, `t0_oversample=16`, 32-duration configuration are included separately.
Uniform configurations sample different integer bin indices and epoch phases:
their coarse-grid ranges are descriptive, **not paired comparisons of the same
ephemerides**. Observed-cadence configurations do share identical ephemerides.

## Model and interpretation

The physical signal comes from exposure-integrated `batman` models with known
unit baselines and quadratic limb darkening `[0.4804, 0.1867]`. The uniform
calculation uses 4,096 intervals per geometric transit and 64-point exposure
quadrature. It uses the pinned cuvarbase template and integral tables, with
float64 evaluation to isolate approximation errors from GPU roundoff.

The catalog includes central, high-impact, grazing, long-period, dense-star,
and eccentric shapes. Shared limb darkening is a controlled assumption, not
an atmosphere model for every stellar class. White-dwarf entries are shape
and resolution stress cases; their physical eclipse depths may violate the
production depth gate, which this amplitude-invariant diagnostic omits.

An additional 72 ephemerides use stored TESS/ZTF times, exposures and relative
errors, with synthetic flux and independent errors of
`0.001 * relative_error`. Thirteen have no sampled signal and are retained
without a retention ratio. This is not a completeness sample conditioned on
observability. Correlated noise, detrending, period errors and native SDE are
outside the calculation.

| CSV quantity | Meaning |
| --- | --- |
| `template_snr_over_oracle` | Best unbinned fixed-template fit relative to the true signal filter. |
| `box_snr_over_oracle` | Best box shape relative to the same oracle; duration and epoch are free. |
| `physical_projection_retention` | Maximum possible SNR retained by the stored bin sums, given the true signal. |
| `physical_binned_over_unbinned` | Actual bin-averaged TLS filter relative to its best unbinned fit. |
| `own_template_binned_over_unbinned` | Compression control injecting the same pointwise TLS template. |
| `coarse_grid_native_selected_over_oracle` | Actual SNR of the trial selected by the native expected-score objective at the true period. |

SNR uses each filter's **actual noise variance**. The coarse kernel's
`mean(T²)` normalization differs from the variance of its `mean(T)` filter;
`native_norm_over_noise` reports that distinction. An individual
`physical_binned_over_unbinned` ratio can exceed one when smoothing improves
a mismatched shape. It still obeys the compressed-oracle information bound.
Isolated bin, epoch and duration losses must not be added: jointly changing
them can change which trial wins. The full [tool documentation](../../../tls_accuracy/README.md)
defines every metric and assumption.

## Evidence and reproduction

[cases.csv](cases.csv) contains 2,014 rows: 1,824 uniform cases and 190
observed-cadence rows, including the 13 unsampled ephemerides. Each of the
59 sampled observed ephemerides has three configuration rows.
[manifest.json](manifest.json) records versions, parameters, source revision
and hashes; its file paths are relative to this directory. Historical source
snapshots preserve the exact diagnostic, tests, template/grid code and kernel
used to define the calculation.

[validation.json](validation.json) records 12 passing mathematical tests,
projection-bound checks, fitting-boundary checks and numerical convergence.
Doubling integration resolution and exposure quadrature changes the checked
SNR quantities by less than **0.0015 percentage points**.
[convergence.csv](convergence.csv) retains the refined run's comparison
columns, matched to the baseline by regime, configuration and offset index.
All packaged-file hashes are in [SHA256SUMS.json](SHA256SUMS.json).

Run from the repository root with NumPy, SciPy and `batman-package` installed:

```sh
python benchmarks/tls_accuracy/diagnose.py \
  --source-revision 11317fb0ff1b68af05ae3f67de5f298c9a90e46b \
  --cadences benchmarks/results/tls_sensitivity_2026-09-09/cadences \
  --out /tmp/cuvarbase-tls-accuracy
python -m pytest -q benchmarks/tls_accuracy/test_diagnose.py
```

For the uniform convergence run, omit `--cadences` and add
`--samples-per-transit 8192 --exposure-nodes 128`, choosing a different output
directory. The source revision is the benchmark's pre-optimization baseline;
this result archive does not establish the optimized kernel's timing or
numerical equivalence.
