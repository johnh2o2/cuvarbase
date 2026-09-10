# Binned TLS accuracy and efficiency tools (2026-09-09)

These tools characterize the earlier binned TLS engine, retained as
`method='binned'`. Their binning losses, kernel speedups and high-impact pilot
do not describe the standard observation-level TLS search. See the
[current benchmark and validation](../../docs/TRANSIT_BENCHMARKS.md).

`diagnose.py` measures expected signal-to-noise retention at the **true period**.
It separates template shape, phase compression, and coarse epoch/duration
sampling. It does not measure detection completeness, a false-positive rate,
native SDE, GTLS sensitivity, or execution-time speedups. The corresponding
historical complete-search study is in `../tls_sensitivity`.

The [2026-09-09 published diagnostic](../results/tls_accuracy_2026-09-09/accuracy/README.md)
contains the validated regime table, numerical outputs and provenance.

The sibling `kernel_benchmark.py` measures the effect of the CUDA optimization
on identical inputs and search settings. `high_impact.py` runs a focused
complete-search comparison for high-impact transits. Those GPU experiments
answer different questions from this CPU diagnostic.

Run on a CPU with NumPy, SciPy, and `batman-package` installed:

```sh
python benchmarks/tls_accuracy/diagnose.py \
  --source-revision 11317fb0ff1b68af05ae3f67de5f298c9a90e46b \
  --cadences benchmarks/results/tls_sensitivity_2026-09-09/cadences \
  --out /path/outside/the/repository/tls-accuracy
python -m pytest -q benchmarks/tls_accuracy/test_diagnose.py
```

The run writes per-case and summary CSVs, a manifest with parameters, versions
and hashes, and snapshots of the model/grid/kernel sources from the requested
Git revision. It deliberately does not put generated results in this source
directory. Default work is 19 physical regimes, three frozen search settings,
32 phase offsets, and 72 additional injections into three observed cadences.

## What is held fixed

The signal is a noiseless, exposure-integrated `batman` transit. Exposure
integration uses 64-point Gauss-Legendre quadrature; uniform sampling resolves
the geometric transit with 4,096 intervals. All arithmetic is float64, while
the template and its integrated tables are the actual float32 tables generated
by the pinned source. Period rounding, GPU folding roundoff and accumulation
roundoff are deliberately excluded. Both signals and templates use known unit
out-of-transit baselines and quadratic limb darkening `[0.4804, 0.1867]`.

The catalog includes Sun/Earth and Sun/Jupiter shapes, high impact parameters,
long periods, dense M dwarfs, and eccentric periastron transits. Its white-dwarf
examples are **shape and resolution stress cases**: the shared limb-darkening
law is a controlled assumption, not an atmosphere model for a white dwarf;
deep physical eclipses may also violate the production depth gate. The shape
ratios are amplitude invariant and do not apply that gate. Nothing in this
catalog establishes the occurrence rate or observability of these systems.

Observed cases use only the stored TESS/ZTF times, exposure times and relative
errors. Eight predetermined random epochs per shape are retained, including
epochs with no sampled signal. No observed flux, fitted detrending, correlated
noise, or real survey selection is modeled. The quoted SNR denominator uses
independent errors of `0.001 * relative_error`; kernel weights include their
`1e-10` regularizer. Signal amplitude cancels from every retention ratio.

## Reading the quantities

For a noiseless flux deficit `s`, filter `f`, independent errors `sigma`, and
kernel weights `w = 1 / (sigma**2 + 1e-10)`, the expected SNR is

```text
sum(w * s * f) / sqrt(sum(w**2 * sigma**2 * f**2)).
```

The oracle is `sqrt(sum(s**2 / sigma**2))`. The uniform calculation replaces
sums with integrals. The native coarse kernel instead divides its squared
numerator by `sum(B * mean(T**2))`. That is different from the actual variance
of its filter `mean(T)`. `native_norm_over_noise` quantifies the difference;
native score amplitudes must not be read as calibrated significance.

| Output | Meaning |
| --- | --- |
| `template_snr_over_oracle` | Best unbinned fixed-template fit, allowing duration and epoch to vary, relative to the true signal filter. |
| `box_snr_over_oracle` | Optimized box shape relative to the same oracle. Its duration is free, rather than fixed to the full transit width. |
| `physical_projection_retention` | Best possible filter of the stored bin sums, assuming the true signal shape is known. This bounds the information loss from summation, separately from TLS's approximate template weights. |
| `physical_binned_over_unbinned` | Additional compression effect, holding that best TLS template's epoch and width fixed. Can exceed one if smoothing improves a mismatched shape. |
| `own_template_binned_over_unbinned` | Compression control: inject the same pointwise TLS template at the true geometric width, then compare its bin-averaged filter. |
| `own_template_uncapped_retention` | Same control if the requested automatic resolution could exceed 8,192 bins; a hypothetical diagnostic, not a supported production setting. |
| `epoch_grid_only_retention` | Uniform cases: fix the template width to the true geometric duration and discretize epoch, relative to the same filter at the true center. |
| `duration_grid_only_retention` | Uniform cases: discretize width, keeping the best continuous epoch fixed. Catalog signals are symmetric around conjunction. |
| `coarse_grid_native_selected_over_oracle` | Uniform cases: at the true period, select epoch and duration with the native coarse expected-score objective, then evaluate that filter's actual SNR. |
| `coarse_grid_best_snr_over_oracle` | The largest actual SNR among those same coarse-grid trials. It distinguishes grid/compression loss from the native normalization's choice. |

Uniform cases cover every sub-bin offset at even spacing and vary the integer
phase-bin index independently, to sample the epoch grid as well. Sampled epoch
phases differ between configurations, so their coarse-grid ranges are
descriptive distributions, **not paired speed/accuracy comparisons on the same
ephemerides**. The observed-cadence cases use the same epoch in every setting.
Continuous TLS fits optimize width and epoch. The box fit optimizes the integrated signal
over both boundaries; observed-cadence boxes exhaust all contiguous intervals
whose endpoints have nonzero signal. These are oracle-assisted shape controls,
not a comparison to a particular BLS implementation.

The grid calculation searches all native epoch trials overlapping the
deterministic signal, including every allowed log-spaced duration. Other
epochs have zero expected numerator and cannot win this calculation. Real
noise has nonzero numerator everywhere, so this pruning is not a proposed
search optimization. No refinement, candidate pruning across periods, or
periodogram standardization is simulated.
The isolated epoch, duration and bin losses must not be added; their joint
effect can change which coarse trial wins.

`duration_below_prior`, `bin_cap`, and `epoch_cap` are separate flags. A narrow
transit excluded by the duration prior cannot be repaired by increasing the
number of bins alone. The bin cap assumes a GPU supporting the implementation
maximum of 8,192; a smaller device shared-memory limit can reduce it further.
`depth_gate_caveat` flags the compact-star stress cases, which are not a literal
production search simulation at their physical eclipse depth.

The mathematical tests check weighted projection, an analytic optimized box
fit to a trapezoid, exhaustive observed interval fitting, circular contact
geometry, bin caps, and the distinction between mean squares and squared means.
For publication, also rerun with finer integration and phase-offset grids and
compare the resulting retention estimates. A small loss of expected SNR is
not a bound on missed detections near a chosen threshold.

## GPU implementation check

`kernel_benchmark.py` compares identical lightcurves and search settings using
the original CUDA kernel and the optimized kernel. It alternates timed calls
after warming both variants, retains every repetition, and compares period
scores, candidates and SDE. Timing includes the public batch API's host work,
transfers, search, refinement and results; it excludes imports, compilation,
grid creation and disk I/O. It does not measure recovery equivalence with GTLS.

For the full-grid, fine-resolution ZTF comparison, on a CUDA installation:

```sh
git show 11317fb0ff1b68af05ae3f67de5f298c9a90e46b:cuvarbase/kernels/tls_fast.cu \
  > /tmp/cuvarbase-tls-baseline.cu
python benchmarks/tls_accuracy/kernel_benchmark.py \
  --inputs benchmarks/results/transit_2026-09-08/inputs \
  --profile ztf --sources 16 --period-limit 0 --reps 5 \
  --nbins 8192 --t0-oversample 16 --n-durations 32 \
  --baseline-kernel /tmp/cuvarbase-tls-baseline.cu \
  --out /tmp/cuvarbase-tls-kernel-ztf.json
```

Without `--baseline-kernel`, the reference is the current source compiled with
empty-bin traversal disabled. That is useful for development, but differs
from timing the archived original source. [Published validation and all
configurations](../results/tls_accuracy_2026-09-09/kernel/README.md).

## Focused high-impact recovery

`high_impact.py` implements the frozen high-impact TESS pilot in stages:
protocol freeze, input generation, calibration runs, threshold freeze,
held-out runs and analysis. Each GPU method/split runs in a separate process.
Held-out execution requires the frozen thresholds; all failures remain in the
results. Its scalar analysis can be reproduced without a GPU.

The [published pilot](../results/tls_accuracy_2026-09-09/high-impact/README.md)
contains the exact configurations, source pins, package versions and commands
for regenerating inputs or recomputing the analysis. Keep generated arrays
outside the repository. This focused experiment is too small for a tight
equivalence claim, and its operational API times are not a replacement for the
exclusive, repeated timing benchmark.
