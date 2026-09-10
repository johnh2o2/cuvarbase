# Selected-grid numerical stress tests

These 25 development fixtures test numerical behavior beyond the independent
full-grid population. **19/25** passed the complete corrected-GTLS/cuvarbase
comparison; **17/25** passed the untouched-GTLS comparison. Every original
outcome remains in `cases.csv` and `comparisons.json.gz`. The five expected
input/reference exceptions and one additional coarse-search discrepancy are
listed below; none is counted as an exact numerical success.

The period grids contain the true period and aliases. These are mathematical
stress tests, not blind recovery measurements or population-level sensitivity
bounds. A shared missed signal establishes numerical agreement only. SDE is
descriptive on these selected grids; no SDE threshold is a pass criterion for
the two stronger controls.

The fixtures include ordinary, high-impact and grazing transits, shortened
eccentric transits, dense M dwarfs, a compact star, heteroscedastic and correlated
noise, phase wrap and ties, large time origins, long periods, flat data and
invalid rows. Passband baselines are assumed removed and injected depths and
shapes are achromatic. Both APIs receive only `t`, `y` and `dy`.

## Long-period examples and fixed-noise controls

The four rows below use a 365.25-day signal on **77,888 observations** in eight
synthetic repetitions of an observed TESS campaign, spaced 180 days apart.
The baseline is 1,285.75675 days; four transit events are sampled. The solar
host has an approximately 7.98-hour transit and the 0.1-solar-mass/radius host
a 1.73-hour transit. Both use 97 selected periods spanning 0.6–730.5 days.

| Host | White-noise oracle SNR | Selected period, days | Selection relative to truth | Corrected GTLS / cuvarbase | True-grid residual rank |
| --- | ---: | ---: | --- | --- | ---: |
| Solar | 10 | 547.875 | 3:2 alias | Exact | 25 |
| Dense M dwarf | 10 | 223.718542 | Wrong period | Exact | 26 |
| Solar | 20 | 121.75 | 1:3 alias | Final results exact; coarse discrepancy | Unavailable from collected original arrays |
| Dense M dwarf | 20 | 365.244890 | Fundamental; drift 0.25 transit widths | Exact | 2 |

Untouched GTLS selects the same period in all four rows. The corrected-GTLS
SDEs are approximately 1.7620, 1.4813, 3.2434 and 3.6912, respectively. The
stronger controls preserve each original physical signal, period grid and
noise realization, halve the Gaussian **and** OU noise amplitudes, and halve
`dy`. The quoted oracle SNR excludes OU noise. These are two predeclared
controlled examples; they do not estimate blind long-period completeness.
The strict fundamental criterion is accumulated period drift no larger than
half the physical transit duration. Aliases are reported separately.

The solar SNR-20 run differs at one coarse residual (absolute difference
`3.9872248e-8`) and its winning start/width, changing the coarse normalized
power. Its complete **final** corrected-reference spectra, public fields and
final fit match cuvarbase. This failed the original strict gate and stopped
the execution before timing; its receipt is preserved as a failed exact
comparison. It is not silently replaced by a later run.

A separate [repeatability diagnostic](diagnostic/README.md) reproduces that
difference by changing only the shared float32 flux cumulative sum. Native
repeats can also change a depth-cutoff decision, mask and final SDE. Identical
intermediate inputs give identical native/fused window scores in the inspected
cases. All nine diagnostic runs retain the same selected alias and final fit;
the true period and alias tie for the minimum residual. These later repeats
do not turn the original failed comparison into a pass.

## Explicit exceptions and retained evidence

- Flat data: untouched GTLS returns period 0.6 with nonfinite SDE; the corrected
  reference raises a division-by-zero error. cuvarbase returns no candidate.
- Invalid rows: both native variants clean the data; cuvarbase explicitly
  rejects the supplied NaN. This tests API behavior rather than search parity.
- Three sparse/long-period inputs hit native zero-width cache construction
  failures. cuvarbase omits unrepresentable cache rows and completes; these
  are reference-unsupported extensions, not parity successes.

The fixture named `compact_star_native_extension` actually completes and
matches both references; its retained name does not imply a native failure.
`source_identities.json` preserves the different development source versions.
The [source notes](../sources/README.md) disclose the unretained historical
21-case generator version; its exact numerical inputs remain in the bank.

`array_digests.json.gz` preserves all 75 records' numerical identities and
compact derived metrics. Sixty-three available NPZ archives were independently
checked array by array. The M-control corrected archive was restored from its
byte-identical native archive after verifying the original no-op receipt and
expected container hash. The three solar-control output containers were not
collected; their original records and comparator hashes survive, and their
unknown truth-grid ranks are left unavailable. This collection loss is
separate from the observed coarse numerical discrepancy.

The invalid-row fixture makes each native variant generate 534,984 predicted
transit times across its extreme input time span. Those two oversized lists
are represented by their float64 shape and numerical hash, plus the original
JSON-value hash, under `transit_times_identity`. Their original record hashes
remain unchanged. This compact representation removes duplicate diagnostic
output; it does not change any comparison or outcome.

Restore inputs with [inputs.py](../../../tls_reference/README.md) using study
labels `selected_grid`, `long_period` or `stronger_controls` and the matching
original manifests in this directory. These fixtures have no independent
study seal; use the maintained per-case `validate.py run --replay --positive-origin` workflow.
The [source archive](../sources/README.md) includes the exact original search
instrumentation and the result packager. `files.json` hashes the original
compact stress files; `publication_files.json` additionally covers this README.

For example, restore and rerun the M-dwarf control using new output directories:

```sh
python benchmarks/tls_reference/inputs.py restore \
  --bank benchmarks/results/tls_reference_2026-09-10/inputs --study stronger_controls \
  --manifest benchmarks/results/tls_reference_2026-09-10/stress/stronger_controls_inputs.json \
  --out reproduced-controls
for backend in gtls gtls_corrected candidate; do
  python benchmarks/tls_reference/validate.py run \
    --case reproduced-controls/dense_long_mdwarf_snr20.npz \
    --backend "$backend" --engine-root . --replay --positive-origin \
    --out "reproduced-control-results/$backend"
done
python benchmarks/tls_reference/validate.py compare \
  --reference reproduced-control-results/gtls_corrected/record.json \
  --candidate reproduced-control-results/candidate/record.json \
  --threshold 8 --out reproduced-control-comparison.json
```

The threshold comparison is recorded for transparency; it is not a selected-grid
recovery acceptance criterion. A rerun is regression evidence on the original
inputs and does not create another independent scientific sample.
