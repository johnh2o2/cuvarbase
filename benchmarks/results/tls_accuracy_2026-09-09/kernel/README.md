# TLS kernel efficiency at fixed search settings

These 9 September 2026 measurements concern the earlier phase-binned TLS
engine, retained as `method='binned'`. API defaults below refer to that
engine; the [current transit report](../../../../docs/TRANSIT_BENCHMARKS.md)
covers the observation-level default. The timings below remain historical
measurements of this specific kernel optimization.

The optimized search skips empty phase-bin runs when the lightcurve has fewer
than one observation per four bins. It retains the reference kernel's
successive float32 coordinate additions, weighted histogram, template,
duration/epoch grids, score normalization, and exact candidate refinement.
Denser lightcurves use the original traversal.

The table compares the frozen pre-optimization CUDA kernel from commit
`11317fb0ff1b68af05ae3f67de5f298c9a90e46b` with the optimized kernel on the same A40.
Both run through the same Python host code. These are batch API measurements:
16 fixed lightcurves (eight injections and eight nulls), the complete period
grid, and five alternating paired repetitions after both variants warm.
Times are median batch-call durations divided by 16. Source preparation on the
host, allocations, transfers, synchronization, GPU search/refinement, and host
results are included; imports, compilation, context setup, grid creation,
synthetic data generation, and disk I/O are excluded.

| Cadence | Settings | Reference seconds/source | Optimized seconds/source | Speedup |
|---|---|---:|---:|---:|
| tess_200s | fine | 0.038244 | 0.038033 | 1.006× |
| tess_200s | original | 0.001621 | 0.001604 | 1.011× |
| tess_gap | fine | 1.203128 | 1.204362 | 0.999× |
| tess_gap | original | 0.025036 | 0.025594 | 0.978× |
| ztf | fine | 3.759983 | 2.899791 | 1.297× |
| ztf | original | 0.061261 | 0.060357 | 1.015× |

The fine ZTF batch uses about 23% less time (1.30× faster). Fine TESS timings
are effectively unchanged. At the original settings the differences are
small relative to observed call-to-call timing variation; individual JSONs
retain all five paired measurements, and `summary.json` gives their ranges.

`fine` uses 8,192 bins, epoch oversampling 16, and 32 durations.
`original` means the earlier benchmark settings: automatic bins, epoch
oversampling 4, and 16 durations. These are explicit benchmark settings;
the public API defaults are epoch oversampling 3 and 15 durations. Both
settings refine 50 candidates with epoch oversampling 33.

For numerical parity, 128 existing lightcurves per cadence (64 injections and
64 nulls) were searched twice with each kernel at the fine settings. TESS200s
used all 3,084 trial periods; gapped TESS and ZTF used 4,096 evenly selected
periods each. Full-grid timing calls also retained numerical comparisons.
The timing sources are subsets of these cohorts: 384 distinct lightcurves,
not an expanded recovery sample.

Across the recorded comparisons, primary periods and valid-period masks all
matched. The maximum relative delta-chi-squared difference was
1.14e-06; the maximum absolute native SDE
difference was 1.87e-05. The relative-score
comparison divides by `max(abs(reference delta-chi-squared), 1)`. Full counts,
any changes in coarse best-fit parameters, and reference-versus-reference repeated
run differences are in `summary.json`. Eleven coarse epoch/duration choices
changed only at the original settings, where sparse traversal is disabled;
the reference also changed one coarse choice between repeated runs. Fine ZTF
had no coarse changes. The score differences are comparable to the
reference's own repeated-run variation; exact tie identity was not recorded.
All 169 TLS tests passed; `tests.log`
includes the warnings. A compact synthetic regression checks the numerical
sensitivity of template-tail integral subtraction.

This is an engineering runtime and numerical-parity check using the earlier
[transit benchmark inputs](../../transit_2026-09-08/inputs/). It does not
establish new recovery-rate or false-positive equivalence, or reduce the
scientific approximation from phase binning.

To reproduce one full-grid timing on a CUDA machine with the package and its
TLS dependencies installed, run from the repository root:

```sh
python benchmarks/tls_accuracy/kernel_benchmark.py \
  --inputs benchmarks/results/transit_2026-09-08/inputs \
  --baseline-kernel benchmarks/results/tls_accuracy_2026-09-09/kernel/baseline_tls_fast.cu \
  --profile ztf --sources 16 --period-limit 0 --reps 5 \
  --nbins 8192 --t0-oversample 16 --n-durations 32 \
  --out ztf_fine.json
```

For the original benchmark settings use `--nbins 0 --t0-oversample 4
--n-durations 16`. For a parity cohort use the fine settings with
`--sources 128 --period-limit 4096 --reps 2`. Individual JSONs retain complete
configuration, input-array, grid, source, package, hardware, and timing
provenance. `grid-verification.json` confirms that the recorded input grids
reconstruct exactly in the recorded Linux/NumPy environment; float64
transcendental results can differ by platform. `source-hashes.json` identifies
the tested implementations and
`input-verification.json` checks the published inputs and installed sources
against the run records. `SHA256SUMS.json` inventories these compact evidence
files.
