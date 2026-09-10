# cuvarbase TLS and GTLS

The standard cuvarbase TLS engine uses the numerical search of [pinned public GTLS](https://github.com/Farthing-0/GTLS/tree/74e449c325792a763dde4fbffab98039c5e8c111), including full refinement. The comparison now asks whether an optimized implementation produces the same search results. [Measurements and validation](TRANSIT_BENCHMARKS.md).

| Stage | Standard cuvarbase TLS | Pinned GTLS, full mode |
| --- | --- | --- |
| Data | Individual observations | Individual observations |
| Template/cache | Native GTLS template samples and overshoot | Same |
| Duration/epoch trials | Native broad domain and sample-window trials | Same logical policy; physical grouping depends on available memory |
| Depth/residual calculation | Native arithmetic, with repeated work removed | Original GPU kernels |
| Candidate selection | Native top-candidate/harmonic policy; finite entries ranked before refinement | Masked entries can enter the first candidate list as NaN |
| Refinement | Every sample start, including native full-stage residual arithmetic | Same |
| Flux cumulative sums | Replayed native row scans | Python-dispatched native row scans |
| Residual storage | Tile winners, reduced on device | Full duration-by-epoch residual tensor |
| Output work | cuvarbase result contract | Additional native SNR/pink-noise diagnostics |

cuvarbase fixes logical duration groups independently of physical workspace size. The validation records the native group policy as well as comparing the production default. Concurrent GTLS workers can change available memory and therefore its groups; throughput settings only qualify for the strict numerical comparison when their complete outputs still match the single-worker reference.

Both packages receive the same float64 positive-origin timestamps, flux, uncertainties and full period grid. cuvarbase restores epochs to the caller's original time system. Valid zero/negative input times are preserved. It rejects malformed input before GPU work and omits unrepresentable zero-sample cache rows in cases where native GTLS otherwise fails; successful native cases retain their usable cache rows.

Automatic grids retain the requested period domain even when it contains fewer than 100 periods; pinned GTLS can silently reset a small grid to default solar-host bounds. cuvarbase also validates the automatic grid's stellar range instead of silently clamping it. The [API guide](source/tls.rst) gives those bounds and the explicit-period alternative. These policies are intentional input-handling differences; the benchmark supplies identical period arrays.

Whole-spectrum agreement is stronger than agreement of a single SDE or a pooled recovery percentage. The tests compare residuals, masks, candidate/harmonic ranks, refinements and final selections before timing. The independent injection/null checks report each astrophysical regime separately. They do not establish that either implementation detects every possible transit or that a fixed SDE has a universal false-alarm rate.

The main study and separate null supplement give **184 exact corrected-reference comparisons**. The [long-control diagnostic](../benchmarks/results/tls_reference_2026-09-10/stress/diagnostic/README.md) additionally shows that the shared float32 prefix scans can vary across runs, changing depth gates, masks and SDE. Identical saved intermediate arrays produce identical native and fused window scores; the original strict stress failure remains recorded. Replaying native operations therefore does not guarantee identical floating-point outputs for every input and execution.

## Invalid-candidate correction

Validation found a defect in pinned GTLS's host candidate selection. Sorting masked scores can place masked periods in its first refinement list; converting that list to a GPU array turns those periods into NaN. The GPU then performs invalid integer conversions and can return finite scores for these nonexistent trial periods. Assigning them back into the spectrum clears their masks and changes its normalization and potentially its selected period.

cuvarbase excludes masked or nonfinite period/score entries **before** sorting, preserving the native top-100 and next-100-above-one-day policy among valid entries. Harmonics remain real trial periods: a full search can legitimately fit one that the coarse stage had masked. A newly valid winner uses its finite period value.

The validation separates untouched public GTLS from a separately recorded native source with this host-mask correction. Exact differential checks use the corrected source; recovery comparisons retain the untouched implementation's outcomes. The correction changes neither the transit template nor its resolution. It is not a speed optimization, and the benchmark competitor remains the public package.

## Interpreting the speedup

Full public-call times measure the cost a user pays. A separate common search boundary ends after the final GPU winner is selected, before physical-parameter and noise diagnostics. Native GTLS computes extra pink-noise SNR diagnostics that cuvarbase does not return; their cost is shown separately and is not described as a faster search kernel.

The optimization retains observation-level information. It removes repeated arithmetic, large intermediate allocations and per-row Python dispatch. The [numerical explanation](TLS_NUMERICS.md) describes why an ordinary vectorized cumulative sum would not be a safe numerical substitution.

## Historical comparisons

The [September 9 sensitivity study](../benchmarks/results/tls_sensitivity_2026-09-09/README.md) compared cuvarbase's earlier binned engine against GTLS `fast=True`. Its bounded population results and large speed ratios remain archived, but do not describe this default full-to-full comparison. The [subsequent accuracy audit](../benchmarks/results/tls_accuracy_2026-09-09/README.md) found narrow-transit losses that motivated replacing the default numerical strategy.

The [provenance audit](BENCHMARK_PROVENANCE.md) records retired headlines and earlier CPU TLS failures. Failed calls are never successful timing denominators. Actual PyPI cuvarbase 0.2.5 contains no TLS implementation.
