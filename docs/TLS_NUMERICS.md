# TLS numerical accuracy

The standard TLS engine evaluates individual observations using the public GTLS numerical objective. **It does not phase-bin the data.** The broad native duration domain and full candidate/harmonic refinement are automatic; thin transits do not require a separate accuracy preset. [Current benchmark and validation](TRANSIT_BENCHMARKS.md).

## What is preserved

The implementation retains GTLS's transit-template cache, integer sample-window widths, epoch trials, depth estimate, native row-wise cumulative-sum operations, spectrum normalization and full refinement. Candidate ranking first excludes masked/nonfinite entries, correcting a [native host-mask defect](GTLS_COMPARISON.md#invalid-candidate-correction). It also preserves the different residual arithmetic used by GTLS's coarse and refinement stages. Simply reusing the coarse kernel at a finer stride would not reproduce those results.

Logical duration groups are fixed independently of physical GPU workspace chunks. A smaller workspace processes the same trials in smaller pieces. Explicit `qmin`/`qmax` overrides remain per-period bounds, including during refinement.

Times are shifted in float64 to a common positive origin so zero/negative relative timestamps remain usable. Comparison runs give GTLS the same shifted input. Returned epochs are restored to the caller's time system. This avoids native GTLS's input-cleaning rule that otherwise drops nonpositive timestamps.

## Why it is faster

The kernels reuse repeated residual calculations and reduce winning trials on the GPU instead of storing the entire duration-by-epoch residual tensor. Reusable CUDA graphs replay the original row-wise cumulative sums; they remove Python dispatch overhead while preserving the original scan arithmetic. An ordinary matrix-axis cumulative sum would change rounding and some threshold decisions, so it is not used for the default flux prefix.

Physical workspaces and retained scan plans are bounded. These changes affect execution, not the template, trial set or detection rule. [Implementation comparison](GTLS_COMPARISON.md).

## What “the same sensitivity” means

Whole-spectrum and refinement comparisons check numerical equivalence before comparing detections or timing. Exact agreement is against GTLS with the disclosed host-mask correction; untouched GTLS outcomes and any changed decisions are reported separately. Paired independent injections include ordinary and thin-transit regimes; noise-only cases check decision agreement. Equality of the complete spectrum implies the same threshold decisions on those inputs without tuning two separate thresholds to match aggregate recovery.

All **184 independent inputs** in the main study and separate null supplement matched the corrected reference exactly. A selected-grid stress test with 77,888 observations exposed a shared numerical limit: long float32 cumulative sums can vary across GPU executions. Changing only those sums reproduced the differing depth gates and coarse winner, while native and fused kernels gave bitwise-identical scores on identical intermediate arrays. Native repeats also changed masks and final SDE; all repeated searches selected the same period. Graph replay can vary too. The [retained diagnostic](../benchmarks/results/tls_reference_2026-09-10/stress/diagnostic/README.md) preserves the original failed exact comparison. This behavior is consistent with [NVIDIA's documented floating-point scan variability](https://github.com/NVIDIA/cccl/blob/v2.3.2/cub/cub/device/device_scan.cuh); the default does not promise universal bitwise repeatability.

This is a GTLS-compatible numerical search, not an exposure-integrated physical oracle. It retains GTLS's sample-index template approximation on irregular cadences and its finite search domain. Neither implementation can recover an unsampled transit or promise detection at arbitrary noise levels. SNR in cuvarbase remains `sqrt(delta chi2)` in input-error units; it is not GTLS's differently defined reported SNR.

## The explicit binned option

`method='binned'` preserves the earlier fast engine. Its weighted phase bins retain a transit-shaped template, but compress observations within each bin. Its duration prior and epoch grid also differ from GTLS. Candidate refinement cannot rescue every period missed by that coarse search.

The [September 9 audit](../benchmarks/results/tls_accuracy_2026-09-09/README.md) measured meaningful losses for narrow transits, including regimes beyond a universal 1–2% SNR-loss claim. Those findings motivated the new default. The old bin-cap discussion, high-impact pilot and large binned-versus-GTLS timing ratios remain reproducible historical results; they do not describe the standard observation-level engine.
