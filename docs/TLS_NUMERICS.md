# TLS shape, phase bins and detection accuracy

cuvarbase's fast TLS search retains a limb-darkened transit shape. Phase binning approximates where observations fall along that shape. Coarse bins can erase some of the information that distinguishes a transit from a box, but binning does not itself change the template into BLS.

![The cuvarbase transit template, two phase-bin resolutions and a box](figures/tls_phase_binning.png)

This example has a five-day period and a nominal 3.11-hour transit. The current benchmark settings use 512 phase bins: 14.1 minutes per bin, or about 13 bins across the transit. The rounded bottom remains visible; the ingress and egress are coarsened. At 4,096 bins, each bin spans 1.76 minutes. The vertical scale is normalized to the transit depth.

## Where this differs from canonical TLS

The [original TLS paper](https://doi.org/10.1051/0004-6361/201834672) searches unbinned, phase-folded observations using a transit-shaped template. cuvarbase's fast engine uses two stages:

1. For each trial period, fold the observations into weighted phase bins. Search transit durations and epochs using integrated template lookup tables, and solve for depth analytically.
2. Refine the selected candidate periods against individual observations. The benchmark uses the public top-50 refinement.

The first stage saves repeated observation-level work. Its approximation loses the individual positions and flux variation inside each bin. Integrating the template over a bin reduces discretization error; it does not recover that missing information. The kernel averages both the template and its square, which also differs from evaluating the model at each observation's actual phase.

Refinement improves the selected candidates. It cannot recover a period excluded by the coarse search, and the reported SDE still comes from the coarse spectrum. This is why accurate fitted parameters alone do not demonstrate accurate detection sensitivity.

The fast engine also uses its own epoch and duration grids, a fixed fiducial transit shape scaled in duration and depth, and its own ranking/refinement implementation. The analytic depth solution is valid for the chosen least-squares model; phase compression and search sampling are separate approximations. cuvarbase TLS, public GTLS and the canonical CPU package are related searches, with different numerical implementations.

## How fine are the benchmark bins?

With epoch oversampling 4, the automatic rule requests at least four bins across the shortest allowed transit, rounds upward to a power of two, and imposes a 256-bin floor. These cadence examples use 256–1,024 bins over the entire phase cycle. Typical central transits span roughly 8–16 bins; the shortest searched durations generally span 4–8, with more at short periods because of the floor. The epoch grid is a separate control, stepping by approximately one quarter of the tested duration here.

These are explicit benchmark settings. The public API defaults to epoch oversampling 3; reproducing the study requires the recorded configuration rather than an unspecified default call.

A small number of bins across ingress does not imply an equally large loss of total detection SNR: the broad transit bottom also carries signal. However, ingress-sensitive measurements, narrow transits and marginal detections can be more demanding than this average picture.

## What the isolated binning check shows

We evaluated the actual cuvarbase template at the known period, epoch and duration of 384 earlier synthetic injections on the observed TESS and ZTF cadences. We compared pointwise and bin-averaged filters using their actual white-noise variance. This isolates compression; it does not search for a period or estimate a recovery rate.

| Cadence | Median SNR loss, current bins | 95th-percentile loss | Largest observed loss | Largest loss at 4,096 bins |
|---|---:|---:|---:|---:|
| TESS, one dense sector | 0.33% | 1.02% | 1.64% | 0.09% |
| TESS, separated sectors | 0.19% | 0.65% | 1.09% | 0.13% |
| ZTF, sparse g/r | 0.29% | 1.09% | 2.05% | 0.20% |

These values support small binning losses for the tested shapes and cadences at known ephemerides. They do not bound losses from the complete search, correlated noise, threshold calibration or a different population of transits. Smoothing occasionally improves the match to an injected shape that differs from the fiducial template; such negative measured losses do not imply information was created.

Finer bins, closer epoch steps and more durations all cost computation. The independent sensitivity study uses these three configurations, each retaining top-50 observation-level refinement:

| Configuration | Phase bins | Epoch oversampling | Durations |
|---|---:|---:|---:|
| Original benchmark | Automatic, 256–1,024 here | 4 | 16 |
| Intermediate | 4,096 | 8 | 16 |
| Fine diagnostic reference | 8,192 | 16 | 32 |

These are complete-search resolution alternatives: they change more than binning alone. Each receives independent null calibration, followed by recovery and false-positive tests on new lightcurves. The fine setting is a convergence reference, not an exact unbinned implementation.

The completed independent experiment adds a net 29, 35 and 6 detections out of 2,048 when moving from original to fine sampling on dense TESS, separated TESS and ZTF: about 1.4, 1.7 and 0.3 percentage points. These are changes to the complete search, not binning alone. All three settings meet the predeclared recovery-loss bound against GTLS on all three cadences. The joint recovery / false-positive matching criterion passes for the fine dense-TESS setting and every separated-TESS setting; ZTF remains inconclusive because the false-positive difference is not constrained tightly enough. This does not show that the original grid is inadequate. [Full results, resolution costs and the secondary BLS control](../benchmarks/results/tls_sensitivity_2026-09-09/README.md).

The secondary BLS control gives another useful check: original-grid TLS detects 881 versus 765 injections on dense TESS, 1,137 versus 1,142 on separated TESS, and 1,620 versus 1,532 on ZTF, out of 2,048 each. Observed false-positive rates differ by less than 0.2 percentage points. The binned transit search therefore retains distinct detection behavior on these cases. This one fixed BLS configuration does not isolate the template shape or establish a universal TLS advantage; it uses a different ranker and numerical search.

The [per-injection binning measurements](../benchmarks/results/tls_sensitivity_2026-09-09/binning-diagnostic.csv), [diagnostic tool](../benchmarks/tls_sensitivity/binning.py) and [figure generator](../benchmarks/tls_sensitivity/plot_binning.py) are retained. Numerical implementation: [host search](../cuvarbase/tls.py), [fast kernel](../cuvarbase/kernels/tls_fast.cu) and [template tables](../cuvarbase/tls_models.py).
