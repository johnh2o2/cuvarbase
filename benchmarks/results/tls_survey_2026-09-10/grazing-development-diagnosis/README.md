# Eight grazing, smeared development transits

This post-freeze diagnosis explains a development warning, not final detection
performance. All eight predeclared development cases in this regime are
included. The eight saved TLS peaks all miss the declared period criterion;
the selected BLS method/ranker recovers cases 1, 3, 5, 6 and 7. These 0/8 and
5/8 are **period-recovery counts**, not newly evaluated thresholded detections
at a common false-positive rate. No search was
rerun, no calibration or held-out outcomes were inspected, and no scientific
sources, settings, thresholds, tolerances, inputs or plans were changed.

The evidence points to the native absolute-depth gate interacting with exposure
smearing and sample-window averaging. It does **not** establish a complete
causal explanation of the actual GPU decisions. The numerical treatment of
very small deficits and the native candidate/ranking procedure remain unresolved.

## Input and calculation provenance

All eight original cloud-development NPZs were downloaded without computation
on the rental. Their bytes match the original manifest
`a1d18d6cf2d09fc4450a6f4ce9cf6f794e05685f65c755bbac5e7429c3116328`.
The older local development cohort was not substituted. Inputs are retained in
`inputs/`; `diagnosis.json` pins every input and the existing development
search/SNR receipts. `diagnose_eight.py` contains the bounded CPU calculation;
`eight-cases.csv` is a compact table. `run-manifest.json` records the exact
command, environment, source/receipt hashes, eight input hashes and output
hashes. The eight NPZs remain outside git; an archived development bank can
supply their original bytes for reproduction.

Run with an environment containing NumPy, SciPy and batman-package:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python diagnose_eight.py --repo /path/to/cuvarbase --study /path/to/study \
  --manifest /path/to/original/development/manifest.json \
  --inputs /path/to/original/eight/npzs --output /path/to/diagnostic-output
```

The study folder supplies `scientific/development-snr-final.json`,
`scientific/bls-response-final.json`, and the four
`scientific/dev-promoted-search-{0,1,2,3}.json` receipts. The manifest's fixed
SHA guard prevents substitution of an older or newly generated cohort.

The new CPU calculations reproduce the saved known-period native-family and
ideal-box expected white SNR to within 1e-8 absolute. That is an explanatory
cross-check, not an amendment to any scientific acceptance tolerance. The
physical signal, errors, noisy flux and period grid come from the original
NPZs. Only intrinsic/smeared midpoint depths are newly evaluated with the
unchanged physical model and 64 exposure quadrature nodes.

## What the physical inputs establish

These are Earth-size planets around a solar star, with impact parameters
0.99905–1.00215 and no full-transit interval. Geometric contact durations are
18.54–24.69 minutes. Every observation has a **30-minute exposure**, with
every ninth timestamp retained from the 200-second TESS cadence. This is one
fixed exposure length, not a mixed-exposure population. There are 1,082 samples,
7–17 nonzero transit samples and 4–11 observed events per case.

Exposure integration reduces the midpoint depth to 36.9–49.4% of the intrinsic
midpoint depth. Stored sampled peaks are only 6.65–11.42 ppm, although the
unintegrated midpoint depths are 14.30–23.77 ppm. Error amplitudes were scaled
by the existing generator to produce target white signal norms 6/8/10/12;
median errors are 1.62–4.55 ppm. These are controlled synthetic detectability
cases, not estimates of occurrence or realistic TESS population recovery.

All eight have a supplied period within the frozen recovery tolerance. Their
nearest-grid drift is 0.010–0.301 of the allowed half-duration. Therefore a
structurally unreachable period grid does not explain these eight misses.

## The absolute floor is on a window mean

The unchanged native kernel requires an **unweighted sample-window mean
deficit strictly greater than 10 ppm**, before applying template overshoot
scaling. This is neither the physical peak depth nor the reported fitted
depth. See `tls_ref_mean_depth`, `tls_ref_window`, and `tls_ref_full_window` in
`cuvarbase/kernels/tls_reference.cu`; the default is in
`cuvarbase/tls_reference.py:raw_search`. Flux is checked but not renormalized by
`tls_reference_math.preprocess_inputs`.

For each original signal we enumerated every cyclic sample start and every
width in the host translation of the frozen coarse logical chunk's admissible
width union near the truth. This is broader than the coarse epoch-stride
schedule. The table gives the **maximum** float64 mean, not just the
mean for the best-shaped filter. The same noiseless maxima occur at the true
period and its nearest supplied trial period.

| Case | Target white SNR | Coarse-envelope maximum noiseless mean, ppm | Coarse-envelope maximum noisy mean at truth, ppm | Native-family expected white SNR | Selected BLS known-period expected white SNR | BLS period recovery |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 6 | 9.666 | 11.007 | 5.862 | 5.894 | No |
| 1 | 8 | 6.715 | 6.992 | 7.826 | 7.422 | Yes |
| 2 | 10 | 5.937 | 6.394 | 9.770 | 9.753 | No |
| 3 | 12 | 8.760 | 10.016 | 11.518 | 11.032 | Yes |
| 4 | 6 | 7.035 | 5.890 | 5.934 | 5.833 | No |
| 5 | 8 | 9.286 | 8.186 | 7.976 | 7.865 | Yes |
| 6 | 10 | 8.824 | 7.644 | 9.479 | 9.343 | Yes |
| 7 | 12 | 8.063 | 8.447 | 11.974 | 11.962 | Yes |

Thus the noiseless signals would fail this mean-depth gate at truth and the
nearest grid point in ideal float64 arithmetic, even where a sampled peak exceeds
10 ppm. With the original noise added, six still have no ideal-float64
gate-passing window. Cases 0 and 3 have respectively only two and one. At the
nearest supplied period, case 5's maximum noisy mean is 8.745 ppm; the
gate-passing counts remain unchanged for all eight.

This result concerns the **coarse logical chunk's width envelope**, not every
width that a differently grouped full-refinement stage could inspect. A
separate exhaustive enumeration over all 34 cached widths finds seven of the
eight noiseless signals still below 10 ppm at truth and nearest grid. Case 0
reaches 10.050 ppm with width 4, excluded by its coarse near-truth minimum
width of 5. With noise, six cases still have no passing all-cache window;
case 0 has four and case 3 has one. These are separate `all_cache` fields in
the JSON, not a replay of actual full-refinement membership or decisions.

The eight best native-family shapes use 6–15 samples, all within the nominal
near-truth admissibility envelope. Their means are only 4.34–7.83 ppm. Each
best row's signal length equals its scanned width: literal zero-flux padding
is not present in these particular best shapes. Width exclusion or padding
therefore does not explain the good shape-family ceiling in this table.

The BLS kernel fits a weighted centered box and accepts downward signals
without this absolute 10-ppm flux-deficit floor (`bls_common.cuh:bls_value`).
That difference is a supported mechanism for differing sensitivity at small
absolute depth. Removing or changing the native gate was not tested here.

## Why the actual GPU failure is not fully attributed

Native mean depth is formed as `1 - (prefix[end] - prefix[start-1]) / width`
from **float32 cumulative raw flux near one**. Our diagnostic sums deficits in
float64, and does not emulate that prefix operation or its reduction tree.
At the best-family windows, one float32 prefix ULP divided by width corresponds
to approximately 2.03–13.56 ppm. This illustrates numerical sensitivity at the
scale of the gate; it uses the nominal cumulative magnitude for unit flux,
not the actual GPU prefixes, and is not an error bound or measured GPU error.
An independent eight-input inspection found the frontend's epoch shift is
zero and the diagnostic/native translated fold orders agree at truth and
nearest grid in all 16 checks.

Cases 1, 2 and 6 have maximum observed point deficits below 10 ppm even after
rounding the input flux to float32 (8.285, 8.702 and 8.821 ppm). No exact mean
of those rounded values could exceed the gate at any period. Nevertheless,
the receipts report finite selected periods but omit fitted depths and prefix
intermediates. This diagnostic does not trace native gate decisions or attribute
misses to cancellation; that explanation remains a hypothesis. The original spectra are represented
by hashes in these receipts, which also prevents tracing truth-period rank,
coarse candidate membership, or a particular losing template from them.

## Shape SNR, aliases and noise

The existing known-period native cache-family and ideal-box diagnostics use
the same fitted weighted constant and the same white or OU-noise variance.
The OU values evaluate the **fixed white-optimal filters** with OU variance;
they are not separately OU-optimal family ceilings.
They do not compare package SNR/SDE labels. Native versus ideal-box expected
white SNR ranges from −0.552% to +2.293%, with median +0.813%; the median OU
advantage is +0.703%. The retained actual selected BLS configuration has
96.76–100% of ideal-box white SNR at the true period. These ceilings retain
substantial expected signal, including cases with target norm 10 or 12 whose
saved TLS peaks miss the period criterion. This best available native-template
ceiling does not measure the actually admitted, scored or ranked statistic,
or attribute any blind-search miss. It excludes native depth estimation/gating,
coarse scheduling, candidate refinement and rank selection.

The TLS peaks all fail the declared alias criterion as well as exact-period
recovery. BLS case 0 selects a near-half-period peak, but accumulated drift is
2.16 times the allowed alias tolerance. BLS case 2 selects 5.924786 days near
the 5.929636-day truth, but accumulated drift is 2.58 times the allowed
tolerance; another unused ranker selected a recoverable nearby peak, which
does not alter the frozen likelihood result. BLS case 4 selects an unrelated
0.814574-day peak. Its white-normalized observed response to the fixed native
shape is 4.49 versus expected 5.93, consistent with an adverse noise realization;
this is descriptive, not a causal intervention. Both target-SNR-6 cases are
BLS misses. No alias rule or recovery tolerance was relaxed.

The defensible conclusion is that this development subgroup exposes a serious
absolute-depth/window-mean limitation worth reporting, with numerical gate
behavior and native ranking still unresolved. These eight predeclared
development cases cannot quantify final population recovery, establish universal
equivalence, or identify what any one hypothetical change would recover.
