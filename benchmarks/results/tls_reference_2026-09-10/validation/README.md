# Independent full-search confirmation

All **160 of 160** planned inputs passed the exact
corrected-GTLS/cuvarbase numerical and public-result checks. The untouched
GTLS comparison passed on **151 of 160** inputs.
`acceptance.json` records the completed gate, source seal, counts and limits;
`comparisons.json` retains the per-case differences.

Each of eight regimes contains three independent injections at each white-noise
oracle SNR 6, 8, 10 and 12, plus eight independent nulls. The full period arrays
come from the pinned native grid arithmetic at oversampling 3; no true period
was inserted. Both searches received identical arrays with a common positive
time origin and the same search bounds. Separate tests cover automatic-grid
dispatch. The injections use exposure-integrated physical transit signals;
nulls independently draw from the same four latent noise recipes. Their
additional OU noise is excluded from the quoted white-noise oracle SNR.
Recovery is conditional on at least five in-transit observations and two
sampled events; every proposal count is retained in the input manifest.
Passband baselines are assumed already removed. Transit depths and shapes are
achromatic, and both APIs receive only `t`, `y` and `dy`; retained band labels
do not imply a fitted multiband model.

The following outcomes use the predeclared, **uncalibrated SDE threshold 8**.
Recovery additionally requires period drift over the observed baseline to be
no more than half the physical transit duration. These counts are descriptive;
they do not establish a common false-positive rate or useful recovery in every
regime. The order in each cell is untouched GTLS / corrected GTLS / cuvarbase.

| Regime | Recovered / 12: native / corrected / cuvarbase | Nulls above 8 / 8: native / corrected / cuvarbase |
| --- | ---: | ---: |
| TESS, ordinary solar | 1 / 1 / 1 | 0 / 0 / 0 |
| TESS, high impact solar | 3 / 3 / 3 | 0 / 0 / 0 |
| TESS, eccentric solar | 4 / 4 / 4 | 0 / 0 / 0 |
| TESS, 0.1 solar mass/radius | 8 / 8 / 8 | 0 / 0 / 0 |
| ZTF, ordinary solar | 8 / 8 / 8 | 8 / 8 / 8 |
| ZTF, high impact solar | 8 / 8 / 8 | 8 / 8 / 8 |
| ZTF, 0.1 solar mass/radius | 6 / 6 / 6 | 8 / 8 / 8 |
| Gapped TESS, ordinary solar | 6 / 6 / 6 | 4 / 4 / 4 |

`strata.csv` separates all four injection SNR levels and the null mixture in
each regime, retains failures, and supplies exact binomial intervals and
simultaneous discordance bounds. Three injections per SNR and eight nulls per
regime cannot establish a one- or two-percentage-point population margin.
The primary evidence is identical complete numerical searches, supplemented
by these recovery/null cross-checks; a shared miss is not a successful detection.

The corrected reference filters masked/nonfinite candidates before the native
first candidate sort. Native CUDA kernels, templates and scoring formulas are
unchanged. Literal GTLS outcomes remain separate. 151
corrected executions were reused only after a complete trace proved that the
correction was a no-op. All nine differing literal searches selected the same
primary period and made the same strict SDE > 8 decision as the corrected
reference and cuvarbase. Their final power spectra and reported SDE values
differ as well as intermediate arrays; all these differences are preserved.
Fitted native SNRs remained identical in those nine cases.
See the [correction explanation](../../../../docs/GTLS_COMPARISON.md#invalid-candidate-correction).

`array_digests.json.gz` contains every retained numerical-output identity,
dtype, shape, compact fit result, record hash and no-op trace. Full NPZ outputs
were compared before the predeclared retention step. `validation.json`
records verification of the retained archives and explicit receipts for the
matching archives removed after comparison. Large output tensors stay outside
the repository; their complete numerical identities remain in this archive.

The published primary receipts come from a complete reexecution after an
earlier run's result collection failed. All 160 original inputs, seeds, source
hashes, settings, sample counts and gates remained unchanged. Earlier
uncollected output reports are not counted as completed evidence, and repeated
executions are not additional independent samples. The completed main search,
its original acceptance and its on-host compact verification were recovered
from complete members of a truncated final download. Every original record,
comparison and numerical-output digest survived. Of 75 retained NPZ output
archives, 66 were recovered and independently checked; nine were not collected.
Those nine are disclosed collection losses, not predeclared pruning.
The separate 405 matching output archives had already been removed under the
original retention rule. [Collection details](collection.json) preserve the
original acceptance hash, recovery evidence and missing-container identities.

Restore the exact inputs and replay the comparison with the
[maintained tools](../../../tls_reference/README.md). The
[source archive](../sources/README.md) preserves the exact executed scientific
code and distinguishes the original generation environment from the GPU
execution environment. `files.json` hashes the original compact result files.
