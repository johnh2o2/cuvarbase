# What the TLS literature establishes

Primary-source audit, 2026-09-10. This note separates published evidence from the
new cuvarbase experiment; it does not replace the latter's sealed protocol.

## The original result is a recovery advantage

Hippke & Heller (2019), §3.1/Fig. 6, report **93.1% TLS versus 75.7% BLS recovery
at a 1% false-positive rate**: 17.4 percentage points, or approximately 23%
relative improvement. Their experiment used 10,000 signal curves and 10,000
noise curves: three-year, 30-minute sampling, 110 ppm white noise, and three
Earth-sized transits around solar hosts with impact parameters in [0,1] and
Kepler-band quadratic limb darkening. Both searches used the same optimized
period grid. This is substantial reported detection evidence, not a 17% SNR
measurement. [Original paper, §3.1 and Fig. 6](https://arxiv.org/html/1901.02015#S3.SS1)

Reproduction detail remains important: §3.1 describes a positive as the global
highest peak lying within 1% of the injected period; Fig. 6 describes the signal
histogram using the highest SDE *within* that window. Preserve the published
claim while recording that this wording needs experiment-code resolution.
The Astropy 3.1 implementation and 66 durations specified in §3.4 describe a
separate timing comparison; do not silently assign them to §3.1. Nor does the
common period grid justify calling the original BLS poorly sampled.
[Original paper, §§3.1 and 3.4](https://arxiv.org/html/1901.02015)

## What the linked code resolves, and what it does not

The paper links the author's `hippke/tls` repository. This audit inspected its
2019-02-18 snapshot, `160020aa31f4d1364cc73e8031700ef3394bf83b`, and current tree
`1440ca760a785bf06a56619f705539c3a7377dd7`. The tree inspection did not locate
the 10,000-injection comparison driver or its paired output table; this is a
bounded inspection, not proof that no archived script exists elsewhere.
[Historical repository tree](https://github.com/hippke/tls/tree/160020aa31f4d1364cc73e8031700ef3394bf83b)

The historical comparison notebook instead demonstrates K2-110: TLS uses
`model.power()`, while BLS uses 20 durations from 0.05 to 0.2 days and
`autopower(..., frequency_factor=10)`. It is therefore **not a reproduction of
the paper's §3.1 common-grid experiment**, and its settings cannot establish
that experiment's BLS tuning. The historical synthetic test is also different:
one fixed-seed, two-hour-cadence, 5 ppm example with a restricted 360–370-day
search. The FAP unit test checks a lookup value at SDE=7; it does not regenerate
the null population. [Comparison notebook](https://github.com/hippke/tls/blob/160020aa31f4d1364cc73e8031700ef3394bf83b/tutorials/06%20Comparison%20between%20TLS%20and%20BLS.ipynb),
[synthetic test](https://github.com/hippke/tls/blob/160020aa31f4d1364cc73e8031700ef3394bf83b/transitleastsquares/tests/test_synthetic.py),
[FAP test](https://github.com/hippke/tls/blob/160020aa31f4d1364cc73e8031700ef3394bf83b/transitleastsquares/tests/test_FAP.py)

## Why a few percent in SNR is a different claim

Hord et al. (2021; Colón is second author) repeat the original recovery result
and note that realistic shapes can give as little as approximately 3% sensitivity
improvement when BLS is sufficiently sampled, citing Jenkins, Doyle & Cullers
(1996). That remark is not a new controlled TLS-versus-BLS recovery trial.
Their TESS hot-Jupiter companion search uses both default and grazing TLS
templates and finds no new validated companions. They describe the TLS, SPOC,
and QLP sensitivities as comparable indications, while explicitly noting the
absence of a direct TLS–SPOC sensitivity comparison. Their SDE>7 search threshold
is inherited from the original Kepler-like simulation, not independently
calibrated on their TESS noise population. [Hord et al., introduction, §III.1,
and §VI.1](https://arxiv.org/html/2109.08790)

For this study, three quantities must remain distinct:

| Quantity | What it measures | What it cannot establish alone |
| --- | --- | --- |
| Common expected matched-filter SNR | Response of a specified template to a noiseless injected signal under the same timestamps, weights, and nuisance projection | Blind period recovery or the null maximum over a template bank |
| A package's SDE | Its normalized periodogram peak, with package-specific baseline and normalization | A directly comparable SNR or a universal false-positive probability |
| Blind recovery at independently calibrated common FPR | Probability of exceeding a separately estimated null threshold and selecting the correct period | Equivalence outside the tested population |

As an analytic diagnostic, with inverse-variance inner product and the same
weighted-mean subtraction, a searched template `h` has expected response
`rho(h) = <s,h>_w / sqrt(<h,h>_w)`. Relative to the perfectly matched signal,
`rho(h)/rho(s)` is the weighted correlation between signal and template. A
well-placed box can be close to a transit under this metric, especially when
ingress contributes little weight. Finite exposure and sparse sampling change
that correlation. Kipping (2023), §§3.2–3.3, derives separate optimal-box and
matched-trapezoid SNR expressions and their convergence in the box limit.
[Kipping, “SNR of a transit”](https://academic.oup.com/mnras/article/523/1/1182/7179431)

Our interpretation: a percentage SNR change and a percentage-point recovery
change have no fixed conversion. Threshold crossing is nonlinear, and blind
recovery also depends on aliases, grid placement, noise maxima, template-bank
size, and the ranking statistic. This explains why the measurements answer
different questions; it does **not** quantitatively explain away or independently
reproduce the original 17.4-point result.

## Real detections also depend on preparation

TLS Survey I is a useful controlled example of that dependence. For K2-32e,
the authors report SDE 13.2 (TLS) versus 8.9 (BLS) in K2SFF data; in EVEREST
data both recover it, at 26.1 and 21.3 respectively. They also test a hyperfine
nonlinear BLS period grid with over 100,000 trials and restricted durations:
the troublesome short-period alias disappears, while the K2SFF signal remains
at SDE 8.9. These are real-data demonstrations, not a population comparison at
independently calibrated equal FPR. Dividing those SDE values does not yield
a matched-filter SNR gain. [Heller, Rodenbeck & Hippke (2019), §§3.3–4.2](https://arxiv.org/html/1904.00651)

## Consequence for cuvarbase

The original 93.1% versus 75.7% result remains substantial published canonical-TLS recovery evidence. It is not a 17.4% expected-SNR measurement and is not independently reproduced by the present experiment. A small ideal-box/shape response difference does not refute it; a percentage SNR change has no fixed conversion to percentage-point blind recovery.

The completed cuvarbase science report instead measures its pinned GTLS-compatible floating-point search against development-selected native GPU BLS on identical arrays and full grids. Method-specific cuts use a paired calibration bank independent of development and test populations. Common 5%/1% calibrated targets have uncertain realized FPRs, as the independent test-null intervals show. [Per-regime recovery and uncertainty](TRANSIT_BENCHMARKS.md).

At both operating points, the predeclared simultaneous intervals support large positive TLS-minus-BLS recovery differences in four TESS regimes and a severe negative difference for grazing/smeared TESS. Known-period native-family/ideal-box median white advantages of only +0.166% to +1.359% coexist with those blind outcomes. Those family ceilings use common signal/weight/nuisance conventions; package SDE ratios, actual native admission/ranking and an ideal box are different quantities. Sparse/high-impact and gapped signals have large negative response tails. Neither the paper nor these finite tests justify universal dominance or a default 1–2% SNR-loss allowance.

The operative tolerance remained zero before held-out evaluation. The optimized-versus-baseline implementation gate failed on nine of 5,120 original pairs, although selected periods and frozen-threshold decisions agreed. Positive held-out advantages do not retroactively permit approximation losses or erase numerical mismatches. The release consequently keeps the original observation-level baseline default and exposes the complete optimization bundle only through an experimental selector; its separate [release wiring validation](../benchmarks/results/tls_survey_2026-09-10/release-validation/README.md) passed without changing the original failed scientific qualification. [Numerical contract](TLS_NUMERICS.md) · [GTLS versus canonical CPU TLS](GTLS_COMPARISON.md).

[Collected recovery report](../benchmarks/results/tls_survey_2026-09-10/final-report/RECOVERY.md) · [report provenance](../benchmarks/results/tls_survey_2026-09-10/final-report/provenance.json). This editorial update now uses the byte-verified completed science collection; it makes no new literature replication or statistical analysis claim. The paper audit and its bounded code-search limitations above are retained unchanged.
