# Survey TLS throughput and detection study

This campaign builds on the September 10 observation-level GTLS compatibility
study. The production deployment criterion is retention of its complete trials,
valid masks, candidate/refinement policy and numerical objective. It does not
introduce a lossy screening path. Approximate experiments remain opt-in.

The earlier box diagnostic compared idealized known-period filters and the
secondary BLS search had one fixed setting. Neither established TLS's practical
advantage over a strong tuned BLS search. This campaign measures that advantage
before assigning any approximation allowance.

## Stages and independence

1. **Development**: eight physical draws per regime (two at each white-noise
   oracle SNR 6, 8, 10 and 12), plus 64 separate development nulls per regime.
   These inputs can expose design defects. Failed development designs remain
   recorded and are not held-out evidence.
2. **Development diagnostics**: enumerate all GTLS sample-index cache templates
   at every observation start at the true period and compare with the optimal
   arbitrary-width box. Both fit a weighted constant and use the same expected
   linear-filter SNR. Select both filters with the diagonal-error objective,
   then report their response using the actual OU covariance variance; the OU
   values are not independently optimized over the template families. The native
   family optimum is an optimistic ceiling: native depth estimation and score
   ranking need not choose that filter. `bls_response.py` measures actual
   noise-free BLS filter response with one further resolution increase.
3. **BLS tuning**: compare overlap 4/8/16/32, duration step .1/.05/.025/.0125, and minimum
   duration factors 1/.5/.25/.125 times the broad native stellar envelope. Reducing
   the minimum width also reduces the fundamental phase-bin width quantum.
   Test normalized BLS power, unnormalized delta-chi-squared likelihood power (using the supplied errors), and median-trend/MAD ranking. Choose maximum
   development recovery at its own calibrated 5% FPR; ties favor finest
   resolution, then delta-chi-squared likelihood power. This is the strongest *tested* setting, not a
   claim that every conceivable BLS implementation has been optimized. The
   32-overlap setting was added before the final freeze after the response
   diagnostic found a 1.37% loss in the previous finest eccentric-TESS case.
   It is explicitly inapplicable to the long gapped-TESS grid because its
   minimum widths exceed available shared memory; the prior finest setting
   already attains the ideal-box response on all gapped development examples.
   The stronger tested setting retains a bounded residual ideal-box gap
   (up to about 0.24% in these development examples).
4. **Freeze** `analyze.py freeze` before generating any final calibration or test
   data. Record all science-source hashes, production-source hashes, settings,
   populations, statistics, execution policy and accuracy tolerances.
5. **Independent calibration**: 512 new nulls per regime and detector. Use the
   strict upper order statistic with rank `ceil((n+1)*(1-alpha))` for primary
   5% and secondary 1% FPR. For n=512 their no-tie marginal rates are
   25/513 and 5/513; ties can make the realized operating point more conservative. This exchangeability guarantee is marginal over calibration
   sets; it does not certify the conditional rate of one realized threshold.
6. **Held-out evaluation**: 256 injections and 256 further nulls per regime (64 injections per SNR).
   Keep every planned injection, including unsampled signals, one-event cases
   and API failures. Detection requires threshold exceedance and period drift
   over the full baseline at most half the physical first-to-fourth-contact
   duration. Report aliases separately. No injected truth is inserted into
   blind grids. Results retain per-SNR, few-point, few-event and grid-unreachable
   subgroups. A small cohort cannot establish 0.1 percentage-point equivalence.

Both detectors receive the same paired null lightcurves, independently drawn
from development and test cohorts, and use separate method-specific thresholds.
Null latent SNR/noise scales are IID draws from the equal four-level mixture;
the injected populations are deliberately balanced over those four levels.
The two detector scores are calibrated separately; SDE, native SNR, BLS power
and oracle SNR are never equated. Report exact marginal binomial intervals and
paired TLS-minus-BLS discordance bounds. Simultaneous paired bounds cover all
regimes, both recovery/FPR outcomes and both operating points using Bonferroni.
No pooled success overrides a weak subgroup. The exact-only production policy
allows zero FPR increase; its separate 0.1 percentage-point absolute cap is
recorded for context and does not grant an operative allowance.

Balanced injection strata need not have equal success probabilities. The
two-sided Clopper–Pearson construction remains valid for their average at the
confidence levels used here; see [Mattner and Tasto, Theorem 1.12](https://arxiv.org/pdf/1403.0229).
The paired construction combines those two-sided discordance bounds by
Bonferroni; it does not require an IID pooled-injection assumption.

## Frozen tolerance rule

An approximation may consume at most **5% of a demonstrated positive TLS
advantage**, with additional absolute ceilings of **0.1% fractional expected
SNR loss** and **0.1 percentage point recovery loss or FPR increase**. A
nonpositive or uncertain subgroup advantage gives zero allowance. This protects
at least 95% of an established advantage and avoids treating even a one-percent
SNR loss as harmless when the entire advantage is around one percent.

The development expected-SNR guard requires positive benefit in every sampled
case under both white and OU variance before considering a one-sided bootstrap
lower mean bound. The recovery guard uses the lower paired development bound.
These are conservative engineering guards, not claims of universal population
coverage. If the development family diagnostic itself cannot establish a
benefit, it cannot justify an approximation budget. The anticipated production
path is exact optimization: no discarded observations, durations, epochs or
candidates and no changed decisions on qualification inputs. Shared native
float32-prefix variability remains explicitly recorded; failed bitwise checks
are not silently assigned a wider tolerance.

## Physical coverage

| Regime | Stellar/geometry sampling | Cadence / period domain |
| --- | --- | --- |
| TESS solar | Solar mass/radius; impact .2–.7 | Observed 200 s sector; injections 2–6 d, blind .6–12.878 d |
| TESS high impact | Solar; impact .94–.96 | Same cadence; 2–6 d |
| TESS eccentric | Solar; e=.7–.8, omega=90°, impact .2–.7 | Same cadence; 6–12 d |
| TESS small M dwarf | .1 solar mass/radius, density 100×solar; impact .2–.7 | Same cadence; 2–6 d |
| ZTF solar | Solar; impact .2–.7 | Observed g/r timestamps over 2,744 d; 2–6 d, blind .6–10 d |
| ZTF high impact | Solar; impact .94–.96 | Same ZTF cadence/domain |
| ZTF small M dwarf | .1 solar mass/radius | Same ZTF cadence/domain |
| Separated TESS, long | Solar; impact .2–.7 | Observed separated sectors over 735 d; 15–25 d, blind .6–27.458 d |
| TESS grazing/smeared | Solar; impact .999–1.003 | Every ninth archived 200 s sample; 1,800 s exposures; 2–6 d |
| HATpi-like short | Solar; impact .2–.96 | Synthetic 30 s cadence, eight-hour nights, absent nights/nightly gaps; .65–2 d, blind .6–5 d |

All use Earth-size planets, fixed quadratic limb darkening [.4804,.1867],
achromatic transit depth, heterogeneous supplied errors and an OU component
with amplitude .25 times median error. The OU time scale is one day for ZTF
and .15 day otherwise. Oracle SNR 6/8/10/12 specifies the preassigned target for
the centered physical signal in *white* noise; correlated noise is extra.
Unsampled signals can realize SNR zero and remain in their assigned target
groups. The expected-SNR diagnostic computes the realized centered signal norm.
Null noise scale is drawn
from the same latent physical mixture independently. There is no observability
rejection, so sparse signals do not disappear before evaluation. TESS/ZTF
cadences are observed timestamps with synthetic flux; HATpi is entirely
synthetic. Errors are rescaled toward the target oracle SNR, so these controlled
ZTF sampling experiments do not forecast Earth/Sun transit yields at actual
ZTF photometric precision. Band offsets are assumed removed.

Period grids use fixed **regime-level** oversampling: 9 for high-impact TESS,
high-impact ZTF, eccentric TESS and HATpi; 24 for grazing/smeared TESS; 3 for the
others. This policy was chosen using physical boundary durations in development,
not the realized injection truth. Both detectors receive the same grid. The
native SDE normalization keeps oversampling setting 3. These duration-informed
benchmark strata do not imply that an operational survey knows impact or
 eccentricity; an unknown-regime survey needs a correspondingly conservative
common grid. Every case records nearest-grid drift relative to the recovery
criterion. The original coarse-grid grazing development failure is retained.

`boundaries.py` checks 32/64/128-node exposure quadrature on observed development
inputs and joint M-dwarf/high-impact/grazing/eccentric boundaries at .65, 10 and
365.25 days with 30/200/1800 s exposures. These annual cases are known-transit
physical diagnostics, not blind annual-period recovery or throughput evidence.
The existing numerical stress archive supplies separate annual-period tests.

Unsupported claims include universal sensitivity, arbitrary stellar/planetary
populations, omega outside the sampled 90°, limb-darkening mismatch,
chromatic/multiband fitting, real survey-flux systematics, transit-timing
variations, eclipsing-binary rejection, starspot distributions, annual blind
search completeness and real-HATpi recovery. The declared two stellar-density
points and narrow planet-radius population are deliberate finite coverage,
not a physical continuum.

## Reproduction

Use Python with numpy, scipy and batman-package for generation/analysis;
current cuvarbase, CuPy and PyCUDA are additionally required on the single GPU.
Generate final development after preserving any rejected development design:

```sh
python benchmarks/tls_survey/generate.py --split development --count 8 --out STUDY/development
python benchmarks/tls_survey/generate.py --split development_nulls --count 64 --out STUDY/development-nulls
python benchmarks/tls_survey/development.py --manifest STUDY/development/manifest.json --out STUDY/development-snr.json
python benchmarks/tls_survey/boundaries.py --manifest STUDY/development/manifest.json --out STUDY/boundaries.json
python benchmarks/tls_survey/bls_response.py --manifest STUDY/development/manifest.json --out STUDY/bls-response.json
python benchmarks/tls_survey/run.py --manifest STUDY/development/manifest.json --methods tls bls_medium bls_fine bls_finest bls_strong --shard-count 4 --shard-index 0 --out STUDY/development-search-0.json
```

Run shards 0–3 in separate processes on the same GPU, each with its own receipt;
repeat for development nulls. The runner enforces one numerical-library thread
per process. Four processes share the same CPU/memory allocation and GPU.
Execution timings here are scientific-run receipts, not sustained-throughput
measurements. `throughput.py` provides the latter with separately tuned batch
and worker settings.

```sh
python benchmarks/tls_survey/analyze.py freeze --development STUDY/development-search-*.json --development-nulls STUDY/development-null-search-*.json --snr STUDY/development-snr.json --out STUDY/seal.json
python benchmarks/tls_survey/generate.py --split calibration --count 512 --seal STUDY/seal.json --out STUDY/calibration
python benchmarks/tls_survey/run.py --manifest STUDY/calibration/manifest.json --methods tls bls_medium bls_fine bls_finest bls_strong --seal STUDY/seal.json --shard-count 4 --shard-index 0 --out STUDY/calibration-search-0.json
python benchmarks/tls_survey/analyze.py calibrate --seal STUDY/seal.json --results STUDY/calibration-search-*.json --out STUDY/thresholds.json
```

After all calibration shards finish and thresholds are frozen, generate
`--split injections --count 256` and `--split nulls --count 256`, run all four
shards with the original seal, then:

```sh
python benchmarks/tls_survey/analyze.py analyze --seal STUDY/seal.json --thresholds STUDY/thresholds.json --injections STUDY/injection-search-*.json --nulls STUDY/null-search-*.json --out STUDY/recovery.json
```

Final-data execution runs only the selected BLS configuration per regime.
A successful degenerate/all-masked TLS result with no candidate and SDE=0 is a valid nondetection with score zero; actual API errors or unavailable scores fail calibration rather than silently lowering a threshold.
An unused diagnostic ranker's failure does not invalidate the selected detector.
Atomic resumable receipts retain every input identity, planned population,
source version, candidate, spectrum/validity hashes and API failure. Analysis
rejects missing counts, duplicate cases, unpaired inputs and source drift.

After reviewing the concrete seal, `campaign.py` can run the remaining stages
as a detached process. Supply the reviewed SHA literally; it verifies the seal
and sources before each stage, preserves interrupted input generation, resumes
search receipts, and exports an exact deduplicated array bank when requested:

```sh
python benchmarks/tls_survey/campaign.py --seal STUDY/seal.json --seal-sha256 REVIEWED_SHA256 --work STUDY/heldout --export-bank
```

Restart the same command after an interruption. An exclusive local lock prevents
two controllers sharing one work directory. `campaign.json` records workers,
commands, logs, heartbeats and failures. This controller never provisions or
terminates a rented resource; its owner must separately enforce the authorized
cost limit and download artifacts before termination.

`exactness.py` supplies a separate implementation qualification on every final
injection and independent test null (5,120 paired inputs for this design).
Freeze its auxiliary plan before held-out generation and review its SHA beside
the science seal. After the science campaign completes, one worker runs the
immutable pre-optimization TLS checkout on those exact inputs and compares to
the **original** candidate receipts: complete available period/chi2 spectra and
validity hashes, chosen period/SDE, recovery and both frozen-threshold decisions.

```sh
python benchmarks/tls_survey/exactness.py freeze --seal STUDY/seal.json --baseline-root BASELINE --campaign STUDY/heldout --out STUDY/exactness-plan.json
python benchmarks/tls_survey/exactness.py run --seal STUDY/seal.json --plan STUDY/exactness-plan.json --plan-sha256 REVIEWED_PLAN_SHA256 --campaign STUDY/heldout --out STUDY/exactness-results.json
```

Any mismatch withholds aggregate exactness qualification. The first ten
mismatching inputs receive two further native-baseline diagnostic runs; each
original outcome is persisted before those runs and cannot be replaced by a
later matching repeat. The plan records the development-based extra cost
estimate (about 4.84 GPU hours / $2.37 at $0.49 per hour), separately from measured
sustained throughput. Finite paired checks do not establish universal physical
or numerical equivalence.

The retained [development implementation comparison](../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/development-promoted-baseline-parity.json")
has 79 exact results out of 80. One HATpi-like case changed its chi2 hash and SDE
by about 3.34e-6 while retaining its period, valid mask and recovery/alias
decisions; all 32 cases using the new short-row path matched. This is a recorded
failure of aggregate bitwise equality, with no numerical tolerance relaxed and
no cause assigned from the regime alone.

The same auxiliary plan pins `heldout_snr.py`. This CPU-only wrapper applies
the unchanged development filter definitions to all 2,560 held-out injections,
without relabeling their split or tuning any setting. Its scientific fields
matched the original diagnostic exactly on all 80 development inputs. Run it
after the science campaign and before sustained throughput measurement:

```sh
python benchmarks/tls_survey/heldout_snr.py run --seal STUDY/seal.json --plan STUDY/exactness-plan.json --plan-sha256 REVIEWED_PLAN_SHA256 --manifest STUDY/heldout/inputs-injections/manifest.json --out STUDY/heldout-snr.json
```

Join descriptive filter ceilings to original detections by input name/hash and
regime. They explain physical/sampling losses and remain distinct from package
SDE/SNR, actual native depth/ranking, and the predeclared blind recovery endpoint.
