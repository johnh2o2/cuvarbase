# Benchmark protocol v1 — BLS & TLS, survey-scale, QLP-anchored (DRAFT for maintainer sign-off)

Date: 2026-06-12. Status: DRAFT — no pod spend until approved.
Raw research behind every factual claim here:
analysis/v1.0-qlp-research-jun2026.json (adversarially verified,
verbatim quotes) and analysis/v1.0-docs-audit-jun2026.json
(benchmark-recon section).

## 0. The questions this campaign answers

1. **How much faster/cheaper are QLP's BLS runs on v1.0 vs what they
   run today?** (master-era cuvarbase; deliverable in s/star,
   GPU-hours/sector, $/sector)
2. **Is TLS now feasible in the QLP?** (GPU-hours/sector at measured
   t_TLS; never been done at this scale anywhere)
3. **What science does the speedup translate into?** (quantified,
   cited; see §7)
4. **Where is the speedup most advantageous?** (baseline × cadence ×
   N_freq response surface across canonical surveys)

## 1. Verified QLP ground truth (what we replicate)

From QLP DRN 003 (Kunimoto et al. 2023, RNAAS 7, 28, arXiv:2302.01293
— read verbatim) and companions:

- Since **Sector 59**, QLP's production search **is cuvarbase BLS**
  (GitHub footnote): "~40 times faster" than VARTOOLS "despite
  searching up to ~10 times more frequencies"; "an entire sector in
  only ~1 day"; **1.9 s/star** (S58 Cam1 CCD4, 77,567 stars) and
  **4.2 s/star** (Cam4 CCD4, 38,486 stars) on multi-sector light
  curves; recovered 392/468 TOIs vs VARTOOLS's 380. Still in
  production as of DRN 004 (Petitpas et al. 2026).
- Their cuvarbase config (footnote 2): **samples_per_peak=2,
  dlogq=0.1, noverlap=3**, Keplerian duration assumption with TIC
  stellar density (solar fallback), durations **0.5–2.0×** circular
  (= our qmin_fac/qmax_fac defaults), P_min at a=2–3 R*, P_max
  baseline-limited, S/N > 9 + S/N_pink > 9 + ≥2 transits.
- Scale: **~1M light curves/sector** (T < 13.5), FFI native cadence —
  30-min (S1–26), 10-min (S27–55), **200-s (S56+)**; no QLP document
  states pre-search binning. Multi-sector light curves re-searched
  each sector. Hardware: **not stated** in any QLP source (so we
  benchmark per-GPU and report GPU-model sensitivity, not "their"
  wall time).

## 2. Comparison matrix

| Search | Contender | Role |
|---|---|---|
| BLS | cuvarbase **v1.0-fixes** (`eebls_transit_gpu`, + adaptive/batch/sparse where applicable) | ours |
| BLS | cuvarbase **origin/master** (060d839, v0.3.0) | "what QLP runs today" proxy |
| BLS | **astropy** `BoxLeastSquares` (CPU, `method='fast'`, `objective='snr'`) | CPU SOTA |
| BLS | fBLS (Shahaf et al. 2022) | literature numbers ONLY (public code is notebook-grade CPU) |
| TLS | cuvarbase **TLS** (experimental, post-rework) | ours |
| TLS | **transitleastsquares** (CPU, `use_threads`=all cores) | reference & CPU SOTA |

Excluded: CETRA (different algorithm; standing no-comparative-claims
constraint), nifty-ls (LS already benchmarked), periodfind (no BLS).

## 3. Scenarios (baseline × cadence response surface)

| # | Scenario | ndata | Baseline | Grid |
|---|---|---|---|---|
| S1 | **QLP single-sector, 30-min** | ~1,300 | 27.4 d | QLP-replica (§1) |
| S2 | **QLP single-sector, 200-s** | ~11,800 | 27.4 d | QLP-replica |
| S3 | **QLP multi-sector** (1 yr stitched, 10-min binned×3? — measure both native & binned) | ~14k–40k | ~350 d | QLP-replica, P_max ~175 d |
| S4 | TESS 2-min SPOC-like | ~19,000 | 27.4 d | Ofir grid |
| S5 | ZTF-like ground sparse | 300 | 3 yr | Ofir grid |
| S6 | HAT-Net-like ground dense | 6,000 | 10 yr | Ofir grid |
| S7 | Kepler LC 4-yr | 65,000 | 4 yr | Ofir grid |

TLS runs S1, S5 (and S2/S3 **binned to ≤3,500 points** — the shared-
memory cap makes native 10-min/200-s cadence infeasible for TLS;
this is reported as an explicit limitation + v1.1 motivation).

## 4. Fairness rules (apples-to-apples)

1. **Master comparison**: two separate venvs/processes (same package
   name + import-time CUDA context forbid co-import). Identical
   freqs/qmin/qmax/dlogq/noverlap inputs; master gets `functions=`
   pre-compiled handles in the warm benchmark (its lack of kernel
   caching is reported as a separate cold-start line, not smuggled
   into the kernel comparison). Note: the published "21–390x vs
   pre-v1.0" numbers conflate compile overhead and are superseded.
2. **astropy matching**: BoxLeastSquares takes absolute-time duration
   arrays evaluated at every period; we pass the **same period grid**
   and a duration set matched to our per-frequency q range at each
   period (piecewise; documented script). Compare **timing and peak
   recovery** (injected signals), never raw power values
   (different normalizations — documented in bls.rst).
3. **TLS matching**: same Ofir period grid both sides (force-override
   our auto grid and TLS's); same n_durations span; report cuvarbase
   TLS at **default T0_OVERSAMPLE=3** AND at reference-matched t0
   fidelity (the reference uses ~33× finer t0 stepping — we must not
   claim a speedup bought by coarser epoch sampling; both fidelity
   points + an injection-recovery parity check at each).
4. **Timing discipline**: end-to-end wall time including H2D/D2H and
   host pre/post (what a pipeline pays), kernel-only time via CUDA
   events as a secondary metric; warm-cache steady-state (≥1 discarded
   warm-up) and cold-start reported separately; median of ≥5 repeats
   with IQR; single otherwise-idle GPU; fixed clocks noted if
   available.
5. **Pinning**: record GPU model/driver/CUDA, python/numpy/astropy/
   tls versions, git SHAs of both cuvarbase trees; commit raw JSON +
   environment to benchmarks/results/.
6. **Correctness gate per scenario**: identical injected transits
   (3 depths × 3 periods per scenario); a contender's timing only
   counts if it recovers the injection at its own nominal threshold.

## 5. Metrics & deliverables

- Per scenario × contender: s/star (median, IQR), LC/s, GPU-hours and
  cloud-$ per 1M-LC sector (A5000 @$0.27/h + one datacenter GPU,
  e.g. A100, for sensitivity), speedup ratios with uncertainty.
- **QLP table**: S1/S2/S3 master-vs-v1.0-vs-astropy + "what 1 sector
  costs" — directly answers question 1. Sanity anchor: our S3
  s/star should be within ~2–5× of DRN 003's published 1.9–4.2 s/star
  (unknown GPU); if not, investigate before publishing.
- **TLS feasibility table**: measured t_TLS/LC × 1M → GPU-h/sector at
  both fidelity settings; side-by-side with measured CPU
  transitleastsquares t/LC × 1M → CPU-core-hours (literature anchor:
  ~10 s/LC K2-scale; largest published TLS searches are ~8×10³ LCs,
  and 121 stars cost ~800 CPU-h in a 2026 study) — answers question 2.
- Response-surface figure: speedup vs (ndata, N_freq) across S1–S7 —
  answers question 4 ("most advantageous where N_freq is large:
  long baselines / fine grids; least where ndata dominates").

## 6. Campaign cost estimate

Single A5000 pod: BLS scenarios ~2–4 h (incl. master venv build +
astropy CPU runs on pod CPU), TLS ~2–6 h (CPU reference is the long
pole: ~10 s/LC × repeats; we sample ~50–100 LCs/scenario for CPU,
full batches for GPU). One optional second pod (A100) ~2 h.
Total ≲ $5 at current rates, plus ~1 day wall time with analysis.

## 7. Science-translation section (cited; written from measurements)

1. **Deeper into the small-planet regime via TLS at scale**: Hippke &
   Heller (2019): at matched SDE=7 (1% FPR), TLS recovers **93.1% vs
   BLS 75.7%** of injected Earth-sized planets around Sun-like stars
   (white noise, K2-like setup); their summary claim ~10% higher
   detection efficiency. The gain is algorithmic and per-LC — our
   contribution is making it *deployable at 10⁶ LCs/sector* (never
   published at even 10⁴). Frame exactly so; no generalization beyond
   their injection setup without our own injections.
2. **Eccentric orbits**: duration scales as √(1−e²)/(1+e·cosφ)
   (Barnes 2007); Burke (2008) shows fixed circular-duration windows
   miss a substantial fraction of eccentric-transit durations. Our
   per-frequency q bounds make the window width a knob whose cost is
   **linear in window width** (verified in-kernel); we quote
   "covering eccentricities up to e at all periastron angles costs
   X× ≈ window factor" with measured $/sector.
3. **Denser/longer grids**: Ofir (2014) — optimal sampling is cubic;
   a uniform grid meeting the same long-period sensitivity is ~330×
   more expensive at 3-yr baselines. QLP's own history is the
   case study: the CPU era capped P at ~56 d on an undersampled
   ≤80K-frequency grid "due to computational expense"; cuvarbase
   removed that cap (10× more frequencies, 40× faster — DRN 003).
4. **Threshold/completeness calibration**: more compute → full-scale
   injection-recovery (Christiansen 2017 methodology) instead of
   analytic thresholds (Jenkins et al. 2002 independent-trials);
   we quote the measured cost of one full injection-recovery pass
   per sector.

## 8. Out of scope / explicitly not claimed

CETRA comparisons; TLS science-readiness claims (module stays
experimental until injection-recovery validation — feasibility ≠
validation); astropy ≥8.0 LS re-run (not released); any reuse of the
non-reproducible 21–390x pre-v1.0 numbers.
