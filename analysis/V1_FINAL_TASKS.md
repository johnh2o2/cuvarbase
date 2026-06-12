# SUPERSEDED 2026-06-12 — see analysis/V1_RELEASE_PUNCHLIST_2.md

All T4 'deferred to v1.1+' items were promoted to v1.0
requirements by maintainer decision; T1/T2 and the promoted items
are now tracked in V1_RELEASE_PUNCHLIST_2.md. This file is kept
for the detailed T1 sub-task lists it contains.

# v1.0 final pass — remaining tasks (June 12, 2026)

Successor to analysis/V1_RELEASE_PUNCHLIST.md (closed 2026-06-12, all
45 items done; GPU validation 608/608 in
analysis/v1.0-rc-gpu-validation/). Master merge / tag move / PyPI
publish remain explicitly deferred — more pre-release work first, per
maintainer direction.

## T1. Documentation refresh + audit  ⟵ NEW (maintainer request)

Audit completed 2026-06-12 (6-auditor sweep + adversarial verification;
findings below). Style/layout constraint: **keep the existing Sphinx
alabaster theme and README structure — content-only updates.**

### T1.a Keplerian-assumption citations (audit: conclusive, paper-verified)
- [ ] Cite **Seager & Mallén-Ornelas (2003), ApJ, 585, 1038**
      (DOI 10.1086/346105, 2003ApJ...585.1038S) for the arcsin
      transit-duration relation: their eq. (3) at i=90°, b=0 reduces
      exactly to cuvarbase's `q = arcsin((f/fmax0)^(2/3))/pi`; their
      eq. (4) is the Kepler's-third-law step in docs/source/bls.rst:32.
      The bls.rst section "A shortcut: assuming orbital mechanics"
      reproduces this derivation without attribution.
- [ ] Cite **Ofir (2014), A&A, 561, A138** (DOI
      10.1051/0004-6361/201220860, arXiv:1307.7330; corrigendum A&A
      597, C2) for the optimal frequency-grid spacing: his eq. (4)
      Δf = q(f)/(S·OS) is exactly `df = q/(oversampling*T)`
      (bls.py:282, bls_frequencies.py:91); Sect. 3.1 gives
      f_min = 2/T (bls.py:207); eqs. (5)-(7) are tls_grids.py:164-185.
- [ ] Insertion points (full list in the audit): bls.py module
      docstring + q_transit/freq_transit/fmax_transit0/fmin_transit
      (currently no docstrings) + transit_autofreq + eebls_transit(_gpu);
      bls_frequencies.py module + _q_transit + keplerian_freq_grid;
      docs/source/bls.rst (open the shortcut section with [SM03]_,
      cite [O2014]_ in period-spacing, add both reference targets).
- [ ] Fix WRONG Ofir titles: tls_grids.py:9-10, docs/TLS_GPU_README.md:101,291
      carry Hippke & Heller's title on Ofir's citation;
      docs/TLS_GPU_IMPLEMENTATION_PLAN.md:868 has an unrelated GW title.
      Correct title: "Optimizing the search for transiting planets in
      long time series".
- [ ] Reconcile fmax0 constant: code 8.6307 vs docs/source/bls.rst 8.612;
      document that it is the derived orbit-at-stellar-surface frequency
      sqrt(G·rho_sun/(3pi)), not a literature value (Ofir uses the
      Roche-limit cutoff = fmax0/3^1.5 — worth a note).

### T1.b README content fixes (no restyle)
- [ ] BLOCKER: PyPI badge + "pip install cuvarbase" point at v0.2.5
      (PyPI latest; CHANGELOG's "0.2.6" is also wrong — 0.2.6 was
      tagged but never published). Until 1.0.0 ships: replace install
      section with `pip install git+...@v1.0.0` and caveat/remove badge.
- [ ] Selling point: move the verified headline numbers (TESS QLP
      production use since Sector 59; 257-354x vs astropy on 7 GPUs;
      4 surveys for ~$33) from lines 56-64 into the first screenful;
      move Citation BibTeX + Personal Note + Future Plans below the
      technical sections. Reorder only — no rewrite of voice/style.
- [ ] Remove claim of nonexistent `periodograms/` module (line 148).
- [ ] Point "examples" line 291 at notebooks/ (LS/CE/PDM walkthroughs
      live there, unreferenced; examples/ has only tls_example.py).
- [ ] Fix Testing section ("tests require a GPU" contradicts the
      CPU-only conftest + CI); singular "module" for experimental TLS;
      reframe "What's New in v1.0" so it isn't relative to master;
      refresh/remove the 8-month-old citation-count claim; mention
      cufinufft optional extra; normalize 3 legacy http:// ADS links.

### T1.c Sphinx site rebuild (same theme/layout) + gh-pages refresh
- [ ] conf.py: drop `matplotlib.sphinxext.only_directives` (removed in
      matplotlib 3.0 — build aborts) and add
      `autodoc_mock_imports = ['pycuda', 'skcuda']` (+pytest if keeping
      the tests page) so API pages build on GPU-less machines.
      Keep alabaster + logo + sidebars exactly as-is.
- [ ] Fix doc-source defects: install.rst comprehensively stale
      (contradicts v1.0 requirements); ce.rst example NameError;
      lomb.rst examples import skcuda.fft needlessly (breaks on
      numpy>=1.24); lomb.rst tau-equation typo; fap significance
      section is a TODO stub despite fap_baluev; bls.rst O(N^2·Nf)
      complexity overstatement; 4 figure scripts use np.int/np.float +
      float linspace num (dead under modern numpy) and
      bls_example_transit.py passes nonexistent fmin_fac/fmax_fac;
      missing blank lines before appended convention sections;
      cuvarbase.tests.rst lists 5 of 14 test modules; conf.py vestigial
      CUDA-8.0/macOS/py2 settings.
- [ ] Figures: plot_directive figures need a GPU to render — either
      regenerate the 4 PNGs on the next pod session and commit them,
      or accept include-source-only pages for now (build treats plot
      failures as warnings).
- [ ] gh-pages republish: site is Sphinx 1.6.3/alabaster from
      2017-10-04 at v0.2.0. publish_docs.sh (repo root, on master)
      is the historical deploy: in-place branch switch + git rm -rf +
      make html + add --all — it committed 169 MB of junk (.eggs with
      a full py2.7 astropy egg, build/, dist/, egg-info, .DS_Store,
      pytest .cache). Rebuild with modern Sphinx + alabaster (layout
      preserved), publish a CLEAN tree (orphan commit or git clean
      step), keep .nojekyll. Update pyproject.toml Documentation URL
      claim only once the site is actually rebuilt.
- [ ] Clean-environment Sphinx render check (carried from punchlist).

## T2. Rigorous benchmark campaign — BLS & TLS vs SOTA + master  ⟵ NEW

Protocol doc: analysis/BENCHMARK_PROTOCOL_V1.md (DRAFT COMPLETE
2026-06-12, QLP research verified — awaiting maintainer sign-off). Key design inputs
from recon (full details in audit):
- master (060d839, v0.3.0) has the SAME eebls_gpu_fast signature but
  no kernel caching/adaptive/batch/sparse and no epoch subtraction →
  master-vs-v1.0 needs two venvs/processes (same package name +
  import-time CUDA context); the in-repo "v0.4 baseline" path now
  uses the kernel cache, so published 21-390x "vs pre-v1.0" numbers
  are NOT reproducible from the current tree — campaign supersedes them.
- Apples-to-apples rules: matched period grids and duration ranges
  (astropy BoxLeastSquares uses absolute-time durations at every
  period vs our fractional per-frequency q — match work explicitly,
  compare timing + peak recovery, never raw power values); report
  warm-cache and cold-start separately; include H2D/D2H transfers in
  end-to-end timings; median of >=5 repeats with dispersion; pin all
  versions; identical injected signals for recovery checks.
- Scenario axis (where is the speedup most advantageous): canonical
  baseline x cadence grid — TESS QLP FFI (the centerpiece: their real
  cadence/binning, targets/sector, period grid — research pending),
  TESS 2-min, ZTF, HAT-Net/ground 10-yr, Kepler 4-yr; single-LC and
  survey-throughput (LC/s, GPU-hours/sector, $/sector).
- Comparison matrix: BLS → cuvarbase v1.0 (fast/adaptive/sparse/batch
  as appropriate) vs origin/master GPU vs astropy BoxLeastSquares
  (CPU, method='fast', matched grid) [+ fBLS literature numbers only —
  public code is notebook-grade CPU]. TLS → cuvarbase TLS vs
  transitleastsquares (CPU, use_threads=all, matched Ofir grid).
  No CETRA comparisons (standing constraint; different algorithm).
- QLP impact deliverables: (1) $/GPU-hours per sector BLS, master vs
  v1.0; (2) TLS feasibility: survey_hours = N_lc x t_TLS/N_gpus with
  measured t_TLS at QLP cadence — note TLS ~3,500-point cap vs
  10-min/200-s FFI cadences (binning required; confirm QLP's actual
  search cadence); (3) science translation with literature backing
  (TLS small-planet sensitivity gain per Hippke & Heller 2019; duration-
  window widening for eccentric orbits = linear compute cost, enabled
  by per-frequency q bounds; denser grids/longer baselines).
- [x] Finalize protocol doc (analysis/BENCHMARK_PROTOCOL_V1.md)
- [ ] Maintainer sign-off on protocol BEFORE pod spend
- [ ] Execute on pod(s); archive raw JSON + env pins in benchmarks/
- [ ] Rewrite README/BENCHMARK_RESULTS claims from the new data

## T3. Release closing moves (blocked on explicit maintainer go)
- [ ] Merge v1.0-fixes → master; move v1.0.0 tag; build + smoke-test
      wheel; publish to PyPI (also resolves the T1.b PyPI blocker)
- [ ] Release-day tracker pass: close #14, #15 (after docs rebuild),
      #19 (close-with-tables or add figures), #32 (wontfix per CE
      descope), #28 status comment
- [ ] JOSS submission; ASCL record update; co-maintainer invite
      (@astrobatty) — maintainer actions

## T4. Deferred to v1.1+ (recorded; no v1.0 action)
See the future-work inventory in the audit (summary): TLS v1.1 rework
(beyond the shipped fixes), eebls_gpu_batch large-ndata regression
diagnosis (nsys), scikit-cuda replacement (#63), lazy CUDA context,
sparse-path per-frequency q bounds, PDM batched API + large_run +
benchmark (#33), NUFFT-LRT GPU rewire, true pinned host buffers,
multiharmonic GLS, noverlap for eebls_gpu_fast, estimate_m L1 bound,
nbins-aware block-size heuristic, LS batch_size>1 overhead, kernel
templating merge, selectable power conventions (#17), docstring
audit + notebooks (#29), API renaming (#30), astropy-8.0 re-benchmark,
CETRA benchmark, GPU FTP / astropy method= / LSDB example, wavelets +
spectrograms (README planned features).
