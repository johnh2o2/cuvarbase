# Issue sweep — drafted close comments (execute on release day, after v1.0.0 is live)

Decision (maintainer, Jul 10 2026; reconfirmed for the Sep-2026 state): close
all 10 open issues with evidence comments; open ONE consolidated "v1.1
roadmap" issue (body at the bottom). Order of operations: publish release →
post roadmap issue → close the 10 with the comments below (several reference
the roadmap issue number).

Placeholders to fill on release day: `#ROADMAP` (the roadmap issue number),
`<N>` (the "passed" count from the Phase 5 gate log, 0 skipped). The
NUFFT-LRT status (Phase 4, 2026-09-06) is filled in below: experimental,
validated. Archived pre-1.0 material is cited by the archive tag
`archive/pre-1.0-process` (pushed on release day), never by an `analysis/`
path on `master`.

---

## #14 — Numerically stable false alarm probability for Lomb Scargle → CLOSE (shipped)

v1.0.0 ships the Baluev (2008) analytic FAP upper bound as
`cuvarbase.lombscargle.fap_baluev(t, dy, z, fmax)`, evaluated in log space so
significant peaks no longer underflow to `FAP == 0`.
`LombScargleAsyncProcess.batched_run_const_nfreq(..., only_return_best_freqs=True)`
applies it automatically to each lightcurve's best peak. The documentation's
significance section (long a TODO) now documents it:
https://johnh2o2.github.io/cuvarbase/lomb.html — see "Estimating significance".
Closing as shipped in v1.0.0.

## #15 — Add documentation for PDM → CLOSE (shipped)

The rebuilt documentation site includes the PDM page (`pdm.rst`) with the
modern `(t, y, err)` API, automatic frequency grids, and the four kernel
variants: https://johnh2o2.github.io/cuvarbase/pdm.html. API docstrings render
there as well. Closing as shipped with the v1.0.0 docs rebuild.

## #17 — Document + maybe add some flexibility for powerspectrum convention → CLOSE (shipped)

v1.0.0 adds selectable BLS power conventions: every BLS entry point accepts
`convention='chi2ratio' | 'snr' | 'loglik'`, and `convert_bls_power()`
converts standalone periodograms. `'snr'` is verified equal to astropy's
`objective='snr'` power at the same solution, and `'loglik'`'s relation to
astropy's `objective='likelihood'` is documented and tested. The LS and BLS
doc pages now each carry an explicit "Power-spectrum convention" section.
Closing as shipped in v1.0.0 (see the CHANGELOG entry citing this issue).

## #19 — Add benchmarking to show speedups → CLOSE (shipped)

v1.0.0 publishes a full measured benchmark suite: `docs/BENCHMARK_RESULTS.md`
(BLS vs astropy across 7 GPU architectures, survey-scale Lomb–Scargle vs
nifty-ls, the TLS-vs-GTLS head-to-head, Keplerian-grid savings, survey cost
projections), with raw JSON + configs archived under `benchmarks/results/`
and the written protocol preserved at
https://github.com/johnh2o2/cuvarbase/blob/archive/pre-1.0-process/analysis/BENCHMARK_PROTOCOL_V1.md.
The release notes carry the headline tables. Closing as shipped in v1.0.0.

## #28 — Refactor the cuvarbase codebase for improved quality, efficiency, and usability → CLOSE (v1.0 is this refactor)

Status at v1.0.0, which was effectively this issue's execution: memory
management refactored into `cuvarbase.memory` with genuinely pinned host
buffers; thread-safe LRU kernel caching (34× on per-lightcurve loops);
lazy CUDA context + PEP 562 imports (`import cuvarbase` works GPU-less);
typed exceptions and input validation that raises on bad data; scikit-cuda
and `future` dropped; Python 3.9+ and numpy 2.x; CI (CPU suite, packaging
smoke, flake8); a September 2026 per-method soundness audit whose 25
confirmed defects are fixed with regression tests; the GPU-validated test
suite grew from ~37 tests to <N> (0 skipped at the release gate) with a
14-check release gate. Remaining polish items (docstring audit, notebooks, naming) are
tracked in the v1.1 roadmap (#ROADMAP). Closing — further quality work will
be scoped as concrete issues rather than this umbrella.

## #29 — Refactor and enhance documentation throughout the codebase → CLOSE (superseded; residue → roadmap)

The v1.0.0 release rebuilt the documentation end-to-end: modern Sphinx build
(the site had been frozen on a 2017 build), new TLS narrative page, PDM page,
a real false-alarm-probability section, fixed examples (several had
NameErrors / dead numpy APIs), corrected complexity claims, Keplerian-grid
citations (Seager & Mallén-Ornelas 2003; Ofir 2014), and a rewritten INSTALL.
The remaining items from this umbrella — a full docstring audit and example
notebooks — are tracked in the v1.1 roadmap (#ROADMAP). Closing in favor of
that concrete list.

## #30 — Standardize code conventions and modernize naming → CLOSE (wontfix-unless-2.0)

With 1.0.0 published, the public API is frozen under semver: a broad renaming
pass would be a compatibility break for existing pipelines (including the
TESS QLP production deployment) for cosmetic benefit. Recording the decision
as wontfix-unless-2.0 in the v1.1 roadmap (#ROADMAP) — if a 2.0 ever
happens, naming gets standardized there with a deprecation cycle. Internal
code style is enforced by flake8 in CI as of v1.0.0. Closing.

## #32 — Optimize efficiency of core algorithms (conditional entropy, etc.) → CLOSE (shipped/superseded)

The optimization work this issue asked for shipped across v1.0.0, largely via
@astrobatty's contributions: CE enhancements and bug fixes (PR #61 — with CE
now in maintenance mode and `periodfind` recommended for actively-developed
GPU CE/AOV), fast PDM kernels (PR #62), plus the BLS survey-speed campaign
(2.0–12.7× end-to-end), sparse-BLS vectorization, the survey-scale TLS
engine, and the Sep-2026 pass (kernel caching on every BLS entry point,
device-sized CE `use_fast` grids, pooled PDM buffers, a stacked
multiharmonic LS solve, a numpy solver for the Keplerian frequency grid). Per-algorithm performance work continues as concrete scoped issues
(v1.1 roadmap #ROADMAP) rather than this umbrella. Closing.

## #33 — Elevate and complete phase dispersion minimization capabilities → CLOSE (shipped)

Updating the June checklist: everything on it has now shipped in v1.0.0 —
fast CUDA kernels for all four variants (GPU-validated on A5000, recovery
table archived), the modern `(t, y, err)` API, `batched_run_const_nfreq()`,
memory-capped `large_run()`, Sphinx docs (`pdm.rst`, also closing #15), and
a measured benchmark (`scripts/benchmark_pdm.py`; PDM rows in the 0.2.6
head-to-head — the new `_fast` kernels measure 1.33×). The one residual
(a CPU-PDM/PyAstronomy comparison figure for the docs) moves to the v1.1
roadmap (#ROADMAP). Closing as shipped.

## #63 — Replace abandoned scikit-cuda → CLOSE (shipped)

Done in v1.0.0 — and more thoroughly than the interim plan discussed above:
rather than shipping the compatibility shim, the scikit-cuda dependency is
**gone**. The cuFFT calls (its only use) now go through a minimal in-house
ctypes binding (`cuvarbase._cufft`: Plan/fft/ifft/cufftEstimate1d, lazily
loaded), validated on an RTX A5000 (full LS/NFFT suite green, FFT matches
scipy, performance within ~2% of scikit-cuda's binding). No cuvarbase module
imports scikit-cuda; the numpy shim is deleted; numpy ≥1.24/2.x environments
work. Verified in CI by a build-wheel → clean-venv → import smoke test.
Closing as shipped in v1.0.0. Thanks @astrobatty for pushing on this one.

---

# NEW ISSUE: "v1.1 roadmap" (post the day v1.0.0 ships; label: enhancement)

**Title: v1.1 roadmap — deferred work consolidated from the v1.0 cycle**

v1.0.0 closed the historical umbrella issues (#28, #29, #30, #32, #33) in
favor of this single tracked list. Items are roughly priority-ordered;
none are release-blocking regressions — the v1.0.0 CHANGELOG's "Known
limitations and deferred work" section is the user-facing summary.

**Validation / correctness**
- [ ] NUFFT-LRT (`cuvarbase.nufft_lrt`): shipped in 1.0.0 as EXPERIMENTAL
      after the Sep-2026 fixes and the 2026-09-06 injection-recovery
      re-validation (importable, quarantined from the top-level namespace,
      warning at construction, outside the 1.x stability promise). The
      re-validation passed the correctness gate (default path exact on
      BJD-scale times, epoch search works; Detector A = sequential
      baseline; numbers on the docs page). Remaining before promotion:
      a finer default epoch grid (the default costs 4-9 % completeness
      against `epoch_oversample` ~4 at the longer durations), one return
      convention for `run()` (tuple vs array), a sane `durations=None`
      default, a decision on `dy`, and a cotrend-then-BLS/TLS comparator
      in the harness; then bring the module into the top-level namespace
      and the stability promise
- [ ] Legacy TLS kernel (`use_fast=False`): formal deprecation or removal
      of the legacy path (the fast batch path is the default on every
      entry point)
- [ ] float64 frequency/period grid builders (the grids are built in
      float64 and handed to the kernels as float32 today)
- [ ] Thread-safety of the `*AsyncProcess` objects (undocumented in 1.0:
      the kernel cache is thread-safe, the process objects are not, and
      nothing states a one-process-per-thread rule -- document it or make
      them safe)

**Performance (measured opportunities on record)**
- [ ] TLS fast kernel: XOR-swizzle for the 32-way shared-memory bank
      conflicts when NBINS/n_t0 ≡ 0 mod 32 (~2–4% of trials)
- [ ] TLS: chunk-pipelined GPU/CPU overlap (double-buffering) for
      multi-chunk surveys
- [ ] TLS coarse-kernel rewrite: a band-of-periods-per-block kernel so the
      light curve is not re-folded once per trial period (87-100% of kernel
      time today; the audit measured 2-4× available)
- [ ] PDM `_fast` kernel rewrite (numerically equivalent to the default
      kernels but only 0.7-2.0× on Ada; either make it win or drop it)
- [ ] CE float32 frequency grids on the device (float64 upload today)
- [ ] Multi-GPU dispatch (the one architectural idea worth adopting from the
      GTLS comparison; see docs/GTLS_COMPARISON.md)

**Packaging / infrastructure**
- [ ] Dockerfile rebuild (the 2023 one was deleted in 1.0: it never
      installed cuvarbase and pulled the dropped scikit-cuda); smoke-build
      it on a GPU pod before it ships
- [ ] `cuvarbase._cufft` hardening (the in-house ctypes cuFFT binding:
      library discovery across CUDA layouts, error-code coverage)
- [ ] Stellar-parameter overrides for the Keplerian frequency/duration
      grids (per-target `R_star`/`M_star` in the batch APIs)

**Docs / community**
- [ ] Docstring audit + example notebooks (residue of #29)
- [ ] PDM: CPU-PDM (PyAstronomy) comparison figure for the docs (residue of #33)
- [ ] JOSS paper + Zenodo DOI (needs the published release; QLP's DRN 003
      currently credits cuvarbase only in a GitHub footnote)
- [ ] ASCL record update

**Algorithm wishlist** (formerly the README "Planned Features" section)
- [ ] (Weighted) wavelet transforms
- [ ] Spectrograms (for PDM and GLS)

**Benchmarks to refresh when the ecosystem moves**
- [ ] astropy 8.0 re-benchmark when its LRA-NUFFT Lomb–Scargle default ships
- [ ] CETRA comparison remains out of scope (different algorithm family) —
      revisit only with a matched-statistics protocol

**Decisions recorded**
- API renaming (#30): wontfix-unless-2.0 — the 1.0 API is frozen under
  semver; a renaming pass would break production users (TESS QLP) for
  cosmetic benefit.
- CE stays in maintenance mode; `periodfind` is the recommended actively
  developed GPU CE/AOV package.
