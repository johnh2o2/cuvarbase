# cuvarbase v1.0 — Deep Audit & Release Gameplan

*Generated 2026-06-10 from a 75-agent audit of `v0.2.6..HEAD` (testing/runpod-benchmarks), with adversarial verification of every critical/high finding (29 confirmed, 1 disputed, 0 refuted) and fresh competitive research.*

---

## 1. What changed since v0.2.6 (May 2025)

124 commits, ~26,000 added lines across 128 files. Library surface (~8,500 new lines in `cuvarbase/`):

| Area | What landed | Verdict |
|---|---|---|
| BLS core | Optimized kernel (`bls_optimized.cu`), `eebls_gpu_fast_optimized/adaptive`, thread-safe LRU kernel cache | **needs-work** |
| Sparse BLS | Panahi & Zucker GPU+CPU, now default in `eebls_transit` for ndata<500 | **needs-work** |
| Batch BLS + Keplerian grids | `bls_batch.cu`, `eebls_gpu_batch`, `bls_frequencies.py` | **needs-work** |
| TLS (new) | `tls.py` + grids/models/stats + `tls.cu` (~2,700 lines) | **cut from v1.0** |
| NUFFT-LRT (new) | `nufft_lrt.py` + kernel (Taaki method, Copilot-implemented) | **cut from v1.0** |
| LS / NFFT | Memory-class refactor, cuFINUFFT backend | **needs-work** (refactor itself is clean) |
| Core refactor | `base/`, `memory/`, `periodograms/` subpackages | **not-ready** (packaging) |
| Packaging/docs/CI | pyproject, Docker, CI workflow, large benchmark+docs apparatus | **not-ready** |
| Tests | Sparse-BLS ground-truth suite (excellent), TLS/NUFFT-LRT/batch coverage (illusory) | **needs-work** |

### What is verifiably GOOD (keep and build on)
- **Every v0.2.6 entry point is byte-identical** — full backward compatibility on default paths.
- **mod1_fast overflow fix and the `full_bls` warp-shuffle fix are correct** (verified by re-derivation).
- **Sparse BLS core is correct and exemplarily tested**: CPU/GPU verified line-by-line consistent, wrapped-transit fix merged, 60 ground-truth tests pass locally against an independent O(N²) brute force.
- **`bls_frequencies.keplerian_freq_grid` is numerically identical to `bls.transit_autofreq`** (at qmin_fac=1) — exact equivalence verified.
- **tls_grids Ofir implementation is faithful** (correct durations, exact oversampling scaling); tls_stats SR/SDE core formulas correct.
- **Memory-class extraction is AST-identical to v0.2.6** — no behavior drift in the refactor itself.
- **Batch kernel is a correct port** (line-for-line vs optimized single-LC kernel; GPU corr>0.999).
- **Survey-scale benchmark methodology is sound** and every README table number traces exactly to `benchmark_results_new_features.json`.
- **Branch hygiene**: all historical bugfix branches (memory leak, resource handle, autoprimaryctx, PDM centering, wrapped transits) are confirmed merged; FFA-BLS was honestly archived as a negative result.

---

## 2. Confirmed defects (all adversarially verified, file:line evidence in audit transcripts)

### P0 — Release blockers (v1.0 cannot ship with any of these)

1. **The built wheel/sdist cannot be imported at all.** `pyproject.toml:56` and `setup.py:39` hard-code `packages = ["cuvarbase", "cuvarbase.tests"]`, omitting the new `base/`, `memory/`, `periodograms/` subpackages that `__init__.py` imports unconditionally. Reproduced: built wheel → `ModuleNotFoundError: No module named 'cuvarbase.base'`. RunPod testing never caught it because it ships the source tree. *Fix: `find_packages()`/`packages.find` + a CI job that builds the wheel, installs into a clean venv, imports it.*

2. **Default pip install is broken on any modern environment.** numpy is unbounded, scikit-cuda (last release 2019) breaks on numpy≥1.24, and the new eager `__init__.py` imports drag `skcuda.fft` in at import time (v0.2.6 didn't). So even after fix #1, `import cuvarbase` crashes on numpy 2.x. The repo's own RunPod scripts monkey-patch skcuda. External issue "Replace abandoned scikit-cuda" (astrobatty, Apr 2026) is open. *Fix for v1.0: lazy module imports (PEP 562 `__getattr__`) so BLS/CE/PDM never touch skcuda; ship the numpy-compat shim before any skcuda import; document. Post-1.0: drop skcuda (cupy or cuFFT via pycuda).*

3. **`eebls_transit` — the flagship transit API — crashes on its default path.** For ndata<500 (sparse GPU default) any documented kwarg (`rho`, `samples_per_peak`, `dlogq`, `noverlap`…) raises `TypeError` because `sparse_bls_gpu` has a closed signature. Reproduced. It also **silently ignores the Keplerian `qmin_fac`/`qmax_fac` constraints** on the sparse path — results are not comparable across the ndata=500 threshold. *Fix: kwargs filtering/absorption (small) + pass per-frequency q bounds into the sparse kernels or loudly document.*

4. **`bls_optimized.cu:391` `reduction_max` silently drops half the reduction candidates** (`s > 32` loop bound — the exact bug commit 72ae029 fixed in the sibling kernel). Silently wrong results for `use_optimized=True` via `eebls_gpu`/`eebls_gpu_custom`. *Fix: one character (`>=`), plus a regression test; better, merge bls_optimized.cu into bls.cu so fixes can't diverge again.*

5. **`lomb_scargle_simple` inverts its weights.** It pre-normalizes `dy**-2` and passes the result as `dy`, which `setdata` squares-and-inverts again — largest-error points get the most weight. Pre-existing since ≤v0.2.6, but it ships in v1.0's public API. *Fix: pass `dy` straight through.*

6. **CI is theater and the docs claim otherwise.** The "Tests" workflow never invokes pytest (`pip install -e .` is `continue-on-error` and always fails); README/CHANGELOG claim "automated testing across Python 3.7–3.12". Meanwhile the shipped `test_readme_examples.py` is never collected (a `@mark_cuda_test` on a class) and would crash if it ran (stale `eebls_gpu` tuple contract). *Fix: stub-conftest CPU CI (93 real tests collect and pass locally with pycuda stubs — verified), packaging smoke job, honest claims.*

7. **Metadata is self-contradictory.** Code requires Python ≥3.9 (`importlib.resources.files`); setup.py says ≥3.7, pyproject ≥3.8. `__version__ = "0.4.0"` vs "v1.0" branding; CHANGELOG's top entry is 0.4.0 and describes none of the actual release content; PyPI would render the 2017-era README.rst.

8. **Git/repo hygiene blockers.** Local `v1.0` branch is **23 commits ahead of `origin/v1.0` (unpushed)** — including the TLS merge and sparse-BLS correctness fixes. A junk local branch literally named `origin/v1.0` points at master's tip and makes the ref ambiguous (observed git warnings). *Push, delete the junk ref, then fast-forward v1.0 to HEAD.*

### P0/P1 — Credibility (doc claims that don't survive checking)

- **TLS "35-202× faster than CPU TLS"** (README ×2): no benchmark data exists anywhere in the repo or its history; the number predates a substantial rewrite of the implementation. Remove or re-measure.
- **LS "1.5-62× faster than nifty-ls"**: the 62 endpoint is derivable from nothing in the repo (real measured range: 1.5–12.6×, plus >27× timeout bounds). Also the HAT-Net ">>6×" understates its own data (>15.6×).
- **Fabricated citation**: BENCHMARK_RESULTS.md cites "Barnsley & Sherley (2024)" for nifty-ls — the real paper is **Garrison, Foreman-Mackey, Shih & Barnett, arXiv:2409.08090**. (Ironic, given commit fc8c032 existed to remove fabrications.)
- **Stale "5-90×" adaptive-BLS claims** (README ×3): synthetic ndata<64 result; realistic measurement is 1.4–5.3×. The strongest *defensible* number — **BLS 257–354× vs astropy, consistent across 7 GPU architectures** — is currently cited nowhere (orphaned in `benchmark_results_by_gpu/`).
- **Honest negative results were deleted** in the Feb 2026 doc rewrite (nifty-ls CPU beats GPU LS 7–66× at small Nf). Restore selective disclosure — it's what makes the strong claims believable.
- `docs/FBLS_GPU_SPEC.md` ships verbatim LLM self-correction text ("Wait — this isn't quite right…") and a 3800× claim that omits the m-bins factor (~38× correct); the experiment already lost 14× to Keplerian-grid BLS. Treat FFA as a documented negative result, not a roadmap item.

### Feature verdicts: TLS and NUFFT-LRT

**TLS — algorithmically unsound as shipped; cut from v1.0, rework for v1.1.**
- Both kernels hard-code `n_t0 = 30` epoch trials per period. Transit windows narrower than 1/30 of phase mostly never overlap a tested epoch: the flagship Keplerian mode is below t0 resolution for essentially **all P > ~3.5d** (CPU simulation of the exact kernel loops: 8/8 epochs missed at P=100d).
- Shared-memory layout caps **ndata at 3,592** (48KB) — TESS (~20K) and Kepler (~65K) light curves fail at launch; docs claim 100,000. Git history shows a guard was deliberately removed on a misconception.
- A `chi2=1e30` sentinel for failed periods collapses SDE from 15.3 → 0.06 and sends FAP to 1.0 (reproduced with the real tls_stats).
- `signal_to_noise` is inflated by √n_transits; FAP calibration constants are invented; bitonic sort provably fails to sort every non-power-of-2 size (harmless — downstream is permutation-invariant — but it's pure wasted work with misleading names).
- Zero automated coverage; the promised accuracy test vs `transitleastsquares` never existed.
- *The salvage path is real*: grids/models/stats foundations are mostly sound; needed kernel work is duration-dependent t0 stride, ootr-style chi2 precomputation, shared-memory chunking or global-memory path, sentinel masking, stats fixes, and a golden test vs the reference TLS.

**NUFFT-LRT — neither NUFFT nor GPU; cut from v1.0 (or ship as clearly-experimental with the bugs fixed).**
- All computation is CPU `np.interp` + `np.fft.rfft`; the 6 CUDA kernels are compiled but never called (so it *requires* a GPU+nvcc while computing nothing on them).
- The uniform grid spans `median_dt × 2N`, silently **ignoring all data beyond that span** — for multi-season data (its advertised use case) most of the light curve never enters the computation (reproduced: perturbing season 2 by +100 changed output by 0.0).
- Tests are circular (test local reimplementations), import tests grep strings.
- *Salvage path*: wire `compute_nufft` to the existing, working `cunfft.NFFTAsyncProcess`; use the shift theorem for epoch sweeps; coordinate with Jamila Taaki on validation. This is a genuinely interesting method (correlated-noise transit detection) worth doing right in a later release.

### Disputed (needs GPU profiling, not code reading)
- **Batch BLS 12× TESS-regime regression**: kernel is correct, but the docs' explanation can't produce the observed magnitude; one verifier refuted the launch-config diagnosis. Run `nsys`/`ncu` before publishing any batch guidance. Until then, document batch mode as "for ndata ≲ 1000".

---

## 3. Improvement opportunities (beyond fixes; impact/effort)

**Quick wins (small effort, high value)**
- Route `eebls_gpu_fast`/`eebls_transit` through the existing kernel LRU cache (~150ms compile currently paid per call; the "adaptive" speedup mostly *is* this).
- Subtract `t.min()` host-side in `BLSMemory.setdata`/`BLSBatchMemory` — BJD-scale timestamps currently degrade float32 phase folding (demonstrated 0.705→0.285 power loss).
- Cache cuFINUFFT `Plan` objects per memory object — likely flips the cufinufft backend from 0.7× to >1× and may beat the custom NFFT.
- Return q values from `keplerian_freq_grid` and wire per-frequency q bounds into the batch kernel (GPU side already supports it).
- Stub `conftest.py` (proven pattern from the FFA branch) → 93 real CPU tests in CI immediately.
- Promote `benchmark_new_features.py --tests-only` correctness checks into pytest GPU tests.

**Medium**
- Merge `bls_optimized.cu` into `bls.cu` (the duplicate-kernel drift already caused one shipped bug).
- Centralize block_size validation in `compile_bls`; make the adaptive heuristic consider nbins, not just ndata.
- Restore page-locked host buffers (removed on a false premise in 4e6e232) — transfers currently serialize.
- Vectorize `sparse_bls_cpu` (numpy cumsums, ~100×) so the CPU fallback is usable.
- Sphinx docs: fix fatal conf.py, add autodoc for all new modules.
- Prune: `periodograms/` (crashes on import, written from imagination), `kernels/test_minimal.cu`, `docs/copilot-generated/` (16 stale AI files), root benchmark JSONs → `benchmarks/`, fix or delete broken `scripts/run_benchmark_remote.sh`.

---

## 4. Strategy: brand, scope, and the competitive map (verified June 2026)

### The market, method by method
- **BLS/transits**: TESS QLP has run cuvarbase GPU BLS in production since Sector 59 (Kunimoto+ 2023 — verified quote, links the repo); Sha+ 2026 used it for a 5-year TESS search; GPFC (MNRAS 2024) used it as the GPU baseline. **But** CETRA (Smith+ 2025, GPU/PyCUDA, active) is now the *officially named* PLATO pipeline detection algorithm (Cabrera+ Apr 2026, verified quote) and powered a 10,000-candidate TESS search. "The only GPU transit search" is no longer claimable; "the GPU BLS that searches every TESS sector" is.
- **TLS**: `transitleastsquares` is dormant (last release Nov 2021) yet has 5,233 downloads/month; its GPU feature request has sat open since **2019** with zero takers. **"First GPU TLS" is an uncontested, high-demand claim — which is exactly why it must not ship broken.**
- **LS**: nifty-ls is active but small (48 stars, ~839 dl/mo); its GPU heterobatch is an unmerged PR and it can't do nonuniform frequency grids — precisely where cuvarbase wins. ⚠️ astropy 8.0 (imminent) makes an LRA-NUFFT the default for `method='fast'` — **re-benchmark against astropy 8.0 before publishing v1.0 claims**, and pin the astropy version in benchmark docs.
- **CE**: Katz GCE is **abandonware** (last commit Jul 2020, unanswered install issues, ~7 citations) — do *not* point users there. The defensible referral is **scope-ml/periodfind** (Coughlin/ZTF-SCoPe; GPU CE+AOV+LS+BLS+FPW; PyPI Feb 2026, commits June 2026). Note: periodfind is also the closest *brand* competitor.
- **PDM**: cuvarbase's GPU PDM is apparently **the only one in existence** (verified absence); demand is tiny. Keep in maintenance mode; don't invest.
- **New methods to watch**: FPW (Finkbeiner/Prince/Whitebook 2025 — GP-derived waveform-agnostic phase folding, CPU+GPU, built for ZTF's 1.5B objects, already inside periodfind) targets the CE/PDM use case. nuance (JAX, GP-based) owns the active-star niche.
- **Open territory**: **no GPU Fast Template Periodogram exists anywhere** (verified). As the FTP author, you uniquely own this. It fits a "fixed-shape signal search" brand (transits, RR Lyrae) far better than CE/PDM maintenance does.
- **Rubin/LSST**: data products don't include periods; DP1-era period finding is astropy LS on CPU Dask clusters via LINCC's LSDB. A worked **LSDB + cuvarbase GPU-node example** targets where survey period-finding actually happens.

### Recommended brand

> **cuvarbase: GPU period finding at survey scale — transit search (BLS, TLS) and Lomb-Scargle for millions of light curves.**

Two pillars, both backed by production evidence:
1. **Transits** (spearhead): standard BLS (QLP-proven, 257–354× vs astropy across 7 GPUs) + Keplerian frequency grids (14–24× measured) + sparse BLS for ground-based, with GPU TLS as the v1.1 flagship once correct.
2. **Survey-scale LS** (second pillar): the batched, huge-Nf regime (365K–1.8M frequencies) where nothing else finishes.

De-scope honestly:
- **CE: deprecate** with a pointer to periodfind (not GCE). The unmerged `feature/period-derivative-search` branch already drafted a gce wrapper — supersede it with a deprecation note referencing periodfind.
- **PDM: freeze** ("maintenance only; the only GPU PDM — kept for QLP-era users; PyAstronomy for CPU").
- **NUFFT-LRT: experimental**, off by default, until rebuilt on the real NFFT with Taaki.
- **FFA/fBLS: published negative result** (the archive commit message already says it best).

This is not "all-in on planets" — it's *all-in on the two things with receipts*, with the variability community served by LS (their actual workhorse) rather than by maintaining three also-ran methods.

---

## 5. v1.0 release gameplan

**Phase 0 — stop the bleeding (½ day)**
1. Delete junk local branch `origin/v1.0`; push local `v1.0` (23 unpushed commits) and `testing/runpod-benchmarks`.
2. Fix packaging (`find_packages`), bump `__version__`, reconcile Python floor to ≥3.9 everywhere.

**Phase 1 — correctness & honesty (1–2 weeks)**
3. Fix: `reduction_max` s≥32; `eebls_transit` sparse kwargs + q-constraint handling (or loud documentation); `lomb_scargle_simple` weights; empty-dict `compile_bls` KeyError; block-size validation.
4. Lazy `__init__` imports (PEP 562) + skcuda numpy-shim; verify `import cuvarbase` and BLS-only use on numpy 2.x without skcuda.
5. Delete `periodograms/`, `test_minimal.cu`, `docs/copilot-generated/`; fix/delete `test_readme_examples.py` and circular `test_nufft_lrt_algorithm.py`.
6. README/docs truth pass: drop 35-202×, 1.5-62×, 5-90×; fix the nifty-ls citation; promote the 257–354×/7-GPU and Keplerian-grid numbers; restore the small-Nf nifty-ls disclosure; re-benchmark LS vs astropy 8.0 when it lands.
7. CI: stub-conftest CPU suite (93 tests) + build-wheel-install-import smoke job on every push.
8. Real 1.0.0 CHANGELOG; retire README.rst or regenerate from README.md.

**Phase 2 — scope cut (the decision)**
9. Move TLS and NUFFT-LRT out of the v1.0 surface (don't export; mark experimental in docs; or move to an `experimental/` namespace). Both have public, verified defects that would define the release's reputation if shipped as headline features.
10. Write deprecation notices for CE (→ periodfind) and freeze note for PDM.

**Phase 3 — release ritual**
11. Full GPU suite on RunPod (`gpu-test.sh` default changed to the whole suite), archive the junit report in the release notes.
12. Merge to master (clean fast-forward — verified only 3 README-notice commits diverge), tag v1.0.0, publish to PyPI from the built-and-smoke-tested wheel.
13. Update the ASCL record; consider a JOSS paper for citability (users currently cite a footnote URL).

**v1.1 and beyond (the brand payoff)**
- **TLS done right** (t0 stride ∝ duration, ootr precompute, >3,592-point support, golden tests vs transitleastsquares, measured benchmark) → "first GPU TLS" announcement.
- Batch BLS: profile the TESS regression with nsys; reuse pinned memory; then publish batch guidance.
- cuFINUFFT plan caching; nifty-ls-style astropy `method=` registration; LSDB worked example.
- **GPU Fast Template Periodogram** — the novel-science differentiator nobody else can claim.
- NUFFT-LRT rebuilt on cunfft with Taaki as co-author of the validation.
- Engage astrobatty (4 merged PRs this spring) as a maintainer — the "pass the torch" goal from the README needs exactly this person.

---

## 6. Risk register

| Risk | Mitigation |
|---|---|
| astropy 8.0 LRA-NUFFT changes the LS comparison baseline | Re-run LS benchmarks on 8.0 before tagging; pin versions in docs |
| CETRA captures "GPU transit search" mindshare via PLATO | Lead with QLP/TESS production record + multi-method breadth; benchmark vs CETRA before any comparative claim |
| periodfind overlaps BLS/LS/CE | Differentiate on transit depth (Keplerian grids, TLS, batch) and published, honest benchmarks |
| scikit-cuda decays further | v1.0: lazy imports + shim; v1.x: replace cufft dependency (the only skcuda use) |
| Solo-maintainer bus factor | JOSS paper + contributor onboarding (astrobatty), CI that makes external PRs safe |

---

## 7. Ratified plan — June 11, 2026 (post-overnight-fixes)

**State**: `v1.0-fixes` carries the 15 overnight fix commits (all 15 audit tasks
done; 108 CPU tests pass; wheel builds+imports). A fresh `git fetch` then
revealed the earlier branch audit ran on stale refs: **remote `origin/v1.0`
gained community PRs #57–62 (Feb–Apr 2026, largely astrobatty)** that our line
lacks — `v1.0-fixes` is 43 ahead / 29 behind the real remote v1.0. The junk
local branch `origin/v1.0` that shadowed the remote ref has been deleted.

**What the unmerged remote work contains**:
- PR #62: PDM refactor — fast PDM CUDA kernels + hooks, (t,y,err) run() API
  (backward compatible), docstrings, tests. *Directly satisfies the "PDM:
  maintain and grab efficiency/usability gains" decision.*
- PR #61: CE enhancements — normalization before processing, compute_log_prob,
  32-bit stream-count overflow check, use_fast+weighted guard
- PR #60: remove inline normalization in LombScargleAsyncProcess
- PR #59: improved LS memory estimation (overlaps our memory_requirement fix —
  reconcile, keep the better)
- PR #58: packaging via setuptools packages.find (same intent as our fix —
  reconcile)
- PR #57/#26: normalize-light-curves for LS/PDM (also now in origin/master,
  which is 6 ahead of local master)

**Ratified decisions**:
1. **PDM: maintain + improve** (only GPU PDM in existence). Integrating PR #62
   delivers most of it; follow with docs (issue #15).
2. **CE: deprecation/maintenance notice pointing to scope-ml/periodfind**
   (NOT gce — abandonware). Close PR #48 (gce wrapper) with explanation.
3. **TLS: not shipped as working** — experimental status stands. The v1.1
   flagship is the rework (duration-scaled t0 grid, shared-mem fix, stats
   fixes, golden tests vs transitleastsquares), then a *measured* speedup
   benchmark. "First GPU TLS" is verified uncontested (no GPU TLS exists
   anywhere; hippke/tls#51 open since 2019); the old 35-202x number has no
   backing data and stays dead.
4. **README: second honesty+significance pass** — lead with numbers that carry
   meaning: TESS QLP runs cuvarbase in production since Sector 59 (Kunimoto+
   2023); standard BLS 257–354x vs astropy across 7 GPU architectures; all
   four major surveys (ZTF+HAT-Net+TESS+Kepler LS+BLS) processable for ~$33
   of GPU time; Keplerian grids 4–37x fewer frequencies. Keep the honest
   caveats (nifty-ls wins small problems; batch BLS for ndata<1000).
5. **Versioning: no ceremonious release yet.** Tag `v1.0.0` on the integrated,
   GPU-validated result; v1.1 tracks the TLS rework line. PyPI publish is a
   separate, deliberate step later.

**Next-session execution order**:
1. `git fetch --all --prune` (refs went stale once already).
2. Merge `refs/remotes/origin/v1.0` into `v1.0-fixes`. Conflict guidance:
   keep our lazy PEP-562 `__init__` + skcuda shim; reconcile the two
   packaging fixes (same intent); reconcile LS memory_requirement vs PR #59;
   keep our lomb_scargle_simple weights fix; take PDM #62 and CE #61
   wholesale; re-check that their CE/PDM changes don't reintroduce eager
   skcuda imports.
3. Gate: full CPU suite (expect ≥108 passed, 0 failed) + wheel smoke test +
   flake8 error class.
4. CE deprecation notice (README + ce.py docstring note → periodfind).
5. README significance pass (item 4 above).
6. GitHub hygiene: close PR #55 (already merged via 1ed5639), retarget/close
   #56 (superseded by this integration), close #48 (CE referral changed to
   periodfind), comment on issue #63 (skcuda shim + lazy imports shipped;
   full replacement post-1.0), comment on issue #33 (PDM plan).
7. Push `v1.0-fixes`; fast-forward/push `v1.0` to the integrated result.
8. GPU validation on RunPod (full suite + reduction_max equivalence +
   kernel-cache timing + benchmark_new_features --tests-only), then tag
   v1.0.0.
