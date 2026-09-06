# v1.0.0 Release Runbook

**HOLD: nothing below executes until the explicit, @astrobatty-coordinated go
(maintainer directive, Jul 4 2026, reaffirmed Sep 3 2026).** Phases 4 and 5
(re-validation, freeze, gate, rehearsal) are preparation that needs no go and
touches no shared ref; the "Release day" section is the only part that
merges, tags, publishes or pushes anything other than a staging branch.

Plan of record: `analysis/audit-sep2026/EXECUTION_PLAN.md` (Phases 4-7) and
`analysis/audit-sep2026/RELEASE_READINESS.md` (blockers 1, 2, 3, 6, 11, 12,
14; section 8 Phases B-E). Decisions D1-D4 there stand and are not reopened
here.

## State as staged (Sep 2026)

- Release branch: `v1.0-fixes`. It carries everything since 0.2.5 (PRs
  #57-#68, the July audits) plus the September 2026 work: Phase 1 (25
  confirmed correctness defects fixed, input validation that raises) and
  Phase 2 (performance), GPU-gated at `000c299` on a shared NVIDIA A40
  (1582 passed / 0 failed / 0 skipped), and the Phase 3 hygiene on top
  (API freeze, packaging, test hygiene, docs consistency, repo prune,
  this runbook). CI: CPU suite green on every push.
- Audits of record for 1.0: `analysis/audit-sep2026/` (release readiness,
  algorithm audit, execution plan, NUFFT-LRT campaign, repro scripts). The
  July audits (`tls-audit-jul2026.md`, `claims-trace-jul2026.md`,
  `nufft-lrt-audit-jul2026.md`) and the July gate record
  (`v1.0-release-gate-jul2026/`) are kept in `analysis/` as well; only the
  pruned material listed under the archive tag below is archive-only.
- GPU gate: the September Phase 1-2 gate (`000c299`) is superseded by
  Phase 3 and will be re-run in full on the frozen tree T (Phase 5). No
  gate record for T exists yet; that record is commit T'.
- GPU test count: it flows ONE way. Phase 5 measures it once on the
  candidate tip C (full suite, 0 skipped), writes it into
  `docs/RELEASE_NOTES_v1.0.0.md` as the last content commit -- that commit
  is T -- and the gate run on T must reproduce the same count. The notes
  are part of T, so they are necessarily edited BEFORE T; the gate on T
  confirms the count, it does not produce it.
- Old `v1.0.0` tag: annotated object `afa9741` pointing at `5553248`
  (Jun 11 2026), exists on origin, STALE (303 commits behind at `47e0ae3`,
  Sep 5 2026 -- `git rev-list --count 5553248..v1.0-fixes` for the current
  figure; its message says 0.2.6 was the last PyPI release and cites 568
  tests, both wrong).
  Never published to PyPI; no GitHub Release exists for it (the only
  GitHub Release ever is v0.2.1 from 2021). It is deleted and re-created
  on release day (step 5).
- Archive tag: `archive/pre-1.0-process` exists LOCALLY only (annotated,
  on the last pre-prune commit; created by the Phase 3 orchestrator). It is
  pushed on release day together with `master` and `v1.0.0`; the pruned
  material (`BENCHMARK_PROTOCOL_V1.md`, `GTLS_COMPARISON.md`, the TESS/TLS
  cost analyses and punchlists, the June `v1.0.0-gpu-validation/` record,
  the raw `pr65-resolution-jul2026/` and `kernel-hygiene-jul2026/` files,
  the one-off scripts) is reachable through it, never through `master`.
  Verified Sep 5 2026 at `47e0ae3`: the tag exists locally and is an
  ancestor of `v1.0-fixes` (pre-flight box below).
- Local `master` is stale: `ec53ae8` versus `origin/master` at `060d839`
  (PR #26, `normalize_light_curves`). Every merge step below starts with
  `git reset --hard origin/master`; never merge from the stale local ref.
- Merge shape against `origin/master` (verified Sep 5 2026 with
  `git merge-tree --write-tree origin/master v1.0-fixes`): exactly FOUR
  content conflicts, `README.rst`, `cuvarbase/lombscargle.py`,
  `cuvarbase/pdm.py`, `cuvarbase/utils.py`. All four are resolved by taking
  the `v1.0-fixes` side (master's `README.rst` banner is obsolete because
  the branch's `README.rst` is a pointer stub; PR #26's
  `normalize_light_curves` already exists in the branch's `utils.py` with
  tests and is wired into LS, PDM and CE). `origin/v1.0` (`89d5481`) is an
  ancestor of `v1.0-fixes`, so it fast-forwards.
- Version string: `cuvarbase/__init__.py` `__version__ = "1.0.0"`;
  `pyproject.toml` is the packaging source of truth (`setup.cfg` gone, so
  the wheel tag is `py3-none-any`; always address it by glob:
  `dist/cuvarbase-1.0.0-*.whl`). README.md is the PyPI long description.
- README flip: DONE on `v1.0-fixes` as the last Phase 3 content commit
  (banner removed, `pip install cuvarbase`, absolute links,
  `test_readme_consistency.py` inverted). There is no post-publish README
  step any more; pre-flight only verifies it (the PKG-INFO grep).
- PyPI: latest published version is 0.2.5 (Oct 2023); 0.2.6 was tagged
  but never uploaded. Publishing needs the maintainer's PyPI token: type
  `! twine upload dist/*` yourself in the session so the token never
  enters a transcript, or keep it in `~/.pypirc`.
- NUFFT-LRT (D1): importable as `cuvarbase.nufft_lrt`, quarantined (not in
  the top-level namespace, EXPERIMENTAL warning at first construction).
  Whether it ships "official" or "experimental" is decided by Phase 4
  (below) BEFORE the freeze; either way the docs text is written from the
  measured numbers before T is cut.

## Pre-flight checklist (release day, before step 1; every box or stop)

- [ ] Explicit maintainer go, coordinated with @astrobatty (draft in
      `analysis/release-staging-v1.0.0/astrobatty-message.md`; he is
      expecting "some changes in BLS" -- ask whether anything targets 1.0.0
      before tagging). @xiaziyna has been told what ships for NUFFT-LRT
      (`analysis/release-staging-v1.0.0/xiaziyna-message.md`).
- [ ] `git fetch origin --prune`. The release tree T is the `origin/v1.0-fixes`
      tip AFTER Phase 4 (NUFFT-LRT re-validation) and Phase 5 (freeze +
      gate). `origin/v1.0-fixes` must equal the local branch.
- [ ] CPU CI (GitHub Actions) is green at T and at T'.
- [ ] Gate record commit T' is the ONLY commit after T and touches
      `analysis/` only:
      `git diff --quiet T T' -- . ':!analysis' && echo TREE-OK`
      (T and T' are SHAs from `analysis/v1.0-release-gate-<date>/SUMMARY.md`).
      Anything else after T means: go back to Phase 5.
- [ ] `docs/RELEASE_NOTES_v1.0.0.md`: the leading `<!-- DRAFT ... -->`
      comment is gone; the GPU test count in the notes is the N that Phase 5
      step 0 measured on the candidate tip and wrote at T, and the gate run
      on T reproduced it: the "N passed" line of `suite_full.log` in the
      gate record (0 skipped, 0 failed) equals the count in the notes. (The
      notes were edited before T by construction -- they are part of T; a
      differing gate count means the run is investigated, never the notes
      edited after T.) The "if you fetched the June v1.0.0 tag, run `git
      fetch --tags --force`" line is present.
- [ ] Archive tag present locally and inside the branch history:
      `git tag -l archive/pre-1.0-process` prints the tag and
      `git merge-base --is-ancestor archive/pre-1.0-process v1.0-fixes`
      exits 0 (both verified Sep 5 2026 at `47e0ae3`). It is pushed in
      step 6; a missing or detached tag stops the release.
- [ ] `CHANGELOG.rst` top section is `1.0.0` (no "Unreleased" heading).
- [ ] `docs/source/nufft_lrt.rst` and the release notes say what Phase 4
      decided (official or experimental) and quote its archived numbers.
- [ ] PKG-INFO check on a fresh local build of T (`python -m build`):
      ```
      tar -xzOf dist/cuvarbase-1.0.0.tar.gz cuvarbase-1.0.0/PKG-INFO \
        | grep -n "Until v1.0.0\|git+https"
      ```
      MUST print nothing (exit status 1). And
      ```
      tar -xzOf dist/cuvarbase-1.0.0.tar.gz cuvarbase-1.0.0/PKG-INFO \
        | grep -n "0\.2\.5" | grep -v "since 0\.2\.5"
      ```
      MUST print nothing. Decision recorded here: the only permitted
      mention of 0.2.5 in the long description is the release-notes-style
      sentence "first release published to PyPI since 0.2.5" (or a line
      that contains the words `since 0.2.5`). Any other 0.2.5 mention is a
      leftover of the pre-flip README and stops the release.
- [ ] `gh-pages-staging` was rebuilt from T (its orphan commit message
      names T's SHA; `git ls-tree -r --name-only gh-pages-staging | grep
      -c "\.doctrees\|\.buildinfo"` prints 0; `.nojekyll` present).
- [ ] The merge rehearsal (Phase 5, last step) was done against the
      current `origin/master` and recorded four conflicts.

## Phase 3 GPU follow-ups (run on the Phase 5 pod, before the freeze)

**Status (Phase 4 pod, 2026-09-06, NVIDIA A40, CUDA 12.4, py3.11,
cufinufft 2.5.1, tree 954f037 + the one test fix below):** the whole
list was run ahead of Phase 5. `cuvarbase/tests/test_nfft.py`,
`test_lombscargle.py` and `test_nufft_lrt*.py` first (`-rs`, 211
tests): 210 passed, 1 failed -- `TestBatchedMemoryReuse::
test_per_call_use_double_matching_the_process_is_accepted` asserted two
double-precision batched runs bitwise equal; measured on the A40, 5 of
19 double repeats differ by up to 6.7e-15 relative (float64 `atomicAdd`
order), so the test now compares to `rtol=1e-12` and
`docs/source/lomb.rst` no longer claims `use_double=True` is bitwise
stable (commit 2f9736a; the kernels are correct). Then the full suite,
`python -m pytest -p no:cacheprovider -v -rs` from the repo root with no
path: **1785 passed, 1 xfailed, 0 failed, 0 skipped (1,786 collected)
in 8 min 6 s** (`/workspace/logs/p4_full_suite.log` on the pod, copied
to `benchmarks/results/nufft_lrt_validation_2026-09-06/logs/`). Every
item below is therefore ticked by that run; the ones with a note are
the ones that needed a look. Phase 5 re-runs the whole gate on the
frozen commit regardless.

Phase 3 was CPU-only. The following changes were verified by reading and
by CPU tests; each has a device-side check that Phase 5 must run (the
full suite covers most of them, the named tests/spot checks are the
ones to look at if anything fails). Result-changing items are marked
(R); everything else adds validation or changes only non-default paths.

- (R) BLS `eebls_transit` top-K solution re-scan now walks the kernel's own
  bin ladder for float32 `qvals` (commit 1bbc08d): run `eebls_transit` with
  `keplerian_freq_grid(..., return_qvals=True)` output and with
  `qvals=np.float32([0.025]*n)`; every returned `(q, phi)` must be a box the
  kernel evaluated (`q * nbinsf` integral). Default float64 path unchanged.
- (R) BLS host `dnbins` mirrors the device's float32 `floorf(dlogq*nbins)`
  (commit 4e1a69c): at the default `dlogq` (0.2/0.3) results must be
  bit-identical to 000c299; at `dlogq=0.65` with per-frequency bounds giving
  `nbins0=180, nbinsf=296` the host and device counts now agree (476).
- BLS `single_bls` rejects `q` outside `[0, 1]` (47c8a26): the GPU tests that
  evaluate `single_bls` over every returned solution must still pass.
- LS/NFFT: a per-call `use_double` that differs from the process precision
  raises `ValueError` before compile/allocation (c95a7f7); equal values are
  still accepted (`TestBatchedMemoryReuse::test_per_call_use_double_matching_the_process_is_accepted`).
  *Phase 4: the only red test of the list -- its bitwise assertion on the
  double path, not the code; relaxed to rounding (2f9736a), see the status
  note above.*
- (R, kernel) NFFT first mode `k0` is computed on the host and passed to the
  `nfft_shift`/`normalize` kernels as an integer (df87ad1, `cunfft.cu` +
  prepared dtypes): nvcc must compile in both precisions; LS periodograms and
  raw NFFT outputs must be bit-identical to 000c299 on the default grids;
  `test_nfft.py::TestFirstModeIsExactOnTheHost::test_large_k0_band_in_double_matches_exact_dft`
  (two ~280 MB complex128 grids) must pass. *Phase 4: compiled and passed
  in both precisions (all of `test_nfft.py`, `test_lombscargle.py` and the
  cuFINUFFT cross-check green); the NUFFT-LRT campaign ran on this NFFT.*
- LS: `batched_run_const_nfreq` validates the shared grid before compiling
  (b91c43c) -- `TestEntryPointsRaiseBeforeDeviceWork` on device.
- NFFT `precomp_psi=False` routes to `slow_gaussian_grid` instead of raising
  (4fbed72): `test_nfft.py::TestPrecompPsiFalseOnDevice` (2 tests; tolerance
  1e-4 normalised vs the default path, 5e-3 vs direct sums -- relax toward
  the file's nfft_rtol if float32 atomics on the pod exceed it).
- LS Baluev `d_K` follows the per-call `nharmonics` (b67d969):
  `test_lombscargle.py::TestBaluevDKUsesEffectiveNharmonics::test_per_call_nharmonics_sets_d_K_on_device`.
- CE `use_fast=True` + `compute_log_prob=True` raises (56566c9); `run(memory=...)`
  with a mismatching per-call option raises (ec54fe3): run all of
  `test_ce.py` to confirm no legitimate reuse pattern is rejected.
- (R) PDM/CE keep a private copy of the grid used for re-upload detection
  (db94455): `test_ce.py::TestCEPreallocate::test_run_reuploads_in_place_mutated_float32_grid`
  and `test_pdm.py::TestPDMAllocationReuse::test_in_place_mutated_float32_grid_is_reuploaded`.
- CE/PDM reject a constant `y`; PDM rejects a `(t, y)` 2-tuple and zero legacy
  weights (f67a9cc, 0b6077b, 97c14b3): validation only, before device work.
- TLS: `dy` is required, `n_durations` validated on both paths, unknown
  keywords rejected with a FAP hint (0d5cc65); template-table cache keyed on
  whether batman was actually used (377bb71): run `test_tls_basic.py::TestTlsInputGuards`,
  `::TestTemplateTableMemoization`, `::TestBatchPreprocessValidation::test_durations_param_removed`
  (this one only skipped under the stub), and the golden/fast/t0-oversample
  suites for no numerical change on the default path; spot-check that
  `tls_search_gpu(..., n_durations=1, use_fast=False)` raises.
- TLS `tls_transit` smoke, `tls_search` dispatch, fast-vs-legacy parity and
  adaptive-BLS block-size parity tests ported from `scripts/` (72002a7):
  `test_tls_fast.py::TestFastLegacyParity`, `::TestTlsTransitSmoke`,
  `test_bls.py::TestAdaptiveBlockSize`.
- cuFINUFFT on-device cross-check `test_lombscargle.py::TestCufinufftBackendOnDevice`
  (needs `pip install cufinufft`); the zero-skip gate now depends on it.
- NUFFT-LRT: EXPERIMENTAL warning at construction, not import
  (`python -W error::UserWarning -c "import cuvarbase.nufft_lrt"` must succeed;
  `NUFFTLRTAsyncProcess()` must warn once); empty basis raises before device
  work (0ec98b8); `test_nufft_lrt.py` in full; Phase 4 harness prints the null
  std as a calibration constant (8ed2246) instead of a pass/fail against 1.
  *Phase 4: all four `test_nufft_lrt*.py` files green on the A40; the
  harness's null calibration is a quoted constant (see the Phase 4 section
  and `docs/source/nufft_lrt.rst`).*
- Input validation: `test_input_validation.py::test_valid_input_is_unaffected_by_the_validators`
  (rewritten, GPU: validators on vs monkeypatched off, `np.array_equal`).
- Packaging/CI on the pod: `python -m pytest -p no:cacheprovider` with no
  path from the repo root (pyproject testpaths/-rs/--strict-markers with real
  pycuda); `pip install <wheel>[test]` resolves batman-package and
  transitleastsquares; `test_kernel_inventory.py` passes with `wavelet.cu`
  gone; `scripts/setup-remote.sh` and `scripts/benchmark_new_features.py --tests-only`
  still work without the stripped scikit-cuda patch blocks; the docs build with
  `-W` renders the five plot-directive figures.
- Counts to refresh from the gate log: `docs/RELEASE_NOTES_v1.0.0.md` (the
  1,582 / 1,786 sentence) and `README.md` ('1,582 tests'). *Phase 4 measured
  1,785 passed + 1 xfailed of 1,786 collected; Phase 5's gate log on the
  frozen commit is the number to write.*

## Phase 4: NUFFT-LRT re-validation (pod; before the freeze; no go needed)

**DONE 2026-09-06** (pod #2, NVIDIA A40 `bd501r0q7qz8tt`, $0.49/h;
harness v2 = commit 2f9736a after a 60-agent adversarial review; 8
processes, 12:13-15:43 UTC). Archive:
`benchmarks/results/nufft_lrt_validation_2026-09-06/` (merged JSON with
per-light-curve records, summary, process logs, both suite logs, the
null-calibration check, the launch script, README). Docs:
`docs/source/nufft_lrt.rst` *When is this the right tool?* and
*Validation status* carry the measured tables. **D1 = EXPERIMENTAL
(validated, API not frozen)**: the default path passed the correctness
gate (BJD-scale times identical to 5e-8 for every arm, 0/800 decisions
differ; `epochs=None` recovers the injected transit in 99 % of its
detections; non-zero-mean basis identical to 5.5e-7; Detector A =
sequential baseline exactly), but the campaign also showed that the
defaults a 1.x freeze would lock in should still change (the default
epoch grid costs 4-9 % completeness against a finer one; PSD whitening
gave no gain over a flat PSD, and BLS/TLS are 10-12 % more complete in
white noise; `run()` returns a tuple or an array depending on
`epochs`), and a 6-judge panel was unanimous on both points. Status
text updated in the module message and class docstring, `__init__.py`,
README, release notes, CHANGELOG, `cuvarbase.rst`, `scripts/README.md`,
the xiaziyna/astrobatty drafts and `issue-sweep.md`. Nothing in the
namespace or the tests changed. Steps 1-5 below are the record of what
was planned; step 5 (terminate) was done after the post-change device
test run.

Extends `scripts/nufft_lrt_validation.py` and decides D1. It changes
`docs/`, so it precedes T.

1. Pod: `scripts/runpod-create.sh "NVIDIA RTX A5000"` (unpiped; wait for
   "SSH ready"), clone `v1.0-fixes` by SHA (below), `pip install -e '.[test]'
   cufinufft`.
2. Run all four existing configurations and all arms, plus (a) a
   configuration with `t + 2457000.5`, (b) an arm using the public default
   `epochs=None` (the automatic epoch grid), (c) a non-zero-mean basis, and
   (d) at least 200 injections per depth. Keep the null-p95 calibration.
3. Archive the JSON under `benchmarks/results/nufft_lrt_validation_<date>/`
   and fill `docs/source/nufft_lrt.rst` from
   `scripts/summarize_lrt_validation.py`.
4. Decide: official only if the fixed default path passes and the docs
   quote the measured numbers honestly (baseline text: `ALGORITHM_AUDIT.md`
   section 6.4). Otherwise it stays quarantined-experimental with the
   caveats. Either way the module stays out of the top-level namespace for
   1.0 (D1). Update the release notes, CHANGELOG and `xiaziyna-message.md`
   accordingly, commit, push `v1.0-fixes`.
5. Terminate the pod (`scripts/runpod-stop.sh --terminate`, confirm with
   the `myself{pods}` query that only that pod went away).

## Phase 5: freeze + gate (pod, ~4 pod-hours; no go needed)

The sequence is: candidate tip C -> full suite once on C (this measures N)
-> write N into the release notes and strip the DRAFT comment (the last
content commit; that commit is T) -> full suite + gate on T, which must
reproduce N. So T = the last content commit on `origin/v1.0-fixes` after
Phase 4, and it is the commit that carries the count. Once T is cut,
nothing but `analysis/` changes may land on the branch; if anything else
does, that commit is the new T and the gate is re-run (a docs-only commit
cannot change the collected count, so N carries over; if the gate on the
new T ever reports a different N, investigate the run -- never edit the
notes after T).

```bash
# --- local: candidate tip C (everything Phase 4 produced is pushed)
git checkout v1.0-fixes && git pull --ff-only && git status   # clean
C=$(git rev-parse HEAD); echo "C=$C"
DATE=$(date +%Y%m%d)

# --- pod: fresh RTX A5000 (A40/4090 acceptable; record which)
scripts/runpod-create.sh "NVIDIA RTX A5000"        # writes .runpod.env
# on the pod (via scripts/run-remote.sh or an ssh wrapper that exports the
# CUDA env). Clone by SHA -- NEVER scripts/sync-to-runpod.sh (it rsyncs the
# working tree without .git, which is why the July record could not name
# its commit).
git clone https://github.com/johnh2o2/cuvarbase.git /workspace/cuvarbase
cd /workspace/cuvarbase && git checkout --detach "$C"
git rev-parse HEAD                                  # must print C
pip install -e '.[test]' cufinufft
python -c "import pycuda.driver, batman, transitleastsquares, nfft, astropy, cufinufft; print('preflight ok')"
python -c "import cuvarbase; print(cuvarbase.__version__, cuvarbase.__file__)"   # 1.0.0, /workspace/cuvarbase/...

# 0. candidate run on C: measures N once (every test on the device, 0 skipped)
python -m pytest cuvarbase/tests -v -rs 2>&1 | tee suite_candidate.log
tail -3 suite_candidate.log  # "<N> passed in ..." -- no skipped, no failed;
                             # this N goes into the notes, nothing else does

# --- local: write N into the notes = the last content commit = T
#     edit docs/RELEASE_NOTES_v1.0.0.md: the "<N> tests (0 skips)" figure,
#     the GPU and date of the candidate run, and delete the leading
#     <!-- DRAFT ... --> block; nothing else changes in this commit
git commit -am "Release notes: GPU test count from the candidate run at $C"
git push origin v1.0-fixes
T=$(git rev-parse HEAD); echo "T=$T"; git rev-parse "$T^{tree}"

# --- pod: move to T (same pod, same install)
cd /workspace/cuvarbase && git fetch origin && git checkout --detach "$T"
git rev-parse HEAD "HEAD^{tree}"                    # must print T and its tree
git diff --stat "$C" "$T"                           # docs/RELEASE_NOTES_v1.0.0.md only

# 1. full suite on T: must reproduce N exactly
python -m pytest cuvarbase/tests -v -rs 2>&1 | tee suite_full.log
tail -3 suite_full.log       # "<N> passed in ..." -- the SAME N as
                             # suite_candidate.log and the notes; else STOP
# 2. release gate
python scripts/check_release_gate.py 2>&1 | tee release_gate.log   # 14/14
# 3. docs with every figure rendered (warnings are errors)
SPHINXOPTS="-E -a -W --keep-going" make -C docs html 2>&1 | tee docs_build.log
# 4. packaging
python -m build 2>&1 | tee build.log && twine check dist/* | tee twine_check.log
# 5. wheel smoke from OUTSIDE the tree (ci_wheel_smoke.py validates the
#    installed package with pycuda absent, so --no-deps in a bare venv)
python -m venv /tmp/wheelsmoke
/tmp/wheelsmoke/bin/pip install --no-deps dist/cuvarbase-1.0.0-*.whl
(cd /tmp && /tmp/wheelsmoke/bin/python /workspace/cuvarbase/scripts/ci_wheel_smoke.py) | tee wheel_smoke.log
# 6. installed-wheel test run on the device, from outside the tree
python -m venv --system-site-packages /tmp/wheeltest
/tmp/wheeltest/bin/pip install "$(ls dist/cuvarbase-1.0.0-*.whl)[test]"
(cd /tmp && /tmp/wheeltest/bin/python -c "import cuvarbase; print(cuvarbase.__file__)")   # /tmp/wheeltest/..., not the tree
(cd /tmp && /tmp/wheeltest/bin/python -m pytest --pyargs cuvarbase -rs) 2>&1 | tee wheel_pyargs.log   # 0 failed, 0 skipped
# 7. sdist smoke
python -m venv --system-site-packages /tmp/sdisttest
/tmp/sdisttest/bin/pip install "dist/cuvarbase-1.0.0.tar.gz[test]"
(cd /tmp && /tmp/sdisttest/bin/python -c "import cuvarbase; print(cuvarbase.__version__)") | tee sdist_smoke.log
# 8. one real run of each headline entry point from the installed wheel
(cd /tmp && /tmp/wheeltest/bin/python - <<'PY' 2>&1 | tee wheel_run.log
import numpy as np
from cuvarbase.bls import eebls_transit
from cuvarbase.tls import tls_search_batch
from cuvarbase.lombscargle import lomb_scargle_simple
rng = np.random.default_rng(1)
t = np.sort(rng.uniform(0, 30, 2000)); dy = np.full(t.size, 1e-3)
y = 1 - 0.01 * (((t - 3.0) % 2.5) < 0.1) + dy * rng.standard_normal(t.size)
f, p, _ = eebls_transit(t, y, dy); print("eebls_transit best period:", 1 / f[p.argmax()])
print("tls_search_batch:", tls_search_batch([(t, y, dy)])[0].get("period"))
print("lomb_scargle_simple:", len(lomb_scargle_simple(t, y, dy)[1]))
PY
)
# 9. environment record
{ echo "commit $T"; echo "tree $(git rev-parse "$T^{tree}")"; date -u; nvidia-smi; pip freeze; } > env_record.txt
```

Docs site staging (still on the pod, from the step-3 build):

```bash
cd /workspace/cuvarbase
rm -rf docs/build/html/.doctrees docs/build/html/.buildinfo
touch docs/build/html/.nojekyll
tar -C docs/build/html -czf /workspace/site-$T.tgz .
```

Back on the workstation:

```bash
# copy logs + site back (scp via the .runpod.env host/port; no rsync needed).
# runpod-create.sh writes RUNPOD_SSH_HOST / RUNPOD_SSH_PORT / RUNPOD_SSH_USER
# (and optionally RUNPOD_SSH_KEY) into .runpod.env -- source it, as
# scripts/setup-remote.sh and test-remote.sh do.
source .runpod.env
POD="$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST"
SCP="scp -P $RUNPOD_SSH_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${RUNPOD_SSH_KEY:+-i $RUNPOD_SSH_KEY}"
REC=analysis/v1.0-release-gate-$DATE; mkdir -p "$REC"
$SCP "$POD:/workspace/cuvarbase/{suite_candidate,suite_full,release_gate,docs_build,build,twine_check,wheel_smoke,wheel_pyargs,sdist_smoke,wheel_run}.log" "$REC"/
$SCP "$POD:/workspace/cuvarbase/env_record.txt" "$REC"/
$SCP "$POD:/workspace/site-$T.tgz" /tmp/
scripts/runpod-stop.sh --terminate           # only the pod in .runpod.env; confirm with myself{pods}

# commit T' = the gate record only (the logs are tracked through the
# !analysis/**/*.log negation in .gitignore)
cat > "$REC"/SUMMARY.md <<EOS
# v1.0.0 release gate ($DATE)
commit T: $T   tree: $(git rev-parse "$T^{tree}")   candidate tip C: $C
GPU: <from env_record.txt>   suite on T: <N> passed, 0 skipped, 0 failed (suite_full.log)
candidate run on C: <N> passed, 0 skipped, 0 failed (suite_candidate.log) -- same N as the notes
check_release_gate.py: 14/14   docs: -E -a -W clean, all figures   build/twine: ok
wheel smoke + --pyargs from outside the tree: ok   sdist: ok   wheel run: ok
EOS
git add "$REC" && git commit -m "Release gate record for v1.0.0 at $T ($DATE)"
TP=$(git rev-parse HEAD)
git diff --quiet "$T" "$TP" -- . ':!analysis' && echo "T' is analysis-only"
git push origin v1.0-fixes

# rebuild gh-pages-staging as ONE orphan commit from T's docs build, in a
# throwaway worktree so the main checkout is never cleaned
git worktree add --detach /tmp/site-wt "$T" && pushd /tmp/site-wt
git checkout --orphan gh-pages-staging-new && git rm -rfq .
tar -xzf /tmp/site-$T.tgz -C .
test -f .nojekyll && ! find . -name .buildinfo -o -name .doctrees | grep -q . && echo SITE-OK
git add -A && git commit -qm "docs site built from v1.0-fixes @ $T (Sphinx -E -a -W, all figures)"
git branch -M gh-pages-staging          # replaces the July staging branch
popd && git worktree remove --force /tmp/site-wt
git log -1 --format=%s gh-pages-staging  # names T
```

### Merge rehearsal (CPU, right after T'; no shared ref touched)

```bash
git fetch origin --prune
git merge-tree --write-tree origin/master v1.0-fixes | grep -c '^CONFLICT'   # expect 4
git merge-tree --write-tree origin/master v1.0-fixes | grep '^CONFLICT'
#   README.rst, cuvarbase/lombscargle.py, cuvarbase/pdm.py, cuvarbase/utils.py
#   -- any other count or file: STOP, the branch or origin/master moved; re-plan.
git checkout -b rehearsal-1.0.0 origin/master
git merge --no-ff v1.0-fixes -m "rehearsal" ; true          # stops on the 4 conflicts
git checkout --theirs README.rst cuvarbase/lombscargle.py cuvarbase/pdm.py cuvarbase/utils.py
git add README.rst cuvarbase/lombscargle.py cuvarbase/pdm.py cuvarbase/utils.py && git commit -qm "rehearsal"
git diff --quiet v1.0-fixes rehearsal-1.0.0 && echo TREE-IDENTICAL       # must print
rm -rf dist && python3 -m build && twine check dist/*
tar -xzOf dist/cuvarbase-1.0.0.tar.gz cuvarbase-1.0.0/PKG-INFO | grep -n "Until v1.0.0\|git+https"   # nothing
tar -xzOf dist/cuvarbase-1.0.0.tar.gz cuvarbase-1.0.0/PKG-INFO | grep -n "pip install cuvarbase"   # present
git checkout v1.0-fixes && git branch -D rehearsal-1.0.0 && rm -rf dist
```

Record "rehearsal: 4 conflicts, tree identical, PKG-INFO flipped" in
`$REC/SUMMARY.md` (an `analysis/`-only amendment is allowed; anything else
re-opens Phase 5).

## Release day (after the go) -- execute top to bottom, stop at any failure

```bash
# 0. clean state and provenance
git fetch origin --prune --tags
git checkout v1.0-fixes && git pull --ff-only && git status          # clean
git rev-parse HEAD                                                   # == T'
git merge-tree --write-tree origin/master v1.0-fixes | grep -c '^CONFLICT'   # 4

# 1. master = origin/master (the local ref is stale: ec53ae8 vs 060d839)
git checkout master && git reset --hard origin/master

# 2. merge (no-ff, preserves the branch point); four known conflicts
git merge --no-ff v1.0-fixes -m "Merge v1.0-fixes: cuvarbase 1.0.0" ; true
git checkout --theirs README.rst cuvarbase/lombscargle.py cuvarbase/pdm.py cuvarbase/utils.py
git add README.rst cuvarbase/lombscargle.py cuvarbase/pdm.py cuvarbase/utils.py
git commit --no-edit
git diff --quiet v1.0-fixes master && echo TREE-IDENTICAL               # must print; else STOP
MERGE=$(git rev-parse HEAD)

# 3. build from the merge commit in a clean venv (the tag comes after
#    the build proves the tree, but the tree is identical to what the
#    gate ran; N below = the count in the notes, which the gate reproduced)
python3 -m venv /tmp/relbuild && source /tmp/relbuild/bin/activate
pip install -q build twine
rm -rf dist && python -m build && twine check dist/*
tar -xzOf dist/cuvarbase-1.0.0.tar.gz cuvarbase-1.0.0/PKG-INFO | grep -n "Until v1.0.0\|git+https"   # nothing
ls dist/            # cuvarbase-1.0.0.tar.gz  cuvarbase-1.0.0-py3-none-any.whl
deactivate

# 4. wheel smoke from outside the tree
python3 -m venv /tmp/wheelsmoke && /tmp/wheelsmoke/bin/pip install --no-deps dist/cuvarbase-1.0.0-*.whl
(cd /tmp && /tmp/wheelsmoke/bin/python "$OLDPWD"/scripts/ci_wheel_smoke.py)

# 5. re-tag v1.0.0 at the merge commit (delete the stale June tag first)
git tag -d v1.0.0
git push origin :refs/tags/v1.0.0
git tag -a v1.0.0 "$MERGE" -m "cuvarbase 1.0.0: first release since 0.2.5; <N> GPU tests, 0 skipped"
git tag -v v1.0.0 2>/dev/null || git cat-file -p v1.0.0 | head -8

# 6. push master, the tag, the archive tag
git push origin master v1.0.0 archive/pre-1.0-process

# 7. rebuild from the TAG (proves the pushed ref) and publish
rm -rf dist /tmp/tagbuild && git worktree add /tmp/tagbuild v1.0.0
(cd /tmp/tagbuild && source /tmp/relbuild/bin/activate && python -m build && twine check dist/*)
cp /tmp/tagbuild/dist/* dist/ ; git worktree remove /tmp/tagbuild
# maintainer's PyPI token: type this yourself, prefixed with '!' so it
# never enters a transcript:
#   ! twine upload dist/*

# 8. post-publish smoke
python3 -m venv /tmp/relverify && /tmp/relverify/bin/pip install "cuvarbase==1.0.0"
/tmp/relverify/bin/python -c "import cuvarbase; print(cuvarbase.__version__)"   # 1.0.0
# optional but cheap: on a GPU pod, pip install cuvarbase==1.0.0 and repeat
# the Phase 5 step-8 snippet

# 9. GitHub Release from the notes (DRAFT comment already stripped in
#    pre-flight; the notes carry the 'git fetch --tags --force' line for
#    anyone who fetched the June tag)
gh release create v1.0.0 --title "cuvarbase 1.0.0" --notes-file docs/RELEASE_NOTES_v1.0.0.md

# 10. docs site: the clean orphan commit built from T
git push origin gh-pages-staging:gh-pages --force
#     then check https://johnh2o2.github.io/cuvarbase/ (tls.html, nufft_lrt.html
#     exist; whatsnew shows 1.0.0)

# 11. fast-forward the v1.0 branch to the release commit (89d5481 is an ancestor)
git push origin master:v1.0

# 12. contributor messages (maintainer sends; drafts in
#     analysis/release-staging-v1.0.0/): astrobatty-message.md (comment on
#     #63 or email) and xiaziyna-message.md.

# 13. issue sweep -- analysis/release-staging-v1.0.0/issue-sweep.md:
#     (a) open the "v1.1 roadmap" issue; note its number
#     (b) replace #ROADMAP in the drafted comments; fill the test-count and
#         NUFFT-LRT status placeholders from the gate record / Phase 4
#     (c) close #14 #15 #17 #19 #28 #29 #30 #32 #33 #63 with their comments
```

## Post-release (maintainer actions, own timeline)

- Co-maintainer invite for @astrobatty (if he accepts); ASCL record update;
  `pyproject.toml` Documentation URL check; JOSS paper + Zenodo DOI (needs
  the published release; tracked in the roadmap issue).
- Delete `analysis/release-staging-v1.0.0/` (one commit on `master` after
  the messages are sent and the sweep is done).
- Branch sweep, in this order, each verified with `git branch -r --merged
  master` (or `--contains`) before deletion:
  1. `origin/v1.0-fixes`: after a grace period (a week or two, once QLP and
     the contributors have re-pointed anything that tracked it).
  2. Local harness/merged branches: the four `worktree-*` branches (zero
     unique commits), the seven `p3-*` Phase 3 branches, and the merged
     feature/fix branches (`git branch --merged master` lists them; keep
     `gh-pages-staging` until the site is confirmed live, keep
     `feature/ffa-bls-experimental` -- it holds the only FFA code).
  3. Obsolete remote branches -- BUT tell @astrobatty first before touching
     `origin/fix/BLS-kernel` and `origin/bugfix/BLS-kernel` (his PR #65
     history): `origin/devel`, `origin/hotfix` (2018), the merged
     `origin/feature/{bls-survey-speed,tls-fast-survey,period-derivative-search}`,
     `origin/fix/kernel-hygiene-jul2026`, `origin/tls-gpu-implementation`,
     `origin/testing/runpod-benchmarks`, `origin/bugfix/{invalid-resource-handle-after-first-batch,swap-out-pycuda-autoinit}`,
     `origin/copilot/add-search-for-pdot-in-algorithms`,
     `origin/feature/nufft-lrt-experimental` (superseded by the module on
     `master`). Keep `origin/feature/ffa-bls-experimental`, `origin/v1.0`,
     `origin/gh-pages`, `origin/master`.
- 1.0.1 / 1.1 queue (goes into the roadmap issue; from EXECUTION_PLAN.md
  Phase 7 plus the audit's deferred items): TLS coarse-kernel rewrite
  (band-of-periods-per-block; the 2-4x the audit measured as available),
  PDM `_fast` kernel rewrite, **NUFFT-LRT promotion to official (Phase 4
  option C)**: fix the three defaults the 2026-09-06 campaign flagged (a
  finer default epoch grid -- the default loses 4-9 % completeness at
  `epoch_oversample=2`; one return convention for `run()` instead of
  tuple-or-array; revisit the default PSD whitening, which gave no gain
  over a flat PSD; also `durations=None` and `dy`), re-run the
  default-path arm of `scripts/nufft_lrt_validation.py` (~1-2 pod-hours)
  and add a cotrend-then-BLS/TLS comparator, then bring the module into
  the top-level namespace and the stability promise (touch points:
  `__init__.py` `_LAZY_ATTRS`, `test_api_freeze.py`,
  `test_lazy_imports.py`, `_EXPERIMENTAL_MSG`, README, notes, rst);
  the deferred unweighted-CE `dy` relaxation (patch in the Phase 3
  scratchpad; `dy=None` documented for now); bump the GitHub Actions
  majors (Node 20 deprecation annotations); enforce flake8 `F` in CI
  (`scripts/` still has F401/F541/F841 in the older benchmark scripts);
  Dockerfile rebuild (deleted in 1.0),
  `_cufft.py` hardening, stellar-parameter overrides for the Keplerian
  grids, CE float32 grids, float64 grid builders, thread-safety of the
  process objects, multi-GPU dispatch.

## Rollback notes

- PyPI: cannot re-upload the same version. If a bad artifact ships, yank
  1.0.0 (`pip` then skips it unless pinned) and publish 1.0.1. Yank is
  reversible; deletion is not. Prefer yank + patch release.
- GitHub Release / tag: `gh release delete v1.0.0` plus deleting the tag is
  fine if caught immediately (before the announcement); afterwards prefer
  a 1.0.1. Anyone who fetched a deleted tag needs `git fetch --tags
  --force`, which is why the notes say so.
- `master`: the merge is a single `--no-ff` commit; `git revert -m 1
  <merge>` restores the pre-1.0 tree if it ever has to happen, but a
  published 1.0.0 must never be un-merged -- fix forward.
- gh-pages: previous content is the 2017 build (worthless), no rollback
  concern; `gh-pages-staging` stays until the site is confirmed live.
- The archive tag and `origin/v1.0-fixes` are the only refs that hold the
  pre-merge history; do not delete either until the release is confirmed.
