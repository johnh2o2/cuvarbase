# cuvarbase 1.0.0 release gate — 6 September 2026

The Phase 5 gate and merge rehearsal passed. Final-tip CI is checked after
the record amendment/push before reporting the Phase 5 boundary.
Phase 6 has not started and still requires the
explicit maintainer go coordinated with @astrobatty. Messages remain drafts.

## Frozen provenance

- Frozen content commit T: `1032caf029570dc4841db1c594a2cbb1654e8fd8`.
- Frozen Git tree: `b023c3e8d163010dbae2fc0b7cd5204ca04384d1`.
- T is the pushed README/release-notes count correction on `b60e01b`.
- The fresh pod cloned the repository and checked out T by SHA; no working-tree
  sync was used. `checkout.log`, `env_record.txt` and `runner/` record provenance.
- T' denotes the single analysis-only child of T containing this record.
  Resolve its final SHA with `git log -1 --format=%H --
  analysis/v1.0-release-gate-20260906/SUMMARY.md`; its parent must be T.
  Its literal SHA and final CI run are also in the external session memory
  `v1-execution-status-sep2026.md` and the Phase 5 completion report.

## Measured checks

| Check | Outcome | Evidence |
| --- | --- | --- |
| Source full suite on T, repo root, no path | 1,785 passed + 1 xfailed of 1,786 collected; 0 failed, 0 skipped; 10 warnings; 660.45 s | `suite_full.log` |
| Release checks | 14/14, plus dependency preflights | `release_gate.log` |
| Clean GPU Sphinx | `-E -a -W --keep-going`; no warnings; five GPU figures plus two geometry diagrams | `docs_build.log`, `docs_figures.log` |
| Pod sdist/wheel build | passed | `build.log` |
| Strict Twine metadata check | both artifacts passed | `twine_check.log` |
| Flipped README in PKG-INFO | counts/xfail and PyPI install present; obsolete banner/git+ absent | `pkg_info.log` |
| Fresh local build of T / strict Twine / exact README metadata / package inventory | passed, Python 3.13 | `local-preflight/` |
| Bare-venv wheel import smoke | passed, PyCUDA absent | `wheel_smoke.log` |
| Bare-venv sdist import smoke | passed, PyCUDA absent | `sdist_smoke.log` |
| Artifact import provenance | all four imports from their fresh venvs, outside the source tree | `artifact_paths.log` |
| Installed-wheel GPU suite | 1,773 passed / 11 source-only skips of 1,784 collected; 0 failed; 17 warnings; 655.60 s | `wheel_pyargs.log` |
| Installed-sdist GPU suite | 1,773 passed / 11 source-only skips of 1,784 collected; 0 failed; 17 warnings; 508.41 s | `sdist_pyargs.log` |
| Installed-wheel headline runs | passed: BLS period 2.499425896 d; batch TLS 2.500537157 d; LS 25,000 finite powers | `wheel_run.log` |

The one source xfail is
`cuvarbase/tests/test_examples_compile.py::test_notebook_code_cells_compile_without_warnings[Phase Dispersion Minimization.ipynb]`,
a strict expected failure for the notebook's known non-raw TeX label strings.
The source count exactly reproduces Phase 4's measurement, copied as
`suite_candidate_phase4.log` (A40, `954f037` plus the LS tolerance change).
README and release notes already contain this count in T; they were not edited
in response to the frozen gate.

## Docs staging

Local orphan `gh-pages-staging`:
`1fd5c7d357bada18adf68e10407d9a71afe5d880`, tree
`33d09eba25d576b44f6cbd8920d459c700703aa8` (99 files). Its message names T;
`.nojekyll` is present, `.doctrees` and `.buildinfo` are absent. The staging
branch did not exist on origin, so it was not pushed. See `site_staging.json`
and `site_staging.log`. `gh-pages` was not changed.

## CI and rehearsal

- Starting `b60e01b`: GitHub Actions run `34059192136`, all jobs successful.
- T: run `34060055095`, all nine jobs successful (`ci-T.json`).
- Merge rehearsal on the first gate-record commit
  `ea2694ffafbb49363f236133bb16675a64ca00ba` against current `origin/master`
  `060d839035bcc65d7c69a48a3329852a5f6d580b`: **passed**. Exactly four
  conflicts (`README.rst`, `cuvarbase/lombscargle.py`, `cuvarbase/pdm.py`,
  `cuvarbase/utils.py`), all resolved from the release side. Merge commit
  `1cc63fc014e6057dfc074ca4a3e5fd412217e1e0` has tree
  `96f43f6afcb34170c880a9b5fd48c3464621fb7d`, identical to the rehearsed
  gate-record tree. Fresh build, strict Twine, exact frozen README in both
  metadata files, and forbidden-banner checks passed. The owned worktree and
  rehearsal branch were removed. Full evidence: `rehearsal/summary.json`
  and its command logs.
- The runbook permits an analysis-only amendment to add rehearsal evidence;
  this record is that amendment, still the single child of T. Final T'
  receives a further rehearsal after the amendment and must have successful
  exact-SHA GitHub Actions before the Phase 5 completion report. The literal
  final SHA, CI run ID, and final rehearsal result are recorded externally
  in the session memory (a commit cannot embed its own SHA).

## Execution adjustments and failures

1. Per the maintainer's Sep 6 instruction, Phase 4 supplied the measured count;
   no redundant candidate run was done before the T count commit. The full
   frozen source suite used the exact no-path command, unlike the historical
   runbook snippets that supplied `cuvarbase/tests`.
2. `setup-remote.sh` always rsyncs, so the fresh SHA clone was installed directly
   over SSH. `rsync` was installed first. CUDA was exported in every SSH command;
   `OPENBLAS_NUM_THREADS=1` was set throughout.
3. The Makefile overrides environment `SPHINXOPTS`; strict flags were also
   passed as a make command-line assignment so `-W` was actually enforced.
4. Bare artifact smoke venvs received NumPy/SciPy before `--no-deps` artifact
   installs. GPU venvs were fully isolated, received artifact `[test]` extras
   plus explicit `cufinufft`, and verified their import paths. Both wheel and
   sdist ran the shipped GPU suite and the no-PyCUDA packaging smoke.
5. Initial artifact installation under shared `/workspace` was slow. Only that
   setup process tree was deliberately interrupted (SIGTERM, exit 143); no
   test or product gate failed. Its logs were retained as
   `artifact_setup_workspace_attempt.log` and `driver_source_to_build.log`.
   Artifact setup and dependent checks resumed in fresh venvs on local `/tmp`
   (`gate_resume.sh`, `driver_artifacts.log`, `storage_adjustment.log`).
6. The installed wheel has 11 deliberate source-file guard skips: one API
   docs check, one BLS docs check, one examples/notebooks directory guard,
   two empty parameter sets, and six README guards. No optional dependency
   is missing. The source suite has zero skips. The historical installed
   zero-skip expectation is incompatible with these packaged tests; their
   guards were kept intact. The sdist independently produced the same counts and reasons.
   See `installed_skip_reasons.log`.

7. The first local inventory helper assumed a `core/` subpackage and 33 test
   modules. T actually contains `core.py` and 30 test modules. Those two helper
   assumptions failed; comparison against T's Git inventory corrected them,
   and both artifacts contain all 69 tracked Python/kernel/header files.
   The initial and corrected logs are retained under `local-preflight/`;
   no package change was needed.
8. Docs staging was assembled locally while artifact tests ran, from the
   completed strict GPU docs output. The final copied site archive was later
   checked byte-for-byte against every staging Git blob (99 files).

All logs are retained, including the interrupted setup. Nothing outside
`analysis/` may change after T without refreezing.

## Pod lifecycle and phase boundary

One RTX A5000, CUDA 12.4, Python 3.11.10; pod `cyx6rax5q03t94`, $0.27/h,
started `2026-09-06T21:06:59.703Z`, terminated and account confirmed empty
at `2026-09-06T21:59:38.451979+00:00`. Phase 5 estimated spend **$0.24**,
cumulative **$14.18** (elapsed time at the API rate; `pod_cost.json`).
All 30 final pod log files and all artifact SHA-256 hashes were verified after
copying; the final site archive exactly matches the 99-file staging Git tree
(`copy_verification.json`, `pod_logs.sha256`).
The account was empty before creation (`pods-before.json`).
No merge to master, tag creation/move/push, PyPI upload, GitHub Release,
contributor message, or gh-pages push is authorized in this phase.
