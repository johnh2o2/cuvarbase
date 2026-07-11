# v1.0.0 Release Runbook

**HOLD: nothing below executes until the explicit, @astrobatty-coordinated go
(maintainer directive, Jul 4 2026).** Everything above the "Release day"
line is preparation that is already done or safe to redo.

## State as staged (Jul 10 2026)

- Release branch: `v1.0-fixes` (all of PRs #57–#68 + the July audit/docs/
  release-content commits). CI: CPU suite green on every push.
- Pre-release audit: `analysis/tls-audit-jul2026.md` (code) +
  `analysis/claims-trace-jul2026.md` (performance claims) — no correctness
  defects; claims corrected where the audit required.
- GPU release gate on the merged tip: archived in
  `analysis/v1.0-release-gate-jul2026/` (suite + `scripts/check_release_gate.py`
  + docs-figure build, RTX A5000).
- Old `v1.0.0` tag: annotated, points at 5553248 (Jun 11), exists on origin.
  Never published to PyPI; no GitHub Release exists for it (the only GitHub
  Release ever is v0.2.1 from 2021) — safe to delete and re-create.
- Version string: `cuvarbase/__init__.py` `__version__ = "1.0.0"` (already set;
  setup.py reads it). MANIFEST.in present; README.md is the PyPI long
  description.
- PyPI: latest published version is 0.2.5 (Oct 2023). Publishing needs the
  maintainer's PyPI token — run the twine step yourself (e.g. type
  `! twine upload dist/*` in the session so the token never enters the
  transcript), or have it in `~/.pypirc`.

## Pre-flight checklist (verify on release day, before step 1)

- [ ] Explicit maintainer go, coordinated with @astrobatty (he's expecting
      "some changes in BLS"; draft message in
      `analysis/release-staging-v1.0.0/astrobatty-message.md`)
- [ ] `git fetch`; release commit = `origin/v1.0-fixes` tip; CPU CI green there
- [ ] GPU gate record in `analysis/v1.0-release-gate-jul2026/` is FROM that
      commit (re-run the gate if anything landed after it)
- [ ] `docs/RELEASE_NOTES_v1.0.0.md`: delete the leading `<!-- DRAFT -->`
      comment; confirm the GPU test count matches the final gate log
- [ ] CHANGELOG.rst top section is `1.0.0` (no "Unreleased" heading)

## Release day — execute top to bottom

```bash
# 0. clean state
git checkout v1.0-fixes && git pull --ff-only
git status   # must be clean

# 1. merge to master (no-ff, preserves the branch point)
git checkout master && git pull --ff-only
git merge --no-ff v1.0-fixes -m "Merge v1.0-fixes: cuvarbase 1.0.0"
# Dry-run (staging, Jul 10) found exactly ONE conflict: README.rst —
# master's old full README vs the release branch's pointer stub.
# Resolve by taking the release branch's version:
#   git checkout --theirs README.rst && git add README.rst && git commit

# 2. re-tag v1.0.0 at the release commit
git tag -d v1.0.0
git push origin :refs/tags/v1.0.0          # delete the stale remote tag
git tag -a v1.0.0 -m "cuvarbase 1.0.0"
git push origin master v1.0.0

# 3. build + verify artifacts (clean venv)
python3 -m venv /tmp/relbuild && source /tmp/relbuild/bin/activate
pip install -q build twine
python -m build                             # sdist + wheel into dist/
twine check dist/*
# wheel smoke (ci_wheel_smoke.py takes no args — it validates the
# INSTALLED package, so install the wheel into a fresh venv and run the
# script from outside the source tree):
deactivate && python3 -m venv /tmp/wheelsmoke && source /tmp/wheelsmoke/bin/activate
pip install dist/cuvarbase-1.0.0-py3-none-any.whl
(cd /tmp && python "$OLDPWD"/scripts/ci_wheel_smoke.py)

# 4. publish to PyPI  (maintainer token — see note above)
twine upload dist/*

# 5. GitHub Release (strip the draft comment first — pre-flight item)
gh release create v1.0.0 \
  --title "cuvarbase 1.0.0" \
  --notes-file docs/RELEASE_NOTES_v1.0.0.md

# 6. docs site: push the staged clean gh-pages commit
#    (staged branch: gh-pages-staging, built on the release-gate pod WITH
#     figures; contains .nojekyll; single clean orphan commit)
git push origin gh-pages-staging:gh-pages --force

# 7. issue sweep — comments drafted in
#    analysis/release-staging-v1.0.0/issue-sweep.md:
#    (a) open the "v1.1 roadmap" issue; note its number
#    (b) replace #ROADMAP placeholders in the drafted comments with it
#    (c) close #14 #15 #17 #19 #28 #29 #30 #32 #33 #63 with their comments

# 7b. post-publish README flip (single commit to master):
#     - remove the top "current PyPI release is 0.2.5" banner
#     - Installation: replace the git+ URL with `pip install cuvarbase`
#     - update test_readme_consistency.py: the
#       test_readme_install_not_pinned_to_stale_pypi guard inverts once
#       1.0.0 is live (bare `pip install cuvarbase` becomes CORRECT) —
#       repoint it at whatever claim should now be guarded

# 8. post-publish verification
python3 -m venv /tmp/relverify && source /tmp/relverify/bin/activate
pip install cuvarbase
python -c "import cuvarbase; print(cuvarbase.__version__)"   # -> 1.0.0
# on a GPU pod: pip install cuvarbase && a tiny eebls_transit run
# docs: https://johnh2o2.github.io/cuvarbase/ serves the rebuilt site
#       (tls.html exists; whatsnew shows 1.0.0)
```

## Post-release (maintainer actions, own timeline)

- Announce to @astrobatty; co-maintainer invite
- ASCL record update
- JOSS paper (next project; needs the published release + Zenodo DOI —
  see the v1.1 roadmap issue)
- Update `pyproject.toml` Documentation URL if it doesn't already point at
  the rebuilt site

## Rollback notes

- PyPI: cannot re-upload the same version — if a bad artifact ships, yank
  1.0.0 (`pip` will then skip it unless pinned) and publish 1.0.1. Yank is
  reversible; deletion is not. Prefer yank + patch release.
- GitHub Release/tag: `gh release delete v1.0.0` + delete tag is fine if
  caught immediately; after announcement, prefer a 1.0.1.
- gh-pages: previous content is the 2017 build (worthless) — no rollback
  concern.
