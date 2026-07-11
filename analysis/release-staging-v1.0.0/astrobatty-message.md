# Draft: coordination message to @astrobatty (post as a comment on #63 or email)

---

Hey @astrobatty — v1.0.0 is release-ready and I'd like to coordinate timing
with you before I push the button.

Since you last looked: the full audit I promised happened (code review of
everything the models wrote plus a claims audit tracing every published
performance number back to archived raw benchmark data), the GPU release gate
is green on the final merged branch, and the docs site finally got rebuilt
(it was frozen on a 2017 Sphinx build — there's a real TLS page now, and the
PDM page you effectively motivated).

What you'll care about most:

- **Your BLS changes are in** (PR #65, merged Jul 4) — plus the root-cause
  fix for the nondeterministic peaks you reported (a float32 guard that let
  degenerate all-weight boxes through), with regression tests proving
  500 ppm transits survive the fix.
- **BLS got a survey-speed campaign on top**: 2.0–12.7× end-to-end on
  realistic Keplerian grids (fused-noverlap kernels, conflict-scatter
  staging, occupancy-aware chunking). Periodograms are parity-identical.
- **The headline feature is a new survey-scale TLS** — 30–171× faster than
  GTLS on the same GPU at equal SDE, thousands× the reference CPU package,
  golden-tested against `transitleastsquares`.
- **scikit-cuda is fully gone** (in-house ctypes cuFFT binding), so #63
  closes as shipped rather than deferred.

You're credited prominently in the release notes (PDM kernels + batch APIs,
CE enhancements, LS improvements, and the BLS epoch/phase work — PRs #57–#62
and #65). Draft: `docs/RELEASE_NOTES_v1.0.0.md` on the `v1.0-fixes` branch —
I'd welcome a skim before it goes out, especially the PDM/CE sections and
the migration table.

Release plan once you give the nod: merge to master, re-tag v1.0.0, publish
to PyPI (first release since 0.2.5!), GitHub Release, docs push, and a sweep
closing the stale issues in favor of a v1.1 roadmap. Also: would you be up
for a co-maintainer invite? Your track record on this codebase speaks for
itself.

Any changes you want in before the tag, or shall I schedule it for [DATE]?

---

*Notes for the maintainer (not part of the message): he replied "Expect some
changes in BLS, but otherwise sounds good!" on #63 in June — if he has
pending BLS work beyond #65, ask whether it targets 1.0.0 or 1.1 before
tagging.*
