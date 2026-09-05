# Draft: coordination message to @astrobatty (post as a comment on #63 or email)

Refreshed Sep 2026 for the post-audit state of `v1.0-fixes`. Fill the
`<...>` placeholders from the Phase 5 gate record before sending. The
maintainer sends this; nothing here is automated.

---

Hey @astrobatty — v1.0.0 is nearly ready and I'd like to coordinate timing
with you before I push the button. This is a longer note than the July one
because the branch went through a second, much deeper audit in September and
several of the results touch code you wrote or use.

**What happened since July**

A full per-method soundness audit (every algorithm compared against an
independent reference on a GPU, plus a release-hygiene pass) found 25
confirmed defects on `v1.0-fixes`. All 25 are fixed, each with a regression
test that fails on the pre-fix tree, and the whole suite was re-run on a GPU
with zero skips (<N> tests at the freeze; the exact number is in the release
notes). The ones I think you'll care about:

- **Input validation now raises (breaking).** Every public entry point
  raises `ValueError` on non-finite `t`/`y`/`dy`, `dy <= 0`, mismatched
  lengths, empty or too-short light curves, and non-finite/non-positive
  frequency grids. Previously a single NaN gave a finite periodogram with
  the wrong argmax (BLS, CE), an all-NaN spectrum (PDM), or a silent -1
  (Lomb-Scargle). If a pipeline of yours relies on NaNs passing through,
  it needs a mask before the call.
- **BLS:** `eebls_gpu` used a 32-bit thread index and sized its bin
  buffers from a non-monotone bound, so large Keplerian-q grids could
  overflow/overrun (illegal memory access); both fixed. Per-frequency
  `qmin`/`qmax` arrays are now honoured per frequency (they used to
  collapse to one batch-wide window, so most Keplerian solutions fell
  outside their own duration window). The fast kernels now evaluate the
  `qmax` box itself (the ladder stopped one rung short). And the Keplerian
  frequency-grid recursion (`transit_autofreq`, `keplerian_freq_grid`) is
  now solved in numpy instead of a per-frequency Python loop — 10x faster
  at survey sizes, and disclosed as result-changing at the level of
  float64 rounding (grid length identical, frequencies agree to ~1e-15
  relative; the float32 `keplerian_freq_grid` output is bitwise
  identical in every configuration we tested; `method='recursion'` keeps
  the old loop).
- **CE:** the brightest point was binned out of range (index `mag_bins`,
  never clamped — it spilled into the next phase bin / next frequency),
  and the weighted histogram truncated `max_phi` asymmetrically (bins below
  the datum lost mass, the brightest point lost all of it). Both fixed;
  weighted results change. Also: the double+fast kernels no longer crash
  with `misaligned address`, `balanced_magbins`/`widen_mag_range` passed to
  the constructor are no longer ignored, `preallocate()` actually uploads
  the frequency grid, and CE compiles once per process instead of once per
  call.
- **PDM:** an out-of-bounds bin read in `binned_step` when a phase rounds
  to exactly 1.0 (up to 0.02 deviation), and the deprecated `(t, y, w,
  freqs)` format now normalizes the weights instead of returning a flat 1.0
  spectrum. `run()` now pools its device buffers across same-shape calls,
  which is what makes `batched_run_const_nfreq`/`large_run` cheap per chunk.
- **Lomb-Scargle / NFFT:** the w-spectrum was gridded with the psi tables
  of a differently sized grid (every default-path power biased by
  3e-3..2e-2), the NFFT grids were sized so that bands not starting near
  zero returned garbage, and `use_double=True` was less accurate than
  float32 because of a `floorf` on a double. All fixed; float64 now sits at
  ~1e-8 of astropy on dense grids.

**Performance (Phase 2)** was measured on one shared A40, so I only quote
ratios: BLS calls that used to recompile per call (`eebls_gpu`,
`sparse_bls_gpu`, `hone_solution`) go through the kernel cache (30-90x on
small calls), the adaptive/optimized BLS paths use the fused kernel they
were silently missing (~2x GPU time), the CE `use_fast=True` grid is sized
from the device and is now the faster CE path (1.2-8x over the default
kernels depending on N), and the multiharmonic LS host solve is stacked
(50-300x on the solve alone). Everything is bit-neutral except where the
changelog says otherwise.

**API freeze.** 1.0 has one top-level namespace (`cuvarbase.__all__`
equals the lazy attribute list; the accidental `cuvarbase.np`-style names
are gone). Compatibility shims for what shipped in 0.2.5 stay with a
`DeprecationWarning` that says "removed in 2.0": `cuvarbase.core`,
`BLSMemory.allocate_pinned_arrays`, the PDM `(t, y, w, freqs)` 4-tuple, and
`GPUAsyncProcess(reader=, function_kwargs=, device=)`. Anything that never
shipped in a release was simply removed. NUFFT-LRT stays importable as
`cuvarbase.nufft_lrt` but out of the top-level namespace, with a warning at
construction, until its re-validation lands.

You're credited in the release notes (PDM kernels + batch APIs, CE
enhancements, LS improvements, the BLS epoch/phase work — PRs #57-#62 and
#65). Draft: `docs/RELEASE_NOTES_v1.0.0.md` on `v1.0-fixes`; the per-fix
detail is in `CHANGELOG.rst` under "Sep-2026 audit fixes". I'd welcome a
skim before it goes out, especially the PDM/CE sections, the breaking
input-validation paragraph, and the migration table.

**Release sequence** once you give the nod: merge `v1.0-fixes` to `master`
(four known conflicts, all resolved for the branch), delete and re-create
the stale June `v1.0.0` tag on the merge commit, publish to PyPI (first
release since 0.2.5), GitHub Release, docs push, then a sweep closing the
stale issues in favor of a v1.1 roadmap.

Three asks:

1. Skim the notes/changelog and tell me if anything above breaks something
   you run.
2. Would you be up for a co-maintainer invite? Your track record on this
   codebase speaks for itself.
3. Timing: any changes you want in before the tag, or shall I schedule it
   for [DATE]? (After release I'd also like to delete the old
   `fix/BLS-kernel` and `bugfix/BLS-kernel` remote branches — shout if you
   still need them.)

---

*Notes for the maintainer (not part of the message): he replied "Expect some
changes in BLS, but otherwise sounds good!" on #63 in June — if he has
pending BLS work beyond #65, ask whether it targets 1.0.0 or 1.1 before
tagging. Since July the BLS entry points he touched in #65 (`single_bls`,
the epoch/phase reporting) gained the Sep-2026 per-frequency `q` bounds
and the numpy grid solver above; if he has a local branch against the July
tree, point him at the CHANGELOG "Sep-2026 audit fixes" BLS bullets.*
