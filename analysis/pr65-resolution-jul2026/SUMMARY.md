# PR #65 resolution — non-zero-epoch test failures root-caused and fixed (Jul 2026)

Branch: `pr65-resolution` (base: `pr65-latest` = PR #65 head 11e9a5bd, itself
rebased by @astrobatty onto `v1.0-fixes` @ 2cc1f96).
Validation hardware: RunPod RTX A5000, CUDA 12.4, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.

## What attila reported (Jul 3)

With `t = t + 4.5` added to the `data()` test fixture (non-zero epoch:
`floor(min(t))` = 4-5), "most of the test_standard cases fail. Typically,
there is one frequency at which the power difference exceeds the threshold."
He asked us to figure it out, flagged `test_single_bls_bjd_invariance` as
needing a convention update, and declined to touch `hone_solution` (no test).

## Reproduction (his code + the fixture change, commit 71085d6)

`pytest cuvarbase/tests/test_bls.py::TestBLS::test_standard` on the A5000:
**61 failed / 155 passed** of 216 parametrizations. Every failure has
`q_index=0` (q = 10^-1.5, the smallest boxes, ~6-10 points in transit);
`use_optimized`, `nstreams`, `freq_batch_size`, `ignore_negative_delta_sols`
don't matter. Diagnostic dump (`raw/diag_standard.json`, `raw/diag_standard.py`):

```
qi=0 pi=0:  maxdiff=0.0357 at 1/50 freqs (f=0.99997075), p_gpu=0.2319
            p_cpu=0.1962, sol=(q=1/22, phi=0.977127), n_in_box=10,
            exactly ONE point within 1e-4 phase of a box edge
```

`nviol=1/50` trips `mostly_ok` (which allows zero violations at 50
frequencies); on attila's GPU/data the same class of flip evidently exceeded
0.1 and tripped `not_too_bad` -- same mechanism, different die roll.

## Root cause (NOT his epoch conversion)

The float64 phi round-trip his PR adds
(`eebls_gpu`: `(phi + epoch*f) % 1` -> `single_bls`: `(phi - epoch*f) % 1`)
is **bit-exact at float32 precision** -- a float32-faithful CPU simulation of
the whole standard pipeline (`raw/sim_standard.py`) produces identical
diffs (~3e-7) with and without the conversions.

The real mechanism is a **pre-existing fold-order precision asymmetry in
`single_bls`**, exposed by the non-zero epoch re-rolling every float32 phase:

- GPU kernels: `phi = mod1(t*f)` **wraps into [0,1) first** (resolution
  ~1e-7), then computes bin membership.
- `single_bls` (reference): `phi = t32*f32; phi -= phi0; phi -= floor(phi)`
  -- subtracts at magnitude `t*f ~ 362`, where float32 ulp = 3.05e-5.

Drill-down at the failing frequency (`raw/drill_worst.py`): data point
idx=198 has true phase 8.3e-6 *below* the best box edge (float64: genuinely
out of the box; the GPU excludes it). In `single_bls`, `361.97726 -
0.97727275` rounds to exactly `361.0` -> relative phase **exactly 0.0** ->
spuriously *in* the box. One point in a ~10-point box = ~power/n = 0.036
disagreement. The old fold loses up to `ulp(t*f)/2 ~ 1.5e-5` phase at a 1-yr
baseline (2.4e-4 at 10 yr).

Hardware probe (`raw/fma_probe.py`, `raw/fma_probe_result.txt`): of 2M random
times, 341 discriminate `frac(fl32(t*f))` from `fl32(frac(exact t*f))`; the
compiled kernel matched the **plain rounded-product** fold 64/64 -> nvcc does
NOT FMA-contract `mod1(t*f)`, so a wrap-first CPU fold is **bit-identical**
to the kernel's fold. The fix is exact, not a tolerance judgement.

## Additional genuine defects found in the PR (fixed here)

1. **`sparse_bls_cpu`/`sparse_bls_gpu` (commit c304aab)** re-referenced
   solution phases with the **float32-cast** `freqs` while `single_bls`
   converts back with the caller's float64 freq: error `epoch*|f64-f32|` --
   ~1.5e-7 cycles in the tests (passed), but **~0.07 cycles at BJD-scale
   epochs** (~2.45e6 d), i.e. completely wrong reported phases. Now converts
   with float64 frequencies.
2. **`bin_and_phase_fold_custom`** folded with the now-double `freqs`
   (`t * f64`), shifting each phase by up to `|f64-f32|*t ~ 1e-5` vs the
   float32 reference fold -- breaking the pre-PR bit-parity that
   `test_custom`'s 1e-5 assertion relies on. Now folds with the float32-cast
   frequency; double freqs are used **only** for the epoch re-referencing.
   `phi_values` are uploaded as float64 so the in-kernel
   `(phi - epoch*f) % 1` matches `single_bls`'s float64 conversion bit for
   bit (`store_best_sols_custom` signature updated).

## Changes on `pr65-resolution`

- `71085d6` -- fixture `t0=4.5` in `data()` (shift applied **before** the
  model, so the injected transit stays at original-timescale phase phi0);
  `test_ignore_positive_sols` passes phi0 directly (new convention) with the
  hardcoded value updated for the rotated fold (0.8902... -> 0.9223...);
  `test_single_bls_bjd_invariance` uses covariantly shifted phases
  `(phi0 + offset*f) % 1`; new `test_single_bls_phase_is_original_timescale`
  (unshifted phi0 on shifted times must MISS the transit -- guards against
  the convention silently reverting).
- `51410f9` -- the parity fixes:
  - `single_bls`: wrap-first fold (documented, hardware-verified bit-parity
    with the kernels).
  - custom kernel fold + float64 `phi_values` (above).
  - sparse CPU/GPU float64 phase conversion (above).
  - Review minors: `eebls_transit_gpu` always returns a 3-tuple
    (`sols=None` on fast/optimized paths); `eebls_transit
    (use_optimized=True)` respects an explicit `block_size`; `test_transit`
    uses one 3-way `mode` axis (standard/fast/optimized) instead of a
    `use_fast x use_optimized` cross-product; `test_standard`/`test_custom`
    drop the `use_optimized` axis (only `reduction_max` differs between the
    modules -- bin/store kernels are byte-shared via `bls_common.cuh`) in
    favor of focused equivalence tests
    (`test_standard_use_optimized_matches`,
    `test_custom_use_optimized_matches`); new `TestHoneSolution` exercises
    `hone_solution` end-to-end at maximal epoch-phase rotation
    (`(epoch*freq) % 1 = 0.5`), closing the "no test for hone_solution" gap.
    `hone_solution` itself needed **no code change** -- after attila's
    updates it is already convention-consistent, which the new test now
    proves on hardware.
- final commit -- CHANGELOG (convention + fold-order fix entries) + this
  summary.

## Validation (RTX A5000)

- CPU-only suite (pycuda stubbed): `test_bls.py` **96 passed** locally.
- GPU targeted (post-fix): `test_standard` + `test_custom` + equivalence +
  hone tests: **159/159 passed** (vs 61 failures on the same pod minutes
  earlier).
- GPU full suite (with the t0=4.5 fixture): **752 passed, 7 skipped**
  in 6:24 (`raw/full_suite.log`; run on the synced 51410f9 code state,
  code-identical to the final commit, which adds only docstrings,
  CHANGELOG and this analysis directory). The 7 skips are the usual
  environment-dependent ones (e.g. optional deps).
- Release gate `scripts/check_release_gate.py`: **ALL CHECKS PASSED**
  (exit 0; includes reduction_max standard-vs-optimized equivalence at
  corr=1.000000, BLS transit recovery, kernel-cache timing, LS/CE/PDM
  checks; `raw/release_gate.log`).
- Merge cleanliness: `git merge-base v1.0-fixes pr65-resolution` =
  `2cc1f96` = `v1.0-fixes` HEAD -> `git merge v1.0-fixes` is a no-op
  ("already up to date"); the PR remains a clean fast-forward candidate.

## Draft reply to attila (do not post without review)

> Root-caused it -- and it wasn't your epoch conversion at all. Your float64
> round-trip (`(phi + epoch*f) % 1` on return, `(phi - epoch*f) % 1` in
> `single_bls`) is bit-exact at float32 precision; a float32-faithful CPU
> simulation of the whole standard pipeline produces identical diffs with
> and without it.
>
> What your `t += 4.5` change actually exposed is a fold-order precision bug
> in `single_bls` that has been there all along. The GPU kernels wrap the
> phase into [0,1) *before* binning (`mod1(t*f)`, ~1e-7 resolution), but
> `single_bls` subtracted phi0 from the *unwrapped* float32 product
> `t*f` -- at magnitude ~362 for a 1-year baseline, where float32 ulp is
> 3.05e-5. At the failing frequency in `test_standard[...-0-0-1.0]` on our
> A5000 (61/216 cases failed, all q_index=0), point idx=198 has true phase
> 8.3e-6 *below* the best box edge: the GPU correctly excludes it, while
> `single_bls` computes `361.97726 - 0.97727275 -> 361.0` exactly and gets
> relative phase exactly 0.0 -> spuriously in-box. One point in a ~10-point
> box is a ~power/n ~ 0.04 disagreement at one frequency -- your failure
> signature. Shifting t re-rolls every float32 phase, so the old epoch~0
> realization had just been lucky.
>
> The fix is to make `single_bls` wrap first, like the kernels. We probed
> the compiled kernel on hardware to be sure this is exact: nvcc does *not*
> FMA-contract `mod1(t*f)` (64/64 discriminating test points match the plain
> rounded-product fold), so wrap-first is bit-identical to the kernel fold,
> not merely closer.
>
> Two small things we adjusted in your changes while validating, both
> invisible in the tests but large at BJD scale:
> 1. `sparse_bls_cpu`/`sparse_bls_gpu` re-referenced solution phases with
>    the float32-cast `freqs`; `single_bls` converts back with the caller's
>    float64 freq, so phases were off by `epoch*|f64-f32|` -- negligible
>    here but ~0.07 cycles at epoch ~2.45e6 d. Now both convert in float64.
> 2. In `bin_and_phase_fold_custom`, folding with the double `freqs`
>    (`t * f64`) moved each phase by up to `|f64-f32|*t ~ 1e-5` relative to
>    `single_bls`'s float32 fold, occasionally flipping an edge point
>    against `test_custom`'s 1e-5 assertion. The kernel now folds with the
>    float32-cast frequency and keeps the doubles only for the epoch
>    re-referencing; `phi_values` go up as float64 so the in-kernel
>    conversion matches `single_bls` bit for bit.
>
> We also took care of the rest so you don't have to: `t += 4.5` is now
> permanent in the fixture (applied before the model, so the injected
> transit stays at original-timescale phi0), `test_single_bls_bjd_invariance`
> uses covariantly shifted phases as you suggested, `hone_solution` turned
> out to be convention-correct after your updates and now has an end-to-end
> test at maximal epoch rotation ((epoch*freq) mod 1 = 0.5), and
> `eebls_transit_gpu` returns a uniform 3-tuple. On the A5000 the previously
> failing set is now 159/159, and the full GPU suite + release gate pass
> with the non-zero-epoch fixture in place. Thanks for pushing on this --
> the shifted fixture flushed out a real reference-side bug (plus the two
> BJD-scale conversion issues) that the epoch~0 tests could never see.

## Raw evidence

- `raw/test_standard_repro_tail40.log` -- reproduction run (61 failed/216;
  tail-40 capture).
- `raw/diag_standard.py` / `raw/diag_standard.json` -- per-frequency dump +
  edge-point analysis of failing configs.
- `raw/drill_worst.py` -- float32 fold trace of the flip point (idx=198).
- `raw/sim_standard.py` -- float32-faithful CPU pipeline simulation (proves
  the phi round-trip is a numerical no-op).
- `raw/fma_probe.py` / `raw/fma_probe_result.txt` -- hardware
  FMA-contraction probe (kernel fold = plain rounded-product fold, 64/64).
- `raw/full_suite.log`, `raw/release_gate.log` -- final validation on
  `pr65-resolution` (with `t0=4.5` fixture).
