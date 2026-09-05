# A3: NFFT error-floor diagnosis — CAUSE FOUND AND FIXED

> The probe scripts and raw JSON/txt named below were pruned before 1.0; see `analysis/README.md` (tag `archive/pre-1.0-process`).

**Claim under investigation** (from the Jun punchlist work, enshrined in
`estimate_m`'s docstring): the realized NFFT error "floors near ~1e-3
absolute … independent of m, in both single and double precision" — an
*inherent* deconvolution/finite-precision limit. The Jul-2 audit flagged
that this exceeds the implemented L1 truncation bound by ~1e4 at float64
and demanded a real diagnosis.

**Verdict: not inherent. The float64 floor was a kernel defect — a
float32 `PI` literal in `cunfft.cu` used in the phase-factor kernels
(`nfft_shift`, `normalize`) in both precision modes.** Fixed by making
`PI` a double literal under `DOUBLE_PRECISION` (plus typing `modflt`
/`diffmod` with `FLT` instead of hardcoded float32 helpers). After the
fix the float64 error tracks the truncation bound down to ~1e-11.

## Mechanism

`#define PI 3.14159…f` has relative error 2.8e-8. The phase arguments in
`nfft_shift` (`phi = 2π·(i%ng)·k0/ng`, up to `2π·|k0|` ≈ 1571 rad for the
reference config) and `normalize` (`theta_k = 2π·n0·(k0+k)/ng`, up to
`π·|k0+k|`) are *not* range-reduced, so the float-π error becomes an
absolute phase error `δφ ≈ 2.8e-8 · |φ|` ≈ 4.4e-5 rad. That corrupts the
spectrum by `~|G(k)|·δφ`, and the Gaussian deconvolution `exp(b·khat²)`
(b ∝ m) then *amplifies* it — which is why the floor **grew** with m
(1.0e-3 at m=6 → 8.0e-3 at m=16) instead of shrinking as `exp(-mD)`.

## Evidence (RTX A5000, `a3_nfft_error_sweep.py`, before → after)

Reference config: ndata=100, ||y||₁≈67, nf=500, σ=2, centered band
(x0=-0.5, k0=-nf/2), vs exact CPU direct sums.

| experiment | result (before fix) | inference |
|---|---|---|
| A: m-sweep, f64 | err: 2.1e-3 (m=4) → 1.0e-3 (m=6) → 3.4e-3 (m=12) → 8.0e-3 (m=16); bound ratio up to 1e10 | floor not just m-independent — grows with m: fixed error × deconvolution amp |
| B: m-sweep, f32 | ≈ f64 values | error source common to both precisions |
| D: fast-math off | identical to 4+ digits | fast-math ruled out |
| F: slow grid (no `precompute_psi`/`modflt`) | identical to 4 digits (3.372e-3 vs 3.371e-3) | gridding path ruled out → error is in shift/FFT/normalize |
| E1 vs E2 matched band (per-k profiles) | E1 (phases active): up to 3.4e-3; E2 (x0=0, k0=0 → phase kernels are identity): 7.7e-6 in the same khat band | **440× — the error is created by the phase factors** |
| E1 low-mode phase error | err/|G| = 4.3e-5 rad at \|mode\|=10 | matches float-π prediction 2π·250·2.8e-8 = **4.4e-5 rad** |
| G: nf-scale | floor roughly ∝ nf (1.1e-3 @200 → 6.2e-3 @2000) | consistent: phase max ∝ \|k0\| = nf/2 |

After the one-line fix (same sweep, `a3_sweep_after_fix.txt`):

| m (f64, σ=2) | before | after | bound 4e^{-mD}·‖y‖₁ | after/bound |
|---|---|---|---|---|
| 6 | 1.04e-3 | 4.46e-5 | 9.3e-4 | 0.05 |
| 8 | 1.52e-3 | 5.21e-7 | 1.4e-5 | 0.04 |
| 10 | 2.24e-3 | 9.97e-9 | 2.1e-7 | 0.05 |
| 12 | 3.37e-3 | **1.19e-10** | 3.3e-9 | 0.04 |
| 14 | 5.15e-3 | 8.32e-12 | 4.9e-11 | 0.17 |
| 16 | 8.00e-3 | 1.37e-11 | 7.5e-13 | 18 (true f64 roundoff floor ~1e-11) |

σ-sweep after fix: 1.2e-10 (σ=2) → 1.7e-12 (σ=5). Float32 unchanged
(~2e-3 floor) — expected: its π literal is correct for that precision;
the f32 floor is float32 trig on large un-reduced phases plus grid/FFT
roundoff, and is a *genuine* single-precision limit (use_double is the
remedy; documented in `estimate_m`).

## Changes landed

- `cuvarbase/kernels/cunfft.cu`: `PI` double literal under
  `DOUBLE_PRECISION`; `modflt` returns `FLT` (was hardcoded `float`);
  `diffmod` uses `fabs` (was `fabsf`).
- `cuvarbase/cunfft.py::estimate_m`: docstring rewritten — bound is now
  honored at f64; the "inherent 1e-3 floor" text replaced with the real
  story and a single-precision-only caveat.
- `cuvarbase/tests/test_nfft.py`: autoset-m test now also asserts
  tol=1e-6 at f64; new regression test
  `test_double_precision_tracks_truncation_bound` (asserts ≤100× bound
  at m=12; the buggy kernel was ~1e6×).

## Related notes (not in scope, flagged)

- `lomb.cu`, `tls.cu`, `nufft_lrt.cu` define the same float32 `PI`
  literal. Exposure is smaller (lomb.cu's direct-sums kernel is a debug
  path — the production LS pipeline gets its phases from the fixed
  cunfft.cu; tls.cu is float32-only by design; nufft_lrt.cu is not
  currently compiled). Worth the same one-line treatment in a v1.1
  hygiene pass.
- The E2 case (k0=0, modes 0..nf-1 on a σ=2 grid) independently
  confirmed the band-edge behavior documented in the C3 smoke test:
  modes k ≥ nf/2 are outside the Gaussian window's accuracy band, and
  the deconvolution blows up towards khat → π/2. That is a window
  property, not a defect; callers wanting the full band accurate should
  request 2× the modes or raise σ.
