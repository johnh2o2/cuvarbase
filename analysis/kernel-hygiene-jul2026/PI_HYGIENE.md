# Kernel hygiene (Jul 2026): the remaining float32 PI literals

Closes out the item flagged in the cunfft.cu A3 fix (commit 2699525;
see `analysis/v1.0-gpu-batch3-jul2026/A3_DIAGNOSIS.md`): `lomb.cu`,
`tls.cu` and `nufft_lrt.cu` carried the same float32 `PI` literal that
caused the NFFT double-precision error floor. Same treatment applied
(PI is a double literal under `DOUBLE_PRECISION`, float32 literal
otherwise), with per-file exposure assessed first.

All measurements: RTX A5000 (pod `l3v5jd4km6epgb`), CUDA 12.4,
before = kernels at 89d5481, after = this branch. Raw outputs:
`ls_before.json` / `ls_after.json` (from `ls_pi_validation.py`) and
`spot_before.json` / `spot_after.json` (from `tls_nufft_spot.py`).

## Inventory and exposure

| file | PI literal | referenced by | exposure |
|---|---|---|---|
| `lomb.cu` | line 6, float32 (rel. err. +2.7828e-8) | `cossum`/`sinsum` → `lomb_dirsum`, `lomb_dirsum_custom_frq` (the `use_fft=False` direct-sums path; the production NFFT path never touches PI) | **live** in double mode: phase `(t+0.5)*f*2*PI` is un-reduced, so the f64 periodogram is evaluated on a frequency axis stretched by 1+2.78e-8. Observable error ~ `2.78e-8 * f * T` × local dP/dlnf: predicted 8.4e-5 at f=100 c/d, T=30 d; **measured 1.162e-4**. Scales linearly in f·T (a decade baseline at f=50 c/d → ~7e-3). float32 mode: literal value unchanged (same 0x40490FDB), bit-identical output. |
| `tls.cu` | line 27, float32 | **nothing** (dead macro; kernel is float32-only by design, no `DOUBLE_PRECISION`/`FLT` mode exists) | zero — removed the dead macro instead of guarding it |
| `nufft_lrt.cu` | line 6, float32 | **nothing** (dead macro today) | zero via PI, but the file has a real `DOUBLE_PRECISION` mode, so PI moved under the guard (future-proofing, exact cunfft idiom). Live same-class defects fixed alongside: `fmaxf`/`fmodf`/`fabsf` on `FLT` operands truncated doubles to float32 in double mode (the analogue of 2699525's `modflt`/`fabsf` → `FLT`/`fabs` typing). |

Note on the mechanism difference vs cunfft.cu: there the float-π phase
error was *amplified* by the Gaussian deconvolution; here it is exactly
equivalent to a uniform frequency-axis rescaling by 1+2.78e-8 (the
phase is strictly linear in `f`), so the corruption is bounded by
`eps * f * T` of the local periodogram slope — real but much milder,
matching the "lower exposure" call in the A3 diagnosis.

## lomb.cu rigorous gate (RTX A5000)

`ls_pi_validation.py`: n=300, T=30 d, signal f0=97 c/d, grid 90–110 c/d
(nf=3000, spp=5), floating-mean mode, `use_double=True`, direct sums.
Reference `ref64` = float64 CPU port of the kernel (identical op order,
exact `np.pi`); `ref32pi` = same port with `pi = float64(float32(pi))`;
astropy `LombScargle(fit_mean=True, center_data=True)` as an
independent check.

| case | BEFORE max\|ΔP\| vs ref64 | AFTER max\|ΔP\| vs ref64 | improvement |
|---|---|---|---|
| A: low-level API, epoch 4.5 | 1.162e-4 | 3.670e-10 | 316,517× |
| B: low-level API, epoch 2,450,000 (BJD) | 1.162e-4 | 1.222e-8 | 9,505× |
| C: `run()` entry point, epoch 4.5 | 1.162e-4 | 3.669e-10 | 316,587× |
| D: float32 mode | — | before/after max\|ΔP\| = 0.0 (bit-identical) | — |

Diagnosis confirmation: BEFORE matches `ref32pi` to 3.7e-10 (A/C) and
1.2e-8 (B) — i.e. the buggy kernel *is* the float32-π model to
roundoff; AFTER sits 1.162e-4 from `ref32pi`, symmetrically. vs
astropy: AFTER agrees to 2.4e-13 (A), 1.4e-8 (B, f64 roundoff on
BJD-magnitude phase products), 1.4e-13 (C); BEFORE was 1.162e-4 from
astropy in all three.

Note `run()` mean-centers `t` in float64 before upload, so BJD-scale
epochs reach the kernel only through the low-level
`lomb_scargle_async`/`LombScargleMemory` API (case B). The error is
epoch-independent (coherent rescaling), which cases A vs B confirm.

New regression test:
`test_lombscargle.py::test_ls_kernel_direct_sums_double_pi` (f64 GPU
dirsum vs the float64 kernel port at f·T ≈ 3e3; threshold 1e-7;
measured on the pod with `reg_test_probe.py`: after = 1.1185e-10,
buggy kernel = 1.2492e-4).

Full suite after fix: 753 passed / 7 skipped (752 baseline + the new
regression test); `scripts/check_release_gate.py` 14/14 green.

## tls.cu / nufft_lrt.cu lighter gate (same pod)

- Both kernels compile (nufft_lrt.cu in float32 AND double mode; tls.cu
  via `compile_tls`), all existing tls/nufft_lrt tests pass in the full
  suite run.
- TLS spot (`tls_search_gpu`, 800 pts, injected 3.0 d transit):
  period/duration/depth/SDE and the full chi2/power/SR arrays are
  bit-identical before/after (max|Δ| = 0.0) — the removed macro was
  provably dead.
- nufft_lrt matched-filter kernel launched directly vs a numpy float64
  port: float32 mode bit-identical before/after (num = -22.9774398803711
  both). Double mode improved from rel. err. 3.5e-9 / 2.4e-9
  (num/den — the `fmaxf` float32 truncation) to **exactly 0.0** after
  the `FLT` typing fix. `generate_transit_template` f64 max|Δ| vs port:
  0.0. End-to-end `NUFFTLRTAsyncProcess.run()` SNR: identical
  before/after, best period 3.0 d as injected.
