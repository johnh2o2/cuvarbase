# NUFFT-LRT injection-recovery re-validation, 2026-09-06 (Phase 4 of the 1.0 release plan)

The campaign that decided D1 (official vs experimental) for
`cuvarbase.nufft_lrt` in 1.0.0, run on the fixed module (Sep-2026
correctness fixes: float64 epoch subtraction, automatic epoch grid for
`epochs=None`, centred sequential cotrend, Detector A PSD from the
basis-projected residual, `sigma = 4` NFFT, PSD/prior validation,
per-run NFFT buffer reuse).

| | |
|---|---|
| harness | `scripts/nufft_lrt_validation.py` (harness version 2) at commit `2f9736a`; tables by `scripts/summarize_lrt_validation.py` |
| GPU / stack | one NVIDIA A40 (RunPod, CUDA 12.4, Python 3.11, pycuda 2026.1, numpy 2.4.6, cufinufft 2.5.1, batman-package 2.5.3, transitleastsquares 1.32) |
| protocol | 600-point ground-based sampling over 90 d; 32 log-spaced trial periods 2-18 d with P = 5.3 d and 2P on the grid; box transits of 0.22 d at random epochs; `sigma_white = 3e-3`; per configuration and arm: null p95 threshold from 200 signal-free light curves, then 200 injections per depth (4 depths); detection = statistic above threshold and best period within 1 % of P, 2P or P/2 |
| configurations | `white`, `white_bjd` (the white light curves on `t + 2457000.5` d, paired), `red_1x`, `red_3x` (OU red noise, tau 0.8 d, at 1x / 3x sigma_white), `red_sys` (1x red + three shared systematics modes, PCA basis + population prior), `red_sys_nzm` (the same light curves searched with non-zero-mean basis columns, paired) |
| arms | `lrt` (explicit epoch grid), `lrt_auto` (the public default path, `epochs=None`), `lrt_flat` (PSD = ones; red configs), `lrt_marg` (Detector A), `lrt_seq` (least-squares cotrend + filter), `bls` (`eebls_gpu_fast`), `tls` (`tls_search_batch`, delta-chi2 statistic) |
| split | 8 processes (`launch_campaign.sh`), 12:13-15:43 UTC, 79,458 s of process compute in total; started from a clean checkout of `2f9736a` |
| seed | 20260711 (per-configuration sub-seeds: paired configurations share one, so they see identical light curves) |

## Files

- `nufft_lrt_validation_2026-09-06.json` -- the merged campaign (`--merge` of the 8 process JSONs): `meta`, `snr_calibration`, and per configuration the protocol, and per arm the null p95, completeness, epoch recovery, seconds per search, **and the per-light-curve records** (200 null maxima; per injection the statistic, best period, best epoch and decision), from which every number in the docs, the uncertainties (bootstrap of the null threshold + Wilson) and the paired comparisons are recomputed by the summarizer.
- `summary.md` -- `summarize_lrt_validation.py` output (the `--rst` form of the same is in `docs/source/nufft_lrt.rst`).
- `logs/A_white.log` ... `logs/F2_nzm_marg.log` -- the 8 process logs (progress, per-arm null p95, seconds per search, completeness).
- `logs/p4_first_nfft_ls_lrt.log` -- the first device run of the CPU-landed `cunfft.cu` change (`test_nfft.py`, `test_lombscargle.py`, `test_nufft_lrt*.py`): 210 passed, 1 failed -- the failure is a test asserting bitwise equality of two double-precision LS runs (float64 `atomicAdd` order differs at 6.7e-15 relative); fixed in `2f9736a` by comparing to rounding.
- `logs/p4_full_suite.log` -- the full GPU suite at 954f037 + that test fix: 1785 passed, 1 xfailed, 0 failed, 0 skipped (1,786 collected) in 8 min 6 s.
- `launch_campaign.sh` -- the process split used.

## Headline numbers (completeness, 200 injections per depth; see `summary.md` for uncertainties and the paired differences)

| configuration | depths | lrt | lrt_auto (default path) | lrt_flat | lrt_marg | lrt_seq | bls | tls |
|---|---|---|---|---|---|---|---|---|
| white | 0.002/0.003/0.004/0.008 | 13/47/82/99 % | 10/42/74/99 % | -- | -- | -- | 13/60/91/100 % | 16/65/90/100 % |
| white_bjd | same | identical to white (max rel. diff of any statistic 5e-8; 0 of 800 decisions differ) | identical | -- | -- | -- | identical | identical |
| red_1x | 0.004/0.006/0.008/0.016 | 4/25/56/100 % | 4/24/52/99 % | 4/22/54/100 % | -- | -- | 2/17/47/100 % | 4/22/62/100 % |
| red_3x | 0.008/0.016/0.024/0.032 | 0/12/57/89 % | 0/10/48/83 % | 0/18/63/90 % | -- | -- | 0/7/48/88 % | 0/17/62/93 % |
| red_sys | 0.004/0.008/0.016/0.032 | 0/0/6/34 % | 0/0/5/34 % | -- | 3/44/98/100 % | 3/43/98/100 % | 0/0/2/16 % | 0/0/0/0 % |
| red_sys_nzm | same | -- | -- | -- | identical to red_sys (max rel. diff 5.5e-7; 0 of 800 decisions differ) | identical | -- | -- |

Null calibration of the single-template statistic on white noise: the
JSON's 200 draws give mean 0.348, std 1.579; 5000 draws from the same
seed (`snr_calibration` with `n=5000`, run on the same pod) give mean
0.030 +- 0.026, std 1.808 -- the 200-draw sample is an unlucky one, and
the harness now draws 1000 by default. The statistic is exactly odd in
the data (verified: `S(-y) = -S(y)` to 2e-9), so its null mean is zero
by construction; the std is the calibration constant (1.81, unchanged
from the pre-fix campaign's 1.812 and independent of `sigma`).

Cost on the A40 under the 8-process split: 3.4-6.7 s per LRT search
(7,473 templates for the explicit grid, ~5,900 for the default path);
1.2-1.6 ms for BLS and 8-11 ms for TLS. Single-process timings are
~2.5x lower for the LRT arms (0.23 ms per template).

The pre-fix campaign this supersedes: `analysis/audit-sep2026/campaign/`
(60 nulls / 60 injections, `sigma = 2`, explicit epochs only, relative
times, zero-mean basis; `ALGORITHM_AUDIT.md` section 6).
