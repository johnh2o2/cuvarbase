# NUFFT-LRT (Taaki) Audit — July 2026

> **SUPERSEDED (Sep 2026).** This report cleared the live path; the
> September-2026 algorithm audit
> (`analysis/audit-sep2026/ALGORITHM_AUDIT.md`, sections 2 and 6) found
> five confirmed defects it missed, all now fixed. Read it instead of
> this file for the module's current state. Specifically wrong here:
> the "**the statistic and its implementation are correct**" verdict
> (absolute BJD times were cast to float32 before folding and gridding,
> `epochs=None` evaluated a single phase-0 template rather than a
> search, `detector='sequential'` fitted the basis without an intercept,
> `detector='marginal'` estimated its PSD from data that still contained
> the realized systematics, and the default `sigma = 2` left the modes
> `k >= nf/2` aliased at O(1)); **row 4**'s "harmless while dead"
> reading of `kernels/nufft_lrt.cu` (the same float32-absolute-time fold
> was live in `run()`, not only in the dead kernel); and **finding 5**'s
> "mild for shallow transits" (self-whitening costs 24-28% of the
> statistic already at the detection threshold). The July validation
> campaign referenced below was never run; the Sep-2026 campaign is in
> `analysis/audit-sep2026/campaign/` and predates the fixes.

**Scope**: correctness audit of `cuvarbase.nufft_lrt` (462-line host module,
204-line kernel file, 243-line GPU test file + import tests,
`docs/NUFFT_LRT_README.md`, `examples/nufft_lrt_example.py`), contributed by
Jamila Taaki (@xiaziyna), GPU-rewired July 2 2026 (batch 3). Companion to the
TLS audit (`analysis/tls-audit-jul2026.md`). A validation campaign
(injection-recovery vs BLS/TLS in white and correlated noise, guided by the
Taaki, Kamalabadi & Kemball 2020 methodology) is reported in the validation
section below when complete.

## What the method is (as implemented)

A frequency-domain whitened matched filter: the lightcurve `y` (demeaned) and
each box-transit template (demeaned, generated per trial period/epoch/
duration) are transformed with the GPU adjoint NFFT over the full non-uniform
baseline; the detection statistic is

    SNR = Re Σ_k Y_k T_k* / P(k)  /  sqrt( Σ_k |T_k|² / P(k) )

with `P(k)` either supplied or estimated from the data as a
boxcar-smoothed `|Y_k|²` with a median-scaled floor. The whitening by `P(k)`
is the correlated-noise handling — the method's reason to exist vs BLS/TLS,
which assume white (per-point-σ) noise.

## Verdict

**The statistic and its implementation are correct** for what they claim to
be (a PSD-whitened matched filter with the PSD estimated from the raw
adjoint-NFFT periodogram). The July-2 rewire's properties re-verified from
the current source: full-baseline adjoint NFFT on device with absolute-t
phase convention (device-verified corr=1.0 vs exact adjoint DFT in the
batch-3 record); data and template share the transform so the common phase
and per-mode deconvolution error largely cancel in the whitened correlation;
demeaning of both `y` and template kills the k=0 mode; box template fold and
sign conventions are consistent (dip = −depth ↔ positive SNR); the epoch
axis is honest (no epoch marginalization — you search the grid you supply).

The findings below are (a) genuine-but-secondary defects, (b) statistical
caveats any user must understand (→ the when-to-use docs), and (c) stale
documentation.

## Findings

| # | Severity | Where | Finding | Disposition |
|---|----------|-------|---------|-------------|
| 1 | **Medium (docs, stale)** | `docs/NUFFT_LRT_README.md` banner | The warning banner still describes the two pre-July-2 defects ("computes on the CPU", "median(dt)*nf ... silently ignored") as current, and cites an internal analysis doc. Both were fixed by the GPU rewire; the honest current caveat is "no full injection-recovery validation yet". | **Fix now**: banner rewritten to the true state. |
| 2 | Low (bias) | host PSD smoothing (`run`, `estimate_psd=True`) | `np.convolve(psd, ones(k)/k, mode='same')` has no edge correction: the first/last k/2 bins average with implicit zeros, depressing `P(k)` there and overweighting those bins by up to ~2× in the whitened sums. (The dead GPU kernel version divides by the actual neighbor count — the host path regressed the convention.) Low practical impact — the affected bins are the demeaned k≈0 modes and the top-of-band modes — but it is a bias with a one-line fix. | **Fix now**: count-normalized smoothing + CPU regression test. |
| 3 | Low (cleanup) | `nufft_lrt.py` | Unused imports (`sys`, `NFFTMemory`) — F401. | **Fix now**: removed. |
| 4 | Note (dead code) | `kernels/nufft_lrt.cu`, `NUFFTLRTMemory` | The six CUDA kernels and the memory class are compiled by tests but never invoked by `run()` (the matched-filter reduction is an O(nf) host sum, negligible next to the per-template NFFT — documented in the module). The dead `compute_frequency_weights` kernel additionally encodes the retired one-sided-rfft 1/2/1 weighting, and the dead template kernel folds float32 absolute times (BJD-unsafe). Harmless while dead; a future batched-matched-filter pass should rewrite rather than revive them. | Recorded; kernels stay (documented as unwired). |
| 5 | **Statistical caveat (by design)** | PSD estimation | `P(k)` is estimated from the *signal-containing* data: a strong transit's own harmonic comb inflates the PSD at exactly the template's support, partially self-whitening the signal and depressing its own SNR (mild for shallow transits, grows with depth/duty cycle). The 5-bin boxcar on a single periodogram realization is also a high-variance PSD estimate, and for irregular sampling `|Y_k|²` is the *window-convolved* power, not the true noise PSD. These are properties of the published approach class, not implementation bugs — but users must know them. | → when-to-use docs; validation quantifies the effect. |
| 6 | **Practical constraint** | `run()` cost model | One full GPU adjoint-NFFT round trip **per template** (period × duration × epoch), plus one for the data: a survey-style grid (10⁴ periods × 10 durations × epoch scan) is millions of NFFTs and is not what this tool is for. Practical today at ≲10³–10⁴ templates: single-candidate vetting, small focused grids, or re-scoring BLS/TLS candidates under a realistic noise model. | → when-to-use docs (headline guidance). |
| 7 | Note | `nf` default (`2·len(t)`) | Max template frequency is `nf/T_span`; resolving a duration `d` needs `nf ≳ T_span/d` — for sparse long-baseline data the default underesolves short transits. The README's "increase nf for very gappy data" hint exists; make the criterion explicit in docs. | → docs. |
| 8 | Test gap | `test_nufft_lrt.py` | Behavioral smoke tests only (detection, shapes, white-noise sanity): **no correlated-noise test — the method's entire premise** — and no comparison against a white-noise matched filter or BLS. | Validation campaign adds these (correlated-noise injection-recovery); a fast red-noise regression test lands with it. |

## Test/validation status

- Full suite (release gate, Jul 10): all `test_nufft_lrt*` tests pass on
  device (they are part of the 796/796 run).
- Batch-3 device validation (Jul 2): adjoint NFFT corr=1.0 vs exact DFT in
  the σ=2 guaranteed band; two-season detection exact; phase-convention
  docstring fixed.
- Outstanding (this campaign): injection-recovery in white + correlated
  noise vs BLS/TLS; recreation of the published validation experiment where
  feasible; then a decision on the experimental flag.
