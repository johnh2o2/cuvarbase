# Draft: message to @xiaziyna about NUFFT-LRT in cuvarbase 1.0.0

Send BEFORE tagging (D1: "message the contributor before tagging either
way"). Fill the `<...>` placeholders after Phase 4 decides official vs
experimental. Email or a GitHub mention on the release PR both work. The
maintainer sends this; nothing here is automated.

---

Hi Jamila,

cuvarbase 1.0.0 is close to release and your NUFFT likelihood-ratio transit
search is part of it, so I wanted to tell you exactly what is shipping, what
changed, and ask for your eyes on two things.

**What was fixed.** A September audit compared every method in the package
against independent references on a GPU. For the LRT module it turned up
six things, all now fixed with regression tests:

1. Absolute (BJD-scale) times were cast to float32 before the fold and the
   NFFT, so a light curve at t ~ 2.457e6 gave a different statistic than the
   same light curve with a subtracted epoch (correlation ~0.5, different
   argmax). `run()` now subtracts `floor(min(t))` in float64 first and
   shifts any supplied `epochs` into the same frame.
2. `epochs=None` (the public default) evaluated one phase-0 template per
   (period, duration) cell rather than searching over epoch. It now scans
   an automatic epoch grid (`clip(ceil(2P/duration), 8, 96)` epochs per
   cell) and returns `(statistic, best_epoch)`; this is a signature change
   and is flagged as breaking in the changelog.
3. `detector='sequential'` fit the systematics basis without an intercept on
   un-demeaned data, so any basis column with a non-zero mean pulled the
   statistic down (a 1% column mean on relative flux took the peak from ~25
   to ~5). Basis and data are now centred before the least-squares solve.
4. `detector='marginal'` (Detector A) with `estimate_psd=True` estimated the
   PSD from `y - V mu`, which still contained the realized systematics, so
   the whitening removed the transit along with them and Detector A trailed
   its own sequential baseline. The PSD now comes from the basis-projected
   residual; Detector A matches the sequential baseline again.
5. The NFFT oversampling default was `sigma = 2`, which put the upper half
   of the returned modes outside the Gaussian window's accuracy band (O(1)
   aliasing error, even in double precision, and run-to-run jitter).
   Default is now `sigma = 4`, matching the rest of the library.
6. Supplied PSDs are validated and floored (`eps_floor` default 1e-3 of the
   positive median; a zero bin used to return ~1e6 or NaN), a singular
   `coeff_prior_cov` is handled in the correct limit (`C (I + G C)^-1` via a
   solve instead of `pinv(pinv(C) + G)`), and `run()` validates its inputs
   (and warns that `dy` is not used — the noise model is the PSD).

On top of that, one NFFT buffer set is now allocated per `run()` and
reused for the data, the basis and every template (about 0.15-0.3 ms per
template on an A40), which is what makes the injection-recovery campaign
below affordable.

**How it ships in 1.0.0.** The module is importable as `cuvarbase.nufft_lrt`
(`NUFFTLRTAsyncProcess`, `detector='matched' | 'marginal' | 'sequential'`).
It is deliberately kept out of the top-level `cuvarbase` namespace, emits a
warning the first time a process object is constructed, and sits outside
the 1.x API-stability promise, so its signature can still move in 1.1 if
the validation says it should. The docs are honest about the statistic:
it is a whitened correlation, not an N(0, 1) SNR (its null standard
deviation is 1.8-2.7 for ground-based sampling even with the true PSD and
grows with `nf`), so thresholds have to be calibrated per configuration,
and the validation script shows how.

**Before tagging** we re-run the injection-recovery campaign on the fixed
code: all four existing configurations and all detector arms, plus a
configuration with `t + 2457000.5`, an arm using the `epochs=None` default,
a non-zero-mean basis, and at least 200 injections per depth so 0.05-level
differences resolve. That run decides whether the module is labelled
"official" or "experimental" in 1.0.0; I don't want to promise which until
the numbers are in. <Phase 4 outcome: "It came back as ..., the archived
results are at benchmarks/results/nufft_lrt_validation_<date>/ and the
summary is in the docs page.">

**Credit.** You are named as the contributor of the NUFFT-LRT search in the
README acknowledgements (with Taaki, Kamalabadi & Kemball 2020 and the
reference implementation linked), in the release notes, and in the
CHANGELOG entry for the module.

**Two asks, if you have the time:**

1. Would you look over the docs page for the module (`docs/source/nufft_lrt.rst`
   on the `v1.0-fixes` branch; it renders as `nufft_lrt.html` on the
   rebuilt site) and the `detector=` API? In particular whether the
   description of the three detectors, the PSD convention and the return
   shapes say what you'd want them to say.
2. How would you like the method framed and cited? Right now the README and
   the docs cite Taaki, Kamalabadi & Kemball (2020) and link the reference
   implementation; if there is a preferred citation, an author-list form,
   or a name for the method you'd rather we used, I'll match it.

Thank you again for contributing this — it is the one method in the package
that treats systematics as part of the model rather than something to
detrend away first, and I'd like 1.0 to present it accurately.

Best,
John

---

*Notes for the maintainer (not part of the message): the six items map to
ALGORITHM_AUDIT.md defects 5 (`lrt-bjd-float32`), 6 (`lrt-epochs-none`),
21 (`lrt-sequential-intercept`), 22 (`lrt-detectorA-defeated`), 24
(`lrt-upper-half-band`) and the ids 121/122 PSD/prior items; the buffer
reuse is LRT-1. If Phase 4 keeps the module experimental, say so plainly
in the placeholder and mention that Detector A promotion is queued for
1.1 in the roadmap issue. The July audit
(`analysis/nufft-lrt-audit-jul2026.md`) predates Detector A and is only
reachable through the archive tag; do not cite it as current.*
