# Draft: message to @xiaziyna about NUFFT-LRT in cuvarbase 1.0.0

Send BEFORE tagging (D1: "message the contributor before tagging either
way"). Phase 4 decided on 2026-09-06: EXPERIMENTAL in 1.0.0 (validated,
API not frozen); the text below reflects that. Email or a GitHub mention
on the release PR both work. The maintainer sends this; nothing here is
automated.

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
the 1.x API-stability promise, so its signature can still move in 1.1.
The docs are honest about the statistic: it is a whitened correlation,
not an N(0, 1) SNR (its null standard deviation is 1.81 for the
validation's ground-based sampling even with the true PSD, and grows
with `nf`), so thresholds have to be calibrated per configuration, and
the validation script shows how.

**The re-validation.** Before tagging I re-ran the injection-recovery
campaign on the fixed code (200 injections per depth and 200 null light
curves per threshold, 600-point ground-based sampling over 90 d, one
A40): all four earlier noise configurations and every detector arm,
plus a configuration on absolute timestamps (`t + 2457000.5`), an arm
using the `epochs=None` default exactly as a user would call it, and a
non-zero-mean basis. The archived results are at
`benchmarks/results/nufft_lrt_validation_2026-09-06/` and the tables are
on the docs page. What it showed:

- The fixes hold on device. BJD-scale times give the same statistics as
  relative times to 5e-8 for every method (0 of 800 detection decisions
  differ); the `epochs=None` search finds the injected transit (99 % of
  its detections within half a duration of the true mid-time); a
  non-zero-mean basis changes nothing (5.5e-7).
- With a shared-systematics basis, Detector A and the sequential
  cotrend + filter recover 3/44/98/100 % of transits at depths
  0.4/0.8/1.6/3.2 % where basis-free BLS recovers 0/0/2/16 % and TLS
  nothing. After the PSD fix Detector A no longer trails the sequential
  baseline -- but it equals it exactly (zero discordant decisions out of
  800), so the marginalization itself bought nothing measurable at these
  sample sizes.
- In OU red noise (1x and 3x the white level) the whitened filter is
  6-10 +- 3 % more complete than BLS at the transition depths and about
  as complete as TLS; a flat-PSD matched filter does as well (1x) or
  better (3x, by 6 +- 2 %), so the gain over BLS comes from the
  full-baseline matched filter rather than from the PSD whitening. In
  white noise BLS and TLS are 10-12 +- 3 % more complete.
- The default epoch grid costs the default call 4-9 % of completeness
  against a twice-finer explicit grid at the transition depths.

**The decision.** The correctness gate passed, so the label is not
about validation any more. I am still shipping it as experimental in
1.0.0, because going official would freeze `run()` for the whole 1.x
series, and the campaign itself says three things should change first:
the default epoch grid should be finer, `run()` should have one return
convention (it returns `(statistic, best_epoch)` for `epochs=None` and a
plain array otherwise), and the whitening default deserves a rethink
given the flat-PSD result. The docs page and the warning say exactly
that ("validated; API may still change"), not "unvalidated". Promotion
in 1.1 is the plan once those land, and I would value your view on each
of them -- especially whether you see a regime where PSD whitening
should beat the flat filter, and whether the Detector A prior API is
worth keeping given the parity with the sequential detector.

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
ALGORITHM_AUDIT.md defects 5 (`lrt-bjd-float32`), 6 (`lrt-epochs-none`), 21
(`lrt-sequential-intercept`), 22 (`lrt-detectorA-defeated`), 24
(`lrt-upper-half-band`) and the ids 121/122 PSD/prior items; the buffer
reuse is LRT-1. Phase 4 (2026-09-06) kept the module experimental; the
promotion items are listed in `issue-sweep.md` for the roadmap issue. The
July audit (`analysis/nufft-lrt-audit-jul2026.md`) predates Detector A
and is only reachable through the archive tag; do not cite it as
current.*
