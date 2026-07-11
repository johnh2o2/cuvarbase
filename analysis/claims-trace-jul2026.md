# Performance-Claims Trace — cuvarbase v1.0-fixes @ 72f3663 (July 2026)

Companion to `analysis/tls-audit-jul2026.md` (Phase 1b of the pre-release
audit). Every quantitative performance claim in README.md, CHANGELOG.rst,
`analysis/TLS_COST_ANALYSIS.md`, and `analysis/GTLS_COMPARISON.md` was traced
to the raw archives under `benchmarks/results/`; all ratios were **recomputed
from the raw JSON/txt files** — prose in analysis docs was treated as under
audit, not as ground truth.

## Bottom line

The flagship results **trace exactly to archived raw data**:

- TLS **30–171×** vs GTLS (same A5000, matched search + epoch density, equal
  SDE, 200–2000-d baselines; recomputed 30.0 / 55.3 / 85.9 / 123.7 / 171.0)
- Cross-check **23–40×** vs GTLS's own published RTX-4090 numbers
- BLS survey end-to-end **2.0 / 2.2 / 12.7 / 3.0×** (recomputed 1.948 / 2.156
  / 12.683 / 3.021) and kernel-only **2.9–9.2×**
- TLS regime table (all 3 GPUs), fidelity table (0.97–0.99× / 1.01–1.03× SDE,
  100% recovery), cold single-shot table (2.6–34×), and the 34× kernel-cache
  loop claim (recomputed 34.4×)

## Fix-before-release list (applied in the Phase-2b content pass)

1. **CHANGELOG.rst 1.0.0 TLS entry carries a stale pre-campaign estimate**
   ("~22× matched to ~190× default vs GTLS"): written Jul 6 14:53 against
   GTLS's *published* 4090 numbers with an unarchived 1.48 s/LC input,
   *before* the same-GPU campaign that evening. Replace with the archived
   same-GPU 30–171× presentation (+ the 23–40× published-numbers cross-check).
2. **README.md:14 says "30-170x" while README.md:47 says "30–171x"** for the
   same experiment (data endpoint: 171.0). Unify.
3. **README.md:233 adaptive block sizing "~1.3×" is a cherry-pick** — the
   archived data (bls_adaptive_keplerian jun2026 JSON) gives median 1.08×,
   range 0.48–1.33× (slower on 2 of 12 configs). CHANGELOG's "~1.0–1.3×" is
   the honest phrasing; README must match.
4. **"6–15×" matched-fidelity cost is only half-archived and its floor is
   wrong**: archived points are 5.31× / 5.95× / 11.9×; the 14.6× (TESS-yr)
   and 8.4× / 176.8→1479 ms (Kepler) figures have no raw file
   (`scripts/tls_matched_timing.py` printed to stdout only). Also feeds the
   $111/M Kepler cost figure. → Re-run `tls_matched_timing.py` on the
   Phase-3 gate pod and archive; restate the range as "~5–15×" meanwhile.
5. **CHANGELOG "~1,000–3,000× vs reference TLS"** needs the CPU-reference
   caveat (two archived CPU timings for the same config differ 2.7×; claim
   survives the faster reference at ~1,066×, but say so).
6. **Provenance gap in `benchmarks/results/tls_survey_jul2026/`**:
   `tls_survey_a5000_final.json` (block-size-tuned; matches every published
   cell) vs the 3.7–9.4×-slower first run `tls_survey_a5000.json` (which
   uniquely holds the legacy-kernel and reference-CPU columns) is
   undocumented, as is the published V100 column mixing `v100.json` (3
   regimes) + `v100b.json` re-run (tess-yr/kepler). → Add a provenance
   README to the results directory.
7. **GTLS_COMPARISON.md nits**: SDE table prints "—" for GTLS at 1500 d but
   `results_gtls_skip8_big.json` HAS it (104.18, −0.6% — omission
   *understates* the evidence); "−0.4%" at 500 d is −0.5% and "−1%" at 200 d
   is −1.4% (still within the claimed 1–3%); "super-quadratic exponent
   2.5–2.7" is measured for gtls_full only (skip8, the headline comparator,
   scales ~1.9–2.2); "$24/M TESS-yr" recomputes to ~$20/M from the published
   inputs; "1.47 s" at qmin=2e-4 is 1.48 s archived and the "0.059 s at
   qmin=4e-3" point is unarchived (archived sensible config is qmin=2e-3 →
   0.032 s).
8. **TESS 12.7× disclosure**: roughly half that gain is the in-library fix
   for a default-environment BLAS/CFS-throttling pathology (5.8× vs a
   thread-pinned baseline). SUMMARY.md discloses this; add a parenthetical
   where README/CHANGELOG quote 12.7×.
9. Minor untraceables (non-headline; archive opportunistically): "2nd star
   onward ~0.5–2 s" (GTLS cold section), GTLS-own-grid count 191,837, chi2
   corr 0.998, medfilt/duration-grid micro-timings.

## Verdict key from the full trace (35 claims audited)

TRACES: 26 · TRACES-with-nits: 3 · PARTIAL (unarchived inputs): 4 ·
DISCREPANCY (wording/cherry-pick): 2 (items 1 and 3 above) ·
UNTRACEABLE-but-non-headline: several micro-claims (item 9).

The full 35-row trace table with per-claim doc:line ↔ data-file ↔ recomputed
value lives in the session record of this audit; the actionable subset is the
numbered list above.
