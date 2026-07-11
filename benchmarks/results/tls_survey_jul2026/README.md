# TLS survey benchmark archives — provenance (July 2026)

Which file is authoritative for which published number (added by the
pre-release claims audit, `analysis/claims-trace-jul2026.md`):

- **`tls_survey_a5000_final.json` is authoritative for the published A5000
  fast-path regime table** (tess-ffi 1.25 ms/LC … kepler-4yr 167.6 ms/LC).
  It was run at 18:06 after the compute-capability-aware block-size tuning
  (`_band_block_size`) landed; every published cell matches it exactly.
- **`tls_survey_a5000.json` (17:37) is the pre-tuning first run** — the same
  fast-path configs run 3.7–9.4× slower there. It is retained because it is
  the only file holding the **legacy-kernel and reference-CPU columns**
  (e.g. reference `transitleastsquares` 14.7 s/LC on tess-ffi with all pod
  cores). Do not quote its fast-path timings.
- **The published V100 column mixes two runs**: `tls_survey_v100.json`
  (tess-ffi / k2 / tess-2min) and the `tls_survey_v100b.json` re-run
  (tess-yr 16.5 ms, kepler-4yr 145.6 ms; the first run measured 17.0 /
  146.5 — a ~3% immaterial difference).
- `tls_survey_rtx4000ada.json`: single run, matches the published Ada column.
- `fidelity_raw_a5000.txt`: the SDE-parity experiment behind the
  0.97–0.99× / 1.01–1.03× fidelity claims and the archived matched-cost
  points (tess-ffi 5.3–6.0×, k2 11.9×). Note its reference-CPU timings are
  from a slow pod CPU (46–61 s/LC) — 2.7× slower than the reference column
  in `tls_survey_a5000.json` for the same config; speedup-vs-CPU claims are
  therefore quoted as "thousands×" with the multiplier marked CPU-dependent.
- `matched_timing_a5000_jul2026.txt`: the re-measured matched-fidelity
  costs from the v1.0.0 release-gate pod (TESS-yr 12.8×, 25.3→325.2 ms/LC;
  Kepler-4yr 8.1×, 188.3→1520.5 ms/LC; 100% recovery both fidelities). The
  original session printed 14.6×/8.4× (176.8→1479 ms) but was not archived;
  the published "~5–13×" range uses the archived measurements.
