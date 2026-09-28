# Survey recovery and implementation qualification

These tables format the existing sealed analysis. Rates remain separate by regime; interval bounds are copied from the source JSON. No new inferential statistics or pooled detection rates are calculated.

Native TLS versus the selected native GPU BLS measures blind detection at separately calibrated operating points. Baseline versus optimized TLS exactness is a separate comparison of the original held-out executions. Package SDE, BLS power and expected matched-filter SNR are not interchangeable.

Finite synthetic-flux population on fixed observed or synthetic cadences. Exact implementation qualification is separate. No universal completeness or sub-percentage noninferiority established. Marginal calibrated target FPR is not certainty about realized conditional FPR.

## Frozen BLS control

| Regime | Selected configuration | Ranker |
| --- | --- | --- |
| tess_solar | bls_strong | likelihood |
| tess_highimpact | bls_strong | detrended |
| tess_eccentric | bls_strong | likelihood |
| tess_mdwarf | bls_strong | detrended |
| ztf_solar | bls_medium | raw |
| ztf_highimpact | bls_strong | likelihood |
| ztf_mdwarf | bls_strong | likelihood |
| tess_gap_long | bls_fine | likelihood |
| tess_grazing_smeared | bls_medium | likelihood |
| hatpi_short | bls_strong | likelihood |

## Primary operating point: 5% target FPR

Recovery and observed FPR cells show successes/denominator, rate, and the existing 95% marginal interval. Failed executions remain in each planned denominator; a failure is not a detection.

| Regime | TLS recovery | BLS recovery | TLS observed FPR | BLS observed FPR |
| --- | --- | --- | --- | --- |
| tess_solar | 73/256 (28.52%; 23.07–34.47%) | 40/256 (15.62%; 11.40–20.66%) | 14/256 (5.47%; 3.02–9.01%) | 20/256 (7.81%; 4.84–11.81%) |
| tess_highimpact | 128/256 (50.00%; 43.71–56.29%) | 53/256 (20.70%; 15.91–26.19%) | 17/256 (6.64%; 3.92–10.42%) | 18/256 (7.03%; 4.22–10.88%) |
| tess_eccentric | 83/256 (32.42%; 26.73–38.53%) | 28/256 (10.94%; 7.39–15.42%) | 19/256 (7.42%; 4.53–11.35%) | 9/256 (3.52%; 1.62–6.57%) |
| tess_mdwarf | 154/256 (60.16%; 53.87–66.20%) | 60/256 (23.44%; 18.39–29.11%) | 10/256 (3.91%; 1.89–7.07%) | 15/256 (5.86%; 3.32–9.48%) |
| ztf_solar | 183/256 (71.48%; 65.53–76.93%) | 197/256 (76.95%; 71.30–81.97%) | 8/256 (3.12%; 1.36–6.06%) | 10/256 (3.91%; 1.89–7.07%) |
| ztf_highimpact | 147/256 (57.42%; 51.11–63.56%) | 159/256 (62.11%; 55.86–68.08%) | 16/256 (6.25%; 3.61–9.95%) | 13/256 (5.08%; 2.73–8.53%) |
| ztf_mdwarf | 103/256 (40.23%; 34.18–46.52%) | 124/256 (48.44%; 42.17–54.74%) | 15/256 (5.86%; 3.32–9.48%) | 8/256 (3.12%; 1.36–6.06%) |
| tess_gap_long | 40/256 (15.62%; 11.40–20.66%) | 59/256 (23.05%; 18.03–28.70%) | 12/256 (4.69%; 2.45–8.04%) | 13/256 (5.08%; 2.73–8.53%) |
| tess_grazing_smeared | 1/256 (0.39%; 0.01–2.16%) | 109/256 (42.58%; 36.44–48.89%) | 11/256 (4.30%; 2.16–7.56%) | 12/256 (4.69%; 2.45–8.04%) |
| hatpi_short | 3/256 (1.17%; 0.24–3.39%) | 0/256 (0.00%; 0.00–1.43%) | 10/256 (3.91%; 1.89–7.07%) | 4/256 (1.56%; 0.43–3.95%) |

Paired differences below are TLS minus BLS in percentage points. Both marginal and the existing simultaneous-family bounds are shown; an interval crossing zero does not establish an advantage. These intervals do not establish sub-percentage equivalence.

| Regime | Recovery: marginal | Recovery: simultaneous | FPR: marginal | FPR: simultaneous |
| --- | --- | --- | --- | --- |
| tess_solar | +12.89 [+5.30, +20.00] | +12.89 [+1.37, +23.50] | -2.34 [-8.36, +3.77] | -2.34 [-11.38, +6.87] |
| tess_highimpact | +29.30 [+21.37, +36.14] | +29.30 [+17.05, +39.77] | -0.39 [-7.28, +6.51] | -0.39 [-10.73, +9.98] |
| tess_eccentric | +21.48 [+13.55, +28.61] | +21.48 [+9.34, +32.15] | +3.91 [-2.85, +10.51] | +3.91 [-6.28, +13.80] |
| tess_mdwarf | +36.72 [+28.31, +43.82] | +36.72 [+23.67, +47.52] | -1.95 [-8.35, +4.52] | -1.95 [-11.55, +7.79] |
| ztf_solar | -5.47 [-10.81, +0.14] | -5.47 [-13.55, +3.03] | -0.78 [-5.79, +4.27] | -0.78 [-8.35, +6.85] |
| ztf_highimpact | -4.69 [-12.06, +2.85] | -4.69 [-15.71, +6.67] | +1.17 [-3.72, +6.00] | +1.17 [-6.22, +8.47] |
| ztf_mdwarf | -8.20 [-16.77, +0.64] | -8.20 [-20.98, +5.14] | +2.73 [-2.98, +8.32] | +2.73 [-5.89, +11.14] |
| tess_gap_long | -7.42 [-13.46, -1.06] | -7.42 [-16.50, +2.21] | -0.39 [-5.86, +5.09] | -0.39 [-8.63, +7.88] |
| tess_grazing_smeared | -42.19 [-49.37, -33.54] | -42.19 [-53.06, -28.72] | -0.39 [-6.62, +5.85] | -0.39 [-9.75, +9.00] |
| hatpi_short | +1.17 [-1.51, +3.75] | +1.17 [-3.05, +5.55] | +2.34 [-2.32, +6.86] | +2.34 [-4.72, +9.23] |

Independent calibration used 512 nulls per method and regime, with strict threshold exceedance. Stored attainable marginal FPR bound(s): 4.8733%. This discrete bound is marginal over calibration sets, not certainty about the conditional FPR of the realized threshold. Ties can make the operating point more conservative. All scores, ranks, exceedance counts, tie counts and zero-score counts are in [thresholds.csv](thresholds.csv).

| Regime | Method | Failed injections | Failed nulls | Ties at threshold | Extra tie conservatism |
| --- | --- | --- | --- | --- | --- |
| tess_solar | tls | 0 | 0 | 1 | False |
| tess_solar | bls | 0 | 0 | 1 | False |
| tess_highimpact | tls | 0 | 0 | 1 | False |
| tess_highimpact | bls | 0 | 0 | 1 | False |
| tess_eccentric | tls | 0 | 0 | 1 | False |
| tess_eccentric | bls | 0 | 0 | 1 | False |
| tess_mdwarf | tls | 0 | 0 | 1 | False |
| tess_mdwarf | bls | 0 | 0 | 1 | False |
| ztf_solar | tls | 0 | 0 | 1 | False |
| ztf_solar | bls | 0 | 0 | 1 | False |
| ztf_highimpact | tls | 0 | 0 | 1 | False |
| ztf_highimpact | bls | 0 | 0 | 1 | False |
| ztf_mdwarf | tls | 0 | 0 | 1 | False |
| ztf_mdwarf | bls | 0 | 0 | 1 | False |
| tess_gap_long | tls | 0 | 0 | 1 | False |
| tess_gap_long | bls | 0 | 0 | 1 | False |
| tess_grazing_smeared | tls | 0 | 0 | 1 | False |
| tess_grazing_smeared | bls | 0 | 0 | 1 | False |
| hatpi_short | tls | 0 | 0 | 1 | False |
| hatpi_short | bls | 0 | 0 | 1 | False |

### Primary target white-noise oracle SNR subgroups

These are preassigned latent target SNR levels. Unsampled signals can have realized SNR zero and remain in their assigned groups. The held-out diagnostic computes the realized centered signal norm; none of these quantities is a package-reported detection score.

| Regime | Level | TLS recovery | BLS recovery |
| --- | --- | --- | --- |
| tess_solar | 6.0 | 1/64 (1.56%; 0.04–8.40%) | 0/64 (0.00%; 0.00–5.60%) |
| tess_solar | 8.0 | 5/64 (7.81%; 2.59–17.30%) | 2/64 (3.12%; 0.38–10.84%) |
| tess_solar | 10.0 | 22/64 (34.38%; 22.95–47.30%) | 6/64 (9.38%; 3.52–19.30%) |
| tess_solar | 12.0 | 45/64 (70.31%; 57.58–81.09%) | 32/64 (50.00%; 37.23–62.77%) |
| tess_highimpact | 6.0 | 2/64 (3.12%; 0.38–10.84%) | 0/64 (0.00%; 0.00–5.60%) |
| tess_highimpact | 8.0 | 20/64 (31.25%; 20.24–44.06%) | 1/64 (1.56%; 0.04–8.40%) |
| tess_highimpact | 10.0 | 47/64 (73.44%; 60.91–83.70%) | 16/64 (25.00%; 15.02–37.40%) |
| tess_highimpact | 12.0 | 59/64 (92.19%; 82.70–97.41%) | 36/64 (56.25%; 43.28–68.63%) |
| tess_eccentric | 6.0 | 0/64 (0.00%; 0.00–5.60%) | 0/64 (0.00%; 0.00–5.60%) |
| tess_eccentric | 8.0 | 5/64 (7.81%; 2.59–17.30%) | 0/64 (0.00%; 0.00–5.60%) |
| tess_eccentric | 10.0 | 27/64 (42.19%; 29.94–55.18%) | 2/64 (3.12%; 0.38–10.84%) |
| tess_eccentric | 12.0 | 51/64 (79.69%; 67.77–88.72%) | 26/64 (40.62%; 28.51–53.63%) |
| tess_mdwarf | 6.0 | 6/64 (9.38%; 3.52–19.30%) | 0/64 (0.00%; 0.00–5.60%) |
| tess_mdwarf | 8.0 | 30/64 (46.88%; 34.28–59.77%) | 1/64 (1.56%; 0.04–8.40%) |
| tess_mdwarf | 10.0 | 55/64 (85.94%; 74.98–93.36%) | 21/64 (32.81%; 21.59–45.69%) |
| tess_mdwarf | 12.0 | 63/64 (98.44%; 91.60–99.96%) | 38/64 (59.38%; 46.37–71.49%) |
| ztf_solar | 6.0 | 12/64 (18.75%; 10.08–30.46%) | 19/64 (29.69%; 18.91–42.42%) |
| ztf_solar | 8.0 | 52/64 (81.25%; 69.54–89.92%) | 54/64 (84.38%; 73.14–92.24%) |
| ztf_solar | 10.0 | 58/64 (90.62%; 80.70–96.48%) | 62/64 (96.88%; 89.16–99.62%) |
| ztf_solar | 12.0 | 61/64 (95.31%; 86.91–99.02%) | 62/64 (96.88%; 89.16–99.62%) |
| ztf_highimpact | 6.0 | 5/64 (7.81%; 2.59–17.30%) | 4/64 (6.25%; 1.73–15.24%) |
| ztf_highimpact | 8.0 | 31/64 (48.44%; 35.75–61.27%) | 39/64 (60.94%; 47.93–72.90%) |
| ztf_highimpact | 10.0 | 52/64 (81.25%; 69.54–89.92%) | 59/64 (92.19%; 82.70–97.41%) |
| ztf_highimpact | 12.0 | 59/64 (92.19%; 82.70–97.41%) | 57/64 (89.06%; 78.75–95.49%) |
| ztf_mdwarf | 6.0 | 3/64 (4.69%; 0.98–13.09%) | 5/64 (7.81%; 2.59–17.30%) |
| ztf_mdwarf | 8.0 | 28/64 (43.75%; 31.37–56.72%) | 32/64 (50.00%; 37.23–62.77%) |
| ztf_mdwarf | 10.0 | 31/64 (48.44%; 35.75–61.27%) | 40/64 (62.50%; 49.51–74.30%) |
| ztf_mdwarf | 12.0 | 41/64 (64.06%; 51.10–75.68%) | 47/64 (73.44%; 60.91–83.70%) |
| tess_gap_long | 6.0 | 0/64 (0.00%; 0.00–5.60%) | 0/64 (0.00%; 0.00–5.60%) |
| tess_gap_long | 8.0 | 4/64 (6.25%; 1.73–15.24%) | 8/64 (12.50%; 5.55–23.15%) |
| tess_gap_long | 10.0 | 10/64 (15.62%; 7.76–26.86%) | 20/64 (31.25%; 20.24–44.06%) |
| tess_gap_long | 12.0 | 26/64 (40.62%; 28.51–53.63%) | 31/64 (48.44%; 35.75–61.27%) |
| tess_grazing_smeared | 6.0 | 0/64 (0.00%; 0.00–5.60%) | 11/64 (17.19%; 8.90–28.68%) |
| tess_grazing_smeared | 8.0 | 0/64 (0.00%; 0.00–5.60%) | 29/64 (45.31%; 32.82–58.25%) |
| tess_grazing_smeared | 10.0 | 1/64 (1.56%; 0.04–8.40%) | 33/64 (51.56%; 38.73–64.25%) |
| tess_grazing_smeared | 12.0 | 0/64 (0.00%; 0.00–5.60%) | 36/64 (56.25%; 43.28–68.63%) |
| hatpi_short | 6.0 | 0/64 (0.00%; 0.00–5.60%) | 0/64 (0.00%; 0.00–5.60%) |
| hatpi_short | 8.0 | 0/64 (0.00%; 0.00–5.60%) | 0/64 (0.00%; 0.00–5.60%) |
| hatpi_short | 10.0 | 0/64 (0.00%; 0.00–5.60%) | 0/64 (0.00%; 0.00–5.60%) |
| hatpi_short | 12.0 | 3/64 (4.69%; 0.98–13.09%) | 0/64 (0.00%; 0.00–5.60%) |

### Primary sampling subgroups

Sampling groups overlap; their counts must not be added. An unrepresented group has no estimated recovery interval.

| Regime | Level | TLS recovery | BLS recovery |
| --- | --- | --- | --- |
| tess_solar | unsampled | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_solar | one_event | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_solar | two_events | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_solar | three_plus_events | 73/256 (28.52%; 23.07–34.47%) | 40/256 (15.62%; 11.40–20.66%) |
| tess_solar | one_to_four_points | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_solar | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_highimpact | unsampled | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_highimpact | one_event | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_highimpact | two_events | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_highimpact | three_plus_events | 128/256 (50.00%; 43.71–56.29%) | 53/256 (20.70%; 15.91–26.19%) |
| tess_highimpact | one_to_four_points | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_highimpact | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_eccentric | unsampled | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_eccentric | one_event | 0/3 (0.00%; 0.00–70.76%) | 0/3 (0.00%; 0.00–70.76%) |
| tess_eccentric | two_events | 36/117 (30.77%; 22.57–39.97%) | 15/117 (12.82%; 7.36–20.26%) |
| tess_eccentric | three_plus_events | 47/136 (34.56%; 26.62–43.19%) | 13/136 (9.56%; 5.19–15.79%) |
| tess_eccentric | one_to_four_points | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_eccentric | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_mdwarf | unsampled | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_mdwarf | one_event | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_mdwarf | two_events | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_mdwarf | three_plus_events | 154/256 (60.16%; 53.87–66.20%) | 60/256 (23.44%; 18.39–29.11%) |
| tess_mdwarf | one_to_four_points | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_mdwarf | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_solar | unsampled | 0/2 (0.00%; 0.00–84.19%) | 0/2 (0.00%; 0.00–84.19%) |
| ztf_solar | one_event | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_solar | two_events | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_solar | three_plus_events | 183/254 (72.05%; 66.10–77.48%) | 197/254 (77.56%; 71.92–82.54%) |
| ztf_solar | one_to_four_points | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_solar | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_highimpact | unsampled | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_highimpact | one_event | 0/1 (0.00%; 0.00–97.50%) | 0/1 (0.00%; 0.00–97.50%) |
| ztf_highimpact | two_events | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_highimpact | three_plus_events | 147/255 (57.65%; 51.33–63.79%) | 159/255 (62.35%; 56.09–68.32%) |
| ztf_highimpact | one_to_four_points | 0/2 (0.00%; 0.00–84.19%) | 0/2 (0.00%; 0.00–84.19%) |
| ztf_highimpact | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_mdwarf | unsampled | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_mdwarf | one_event | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_mdwarf | two_events | 0/3 (0.00%; 0.00–70.76%) | 0/3 (0.00%; 0.00–70.76%) |
| ztf_mdwarf | three_plus_events | 103/253 (40.71%; 34.60–47.04%) | 124/253 (49.01%; 42.70–55.35%) |
| ztf_mdwarf | one_to_four_points | 0/18 (0.00%; 0.00–18.53%) | 2/18 (11.11%; 1.38–34.71%) |
| ztf_mdwarf | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_gap_long | unsampled | 0/1 (0.00%; 0.00–97.50%) | 0/1 (0.00%; 0.00–97.50%) |
| tess_gap_long | one_event | 0/28 (0.00%; 0.00–12.34%) | 0/28 (0.00%; 0.00–12.34%) |
| tess_gap_long | two_events | 5/125 (4.00%; 1.31–9.09%) | 9/125 (7.20%; 3.35–13.23%) |
| tess_gap_long | three_plus_events | 35/102 (34.31%; 25.19–44.37%) | 50/102 (49.02%; 38.99–59.11%) |
| tess_gap_long | one_to_four_points | 0/1 (0.00%; 0.00–97.50%) | 0/1 (0.00%; 0.00–97.50%) |
| tess_gap_long | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_grazing_smeared | unsampled | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_grazing_smeared | one_event | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_grazing_smeared | two_events | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_grazing_smeared | three_plus_events | 1/256 (0.39%; 0.01–2.16%) | 109/256 (42.58%; 36.44–48.89%) |
| tess_grazing_smeared | one_to_four_points | 0/1 (0.00%; 0.00–97.50%) | 0/1 (0.00%; 0.00–97.50%) |
| tess_grazing_smeared | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| hatpi_short | unsampled | 0/10 (0.00%; 0.00–30.85%) | 0/10 (0.00%; 0.00–30.85%) |
| hatpi_short | one_event | 0/2 (0.00%; 0.00–84.19%) | 0/2 (0.00%; 0.00–84.19%) |
| hatpi_short | two_events | 0/3 (0.00%; 0.00–70.76%) | 0/3 (0.00%; 0.00–70.76%) |
| hatpi_short | three_plus_events | 3/241 (1.24%; 0.26–3.59%) | 0/241 (0.00%; 0.00–1.52%) |
| hatpi_short | one_to_four_points | 0/0 — unrepresented | 0/0 — unrepresented |
| hatpi_short | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |

## Secondary operating point: 1% target FPR

Recovery and observed FPR cells show successes/denominator, rate, and the existing 95% marginal interval. Failed executions remain in each planned denominator; a failure is not a detection.

| Regime | TLS recovery | BLS recovery | TLS observed FPR | BLS observed FPR |
| --- | --- | --- | --- | --- |
| tess_solar | 53/256 (20.70%; 15.91–26.19%) | 19/256 (7.42%; 4.53–11.35%) | 7/256 (2.73%; 1.11–5.55%) | 4/256 (1.56%; 0.43–3.95%) |
| tess_highimpact | 112/256 (43.75%; 37.58–50.06%) | 33/256 (12.89%; 9.04–17.62%) | 6/256 (2.34%; 0.86–5.03%) | 7/256 (2.73%; 1.11–5.55%) |
| tess_eccentric | 73/256 (28.52%; 23.07–34.47%) | 6/256 (2.34%; 0.86–5.03%) | 6/256 (2.34%; 0.86–5.03%) | 1/256 (0.39%; 0.01–2.16%) |
| tess_mdwarf | 144/256 (56.25%; 49.94–62.42%) | 37/256 (14.45%; 10.38–19.37%) | 1/256 (0.39%; 0.01–2.16%) | 3/256 (1.17%; 0.24–3.39%) |
| ztf_solar | 176/256 (68.75%; 62.68–74.38%) | 192/256 (75.00%; 69.23–80.18%) | 0/256 (0.00%; 0.00–1.43%) | 1/256 (0.39%; 0.01–2.16%) |
| ztf_highimpact | 131/256 (51.17%; 44.87–57.45%) | 156/256 (60.94%; 54.67–66.95%) | 1/256 (0.39%; 0.01–2.16%) | 2/256 (0.78%; 0.09–2.79%) |
| ztf_mdwarf | 98/256 (38.28%; 32.30–44.54%) | 122/256 (47.66%; 41.40–53.97%) | 5/256 (1.95%; 0.64–4.50%) | 1/256 (0.39%; 0.01–2.16%) |
| tess_gap_long | 23/256 (8.98%; 5.78–13.18%) | 47/256 (18.36%; 13.81–23.66%) | 2/256 (0.78%; 0.09–2.79%) | 0/256 (0.00%; 0.00–1.43%) |
| tess_grazing_smeared | 0/256 (0.00%; 0.00–1.43%) | 97/256 (37.89%; 31.92–44.14%) | 1/256 (0.39%; 0.01–2.16%) | 0/256 (0.00%; 0.00–1.43%) |
| hatpi_short | 0/256 (0.00%; 0.00–1.43%) | 0/256 (0.00%; 0.00–1.43%) | 5/256 (1.95%; 0.64–4.50%) | 1/256 (0.39%; 0.01–2.16%) |

Paired differences below are TLS minus BLS in percentage points. Both marginal and the existing simultaneous-family bounds are shown; an interval crossing zero does not establish an advantage. These intervals do not establish sub-percentage equivalence.

| Regime | Recovery: marginal | Recovery: simultaneous | FPR: marginal | FPR: simultaneous |
| --- | --- | --- | --- | --- |
| tess_solar | +13.28 [+5.63, +20.44] | +13.28 [+1.67, +23.95] | +1.17 [-3.02, +5.28] | +1.17 [-5.20, +7.45] |
| tess_highimpact | +30.86 [+22.82, +37.77] | +30.86 [+18.42, +41.43] | -0.39 [-5.26, +4.50] | -0.39 [-7.76, +7.01] |
| tess_eccentric | +26.17 [+17.78, +33.61] | +26.17 [+13.30, +37.26] | +1.95 [-1.73, +5.46] | +1.95 [-3.68, +7.50] |
| tess_mdwarf | +41.80 [+33.16, +48.98] | +41.80 [+28.35, +52.67] | -0.78 [-3.75, +2.28] | -0.78 [-5.54, +3.98] |
| ztf_solar | -6.25 [-11.41, -0.77] | -6.25 [-14.10, +2.07] | -0.39 [-2.47, +1.69] | -0.39 [-4.03, +3.10] |
| ztf_highimpact | -9.77 [-17.13, -2.04] | -9.77 [-20.76, +1.92] | -0.39 [-2.47, +1.69] | -0.39 [-4.03, +3.10] |
| ztf_mdwarf | -9.38 [-18.08, -0.36] | -9.38 [-22.35, +4.24] | +1.56 [-1.94, +4.91] | +1.56 [-3.81, +6.87] |
| tess_gap_long | -9.38 [-15.45, -2.90] | -9.38 [-18.52, +0.46] | +0.78 [-1.63, +3.14] | +0.78 [-3.09, +4.82] |
| tess_grazing_smeared | -37.89 [-45.02, -29.42] | -37.89 [-48.72, -24.74] | +0.39 [-1.69, +2.47] | +0.39 [-3.10, +4.03] |
| hatpi_short | +0.00 [-1.70, +1.70] | +0.00 [-3.10, +3.10] | +1.56 [-1.94, +4.91] | +1.56 [-3.81, +6.87] |

Independent calibration used 512 nulls per method and regime, with strict threshold exceedance. Stored attainable marginal FPR bound(s): 0.9747%. This discrete bound is marginal over calibration sets, not certainty about the conditional FPR of the realized threshold. Ties can make the operating point more conservative. All scores, ranks, exceedance counts, tie counts and zero-score counts are in [thresholds.csv](thresholds.csv).

| Regime | Method | Failed injections | Failed nulls | Ties at threshold | Extra tie conservatism |
| --- | --- | --- | --- | --- | --- |
| tess_solar | tls | 0 | 0 | 1 | False |
| tess_solar | bls | 0 | 0 | 1 | False |
| tess_highimpact | tls | 0 | 0 | 1 | False |
| tess_highimpact | bls | 0 | 0 | 1 | False |
| tess_eccentric | tls | 0 | 0 | 1 | False |
| tess_eccentric | bls | 0 | 0 | 1 | False |
| tess_mdwarf | tls | 0 | 0 | 1 | False |
| tess_mdwarf | bls | 0 | 0 | 1 | False |
| ztf_solar | tls | 0 | 0 | 1 | False |
| ztf_solar | bls | 0 | 0 | 1 | False |
| ztf_highimpact | tls | 0 | 0 | 1 | False |
| ztf_highimpact | bls | 0 | 0 | 1 | False |
| ztf_mdwarf | tls | 0 | 0 | 1 | False |
| ztf_mdwarf | bls | 0 | 0 | 1 | False |
| tess_gap_long | tls | 0 | 0 | 1 | False |
| tess_gap_long | bls | 0 | 0 | 1 | False |
| tess_grazing_smeared | tls | 0 | 0 | 1 | False |
| tess_grazing_smeared | bls | 0 | 0 | 1 | False |
| hatpi_short | tls | 0 | 0 | 1 | False |
| hatpi_short | bls | 0 | 0 | 1 | False |

### Secondary target white-noise oracle SNR subgroups

These are preassigned latent target SNR levels. Unsampled signals can have realized SNR zero and remain in their assigned groups. The held-out diagnostic computes the realized centered signal norm; none of these quantities is a package-reported detection score.

| Regime | Level | TLS recovery | BLS recovery |
| --- | --- | --- | --- |
| tess_solar | 6.0 | 0/64 (0.00%; 0.00–5.60%) | 0/64 (0.00%; 0.00–5.60%) |
| tess_solar | 8.0 | 3/64 (4.69%; 0.98–13.09%) | 1/64 (1.56%; 0.04–8.40%) |
| tess_solar | 10.0 | 12/64 (18.75%; 10.08–30.46%) | 1/64 (1.56%; 0.04–8.40%) |
| tess_solar | 12.0 | 38/64 (59.38%; 46.37–71.49%) | 17/64 (26.56%; 16.30–39.09%) |
| tess_highimpact | 6.0 | 1/64 (1.56%; 0.04–8.40%) | 0/64 (0.00%; 0.00–5.60%) |
| tess_highimpact | 8.0 | 12/64 (18.75%; 10.08–30.46%) | 0/64 (0.00%; 0.00–5.60%) |
| tess_highimpact | 10.0 | 41/64 (64.06%; 51.10–75.68%) | 6/64 (9.38%; 3.52–19.30%) |
| tess_highimpact | 12.0 | 58/64 (90.62%; 80.70–96.48%) | 27/64 (42.19%; 29.94–55.18%) |
| tess_eccentric | 6.0 | 0/64 (0.00%; 0.00–5.60%) | 0/64 (0.00%; 0.00–5.60%) |
| tess_eccentric | 8.0 | 4/64 (6.25%; 1.73–15.24%) | 0/64 (0.00%; 0.00–5.60%) |
| tess_eccentric | 10.0 | 22/64 (34.38%; 22.95–47.30%) | 0/64 (0.00%; 0.00–5.60%) |
| tess_eccentric | 12.0 | 47/64 (73.44%; 60.91–83.70%) | 6/64 (9.38%; 3.52–19.30%) |
| tess_mdwarf | 6.0 | 4/64 (6.25%; 1.73–15.24%) | 0/64 (0.00%; 0.00–5.60%) |
| tess_mdwarf | 8.0 | 23/64 (35.94%; 24.32–48.90%) | 0/64 (0.00%; 0.00–5.60%) |
| tess_mdwarf | 10.0 | 54/64 (84.38%; 73.14–92.24%) | 7/64 (10.94%; 4.51–21.25%) |
| tess_mdwarf | 12.0 | 63/64 (98.44%; 91.60–99.96%) | 30/64 (46.88%; 34.28–59.77%) |
| ztf_solar | 6.0 | 9/64 (14.06%; 6.64–25.02%) | 14/64 (21.88%; 12.51–33.97%) |
| ztf_solar | 8.0 | 48/64 (75.00%; 62.60–84.98%) | 54/64 (84.38%; 73.14–92.24%) |
| ztf_solar | 10.0 | 58/64 (90.62%; 80.70–96.48%) | 62/64 (96.88%; 89.16–99.62%) |
| ztf_solar | 12.0 | 61/64 (95.31%; 86.91–99.02%) | 62/64 (96.88%; 89.16–99.62%) |
| ztf_highimpact | 6.0 | 2/64 (3.12%; 0.38–10.84%) | 3/64 (4.69%; 0.98–13.09%) |
| ztf_highimpact | 8.0 | 23/64 (35.94%; 24.32–48.90%) | 37/64 (57.81%; 44.82–70.06%) |
| ztf_highimpact | 10.0 | 48/64 (75.00%; 62.60–84.98%) | 59/64 (92.19%; 82.70–97.41%) |
| ztf_highimpact | 12.0 | 58/64 (90.62%; 80.70–96.48%) | 57/64 (89.06%; 78.75–95.49%) |
| ztf_mdwarf | 6.0 | 3/64 (4.69%; 0.98–13.09%) | 4/64 (6.25%; 1.73–15.24%) |
| ztf_mdwarf | 8.0 | 25/64 (39.06%; 27.10–52.07%) | 31/64 (48.44%; 35.75–61.27%) |
| ztf_mdwarf | 10.0 | 31/64 (48.44%; 35.75–61.27%) | 40/64 (62.50%; 49.51–74.30%) |
| ztf_mdwarf | 12.0 | 39/64 (60.94%; 47.93–72.90%) | 47/64 (73.44%; 60.91–83.70%) |
| tess_gap_long | 6.0 | 0/64 (0.00%; 0.00–5.60%) | 0/64 (0.00%; 0.00–5.60%) |
| tess_gap_long | 8.0 | 2/64 (3.12%; 0.38–10.84%) | 5/64 (7.81%; 2.59–17.30%) |
| tess_gap_long | 10.0 | 5/64 (7.81%; 2.59–17.30%) | 12/64 (18.75%; 10.08–30.46%) |
| tess_gap_long | 12.0 | 16/64 (25.00%; 15.02–37.40%) | 30/64 (46.88%; 34.28–59.77%) |
| tess_grazing_smeared | 6.0 | 0/64 (0.00%; 0.00–5.60%) | 8/64 (12.50%; 5.55–23.15%) |
| tess_grazing_smeared | 8.0 | 0/64 (0.00%; 0.00–5.60%) | 22/64 (34.38%; 22.95–47.30%) |
| tess_grazing_smeared | 10.0 | 0/64 (0.00%; 0.00–5.60%) | 31/64 (48.44%; 35.75–61.27%) |
| tess_grazing_smeared | 12.0 | 0/64 (0.00%; 0.00–5.60%) | 36/64 (56.25%; 43.28–68.63%) |
| hatpi_short | 6.0 | 0/64 (0.00%; 0.00–5.60%) | 0/64 (0.00%; 0.00–5.60%) |
| hatpi_short | 8.0 | 0/64 (0.00%; 0.00–5.60%) | 0/64 (0.00%; 0.00–5.60%) |
| hatpi_short | 10.0 | 0/64 (0.00%; 0.00–5.60%) | 0/64 (0.00%; 0.00–5.60%) |
| hatpi_short | 12.0 | 0/64 (0.00%; 0.00–5.60%) | 0/64 (0.00%; 0.00–5.60%) |

### Secondary sampling subgroups

Sampling groups overlap; their counts must not be added. An unrepresented group has no estimated recovery interval.

| Regime | Level | TLS recovery | BLS recovery |
| --- | --- | --- | --- |
| tess_solar | unsampled | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_solar | one_event | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_solar | two_events | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_solar | three_plus_events | 53/256 (20.70%; 15.91–26.19%) | 19/256 (7.42%; 4.53–11.35%) |
| tess_solar | one_to_four_points | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_solar | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_highimpact | unsampled | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_highimpact | one_event | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_highimpact | two_events | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_highimpact | three_plus_events | 112/256 (43.75%; 37.58–50.06%) | 33/256 (12.89%; 9.04–17.62%) |
| tess_highimpact | one_to_four_points | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_highimpact | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_eccentric | unsampled | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_eccentric | one_event | 0/3 (0.00%; 0.00–70.76%) | 0/3 (0.00%; 0.00–70.76%) |
| tess_eccentric | two_events | 30/117 (25.64%; 18.02–34.54%) | 3/117 (2.56%; 0.53–7.31%) |
| tess_eccentric | three_plus_events | 43/136 (31.62%; 23.92–40.14%) | 3/136 (2.21%; 0.46–6.31%) |
| tess_eccentric | one_to_four_points | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_eccentric | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_mdwarf | unsampled | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_mdwarf | one_event | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_mdwarf | two_events | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_mdwarf | three_plus_events | 144/256 (56.25%; 49.94–62.42%) | 37/256 (14.45%; 10.38–19.37%) |
| tess_mdwarf | one_to_four_points | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_mdwarf | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_solar | unsampled | 0/2 (0.00%; 0.00–84.19%) | 0/2 (0.00%; 0.00–84.19%) |
| ztf_solar | one_event | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_solar | two_events | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_solar | three_plus_events | 176/254 (69.29%; 63.22–74.91%) | 192/254 (75.59%; 69.83–80.74%) |
| ztf_solar | one_to_four_points | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_solar | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_highimpact | unsampled | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_highimpact | one_event | 0/1 (0.00%; 0.00–97.50%) | 0/1 (0.00%; 0.00–97.50%) |
| ztf_highimpact | two_events | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_highimpact | three_plus_events | 131/255 (51.37%; 45.06–57.66%) | 156/255 (61.18%; 54.90–67.19%) |
| ztf_highimpact | one_to_four_points | 0/2 (0.00%; 0.00–84.19%) | 0/2 (0.00%; 0.00–84.19%) |
| ztf_highimpact | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_mdwarf | unsampled | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_mdwarf | one_event | 0/0 — unrepresented | 0/0 — unrepresented |
| ztf_mdwarf | two_events | 0/3 (0.00%; 0.00–70.76%) | 0/3 (0.00%; 0.00–70.76%) |
| ztf_mdwarf | three_plus_events | 98/253 (38.74%; 32.70–45.04%) | 122/253 (48.22%; 41.92–54.57%) |
| ztf_mdwarf | one_to_four_points | 0/18 (0.00%; 0.00–18.53%) | 2/18 (11.11%; 1.38–34.71%) |
| ztf_mdwarf | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_gap_long | unsampled | 0/1 (0.00%; 0.00–97.50%) | 0/1 (0.00%; 0.00–97.50%) |
| tess_gap_long | one_event | 0/28 (0.00%; 0.00–12.34%) | 0/28 (0.00%; 0.00–12.34%) |
| tess_gap_long | two_events | 3/125 (2.40%; 0.50–6.85%) | 6/125 (4.80%; 1.78–10.15%) |
| tess_gap_long | three_plus_events | 20/102 (19.61%; 12.41–28.65%) | 41/102 (40.20%; 30.61–50.37%) |
| tess_gap_long | one_to_four_points | 0/1 (0.00%; 0.00–97.50%) | 0/1 (0.00%; 0.00–97.50%) |
| tess_gap_long | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_grazing_smeared | unsampled | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_grazing_smeared | one_event | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_grazing_smeared | two_events | 0/0 — unrepresented | 0/0 — unrepresented |
| tess_grazing_smeared | three_plus_events | 0/256 (0.00%; 0.00–1.43%) | 97/256 (37.89%; 31.92–44.14%) |
| tess_grazing_smeared | one_to_four_points | 0/1 (0.00%; 0.00–97.50%) | 0/1 (0.00%; 0.00–97.50%) |
| tess_grazing_smeared | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |
| hatpi_short | unsampled | 0/10 (0.00%; 0.00–30.85%) | 0/10 (0.00%; 0.00–30.85%) |
| hatpi_short | one_event | 0/2 (0.00%; 0.00–84.19%) | 0/2 (0.00%; 0.00–84.19%) |
| hatpi_short | two_events | 0/3 (0.00%; 0.00–70.76%) | 0/3 (0.00%; 0.00–70.76%) |
| hatpi_short | three_plus_events | 0/241 (0.00%; 0.00–1.52%) | 0/241 (0.00%; 0.00–1.52%) |
| hatpi_short | one_to_four_points | 0/0 — unrepresented | 0/0 — unrepresented |
| hatpi_short | grid_unreachable | 0/0 — unrepresented | 0/0 — unrepresented |

## Comparable expected-SNR diagnostics

The native family and ideal box are evaluated at the known period with the same sampled signal, weights, and fitted constant. Templates are selected by the white diagonal-error matched-filter objective; their white responses are the enumerated family ceilings. OU values evaluate those same white-selected filters using the actual OU covariance variance, not an independently OU-optimized family maximum. The white native-family optimum is an optimistic ceiling: the actual blind search and native depth/ranking need not attain it. These are descriptive diagnostics, not package SNR/SDE values or a measured blind-search advantage.

Cells show the observed median relative native-family/ideal-box advantage and observed minimum–maximum, in percent; these ranges are not confidence intervals. Finite/total counts expose undefined ratios, including zero-signal cases. Detected/missed groups use original TLS decisions; misses include invalid executions and do not isolate a causal effect. No new tests, approximation allowances, or inferential intervals are calculated.

All held-out injections.

| Regime | Group | White-noise family/box advantage | OU-noise family/box advantage |
| --- | --- | --- | --- |
| tess_solar | all | 256/256 finite; +0.951% [+0.499, +1.408] | 256/256 finite; +0.927% [-1.411, +1.894] |
| tess_highimpact | all | 256/256 finite; +0.978% [+0.304, +1.517] | 256/256 finite; +0.471% [-1.552, +2.182] |
| tess_eccentric | all | 256/256 finite; +0.903% [-18.531, +1.592] | 256/256 finite; +0.663% [-29.780, +2.260] |
| tess_mdwarf | all | 256/256 finite; +1.359% [-2.053, +2.404] | 256/256 finite; +0.890% [-5.138, +2.856] |
| ztf_solar | all | 254/256 finite; +0.589% [-0.851, +2.025] | 254/256 finite; +0.580% [-0.737, +2.767] |
| ztf_highimpact | all | 256/256 finite; +0.166% [-51.353, +2.381] | 256/256 finite; +0.179% [-51.518, +2.432] |
| ztf_mdwarf | all | 256/256 finite; +0.348% [-37.484, +3.317] | 256/256 finite; +0.308% [-37.868, +3.790] |
| tess_gap_long | all | 255/256 finite; +0.953% [-44.885, +2.698] | 255/256 finite; +0.445% [-50.768, +2.412] |
| tess_grazing_smeared | all | 256/256 finite; +0.936% [-3.714, +3.582] | 256/256 finite; +0.708% [-4.038, +4.190] |
| hatpi_short | all | 246/256 finite; +0.904% [-18.204, +1.564] | 246/256 finite; +0.333% [-21.526, +3.539] |

Original TLS decisions at 5% target FPR.

| Regime | Group | White-noise family/box advantage | OU-noise family/box advantage |
| --- | --- | --- | --- |
| tess_solar | tls_detected | 73/73 finite; +0.953% [+0.554, +1.279] | 73/73 finite; +0.910% [-1.102, +1.810] |
| tess_solar | tls_missed_including_failures | 183/183 finite; +0.951% [+0.499, +1.408] | 183/183 finite; +0.938% [-1.411, +1.894] |
| tess_highimpact | tls_detected | 128/128 finite; +0.997% [+0.304, +1.517] | 128/128 finite; +0.507% [-1.552, +1.878] |
| tess_highimpact | tls_missed_including_failures | 128/128 finite; +0.948% [+0.319, +1.433] | 128/128 finite; +0.471% [-1.249, +2.182] |
| tess_eccentric | tls_detected | 83/83 finite; +0.929% [-3.120, +1.477] | 83/83 finite; +0.753% [-6.395, +1.940] |
| tess_eccentric | tls_missed_including_failures | 173/173 finite; +0.885% [-18.531, +1.592] | 173/173 finite; +0.490% [-29.780, +2.260] |
| tess_mdwarf | tls_detected | 154/154 finite; +1.359% [-2.053, +2.339] | 154/154 finite; +0.855% [-5.138, +2.638] |
| tess_mdwarf | tls_missed_including_failures | 102/102 finite; +1.394% [-0.513, +2.404] | 102/102 finite; +0.909% [-4.560, +2.856] |
| ztf_solar | tls_detected | 183/183 finite; +0.613% [-0.851, +1.919] | 183/183 finite; +0.596% [-0.611, +2.767] |
| ztf_solar | tls_missed_including_failures | 71/73 finite; +0.476% [-0.530, +2.025] | 71/73 finite; +0.440% [-0.737, +2.178] |
| ztf_highimpact | tls_detected | 147/147 finite; +0.101% [-1.439, +2.059] | 147/147 finite; +0.160% [-1.520, +2.282] |
| ztf_highimpact | tls_missed_including_failures | 109/109 finite; +0.211% [-51.353, +2.381] | 109/109 finite; +0.206% [-51.518, +2.432] |
| ztf_mdwarf | tls_detected | 103/103 finite; +0.461% [-6.636, +2.862] | 103/103 finite; +0.516% [-7.971, +3.192] |
| ztf_mdwarf | tls_missed_including_failures | 153/153 finite; +0.263% [-37.484, +3.317] | 153/153 finite; +0.219% [-37.868, +3.790] |
| tess_gap_long | tls_detected | 40/40 finite; +0.967% [+0.572, +1.907] | 40/40 finite; +0.562% [-1.840, +2.412] |
| tess_gap_long | tls_missed_including_failures | 215/216 finite; +0.953% [-44.885, +2.698] | 215/216 finite; +0.436% [-50.768, +2.367] |
| tess_grazing_smeared | tls_detected | 1/1 finite; +0.933% [+0.933, +0.933] | 1/1 finite; +0.631% [+0.631, +0.631] |
| tess_grazing_smeared | tls_missed_including_failures | 255/255 finite; +0.940% [-3.714, +3.582] | 255/255 finite; +0.708% [-4.038, +4.190] |
| hatpi_short | tls_detected | 3/3 finite; +0.917% [+0.824, +1.210] | 3/3 finite; +0.060% [-0.002, +1.708] |
| hatpi_short | tls_missed_including_failures | 243/253 finite; +0.903% [-18.204, +1.564] | 243/253 finite; +0.338% [-21.526, +3.539] |

Original TLS decisions at 1% target FPR.

| Regime | Group | White-noise family/box advantage | OU-noise family/box advantage |
| --- | --- | --- | --- |
| tess_solar | tls_detected | 53/53 finite; +0.953% [+0.554, +1.279] | 53/53 finite; +0.958% [-0.961, +1.810] |
| tess_solar | tls_missed_including_failures | 203/203 finite; +0.951% [+0.499, +1.408] | 203/203 finite; +0.922% [-1.411, +1.894] |
| tess_highimpact | tls_detected | 112/112 finite; +0.992% [+0.304, +1.517] | 112/112 finite; +0.507% [-1.552, +1.878] |
| tess_highimpact | tls_missed_including_failures | 144/144 finite; +0.957% [+0.319, +1.433] | 144/144 finite; +0.471% [-1.249, +2.182] |
| tess_eccentric | tls_detected | 73/73 finite; +0.938% [-3.120, +1.477] | 73/73 finite; +0.853% [-6.395, +1.940] |
| tess_eccentric | tls_missed_including_failures | 183/183 finite; +0.877% [-18.531, +1.592] | 183/183 finite; +0.468% [-29.780, +2.260] |
| tess_mdwarf | tls_detected | 144/144 finite; +1.359% [-2.053, +2.339] | 144/144 finite; +0.904% [-5.138, +2.638] |
| tess_mdwarf | tls_missed_including_failures | 112/112 finite; +1.386% [-0.513, +2.404] | 112/112 finite; +0.875% [-4.560, +2.856] |
| ztf_solar | tls_detected | 176/176 finite; +0.621% [-0.851, +1.919] | 176/176 finite; +0.600% [-0.611, +2.767] |
| ztf_solar | tls_missed_including_failures | 78/80 finite; +0.471% [-0.530, +2.025] | 78/80 finite; +0.450% [-0.737, +2.178] |
| ztf_highimpact | tls_detected | 131/131 finite; +0.101% [-1.439, +1.668] | 131/131 finite; +0.149% [-1.520, +1.981] |
| ztf_highimpact | tls_missed_including_failures | 125/125 finite; +0.209% [-51.353, +2.381] | 125/125 finite; +0.224% [-51.518, +2.432] |
| ztf_mdwarf | tls_detected | 98/98 finite; +0.446% [-6.636, +2.862] | 98/98 finite; +0.503% [-7.971, +3.192] |
| ztf_mdwarf | tls_missed_including_failures | 158/158 finite; +0.269% [-37.484, +3.317] | 158/158 finite; +0.233% [-37.868, +3.790] |
| tess_gap_long | tls_detected | 23/23 finite; +0.877% [+0.572, +1.907] | 23/23 finite; +0.561% [-0.730, +2.412] |
| tess_gap_long | tls_missed_including_failures | 232/233 finite; +0.957% [-44.885, +2.698] | 232/233 finite; +0.441% [-50.768, +2.367] |
| tess_grazing_smeared | tls_detected | 0/0 finite — unavailable | 0/0 finite — unavailable |
| tess_grazing_smeared | tls_missed_including_failures | 256/256 finite; +0.936% [-3.714, +3.582] | 256/256 finite; +0.708% [-4.038, +4.190] |
| hatpi_short | tls_detected | 0/0 finite — unavailable | 0/0 finite — unavailable |
| hatpi_short | tls_missed_including_failures | 246/256 finite; +0.904% [-18.204, +1.564] | 246/256 finite; +0.333% [-21.526, +3.539] |

Full native/box SNR distributions are in [snr_descriptive.csv](../../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/final-report/snr_descriptive.csv"); the measured case values and original decision join are in [snr_cases.csv](../../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/final-report/snr_cases.csv").

## Baseline versus optimized TLS: finite implementation qualification

**Aggregate exactness is withheld. Original mismatches or unavailable valid executions remain failures, regardless of diagnostic repeats.**

This checks the full available period/chi-squared/mask hashes, selected period/SDE, recovery and both frozen-threshold decisions. It does not establish universal numerical or physical equivalence. BLS is absent from this comparison.

| Regime | Split | Planned | Compared | Exact | Mismatches | Candidate invalid | Baseline invalid | Pending repeats |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tess_solar | injections | 256 | 256 | 256 | 0 | 0 | 0 | 0 |
| tess_solar | nulls | 256 | 256 | 256 | 0 | 0 | 0 | 0 |
| tess_highimpact | injections | 256 | 256 | 255 | 1 | 0 | 0 | 0 |
| tess_highimpact | nulls | 256 | 256 | 255 | 1 | 0 | 0 | 0 |
| tess_eccentric | injections | 256 | 256 | 255 | 1 | 0 | 0 | 0 |
| tess_eccentric | nulls | 256 | 256 | 254 | 2 | 0 | 0 | 0 |
| tess_mdwarf | injections | 256 | 256 | 256 | 0 | 0 | 0 | 0 |
| tess_mdwarf | nulls | 256 | 256 | 256 | 0 | 0 | 0 | 0 |
| ztf_solar | injections | 256 | 256 | 256 | 0 | 0 | 0 | 0 |
| ztf_solar | nulls | 256 | 256 | 256 | 0 | 0 | 0 | 0 |
| ztf_highimpact | injections | 256 | 256 | 256 | 0 | 0 | 0 | 0 |
| ztf_highimpact | nulls | 256 | 256 | 256 | 0 | 0 | 0 | 0 |
| ztf_mdwarf | injections | 256 | 256 | 256 | 0 | 0 | 0 | 0 |
| ztf_mdwarf | nulls | 256 | 256 | 256 | 0 | 0 | 0 | 0 |
| tess_gap_long | injections | 256 | 256 | 256 | 0 | 0 | 0 | 0 |
| tess_gap_long | nulls | 256 | 256 | 256 | 0 | 0 | 0 | 0 |
| tess_grazing_smeared | injections | 256 | 256 | 256 | 0 | 0 | 0 | 0 |
| tess_grazing_smeared | nulls | 256 | 256 | 256 | 0 | 0 | 0 | 0 |
| hatpi_short | injections | 256 | 256 | 254 | 2 | 0 | 0 | 0 |
| hatpi_short | nulls | 256 | 256 | 254 | 2 | 0 | 0 | 0 |

Individual implementation failures are retained in [exactness_mismatches.csv](exactness_mismatches.csv); the original source JSON retains every diagnostic repeat and any full-array mismatch artifacts.

## Machine-readable tables and provenance

[Recovery/FPR](recovery_fpr.csv), [paired contrasts](paired_contrasts.csv), [all subgroups](../../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/final-report/subgroups.csv"), [thresholds](thresholds.csv), [per-regime exactness](exactness.csv), [provenance](../../../../docs/BENCHMARK_ARCHIVES.md#tls_survey_2026-09-10 "Archived file: benchmarks/results/tls_survey_2026-09-10/final-report/provenance.json").

All interval bounds in the CSVs preserve the original JSON values. Displayed percentages are rounded only for readability.

| Source | SHA256 |
| --- | --- |
| recovery | e4bb50e577e77d9148f044ed92de5d2b1df5966155f419fabe8b16c8f528e7ae |
| seal | 1b81c75bd1a498c0dbed607e3221da1f374fc05be765de6dd2670c8d2f2b0807 |
| exactness | 1931a7a9f7ab7c34f7e882406926da7e44b446c7d8c7fa4b57120dde4fe4c9eb |
| snr | 0b662ca2b11ea286cc9fced82d65152ad0aee2bd41b76ba137a93e449f6b9dd3 |
| renderer | e3fdeb74daa7393ee927d79b62c24f25b47c4904e774d536fdae8800dc8eb97f |
