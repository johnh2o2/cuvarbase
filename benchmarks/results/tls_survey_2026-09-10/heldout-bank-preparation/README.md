# Held-out bank preparation check

The metadata check passed at 2026-09-11 18:37:15 UTC. All three completed manifests match the frozen scientific seal: 5,120 calibration nulls, 2,560 injections and 2,560 separate test nulls, with the planned counts in every regime. Injection generation completed at 18:27:18 UTC, test-null generation at 18:36:18 UTC, and the blind injection search started immediately afterward.

The check confirms the split/null roles, 64-node exposure integration, absence of truth insertion into the trial grid, seven recorded array hashes per lightcurve, and unique names and recorded input-file and flux-array hashes within and across banks. Shared timestamps and other common cadence arrays are allowed.

These are checks of completed manifest metadata and recorded hashes. This check does not rehash the held-out NPZ files or read detection outcomes. Hash disjointness checks reuse of recorded byte identities; it does not establish statistical independence or detection performance. Search-time input verification and final archive verification remain required. The frozen generator defines the separate random streams and populations.

`review.json` preserves the result and manifest identities. `check.py` is the exact Python body executed read-only on the rental, using only the standard library. For reproduction, mount the archived manifests and scientific seal read-only at their original `/workspace/tls-survey/final-campaign` and `/workspace/tls-survey/evidence` paths, then run `python3 check.py`. The printed campaign stage is a dated observation and will differ after the search advances. Do not overwrite the original receipt.
