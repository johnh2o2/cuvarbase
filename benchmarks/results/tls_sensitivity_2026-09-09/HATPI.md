# HATPI cost pilot

A study with the same 4,096 calibration nulls, 2,048 injections and 4,096 independent test nulls per method projects to **$27.29 of GPU search time at native 30-second cadence**, or **$3.46 after five-minute time averaging**, for three cuvarbase TLS grids plus GTLS. The secondary BLS control adds about $0.03 or $0.006 respectively. This is a pricing pilot, not a HATPI sensitivity result.

The synthetic example has **102 clear eight-hour nights within a 196-day season**, 195.33 days between the first and last observations, **97,920 measurements per source**, and **17,377 common trial periods from 0.6 to 12 days**. Five-minute averages reduce each source to **9,792 measurements**. Two injected transits and two nulls are retained; all ten configurations returned valid API outputs.

[HATPI's specifications](https://hatpi.org/about) give 30/45-second exposures and a broad optical band. The observing nights, fluxes and errors here are wholly simulated. No observed HATPI lightcurve was available from the authenticated data service during this experiment, so these numbers must not be described as measured HATPI survey performance.

| Search | Native 30-second data: time / source | Five-minute averages: time / source |
|---|---:|---:|
| BLS v1 | 21.19 ms | 4.58 ms |
| TLS original grid | 65.60 ms | 5.88 ms |
| TLS intermediate grid | 48.95 ms | 31.83 ms |
| TLS fine grid | 0.235 s | 0.209 s |
| Public GTLS, one worker | 19.231 s | 2.232 s |

The pilot uses an A40 and the same prepared-array API boundary as the TLS study. Its software and CPU-quota context are recorded in [hatpi_analysis.json](hatpi_analysis.json); a separate CPU-model snapshot was not retained for this pilot. cuvarbase uses three warmed four-source batch repetitions. To bound pilot cost, GTLS uses three distinct single-source calls after a first-source warmup. The GTLS sample includes one injection and two nulls; the cuvarbase batch includes two of each. Full [records](hatpi-cost), [timing ranges](hatpi_timing.csv), initialization and first-call values are retained. These small, differently aggregated samples support rough pricing, not an apples-to-apples headline speed ratio or an established recovery match.

For planning, allow roughly **$30–40 for the native-cadence experiment** or **$5–10 for the five-minute experiment**, including room for setup and generation. The measured search projections use `$0.49/hour × 10,240 cases × sum of four methods' seconds per source / 3,600`. More seasons, a different period grid, real residual noise, different GTLS memory behavior or extra BLS competitors can change the price. A fixed budget does not guarantee a sensitivity conclusion.

Five-minute time averaging combines adjacent observations once before searching. Phase binning happens separately for every trial period inside TLS. Time averaging can erase information from short ingress, narrow transits and other fast variability; both preprocessing choices would need inclusion in a future recovery test. It cannot be assumed harmless because its timings are cheaper.

HATPI's high observation count increases folding, sorting and per-observation work. Its one-season period grid here is much shorter than the long-baseline ZTF grid, reducing GTLS's per-period host overhead. Those effects pull relative timing in different directions. Even bin-count cost is not universally monotonic: the intermediate TLS grid was faster than the automatic grid on the native pilot, while the fine grid was slower. This pilot did not profile the cause of that difference.

A full HATPI study would first need an observed cadence and a frozen choice of native versus time-averaged inputs. [Generator and worker](../../tls_sensitivity/hatpi_cost.py) · [Verified timing arithmetic](hatpi_analysis.json) · [Combined experiment rental ledger](rental-ledger.json).
