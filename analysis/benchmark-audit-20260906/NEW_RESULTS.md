# Fresh measurement results

All times below are measured warm API wall times on the same A40 host. The figure groups and source filenames are recorded in `selected_timings.csv`.

## Lomb–Scargle, float64 throughout

| Workload | LCs | CPU ms/LC | GPU competitor ms/LC | PyPI 0.2.5 ms/LC | v1.0 ms/LC | CPU/v1.0 | PyPI/v1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 1 | 1.517 | 11.538 | 8.795 | 7.573 | 0.20× | 1.16× |
| small | 32 | 1.398 | 4.471 | N/A | 0.395 | 3.54× | N/A |
| tess | 1 | 5.139 | 5.566 | 11.200 | 8.242 | 0.62× | 1.36× |
| tess | 32 | 4.638 | 5.070 | 7.025 | 0.880 | 5.27× | 7.99× |
| ztf | 1 | 41.852 | 12.473 | N/A | 23.070 | 1.81× | N/A |
| ztf | 32 | 21.056 | 14.966 | N/A | 7.734 | 2.72× | N/A |
| kepler | 1 | 98.793 | 22.745 | 83.752 | 44.941 | 2.20× | 1.86× |
| kepler | 32 | 55.682 | 23.454 | 60.706 | 14.956 | 3.72× | 4.06× |

Ratios below one mean v1.0 is slower. “Best tested” selects only completed candidates passing the sampled accuracy screen. See `validation.json` for precision/error measurements and excluded results.

## Lomb–Scargle, cuvarbase default precision

| Workload | LCs | CPU ms/LC | GPU competitor ms/LC | PyPI 0.2.5 ms/LC | v1.0 ms/LC | CPU/v1.0 | PyPI/v1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 1 | 1.517 | 11.538 | 5.272 | 7.361 | 0.21× | 0.72× |
| small | 32 | 1.398 | 4.471 | N/A | 0.386 | 3.63× | N/A |
| tess | 1 | 5.139 | 5.075 | 11.218 | 7.940 | 0.65× | 1.41× |
| tess | 32 | 4.638 | 5.070 | 6.133 | 0.966 | 4.80× | 6.35× |
| ztf | 1 | 41.852 | 12.473 | N/A | 14.155 | 2.96× | N/A |
| ztf | 32 | 21.056 | 14.966 | N/A | N/A | N/A | N/A |
| kepler | 1 | 98.793 | 22.745 | 79.922 | 26.002 | 3.80× | 3.07× |
| kepler | 32 | 55.682 | 23.454 | 34.740 | 8.160 | 6.82× | 4.26× |

Ratios below one mean v1.0 is slower. “Best tested” selects only completed candidates passing the sampled accuracy screen. See `validation.json` for precision/error measurements and excluded results.

## Shared-time LS batch, float64 throughout

| Workload | LCs | CPU ms/LC | GPU competitor ms/LC | PyPI 0.2.5 ms/LC | v1.0 ms/LC | CPU/v1.0 | PyPI/v1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tess | 32 | 1.437 | 0.421 | 5.429 | 0.910 | 1.58× | 5.97× |

Ratios below one mean v1.0 is slower. “Best tested” selects only completed candidates passing the sampled accuracy screen. See `validation.json` for precision/error measurements and excluded results.

## Shared-time LS batch, cuvarbase default precision

| Workload | LCs | CPU ms/LC | GPU competitor ms/LC | PyPI 0.2.5 ms/LC | v1.0 ms/LC | CPU/v1.0 | PyPI/v1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tess | 32 | 1.437 | 0.405 | 6.242 | 0.994 | 1.45× | 6.28× |

Ratios below one mean v1.0 is slower. “Best tested” selects only completed candidates passing the sampled accuracy screen. See `validation.json` for precision/error measurements and excluded results.

## TLS timing and recovery

| Record | ms/LC | Exact period recovery | Native output finite |
|---|---:|---:|---|
| tls_1500_1_gtls_head.json | 153192.137 | 1/1 | True |
| tls_1500_1_v1_default.json | 184.626 | 1/1 | True |
| tls_1500_1_v1_wide.json | 942.809 | 1/1 | True |
| tls_200_1_cpu.json | 44553.928 | 1/1 | True |
| tls_200_1_gtls_head.json | 3408.473 | 1/1 | True |
| tls_200_1_gtls_pypi.json | 3034.942 | 1/1 | True |
| tls_200_1_v1_default.json | 14.813 | 1/1 | True |
| tls_200_1_v1_wide.json | 83.375 | 1/1 | True |
| tls_27_16_cpu.json | 1037.258 | 5/16 | True |
| tls_27_16_cpu_w8.json | 566.629 | 5/16 | True |
| tls_27_16_gtls_head.json | 357.723 | 4/16 | True |
| tls_27_16_gtls_pypi.json | 372.149 | 0/16 | False |
| tls_27_16_v1_default.json | 1.637 | 5/16 | True |
| tls_27_16_v1_wide.json | 5.822 | 5/16 | True |
| tls_27_1_cpu.json | 2533.366 | 1/1 | True |
| tls_27_1_cpu_t1.json | 3993.895 | 1/1 | True |
| tls_27_1_cpu_t8.json | 2201.783 | 1/1 | True |
| tls_27_1_gtls_head.json | 332.710 | 1/1 | True |
| tls_27_1_gtls_pypi.json | 362.403 | 0/1 | False |
| tls_27_1_v1_default.json | 14.062 | 0/1 | True |
| tls_27_1_v1_wide.json | 14.560 | 1/1 | True |

PyPI cuvarbase 0.2.5 has no TLS. The default v1.0 window is narrower than the wide-window comparison. Neither timing configuration is certified to have equivalent completeness/FPR to GTLS or CPU TLS.

## Small sensitivity diagnostic

| Method | Exact periods | Including aliases | Nulls with native SDE > 7 |
|---|---:|---:|---:|
| CPU TLS | 15/48 | 15/48 | 0/16 |
| GTLS 0.5.1 | 13/48 | 13/48 | 1/16 |
| v1.0 wide | 12/48 | 13/48 | 0/16 |
| v1.0 default | 14/48 | 14/48 | 1/16 |

These are diagnostic counts for this particular injection set, not population completeness estimates. Sixteen nulls are insufficient to validate low false-alarm rates. The score curves do not remove the need for larger paired injections and real noise.
