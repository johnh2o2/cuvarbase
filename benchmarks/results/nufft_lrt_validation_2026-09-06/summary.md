GPU: NVIDIA A40; commit: 2f9736ae4; date: 2026-09-06

### LRT statistic null calibration (white noise, fixed template)

mean = 0.348, std = 1.579 over 200 realizations (the pre-fix Sep-2026 campaign, sigma = 2 NFFT: 0.007, 1.812). Calibration constant of this configuration, not a pass/fail check: the statistic is a whitened correlation, not N(0,1), because the NFFT modes of irregular sampling are not orthogonal; its null std depends on the sampling, nf and the PSD estimator. This is why the thresholds below are empirical null percentiles.

### Protocol

600-point ground-like irregular sampling over 90 d; trial grid 32 periods (injected P = 5.30 d on-grid), box duration 0.22 d; thresholds = 95th percentile of 200 null search maxima; completeness over 200 injections per depth, period hit within 1% (incl. 2:1 aliases). Depths are fractions of the flux; sigma_white = 0.003.

Completeness cells are "p +- sigma" with sigma the quadrature sum of the null-threshold sampling error (bootstrap of the null maxima) and the binomial (Wilson, z = 1) error; the paired-difference rows use the same lightcurves for both arms (McNemar sigma = sqrt(b + c) / n, thresholds fixed).

### White noise

| arm | null p95 | depth 0.002 | depth 0.003 | depth 0.004 | depth 0.008 | ms/search |
|---|---:|---:|---:|---:|---:|---:|
| LRT (explicit epoch grid) | 8.588 | 13 +- 3% | 47 +- 5% | 82 +- 3% | 99 +- 1% | 5737 |
| LRT, default path (epochs=None) | 8.719 | 10 +- 3% | 42 +- 5% | 74 +- 4% | 99 +- 1% | 4491 |
| BLS (eebls_gpu_fast) | 0.038 | 13 +- 3% | 60 +- 4% | 91 +- 2% | 100 +- 0% | 1.55 |
| TLS (tls_search_batch, delta-chi2) | 4.662 | 16 +- 3% | 65 +- 4% | 90 +- 2% | 100 +- 1% | 11.4 |

Paired completeness differences (A - B, same lightcurves):

| A - B | depth 0.002 | depth 0.003 | depth 0.004 | depth 0.008 |
|---|---:|---:|---:|---:|
| lrt - bls | +0 +- 2% | -12 +- 3% | -10 +- 2% | -1 +- 1% |
| lrt_auto - lrt | -4 +- 1% | -5 +- 3% | -8 +- 2% | +0 +- 1% |
| lrt - tls | -4 +- 2% | -18 +- 3% | -9 +- 3% | -0 +- 1% |

Epoch recovery among detections (arms that return a best epoch; "same transit" = within half the injected duration, which any correct-period detection meets; the errors show the grid resolution):

| arm | depth | detections | same transit (within dur/2) | median abs. error (d) | max abs. error (d) |
|---|---:|---:|---:|---:|---:|
| LRT (explicit epoch grid) | 0.002 | 26 | 96% | 0.034 | 0.140 |
| LRT (explicit epoch grid) | 0.003 | 94 | 96% | 0.027 | 0.170 |
| LRT (explicit epoch grid) | 0.004 | 163 | 100% | 0.019 | 0.093 |
| LRT (explicit epoch grid) | 0.008 | 198 | 100% | 0.015 | 0.044 |
| LRT, default path (epochs=None) | 0.002 | 19 | 95% | 0.030 | 0.135 |
| LRT, default path (epochs=None) | 0.003 | 84 | 99% | 0.025 | 0.149 |
| LRT, default path (epochs=None) | 0.004 | 147 | 99% | 0.020 | 0.128 |
| LRT, default path (epochs=None) | 0.008 | 198 | 100% | 0.020 | 0.091 |

(compute: 10253 s)

### White noise, absolute times (BJD-scale, t + 2457000 d)

| arm | null p95 | depth 0.002 | depth 0.003 | depth 0.004 | depth 0.008 | ms/search |
|---|---:|---:|---:|---:|---:|---:|
| LRT (explicit epoch grid) | 8.588 | 13 +- 3% | 47 +- 5% | 82 +- 3% | 99 +- 1% | 5740 |
| LRT, default path (epochs=None) | 8.719 | 10 +- 3% | 42 +- 5% | 74 +- 4% | 99 +- 1% | 4487 |
| BLS (eebls_gpu_fast) | 0.038 | 13 +- 3% | 60 +- 4% | 91 +- 2% | 100 +- 0% | 1.53 |
| TLS (tls_search_batch, delta-chi2) | 4.662 | 16 +- 3% | 65 +- 4% | 90 +- 2% | 100 +- 1% | 11.4 |

Paired completeness differences (A - B, same lightcurves):

| A - B | depth 0.002 | depth 0.003 | depth 0.004 | depth 0.008 |
|---|---:|---:|---:|---:|
| lrt - bls | +0 +- 2% | -12 +- 3% | -10 +- 2% | -1 +- 1% |
| lrt_auto - lrt | -4 +- 1% | -5 +- 3% | -8 +- 2% | +0 +- 1% |
| lrt - tls | -4 +- 2% | -18 +- 3% | -9 +- 3% | -0 +- 1% |

Epoch recovery among detections (arms that return a best epoch; "same transit" = within half the injected duration, which any correct-period detection meets; the errors show the grid resolution):

| arm | depth | detections | same transit (within dur/2) | median abs. error (d) | max abs. error (d) |
|---|---:|---:|---:|---:|---:|
| LRT (explicit epoch grid) | 0.002 | 26 | 96% | 0.034 | 0.140 |
| LRT (explicit epoch grid) | 0.003 | 94 | 96% | 0.027 | 0.170 |
| LRT (explicit epoch grid) | 0.004 | 163 | 100% | 0.019 | 0.093 |
| LRT (explicit epoch grid) | 0.008 | 198 | 100% | 0.015 | 0.044 |
| LRT, default path (epochs=None) | 0.002 | 19 | 95% | 0.030 | 0.135 |
| LRT, default path (epochs=None) | 0.003 | 84 | 99% | 0.025 | 0.149 |
| LRT, default path (epochs=None) | 0.004 | 147 | 99% | 0.020 | 0.128 |
| LRT, default path (epochs=None) | 0.008 | 198 | 100% | 0.020 | 0.091 |

(compute: 10249 s)

### Red noise, sigma_red = sigma_white

| arm | null p95 | depth 0.004 | depth 0.006 | depth 0.008 | depth 0.016 | ms/search |
|---|---:|---:|---:|---:|---:|---:|
| LRT (explicit epoch grid) | 11.364 | 4 +- 2% | 25 +- 4% | 56 +- 5% | 100 +- 1% | 4383 |
| LRT, default path (epochs=None) | 11.186 | 4 +- 2% | 24 +- 4% | 52 +- 5% | 99 +- 1% | 3412 |
| LRT, flat PSD | 1.779 | 4 +- 2% | 22 +- 4% | 54 +- 5% | 100 +- 1% | 4377 |
| BLS (eebls_gpu_fast) | 0.124 | 2 +- 1% | 17 +- 3% | 47 +- 5% | 100 +- 0% | 1.22 |
| TLS (tls_search_batch, delta-chi2) | 12.283 | 4 +- 1% | 22 +- 3% | 62 +- 4% | 100 +- 0% | 8.56 |

Paired completeness differences (A - B, same lightcurves):

| A - B | depth 0.004 | depth 0.006 | depth 0.008 | depth 0.016 |
|---|---:|---:|---:|---:|
| lrt - bls | +2 +- 2% | +8 +- 3% | +10 +- 3% | -0 +- 0% |
| lrt_auto - lrt | +0 +- 1% | -1 +- 2% | -4 +- 2% | -0 +- 0% |
| lrt - lrt_flat | -0 +- 1% | +2 +- 2% | +3 +- 3% | +0 +- 0% |
| lrt - tls | -0 +- 2% | +3 +- 3% | -5 +- 3% | -0 +- 0% |

Epoch recovery among detections (arms that return a best epoch; "same transit" = within half the injected duration, which any correct-period detection meets; the errors show the grid resolution):

| arm | depth | detections | same transit (within dur/2) | median abs. error (d) | max abs. error (d) |
|---|---:|---:|---:|---:|---:|
| LRT (explicit epoch grid) | 0.004 | 7 | 100% | 0.017 | 0.060 |
| LRT (explicit epoch grid) | 0.006 | 50 | 96% | 0.017 | 0.144 |
| LRT (explicit epoch grid) | 0.008 | 113 | 100% | 0.018 | 0.107 |
| LRT (explicit epoch grid) | 0.016 | 199 | 100% | 0.017 | 0.061 |
| LRT, default path (epochs=None) | 0.004 | 7 | 100% | 0.011 | 0.021 |
| LRT, default path (epochs=None) | 0.006 | 48 | 94% | 0.033 | 1.447 |
| LRT, default path (epochs=None) | 0.008 | 104 | 100% | 0.019 | 0.080 |
| LRT, default path (epochs=None) | 0.016 | 198 | 99% | 0.019 | 0.110 |
| LRT, flat PSD | 0.004 | 8 | 88% | 0.020 | 0.960 |
| LRT, flat PSD | 0.006 | 45 | 98% | 0.018 | 0.115 |
| LRT, flat PSD | 0.008 | 107 | 100% | 0.018 | 0.069 |
| LRT, flat PSD | 0.016 | 199 | 100% | 0.018 | 0.061 |

(compute: 12206 s)

### Red noise, sigma_red = 3 sigma_white

| arm | null p95 | depth 0.008 | depth 0.016 | depth 0.024 | depth 0.032 | ms/search |
|---|---:|---:|---:|---:|---:|---:|
| LRT (explicit epoch grid) | 14.121 | 0 +- 0% | 12 +- 4% | 57 +- 6% | 89 +- 3% | 4383 |
| LRT, default path (epochs=None) | 13.972 | 0 +- 0% | 10 +- 3% | 48 +- 5% | 83 +- 4% | 3413 |
| LRT, flat PSD | 5.097 | 0 +- 1% | 18 +- 6% | 63 +- 6% | 90 +- 3% | 4376 |
| BLS (eebls_gpu_fast) | 0.204 | 0 +- 1% | 7 +- 2% | 48 +- 4% | 88 +- 2% | 1.2 |
| TLS (tls_search_batch, delta-chi2) | 34.507 | 0 +- 1% | 17 +- 3% | 62 +- 4% | 93 +- 2% | 8.55 |

Paired completeness differences (A - B, same lightcurves):

| A - B | depth 0.008 | depth 0.016 | depth 0.024 | depth 0.032 |
|---|---:|---:|---:|---:|
| lrt - bls | -0 +- 0% | +6 +- 2% | +10 +- 3% | +1 +- 2% |
| lrt_auto - lrt | +0 +- 0% | -2 +- 1% | -9 +- 2% | -6 +- 2% |
| lrt - lrt_flat | -0 +- 0% | -6 +- 2% | -6 +- 3% | -0 +- 1% |
| lrt - tls | -0 +- 0% | -4 +- 2% | -6 +- 3% | -4 +- 1% |

Epoch recovery among detections (arms that return a best epoch; "same transit" = within half the injected duration, which any correct-period detection meets; the errors show the grid resolution):

| arm | depth | detections | same transit (within dur/2) | median abs. error (d) | max abs. error (d) |
|---|---:|---:|---:|---:|---:|
| LRT (explicit epoch grid) | 0.016 | 25 | 100% | 0.010 | 0.046 |
| LRT (explicit epoch grid) | 0.024 | 114 | 99% | 0.013 | 0.219 |
| LRT (explicit epoch grid) | 0.032 | 178 | 100% | 0.015 | 0.097 |
| LRT, default path (epochs=None) | 0.016 | 21 | 100% | 0.017 | 0.100 |
| LRT, default path (epochs=None) | 0.024 | 96 | 100% | 0.026 | 0.070 |
| LRT, default path (epochs=None) | 0.032 | 166 | 100% | 0.018 | 0.102 |
| LRT, flat PSD | 0.008 | 1 | 100% | 0.095 | 0.095 |
| LRT, flat PSD | 0.016 | 36 | 97% | 0.012 | 2.597 |
| LRT, flat PSD | 0.024 | 126 | 98% | 0.015 | 0.219 |
| LRT, flat PSD | 0.032 | 179 | 100% | 0.015 | 0.069 |

(compute: 12202 s)

### Red noise + shared systematics (PCA basis + population prior)

| arm | null p95 | depth 0.004 | depth 0.008 | depth 0.016 | depth 0.032 | ms/search |
|---|---:|---:|---:|---:|---:|---:|
| LRT (explicit epoch grid) | 11.965 | 0 +- 0% | 0 +- 0% | 6 +- 2% | 34 +- 4% | 5735 |
| LRT, default path (epochs=None) | 11.616 | 0 +- 0% | 0 +- 0% | 5 +- 2% | 34 +- 4% | 4482 |
| LRT Detector A (marginal) | 12.104 | 3 +- 1% | 44 +- 5% | 98 +- 1% | 100 +- 0% | 5524 |
| LRT sequential cotrend | 12.220 | 3 +- 2% | 43 +- 5% | 98 +- 1% | 100 +- 0% | 5412 |
| BLS (eebls_gpu_fast) | 0.188 | 0 +- 0% | 0 +- 0% | 2 +- 1% | 16 +- 3% | 1.51 |
| TLS (tls_search_batch, delta-chi2) | 114.971 | 0 +- 0% | 0 +- 0% | 0 +- 0% | 0 +- 0% | 11.4 |

Paired completeness differences (A - B, same lightcurves):

| A - B | depth 0.004 | depth 0.008 | depth 0.016 | depth 0.032 |
|---|---:|---:|---:|---:|
| lrt - bls | +0 +- 0% | +0 +- 0% | +3 +- 1% | +17 +- 3% |
| lrt_auto - lrt | +0 +- 0% | +0 +- 0% | -0 +- 0% | +0 +- 2% |
| lrt_marg - lrt_seq | +0 +- 0% | +0 +- 0% | +0 +- 0% | +0 +- 0% |
| lrt_seq - bls | +3 +- 1% | +43 +- 5% | +95 +- 7% | +84 +- 6% |
| lrt - tls | +0 +- 0% | +0 +- 0% | +6 +- 2% | +34 +- 4% |

Epoch recovery among detections (arms that return a best epoch; "same transit" = within half the injected duration, which any correct-period detection meets; the errors show the grid resolution):

| arm | depth | detections | same transit (within dur/2) | median abs. error (d) | max abs. error (d) |
|---|---:|---:|---:|---:|---:|
| LRT (explicit epoch grid) | 0.016 | 11 | 100% | 0.016 | 0.052 |
| LRT (explicit epoch grid) | 0.032 | 67 | 100% | 0.017 | 0.080 |
| LRT, default path (epochs=None) | 0.016 | 10 | 100% | 0.009 | 0.086 |
| LRT, default path (epochs=None) | 0.032 | 67 | 100% | 0.017 | 0.104 |
| LRT Detector A (marginal) | 0.004 | 6 | 100% | 0.010 | 0.108 |
| LRT Detector A (marginal) | 0.008 | 87 | 100% | 0.016 | 0.090 |
| LRT Detector A (marginal) | 0.016 | 195 | 100% | 0.014 | 0.052 |
| LRT Detector A (marginal) | 0.032 | 200 | 100% | 0.016 | 0.052 |
| LRT sequential cotrend | 0.004 | 6 | 100% | 0.010 | 0.108 |
| LRT sequential cotrend | 0.008 | 86 | 100% | 0.017 | 0.090 |
| LRT sequential cotrend | 0.016 | 195 | 100% | 0.014 | 0.052 |
| LRT sequential cotrend | 0.032 | 200 | 100% | 0.016 | 0.052 |

(compute: 21204 s)

### Red noise + shared systematics, non-zero-mean basis columns

| arm | null p95 | depth 0.004 | depth 0.008 | depth 0.016 | depth 0.032 | ms/search |
|---|---:|---:|---:|---:|---:|---:|
| LRT Detector A (marginal) | 12.104 | 3 +- 1% | 44 +- 5% | 98 +- 1% | 100 +- 0% | 6725 |
| LRT sequential cotrend | 12.220 | 3 +- 2% | 43 +- 5% | 98 +- 1% | 100 +- 0% | 6595 |

Paired completeness differences (A - B, same lightcurves):

| A - B | depth 0.004 | depth 0.008 | depth 0.016 | depth 0.032 |
|---|---:|---:|---:|---:|
| lrt_marg - lrt_seq | +0 +- 0% | +0 +- 0% | +0 +- 0% | +0 +- 0% |

Epoch recovery among detections (arms that return a best epoch; "same transit" = within half the injected duration, which any correct-period detection meets; the errors show the grid resolution):

| arm | depth | detections | same transit (within dur/2) | median abs. error (d) | max abs. error (d) |
|---|---:|---:|---:|---:|---:|
| LRT Detector A (marginal) | 0.004 | 6 | 100% | 0.010 | 0.108 |
| LRT Detector A (marginal) | 0.008 | 87 | 100% | 0.016 | 0.090 |
| LRT Detector A (marginal) | 0.016 | 195 | 100% | 0.014 | 0.052 |
| LRT Detector A (marginal) | 0.032 | 200 | 100% | 0.016 | 0.052 |
| LRT sequential cotrend | 0.004 | 6 | 100% | 0.010 | 0.108 |
| LRT sequential cotrend | 0.008 | 86 | 100% | 0.017 | 0.090 |
| LRT sequential cotrend | 0.016 | 195 | 100% | 0.014 | 0.052 |
| LRT sequential cotrend | 0.032 | 200 | 100% | 0.016 | 0.052 |

(compute: 13344 s)

### Paired configurations (same lightcurves)

Each pair saw identical noise and injections (shared sub-seed) and differs only in the time origin (white / white_bjd: + 2457000 d, an integer, so every method's floor(min t)-anchored grid keeps its phase) or in the basis column offsets (red_sys / red_sys_nzm). Differences beyond float32 rounding would indicate a time-scale or centring defect.

**white vs white_bjd**

| arm | searches | max rel. diff | median rel. diff | best period differs | detection differs | null p95 (white / white_bjd) |
|---|---:|---:|---:|---:|---:|---:|
| LRT (explicit epoch grid) | 1000 | 5.2e-08 | 0.0e+00 | 3 / 800 | 0 / 800 | 8.588 / 8.588 |
| LRT, default path (epochs=None) | 1000 | 3.5e-08 | 0.0e+00 | 2 / 800 | 0 / 800 | 8.719 / 8.719 |
| BLS (eebls_gpu_fast) | 1000 | 1.3e-06 | 0.0e+00 | 0 / 800 | 0 / 800 | 0.038 / 0.038 |
| TLS (tls_search_batch, delta-chi2) | 1000 | 1.4e-07 | 0.0e+00 | 0 / 800 | 0 / 800 | 4.662 / 4.662 |

**red_sys vs red_sys_nzm**

| arm | searches | max rel. diff | median rel. diff | best period differs | detection differs | null p95 (red_sys / red_sys_nzm) |
|---|---:|---:|---:|---:|---:|---:|
| LRT Detector A (marginal) | 1000 | 5.5e-07 | 1.4e-08 | 3 / 800 | 0 / 800 | 12.104 / 12.104 |
| LRT sequential cotrend | 1000 | 8.4e-08 | 0.0e+00 | 2 / 800 | 0 / 800 | 12.220 / 12.220 |

