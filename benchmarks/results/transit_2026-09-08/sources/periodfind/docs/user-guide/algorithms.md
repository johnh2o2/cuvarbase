# Algorithms

periodfind implements four period-finding algorithms and several feature
extraction tools. All period-finding algorithms evaluate a test statistic over
a 2D grid of trial periods and period derivatives.

## Period-Finding Algorithms

### Conditional Entropy

Conditional Entropy (CE) folds the light curve at each trial period and bins the
result into a 2D phase-magnitude histogram. The conditional entropy of the
magnitude given the phase is computed:

$$H(m \mid \phi) = -\sum_{i,j} p(m_j, \phi_i) \log \frac{p(m_j, \phi_i)}{p(\phi_i)}$$

A **minimum** in the CE periodogram indicates the best-fit period (low entropy
means the magnitudes are tightly concentrated in phase).

**Key parameters:**

- `n_phase` — Number of phase bins (default: 10)
- `n_mag` — Number of magnitude bins (default: 10)
- `phase_bin_extent` — Overlap/smoothing width in phase bins (default: 1)
- `mag_bin_extent` — Overlap/smoothing width in magnitude bins (default: 1)
- `normalize` — CE requires magnitudes in (0, 1), so normalization is enabled by default

**Reference:** Graham et al. (2013), *MNRAS*, 434, 2629.

### Analysis of Variance (AOV)

AOV folds the light curve at each trial period and bins the folded times into
phase bins. It computes an F-statistic comparing the between-bin variance to the
within-bin variance:

$$\Theta_{\mathrm{AOV}} = \frac{s_1^2}{s_2^2} = \frac{\sum n_k (\bar{x}_k - \bar{x})^2 / (K-1)}{\sum \sum (x_{ki} - \bar{x}_k)^2 / (N-K)}$$

A **maximum** in the AOV periodogram indicates the best-fit period.

**Key parameters:**

- `n_phase` — Number of phase bins (default: 10)
- `phase_bin_extent` — Overlap/smoothing width in phase bins (default: 1)

**Reference:** Schwarzenberg-Czerny (1989), *MNRAS*, 241, 153.

### Lomb-Scargle

The Lomb-Scargle periodogram computes a normalized power spectrum for unevenly
sampled data:

$$P(\omega) = \frac{1}{2} \left[ \frac{\left(\sum_i (x_i - \bar{x}) \cos\omega(t_i - \tau)\right)^2}{\sum_i \cos^2\omega(t_i - \tau)} + \frac{\left(\sum_i (x_i - \bar{x}) \sin\omega(t_i - \tau)\right)^2}{\sum_i \sin^2\omega(t_i - \tau)} \right]$$

where $\tau$ is a time offset that orthogonalizes the sine and cosine terms.

A **maximum** in the LS periodogram indicates the best-fit period.

**Key parameters:**

- `center` — LS benefits from mean-subtracted data (default: True)

**Reference:** Lomb (1976), *Ap&SS*, 39, 447; Scargle (1982), *ApJ*, 263, 835.

### Fast Phase-folding Weighted (FPW)

FPW computes a weighted chi-squared reduction that supports per-point
uncertainties. It folds light curves into phase bins and evaluates the
weighted variance ratio.

A **maximum** in the FPW periodogram indicates the best-fit period.

**Key parameters:**

- `n_bins` — Number of phase bins (default: 10)
- `errs` — Per-point uncertainties (optional; uniform if not provided)

**Reference:** Finkbeiner et al. (2025).

### Choosing an Algorithm

| Feature | CE | AOV | LS | FPW |
|---------|-----|-----|-----|------|
| Best for | Non-sinusoidal, arbitrary shapes | Sharp, non-sinusoidal features | Sinusoidal signals | Weighted data with uncertainties |
| Statistic | Minimum (entropy) | Maximum (F-statistic) | Maximum (power) | Maximum (chi-squared) |
| Normalization | Required | Optional | Optional (centering recommended) | Not needed |
| Supports uncertainties | No | No | No | Yes |
| Computational cost | Medium | Medium | Lower | Medium |

### Period Derivatives

All period-finding algorithms accept a `period_dts` array to search over period
derivatives, useful for sources whose period is evolving over time. Setting
`period_dts = np.array([0.0], dtype=np.float32)` reduces to a standard 1D
period search.

### Peaks Mode

All period-finding algorithms support `output='peaks'`, which uses a streaming
greedy algorithm to return the top peaks without materialising the full
periodogram array. This saves significant memory on large period grids.

## Feature Extraction

### Fourier Decomposition

Computes weighted linear least-squares Fourier fits with BIC model selection
over 0-5 harmonics. Returns 14 features per curve:

`[power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]`

This replaces per-source `scipy.optimize.curve_fit` with a direct Cholesky
solve, giving identical results orders of magnitude faster.

### dm-dt Histograms

Computes L2-normalised 2D histograms of pairwise magnitude differences vs.
time differences for each light curve.

### Basic Statistics

Computes 22 summary statistics per light curve: N, median, wmean, chi2red,
RoMS, wstd, NormPeaktoPeakamp, NormExcessVar, medianAbsDev, iqr, i60r, i70r,
i80r, i90r, skew, smallkurt, invNeumann, WelchI, StetsonJ, StetsonK, AD, SW.

### High-Cadence Removal

Utility to filter light curves, keeping only points separated by at least a
given cadence threshold. Available via `periodfind.remove_high_cadence()`.
