"""
Statistical calculations for Transit Least Squares.

Implements the Signal Residue (SR), Signal Detection Efficiency (SDE),
a delta-chi-squared Signal-to-Noise Ratio (SNR), and related metrics.

Definitions (identical to the reference ``transitleastsquares`` package,
so published SDE thresholds transfer):

* ``SR = chi2_min / chi2`` (1 at the best trial period, < 1 elsewhere);
* ``SDE_raw = (1 - mean(SR)) / std(SR)``;
* ``SDE`` is the same z-score after subtracting a running median of SR
  (edge-extended, see :func:`running_median`).

No calibrated false-alarm probability is derived from the SDE: the null
SDE distribution depends on the period grid and the baseline (measured:
4% to 92% of pure-noise light curves exceed SDE = 7 across four common
configurations), so only a per-configuration null bootstrap can be
honest -- see ``tls_search_batch(fap_null_draws=...)``.

References
----------
- Hippke & Heller (2019), A&A 623, A39
- Kovács et al. (2002), A&A 391, 369
"""

import warnings

import numpy as np
from scipy import ndimage, stats


__all__ = [
    'signal_residue',
    'running_median',
    'signal_detection_efficiency',
    'signal_to_noise',
    'false_alarm_probability',
    'odd_even_mismatch',
    'compute_all_statistics',
    'compute_period_uncertainty',
]


def signal_residue(chi2, chi2_null=None):
    """
    Calculate the Signal Residue (SR) of a chi-squared spectrum.

    ``SR = chi2_min / chi2`` -- the definition of the reference
    ``transitleastsquares`` package: SR = 1 at the best trial period and
    decreases towards 0 for worse fits.

    Parameters
    ----------
    chi2 : array_like
        Chi-squared values at each trial period (finite, >= 0)
    chi2_null : float, optional
        Deprecated and ignored (a warning is raised if given). Before
        1.0 the SR was ``1 - chi2 / max(chi2)``, which agrees with the
        reference definition under the null but is up to 2x lower at
        the peak of a strong signal, so SDE thresholds from the
        literature did not transfer.

    Returns
    -------
    SR : ndarray
        Signal residue values in [0, 1]; 1 at the minimum chi2.
    """
    if chi2_null is not None:
        warnings.warn(
            "signal_residue: chi2_null is ignored; since 1.0 the signal "
            "residue is chi2_min / chi2 (reference transitleastsquares "
            "definition)", DeprecationWarning, stacklevel=2)
    chi2 = np.asarray(chi2, dtype=np.float64)
    if chi2.size == 0:
        return chi2.copy()
    # chi2 is a sum of squares; the batch path reconstructs it as
    # chi2_0 - score in float64 from a float32 score, which can dip a
    # hair below zero for a near-perfect fit
    chi2 = np.maximum(chi2, 0.0)
    chi2_min = np.min(chi2)
    with np.errstate(divide='ignore', invalid='ignore'):
        SR = chi2_min / chi2
    # 0/0 at a perfect (noiseless) fit; the best period has SR = 1 by
    # definition
    SR[chi2 == chi2_min] = 1.0
    return SR


def running_median(x, kernel):
    """
    Sliding median of odd width ``kernel`` with the reference package's
    edge handling.

    Interior points use the full centred window. The first and last
    ``kernel // 2`` points, whose window would run off the array, are
    filled with the first / last full-window median -- exactly the
    ``running_median`` of ``transitleastsquares`` (which builds the
    same edge-extended trend with an explicit index matrix), computed
    here with :func:`scipy.ndimage.median_filter`. Zero-padding (what
    ``scipy.signal.medfilt`` does) is NOT equivalent: it drags the
    trend towards zero over the outermost ``kernel // 2`` points and
    inflates the detrended power at the grid edges (measured: null
    peaks landed within 45 points of an edge 2.3x more often than
    uniform).

    Parameters
    ----------
    x : array_like
        Input series (float64 on output)
    kernel : int
        Window width; even values are rounded up to the next odd
        integer. Must satisfy ``kernel <= len(x)``.

    Returns
    -------
    trend : ndarray
        Running median, same length as ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    kernel = int(kernel)
    if kernel % 2 == 0:
        kernel += 1
    n = len(x)
    if kernel > n:
        raise ValueError("running_median: kernel (%d) exceeds the series "
                         "length (%d)" % (kernel, n))
    if kernel <= 1:
        return x.copy()
    h = kernel // 2
    trend = ndimage.median_filter(x, size=kernel, mode='nearest')
    # mode='nearest' only affects the outermost h points; overwrite
    # them with the first/last full-window medians (indices h, n-1-h)
    trend[:h] = trend[h]
    trend[n - h:] = trend[n - 1 - h]
    return trend


def signal_detection_efficiency(chi2, chi2_null=None, detrend=True,
                                kernel_size=None):
    """
    Calculate Signal Detection Efficiency (SDE).

    SDE measures how many standard deviations the peak of the signal
    residue spectrum stands above its mean. Higher SDE = more
    significant detection.

    Parameters
    ----------
    chi2 : array_like
        Chi-squared values at each period, ordered by ascending period
        (the running-median detrend assumes period-ordered neighbours)
    chi2_null : float, optional
        Deprecated and ignored (see :func:`signal_residue`).
    detrend : bool, optional
        Subtract a running median of SR before the z-score (default:
        True)
    kernel_size : int, optional
        Running-median kernel size for detrending. If None (default),
        uses ``min(len(SR)//10 forced odd (min 3), 91)``: small period
        grids keep the length-proportional window, while large grids
        are capped at 91 points -- the fixed-kernel convention of the
        reference ``transitleastsquares`` package (oversampling factor
        3 x SDE_MEDIAN_KERNEL_SIZE 30, forced odd). Passing an explicit
        value overrides the automatic choice (even values are rounded
        up to the next odd integer, as required by the median filter).

    Returns
    -------
    SDE : float
        Signal detection efficiency (z-score)
    SDE_raw : float
        Raw SDE before detrending
    power : ndarray
        Detrended signal residue (``SR - trend + median(SR)``) when
        detrending was applied, else ``SR``

    Notes
    -----
    With ``SR = chi2_min / chi2`` (see :func:`signal_residue`):

    - ``SDE_raw = (max(SR) - mean(SR)) / std(SR) = (1 - mean(SR)) / std(SR)``
    - ``SDE = (max(D) - mean(D)) / std(D)`` with ``D = SR - running_median(SR)``

    which is the reference package's ``spectra()`` statistic (its
    rescaling of the detrended spectrum to touch ``max = SDE`` does not
    change the z-score). A flat spectrum (``std(SR) < 1e-10``) gives
    SDE = 0.

    The SDE is a *contrast* statistic, not a calibrated significance:
    under the null its distribution shifts with the number of trial
    periods and the baseline (measured mean 6.4, std 1.0 for 6157
    periods on a 60-d light curve; 23% of pure-noise light curves above
    7 there, 92% above 7 at 365 d with 43,780 periods). There is no
    fixed SDE threshold with a known false-alarm rate; use the
    per-configuration null bootstrap of ``tls_search_batch``
    (``fap_null_draws``) or your own injection-recovery.

    Following ``transitleastsquares`` (Hippke & Heller 2019), detrending
    is skipped entirely when ``len(SR) <= 2 * kernel_size``; in that
    case the raw SDE and raw SR are returned unchanged.
    """
    chi2 = np.asarray(chi2)

    # Calculate signal residue
    SR = signal_residue(chi2, chi2_null)

    # Raw SDE (before detrending)
    mean_SR = np.mean(SR)
    std_SR = np.std(SR)

    if std_SR < 1e-10:
        SDE_raw = 0.0
    else:
        SDE_raw = (np.max(SR) - mean_SR) / std_SR

    # Detrend with a running median if requested
    if detrend:
        if kernel_size is None:
            kernel_size = max(len(SR) // 10, 3)
            # Ensure odd window
            if kernel_size % 2 == 0:
                kernel_size += 1
            # Cap at the fixed 91-point kernel used by the reference
            # transitleastsquares implementation; an uncapped len//10
            # window makes the median filter O(n*k) ~ O(n^2/10) and
            # takes minutes of CPU at survey-scale period grids
            # (n ~ 1e5).
            kernel_size = min(kernel_size, 91)
        elif kernel_size % 2 == 0:
            # the median filter requires an odd kernel
            kernel_size += 1

        if len(SR) <= 2 * kernel_size:
            # Too few points to estimate a trend; follow the reference
            # transitleastsquares behavior and skip detrending.
            SDE = SDE_raw
            power = SR
        else:
            # Edge-extended running median (reference convention)
            SR_trend = running_median(SR, kernel_size)

            # Detrended signal residue
            SR_detrended = SR - SR_trend + np.median(SR)

            # Calculate SDE on detrended signal
            mean_SR_detrended = np.mean(SR_detrended)
            std_SR_detrended = np.std(SR_detrended)

            if std_SR_detrended < 1e-10:
                SDE = 0.0
            else:
                SDE = ((np.max(SR_detrended) - mean_SR_detrended)
                       / std_SR_detrended)

            power = SR_detrended
    else:
        SDE = SDE_raw
        power = SR

    return SDE, SDE_raw, power


def signal_to_noise(depth, depth_err=None,
                    chi2_null=None, chi2_best=None):
    """
    Calculate signal-to-noise ratio.

    Parameters
    ----------
    depth : float
        Transit depth
    depth_err : float, optional
        Uncertainty in depth. If None, estimated from chi2 values or
        Poisson statistics as a last resort.
    chi2_null : float, optional
        Null hypothesis chi-squared (no transit). Used to estimate
        depth_err when depth_err is not provided.
    chi2_best : float, optional
        Best-fit chi-squared. Used with chi2_null to estimate depth_err.

    Returns
    -------
    snr : float
        Signal-to-noise ratio

    Notes
    -----
    When depth_err is not provided, it is estimated as
    depth / sqrt(chi2_null - chi2_best) if chi2 values are given,
    otherwise this returns 0 -- i.e. the returned SNR is the
    delta-chi-squared significance ``sqrt(chi2_null - chi2_best)`` of
    the transit model over the constant model. The search wrappers pass
    the constant-model ``chi2_0`` (float64) and the refined ``chi2_min``
    of the best fit, so ``SNR = sqrt(chi2_0 - chi2_min)``; this is not
    the reference package's ``depth / std * sqrt(n_in_transit)``. A
    depth_err derived from the full-dataset delta-chi-squared already
    includes every in-transit point across all transits, so no
    additional sqrt(n_transits) scaling is applied.
    """
    if depth_err is None:
        if chi2_null is not None and chi2_best is not None:
            delta_chi2 = chi2_null - chi2_best
            if delta_chi2 > 0:
                depth_err = depth / np.sqrt(delta_chi2)
            else:
                return 0.0
        else:
            return 0.0

    if depth_err < 1e-10:
        return 0.0

    return depth / depth_err


def false_alarm_probability(SDE, method='empirical'):
    """
    Heuristic SDE -> "FAP" map. NOT a calibrated false-alarm
    probability; kept only as an explicit opt-in helper.

    Since 1.0 no TLS result dict carries this number: the audit
    measured 23% of pure-noise light curves receiving ``FAP < 0.01``
    from it (60 d, 6157 periods), a 250x error at SDE = 9, and a null
    SDE distribution that moves with the grid and baseline, so no
    fixed map can be right. Use ``tls_search_batch(fap_null_draws=N)``
    for an empirical, per-configuration false-alarm probability.

    Parameters
    ----------
    SDE : float
        Signal Detection Efficiency
    method : str, optional
        - 'empirical': ad-hoc piecewise heuristic (see Notes); raises
          a UserWarning every call
        - 'gaussian': one-sided Gaussian tail ``1 - Phi(SDE)``, which
          treats the SDE as a standard-normal z-score (it is not: it is
          the maximum over thousands of correlated trials)

    Returns
    -------
    FAP : float
        Heuristic value in [1e-10, 1]

    Notes
    -----
    .. warning::

        The 'empirical' method is a hand-rolled piecewise heuristic:
        1 below SDE = 5, ``10**(-0.5 (SDE - 5))`` (0.1 just below 7),
        then ``10**(-(SDE - 5))`` from 7 upwards (0.01 at 7, so it is
        discontinuous there). It is NOT calibrated against any null
        distribution or published injection-recovery results. Treat the
        returned values as order-of-magnitude indicators at best.
    """
    if method == 'gaussian':
        # Gaussian approximation: FAP = 1 - Phi(SDE)
        FAP = 1.0 - stats.norm.cdf(SDE)
    else:
        warnings.warn(
            "false_alarm_probability: this is an uncalibrated heuristic "
            "(1 below SDE 5, 0.1 just below 7, 0.01 at 7, then 10**-(SDE-5)); "
            "measured null exceedance differs from it by orders of "
            "magnitude. Use tls_search_batch(fap_null_draws=N) for an "
            "empirical FAP.", UserWarning, stacklevel=2)
        # Ad-hoc piecewise heuristic; no published calibration.
        # Values: FAP(5) = 1, FAP(6) = 0.32, FAP(7-) = 0.1, FAP(7) = 0.01
        # (discontinuous), FAP(9) = 1e-4.
        if SDE < 5:
            FAP = 1.0
        elif SDE < 7:
            FAP = 10 ** (-0.5 * (SDE - 5))
        else:
            FAP = 10 ** (-(SDE - 5))

        # Clip to reasonable range
        FAP = np.clip(FAP, 1e-10, 1.0)

    return FAP


def odd_even_mismatch(depths_odd, depths_even):
    """
    Calculate odd-even transit depth mismatch.

    This tests whether odd and even transits have significantly
    different depths, which could indicate:
    - Binary system
    - Non-planetary signal
    - Instrumental effects

    Parameters
    ----------
    depths_odd : array_like
        Depths of odd-numbered transits
    depths_even : array_like
        Depths of even-numbered transits

    Returns
    -------
    mismatch : float
        Significance of mismatch (z-score)
    depth_diff : float
        Difference between mean depths

    Notes
    -----
    High mismatch (>3σ) suggests the signal may not be planetary.
    """
    depths_odd = np.asarray(depths_odd)
    depths_even = np.asarray(depths_even)

    mean_odd = np.mean(depths_odd)
    mean_even = np.mean(depths_even)

    std_odd = np.std(depths_odd) / np.sqrt(len(depths_odd))
    std_even = np.std(depths_even) / np.sqrt(len(depths_even))

    depth_diff = mean_odd - mean_even
    combined_std = np.sqrt(std_odd**2 + std_even**2)

    if combined_std < 1e-10:
        return 0.0, 0.0

    mismatch = np.abs(depth_diff) / combined_std

    return mismatch, depth_diff


def compute_all_statistics(chi2, periods, best_period_idx,
                           depth, duration, n_transits,
                           depths_per_transit=None, kernel_size=None,
                           chi2_null=None, chi2_best=None):
    """
    Compute all TLS statistics for a search result.

    Parameters
    ----------
    chi2 : array_like
        Chi-squared values at each trial period, in ascending period
        order (the SDE detrend assumes period-ordered neighbours)
    periods : array_like
        Trial periods (ascending)
    best_period_idx : int
        Index of best period
    depth : float
        Best-fit transit depth
    duration : float
        Best-fit transit duration
    n_transits : int
        Number of transits at best period
    depths_per_transit : array_like, optional
        Individual transit depths
    kernel_size : int, optional
        Running-median kernel for SDE detrending, passed through to
        :func:`signal_detection_efficiency`. Default (None) uses
        ``min(len(chi2)//10 forced odd, 91)``, following the fixed
        91-point kernel convention of ``transitleastsquares``.
    chi2_null : float, optional
        Constant-model chi-squared ``chi2_0`` for the SNR. Default
        (None) falls back to ``max(chi2)`` over the grid.
    chi2_best : float, optional
        Best-fit chi-squared for the SNR (the refined ``chi2_min`` on
        the batch path). Default (None) uses ``chi2[best_period_idx]``.

    Returns
    -------
    stats : dict
        Dictionary with all statistics:
        - SDE: Signal Detection Efficiency (see
          :func:`signal_detection_efficiency`)
        - SDE_raw: Raw SDE before detrending
        - SNR: ``sqrt(chi2_null - chi2_best)`` (delta-chi-squared
          significance; see :func:`signal_to_noise`)
        - power: Detrended signal residue spectrum
        - SR: Signal residue ``chi2_min / chi2``
        - odd_even_mismatch: Odd/even depth difference (if available)

        No 'FAP' key: see :func:`false_alarm_probability` for why the
        old heuristic was removed and ``tls_search_batch`` for the
        null-bootstrap alternative.
    """
    chi2 = np.asarray(chi2, dtype=np.float64)

    # Signal residue and SDE
    SDE, SDE_raw, power = signal_detection_efficiency(
        chi2, detrend=True, kernel_size=kernel_size)

    SR = signal_residue(chi2)

    # SNR (delta-chi2 of the best fit over the constant model)
    if chi2_null is None:
        chi2_null = np.max(chi2)
    if chi2_best is None:
        chi2_best = chi2[best_period_idx]
    SNR = signal_to_noise(depth, chi2_null=chi2_null, chi2_best=chi2_best)

    # Compile statistics
    stats = {
        'SDE': SDE,
        'SDE_raw': SDE_raw,
        'SNR': SNR,
        'power': power,
        'SR': SR,
        'best_period': periods[best_period_idx],
        'best_chi2': chi2[best_period_idx],
    }

    # Odd-even mismatch if per-transit depths available
    if depths_per_transit is not None and len(depths_per_transit) > 2:
        depths = np.asarray(depths_per_transit)
        n = len(depths)

        if n >= 4:  # Need at least 2 odd and 2 even
            depths_odd = depths[::2]
            depths_even = depths[1::2]

            mismatch, diff = odd_even_mismatch(depths_odd, depths_even)
            stats['odd_even_mismatch'] = mismatch
            stats['odd_even_depth_diff'] = diff
        else:
            stats['odd_even_mismatch'] = 0.0
            stats['odd_even_depth_diff'] = 0.0

    return stats


def compute_period_uncertainty(periods, chi2, best_idx, threshold=1.0):
    """
    Estimate period uncertainty using FWHM approach.

    Parameters
    ----------
    periods : array_like
        Trial periods, ascending (the neighbour walk assumes sorted
        periods; the search wrappers sort user grids before calling)
    chi2 : array_like
        Chi-squared values
    best_idx : int
        Index of minimum chi²
    threshold : float, optional
        Chi² increase threshold for FWHM (default: 1.0)

    Returns
    -------
    uncertainty : float
        Period uncertainty (half-width at threshold)

    Notes
    -----
    Finds the width of the chi² minimum at threshold above minimum.
    Default threshold=1 corresponds to 1σ for Gaussian errors.
    """
    periods = np.asarray(periods)
    chi2 = np.asarray(chi2)

    chi2_min = chi2[best_idx]
    chi2_thresh = chi2_min + threshold

    # Find points below threshold
    below = chi2 < chi2_thresh

    if not np.any(below):
        # If no points below threshold, use grid spacing
        if len(periods) > 1:
            return np.abs(periods[1] - periods[0])
        else:
            return 0.1 * periods[best_idx]

    # Find continuous region around best_idx
    # Walk left from best_idx
    left_idx = best_idx
    while left_idx > 0 and below[left_idx]:
        left_idx -= 1

    # Walk right from best_idx
    right_idx = best_idx
    while right_idx < len(periods) - 1 and below[right_idx]:
        right_idx += 1

    # Uncertainty is half the width
    width = periods[right_idx] - periods[left_idx]
    uncertainty = width / 2.0

    return uncertainty
