"""
Statistical calculations for Transit Least Squares.

Implements Signal Detection Efficiency (SDE), Signal-to-Noise Ratio (SNR),
False Alarm Probability (FAP), and related metrics.

References
----------
.. [1] Hippke & Heller (2019), A&A 623, A39
.. [2] Kovács et al. (2002), A&A 391, 369
"""

import numpy as np
from scipy import signal, stats


def signal_residue(chi2, chi2_null=None):
    """
    Calculate Signal Residue (SR).

    SR is the ratio of chi-squared values, normalized to [0, 1].
    SR = chi²_null / chi²_signal, where 1 = strongest signal.

    Parameters
    ----------
    chi2 : array_like
        Chi-squared values at each period
    chi2_null : float, optional
        Null hypothesis chi-squared (constant model)
        If None, uses maximum chi2 value

    Returns
    -------
    SR : ndarray
        Signal residue values [0, 1]

    Notes
    -----
    Higher SR values indicate stronger signals.
    SR = 1 means chi² is at its minimum (perfect fit).
    """
    chi2 = np.asarray(chi2)

    if chi2_null is None:
        chi2_null = np.max(chi2)

    SR = chi2_null / (chi2 + 1e-10)

    # Clip to [0, 1] range
    SR = np.clip(SR, 0, 1)

    return SR


def signal_detection_efficiency(chi2, chi2_null=None, detrend=True,
                                window_length=None):
    """
    Calculate Signal Detection Efficiency (SDE).

    SDE measures how many standard deviations above the noise
    the signal is. Higher SDE = more significant detection.

    Parameters
    ----------
    chi2 : array_like
        Chi-squared values at each period
    chi2_null : float, optional
        Null hypothesis chi-squared
    detrend : bool, optional
        Apply median filter detrending (default: True)
    window_length : int, optional
        Window length for median filter (default: len(chi2)//10)

    Returns
    -------
    SDE : float
        Signal detection efficiency (z-score)
    SDE_raw : float
        Raw SDE before detrending
    power : ndarray
        Detrended power spectrum (if detrend=True)

    Notes
    -----
    SDE is essentially a z-score:
    SDE = (1 - ⟨SR⟩) / σ(SR)

    Typical threshold: SDE > 7 for 1% false alarm probability
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
        SDE_raw = (1.0 - mean_SR) / std_SR

    # Detrend with median filter if requested
    if detrend:
        if window_length is None:
            window_length = max(len(SR) // 10, 3)
            # Ensure odd window
            if window_length % 2 == 0:
                window_length += 1

        # Apply median filter to remove trends
        SR_trend = signal.medfilt(SR, kernel_size=window_length)

        # Detrended signal residue
        SR_detrended = SR - SR_trend + np.median(SR)

        # Calculate SDE on detrended signal
        mean_SR_detrended = np.mean(SR_detrended)
        std_SR_detrended = np.std(SR_detrended)

        if std_SR_detrended < 1e-10:
            SDE = 0.0
        else:
            SDE = (1.0 - mean_SR_detrended) / std_SR_detrended

        power = SR_detrended
    else:
        SDE = SDE_raw
        power = SR

    return SDE, SDE_raw, power


def signal_to_noise(depth, depth_err=None, n_transits=1):
    """
    Calculate signal-to-noise ratio.

    Parameters
    ----------
    depth : float
        Transit depth
    depth_err : float, optional
        Uncertainty in depth. If None, estimated from Poisson statistics.
        **WARNING**: The default Poisson approximation is overly simplified
        and may not be accurate for real data with systematic noise, correlated
        errors, or stellar activity. Users should provide actual depth_err values
        computed from their data for more accurate SNR calculations.
    n_transits : int, optional
        Number of transits (default: 1)

    Returns
    -------
    snr : float
        Signal-to-noise ratio

    Notes
    -----
    SNR improves as sqrt(n_transits) for independent transits.

    The default depth_err estimation (depth / sqrt(n_transits)) assumes:
    - Pure Poisson (photon) noise
    - No systematic errors
    - Independent transits
    - White noise

    For realistic astrophysical data, these assumptions are rarely valid.
    Always provide depth_err when available for accurate results.
    """
    if depth_err is None:
        # Rough estimate from Poisson statistics
        # WARNING: This is a simplified approximation - see docstring
        depth_err = depth / np.sqrt(n_transits)

    if depth_err < 1e-10:
        return 0.0

    snr = depth / depth_err * np.sqrt(n_transits)

    return snr


def false_alarm_probability(SDE, method='empirical'):
    """
    Estimate False Alarm Probability from SDE.

    Parameters
    ----------
    SDE : float
        Signal Detection Efficiency
    method : str, optional
        Method for FAP estimation (default: 'empirical')
        - 'empirical': From Hippke & Heller calibration
        - 'gaussian': Assuming Gaussian noise

    Returns
    -------
    FAP : float
        False Alarm Probability

    Notes
    -----
    Empirical calibration from Hippke & Heller (2019):
    - SDE = 7 → FAP ≈ 1%
    - SDE = 9 → FAP ≈ 0.1%
    - SDE = 11 → FAP ≈ 0.01%
    """
    if method == 'gaussian':
        # Gaussian approximation: FAP = 1 - erf(SDE/sqrt(2))
        FAP = 1.0 - stats.norm.cdf(SDE)
    else:
        # Empirical calibration from Hippke & Heller (2019)
        # Rough approximation based on their Figure 5
        if SDE < 5:
            FAP = 1.0  # Very high FAP
        elif SDE < 7:
            FAP = 10 ** (-0.5 * (SDE - 5))  # ~10% at SDE=5, ~1% at SDE=7
        else:
            FAP = 10 ** (-(SDE - 5))  # Exponential decrease

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
                           depths_per_transit=None):
    """
    Compute all TLS statistics for a search result.

    Parameters
    ----------
    chi2 : array_like
        Chi-squared values at each period
    periods : array_like
        Trial periods
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

    Returns
    -------
    stats : dict
        Dictionary with all statistics:
        - SDE: Signal Detection Efficiency
        - SDE_raw: Raw SDE before detrending
        - SNR: Signal-to-noise ratio
        - FAP: False Alarm Probability
        - power: Detrended power spectrum
        - SR: Signal residue
        - odd_even_mismatch: Odd/even depth difference (if available)
    """
    # Signal residue and SDE
    SDE, SDE_raw, power = signal_detection_efficiency(chi2, detrend=True)

    SR = signal_residue(chi2)

    # SNR
    SNR = signal_to_noise(depth, n_transits=n_transits)

    # FAP
    FAP = false_alarm_probability(SDE)

    # Compile statistics
    stats = {
        'SDE': SDE,
        'SDE_raw': SDE_raw,
        'SNR': SNR,
        'FAP': FAP,
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
        Trial periods
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


def pink_noise_correction(snr, n_transits, correlation_length=1):
    """
    Correct SNR for correlated (pink) noise.

    Parameters
    ----------
    snr : float
        White noise SNR
    n_transits : int
        Number of transits
    correlation_length : float, optional
        Correlation length in transit durations (default: 1)

    Returns
    -------
    snr_pink : float
        Pink noise corrected SNR

    Notes
    -----
    Pink noise (correlated noise) reduces effective SNR because
    neighboring points are not independent.

    Correction factor ≈ sqrt(correlation_length / n_points_per_transit)
    """
    if correlation_length <= 0:
        return snr

    # Approximate correction
    correction = np.sqrt(correlation_length)
    snr_pink = snr / correction

    return snr_pink
