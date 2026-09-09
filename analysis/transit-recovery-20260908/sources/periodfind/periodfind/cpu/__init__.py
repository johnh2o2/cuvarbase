"""CPU implementations of period-finding algorithms.

Provides the same API as the CUDA-backed Cython wrappers (ce.pyx, aov.pyx,
ls.pyx), but uses a Rust+Rayon backend that runs on the CPU.
"""

import numpy as np
from periodfind_cpu import (
    calc_aov_batched,
    calc_aov_peaks_batched,
    calc_basic_stats_batched,
    calc_ce_batched,
    calc_ce_peaks_batched,
    calc_fourier_batched,
    calc_fpw_batched,
    calc_fpw_peaks_batched,
    calc_ls_batched,
    calc_ls_peaks_batched,
    calc_mf_batched,
    calc_mf_features_batched,
    calc_mf_peaks_batched,
    calc_mhf_batched,
    calc_mhf_peaks_batched,
    calc_mhf_per_k_batched,
    calc_vn_batched,
    calc_vn_peaks_batched,
    compute_dmdt_batched,
    find_top_peaks_batched,  # noqa: F401  (public API)
    remove_high_cadence_batched,
)

# BLS support is optional (requires periodfind_cpu built with BLS feature)
try:
    from periodfind_cpu import calc_bls_batched, calc_bls_peaks_batched
    _HAS_BLS = True
except ImportError:
    _HAS_BLS = False

from periodfind import Periodogram, Statistics
from periodfind._utils import ensure_float32, prepare_magnitudes, validate_inputs

N_PEAKS_DEFAULT = 32
MIN_DISTANCE_DEFAULT = 1


def _unravel_peaks(peak_indices, peak_values, periods, period_dts, n_peaks):
    """Convert flat peak indices + values into a list of lists of Statistics."""
    n_curves = peak_indices.shape[0]
    n_pdts = len(period_dts)
    results = []
    for ci in range(n_curves):
        curve_peaks = []
        for pi in range(n_peaks):
            flat_idx = peak_indices[ci, pi]
            if flat_idx < 0:
                break
            p_idx = int(flat_idx // n_pdts)
            d_idx = int(flat_idx % n_pdts)
            curve_peaks.append(
                Statistics(
                    params=[float(periods[p_idx]), float(period_dts[d_idx])],
                    value=float(peak_values[ci, pi]),
                    mean=0.0,
                    std=1.0,
                    median=0.0,
                    mad=1.0,
                )
            )
        results.append(curve_peaks)
    return results


class ConditionalEntropy:
    """Conditional Entropy based light curve analysis (CPU backend).

    Parameters
    ----------
    n_phase : int, default=10
        The number of phase bins in the histogram.
    n_mag : int, default=10
        The number of magnitude bins in the histogram.
    phase_bin_extent : int, default=1
        Effective width (in bins) of each phase bin (overlap/smoothing).
    mag_bin_extent : int, default=1
        Effective width (in bins) of each magnitude bin (overlap/smoothing).
    """

    def __init__(self, n_phase=10, n_mag=10, phase_bin_extent=1, mag_bin_extent=1):
        self.n_phase = n_phase
        self.n_mag = n_mag
        self.phase_bin_extent = phase_bin_extent
        self.mag_bin_extent = mag_bin_extent

    def calc(
        self,
        times,
        mags,
        periods,
        period_dts,
        output="stats",
        normalize=True,
        center=False,
        n_stats=1,
        significance_type="stdmean",
        n_peaks=N_PEAKS_DEFAULT,
        min_distance=MIN_DISTANCE_DEFAULT,
    ):
        """Runs Conditional Entropy calculations on a list of light curves.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times.
        mags : list of ndarray
            List of light curve magnitudes.
        periods : ndarray
            Array of trial periods (float32).
        period_dts : ndarray
            Array of trial period time derivatives (float32).
        output : {'stats', 'periodogram', 'peaks'}, default='stats'
            Type of output to return.  ``'peaks'`` uses a streaming greedy
            algorithm to return the top *n_peaks* without materialising the
            full 3-D periodogram array, saving memory on large grids.
        normalize : bool, default=True
            Whether to normalize magnitudes to (0, 1).
        center : bool, default=False
            Whether to center magnitudes to zero mean.
        n_stats : int, default=1
            Number of top Statistics to return.
        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Significance metric.
        n_peaks : int, default=32
            Number of peaks to keep per light curve (used when output='peaks').
        min_distance : int, default=1
            Minimum distance (in flattened period grid samples) between
            accepted peaks.

        Returns
        -------
        list of Statistics, list of Periodogram, or list of list of Statistics
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")

        mags_use = prepare_magnitudes(mags, center, normalize)

        if output == "peaks":
            idx, val = calc_ce_peaks_batched(
                times,
                mags_use,
                periods,
                period_dts,
                self.n_phase,
                self.n_mag,
                self.phase_bin_extent,
                self.mag_bin_extent,
                n_peaks,
                min_distance,
            )
            return _unravel_peaks(idx, val, periods, period_dts, n_peaks)

        ces_ndarr = calc_ce_batched(
            times,
            mags_use,
            periods,
            period_dts,
            self.n_phase,
            self.n_mag,
            self.phase_bin_extent,
            self.mag_bin_extent,
        )

        if output == "stats":
            all_stats = []
            for i in range(len(times)):
                stats = Statistics.statistics_from_data(
                    ces_ndarr[i],
                    [periods, period_dts],
                    False,
                    n=n_stats,
                    significance_type=significance_type,
                )
                all_stats.append(stats)
            return all_stats
        elif output == "periodogram":
            return [Periodogram(data, [periods, period_dts], False) for data in ces_ndarr]
        else:
            raise NotImplementedError(
                f'Output type "{output}" is not implemented. '
                f'Use "stats", "periodogram", or "peaks".'
            )


class AOV:
    """Analysis-of-Variance based light curve analysis (CPU backend).

    Parameters
    ----------
    n_phase : int, default=10
        The number of phase bins.
    phase_bin_extent : int, default=1
        Effective width (in bins) of each phase bin (overlap/smoothing).
    """

    def __init__(self, n_phase=10, phase_bin_extent=1):
        self.n_phase = n_phase
        self.phase_bin_extent = phase_bin_extent

    def calc(
        self,
        times,
        mags,
        periods,
        period_dts,
        output="stats",
        normalize=False,
        center=False,
        n_stats=1,
        significance_type="stdmean",
        n_peaks=N_PEAKS_DEFAULT,
        min_distance=MIN_DISTANCE_DEFAULT,
    ):
        """Runs Analysis-of-Variance calculations on a list of light curves.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times.
        mags : list of ndarray
            List of light curve magnitudes.
        periods : ndarray
            Array of trial periods (float32).
        period_dts : ndarray
            Array of trial period time derivatives (float32).
        output : {'stats', 'periodogram', 'peaks'}, default='stats'
            Type of output to return.
        normalize : bool, default=False
            Whether to normalize magnitudes to (0, 1).
        center : bool, default=False
            Whether to center magnitudes to zero mean.
        n_stats : int, default=1
            Number of top Statistics to return.
        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Significance metric.
        n_peaks : int, default=32
            Number of peaks to keep per light curve (used when output='peaks').
        min_distance : int, default=1
            Minimum distance between accepted peaks.

        Returns
        -------
        list of Statistics, list of Periodogram, or list of list of Statistics
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")

        mags_use = prepare_magnitudes(mags, center, normalize)

        if output == "peaks":
            idx, val = calc_aov_peaks_batched(
                times,
                mags_use,
                periods,
                period_dts,
                self.n_phase,
                self.phase_bin_extent,
                n_peaks,
                min_distance,
            )
            return _unravel_peaks(idx, val, periods, period_dts, n_peaks)

        aovs_ndarr = calc_aov_batched(
            times,
            mags_use,
            periods,
            period_dts,
            self.n_phase,
            self.phase_bin_extent,
        )

        if output == "stats":
            all_stats = []
            for i in range(len(times)):
                stats = Statistics.statistics_from_data(
                    aovs_ndarr[i],
                    [periods, period_dts],
                    True,
                    n=n_stats,
                    significance_type=significance_type,
                )
                all_stats.append(stats)
            return all_stats
        elif output == "periodogram":
            return [Periodogram(data, [periods, period_dts], True) for data in aovs_ndarr]
        else:
            raise NotImplementedError(
                f'Output type "{output}" is not implemented. '
                f'Use "stats", "periodogram", or "peaks".'
            )


class LombScargle:
    """Lomb-Scargle periodogram light curve analysis (CPU backend)."""

    def __init__(self):
        pass

    def calc(
        self,
        times,
        mags,
        periods,
        period_dts,
        output="stats",
        normalize=False,
        center=True,
        n_stats=1,
        significance_type="stdmean",
        n_peaks=N_PEAKS_DEFAULT,
        min_distance=MIN_DISTANCE_DEFAULT,
    ):
        """Runs Lomb-Scargle calculations on a list of light curves.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times.
        mags : list of ndarray
            List of light curve magnitudes.
        periods : ndarray
            Array of trial periods (float32).
        period_dts : ndarray
            Array of trial period time derivatives (float32).
        output : {'stats', 'periodogram', 'peaks'}, default='stats'
            Type of output to return.
        normalize : bool, default=False
            Whether to normalize magnitudes to (0, 1).
        center : bool, default=True
            Whether to center magnitudes to zero mean.
        n_stats : int, default=1
            Number of top Statistics to return.
        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Significance metric.
        n_peaks : int, default=32
            Number of peaks to keep per light curve (used when output='peaks').
        min_distance : int, default=1
            Minimum distance between accepted peaks.

        Returns
        -------
        list of Statistics, list of Periodogram, or list of list of Statistics
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")

        mags_use = prepare_magnitudes(mags, center, normalize)

        if output == "peaks":
            idx, val = calc_ls_peaks_batched(
                times,
                mags_use,
                periods,
                period_dts,
                n_peaks,
                min_distance,
            )
            return _unravel_peaks(idx, val, periods, period_dts, n_peaks)

        ls_ndarr = calc_ls_batched(
            times,
            mags_use,
            periods,
            period_dts,
        )

        if output == "stats":
            all_stats = []
            for i in range(len(times)):
                stats = Statistics.statistics_from_data(
                    ls_ndarr[i],
                    [periods, period_dts],
                    True,
                    n=n_stats,
                    significance_type=significance_type,
                )
                all_stats.append(stats)
            return all_stats
        elif output == "periodogram":
            return [Periodogram(data, [periods, period_dts], True) for data in ls_ndarr]
        else:
            raise NotImplementedError(
                f'Output type "{output}" is not implemented. '
                f'Use "stats", "periodogram", or "peaks".'
            )


class FPW:
    """Fast Phase-folding Weighted (FPW) light curve analysis (CPU backend).

    Computes the FPW statistic (Finkbeiner et al. 2025), a weighted
    chi-squared reduction that supports per-point uncertainties.

    Parameters
    ----------
    n_bins : int, default=10
        The number of phase bins.
    """

    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def calc(
        self,
        times,
        mags,
        periods,
        period_dts,
        errs=None,
        output="stats",
        normalize=False,
        center=False,
        n_stats=1,
        significance_type="stdmean",
        n_peaks=N_PEAKS_DEFAULT,
        min_distance=MIN_DISTANCE_DEFAULT,
    ):
        """Runs FPW calculations on a list of light curves.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times.
        mags : list of ndarray
            List of light curve magnitudes.
        periods : ndarray
            Array of trial periods (float32).
        period_dts : ndarray
            Array of trial period time derivatives (float32).
        errs : list of ndarray or None, default=None
            List of per-point uncertainties (standard deviations).
            If None, uniform uncertainties of 1.0 are assumed.
        output : {'stats', 'periodogram', 'peaks'}, default='stats'
            Type of output to return.
        normalize : bool, default=False
            Unused (accepted for API consistency).
        center : bool, default=False
            Unused (accepted for API consistency).
        n_stats : int, default=1
            Number of top Statistics to return.
        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Significance metric.
        n_peaks : int, default=32
            Number of peaks to keep per light curve (used when output='peaks').
        min_distance : int, default=1
            Minimum distance between accepted peaks.

        Returns
        -------
        list of Statistics, list of Periodogram, or list of list of Statistics
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")

        # Handle uncertainties
        if errs is None:
            errs = [np.ones(len(t), dtype=np.float32) for t in times]
        else:
            ensure_float32(errs, "errs")
            if len(errs) != len(times):
                raise ValueError(
                    f"errs must have the same number of arrays as times, "
                    f"got {len(errs)} and {len(times)}"
                )
            for i, (e, t) in enumerate(zip(errs, times)):
                if len(e) != len(t):
                    raise ValueError(
                        f"errs[{i}] and times[{i}] have different lengths: {len(e)} vs {len(t)}"
                    )

        if output == "peaks":
            idx, val = calc_fpw_peaks_batched(
                times,
                mags,
                errs,
                periods,
                period_dts,
                self.n_bins,
                n_peaks,
                min_distance,
            )
            return _unravel_peaks(idx, val, periods, period_dts, n_peaks)

        fpw_ndarr = calc_fpw_batched(
            times,
            mags,
            errs,
            periods,
            period_dts,
            self.n_bins,
        )

        if output == "stats":
            all_stats = []
            for i in range(len(times)):
                stats = Statistics.statistics_from_data(
                    fpw_ndarr[i],
                    [periods, period_dts],
                    True,
                    n=n_stats,
                    significance_type=significance_type,
                )
                all_stats.append(stats)
            return all_stats
        elif output == "periodogram":
            return [Periodogram(data, [periods, period_dts], True) for data in fpw_ndarr]
        else:
            raise NotImplementedError(
                f'Output type "{output}" is not implemented. '
                f'Use "stats", "periodogram", or "peaks".'
            )


class BoxLeastSquares:
    """Box Least Squares (BLS) transit-detection light curve analysis (CPU backend).

    Searches for periodic box-shaped (flat-bottom) dips in time-series data,
    as described by Kovács, Zucker & Mazeh (2002).

    Parameters
    ----------
    n_bins : int, default=50
        The number of phase bins.
    qmin : float, default=0.01
        Minimum transit duration as a fraction of the period.
    qmax : float, default=0.5
        Maximum transit duration as a fraction of the period.
    """

    def __init__(self, n_bins=50, qmin=0.01, qmax=0.5):
        if not _HAS_BLS:
            raise ImportError(
                "BLS support requires periodfind_cpu compiled with BLS. "
                "Rebuild periodfind_cpu from source to enable BLS."
            )
        self.n_bins = n_bins
        self.qmin = qmin
        self.qmax = qmax

    def calc(
        self,
        times,
        mags,
        periods,
        period_dts,
        errs=None,
        output="stats",
        normalize=False,
        center=False,
        n_stats=1,
        significance_type="stdmean",
        n_peaks=N_PEAKS_DEFAULT,
        min_distance=MIN_DISTANCE_DEFAULT,
    ):
        """Runs BLS calculations on a list of light curves.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times.
        mags : list of ndarray
            List of light curve magnitudes.
        periods : ndarray
            Array of trial periods (float32).
        period_dts : ndarray
            Array of trial period time derivatives (float32).
        errs : list of ndarray or None, default=None
            List of per-point uncertainties (standard deviations).
            If None, uniform uncertainties of 1.0 are assumed.
        output : {'stats', 'periodogram', 'peaks'}, default='stats'
            Type of output to return.
        normalize : bool, default=False
            Unused (accepted for API consistency).
        center : bool, default=False
            Unused (accepted for API consistency).
        n_stats : int, default=1
            Number of top Statistics to return.
        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Significance metric.
        n_peaks : int, default=32
            Number of peaks to keep per light curve (used when output='peaks').
        min_distance : int, default=1
            Minimum distance between accepted peaks.

        Returns
        -------
        list of Statistics, list of Periodogram, or list of list of Statistics
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")

        # Handle uncertainties
        if errs is None:
            errs = [np.ones(len(t), dtype=np.float32) for t in times]
        else:
            ensure_float32(errs, "errs")
            if len(errs) != len(times):
                raise ValueError(
                    f"errs must have the same number of arrays as times, "
                    f"got {len(errs)} and {len(times)}"
                )
            for i, (e, t) in enumerate(zip(errs, times)):
                if len(e) != len(t):
                    raise ValueError(
                        f"errs[{i}] and times[{i}] have different lengths: {len(e)} vs {len(t)}"
                    )

        if not _HAS_BLS:
            raise ImportError(
                "BLS support requires periodfind_cpu compiled with BLS feature. "
                "Rebuild periodfind_cpu from source to enable BLS."
            )

        if output == "peaks":
            idx, val = calc_bls_peaks_batched(
                times,
                mags,
                errs,
                periods,
                period_dts,
                self.n_bins,
                self.qmin,
                self.qmax,
                n_peaks,
                min_distance,
            )
            return _unravel_peaks(idx, val, periods, period_dts, n_peaks)

        bls_ndarr = calc_bls_batched(
            times,
            mags,
            errs,
            periods,
            period_dts,
            self.n_bins,
            self.qmin,
            self.qmax,
        )

        if output == "stats":
            all_stats = []
            for i in range(len(times)):
                stats = Statistics.statistics_from_data(
                    bls_ndarr[i],
                    [periods, period_dts],
                    True,
                    n=n_stats,
                    significance_type=significance_type,
                )
                all_stats.append(stats)
            return all_stats
        elif output == "periodogram":
            return [Periodogram(data, [periods, period_dts], True) for data in bls_ndarr]
        else:
            raise NotImplementedError(
                f'Output type "{output}" is not implemented. '
                f'Use "stats", "periodogram", or "peaks".'
            )


class MatchedFilter:
    """Matched Filter morphology scoring (CPU backend).

    Phase-folds light curves, bins into a profile, and correlates against
    template shapes (sawtooth, sinusoidal, eclipsing) via circular
    cross-correlation.  The combined score (max_corr * R^2 * coverage)
    serves as the periodogram statistic.

    Parameters
    ----------
    num_bins : int, default=20
        The number of phase bins for the folded profile.
    """

    MF_FEATURE_NAMES = [
        "best_sawtooth",
        "best_sinusoidal",
        "best_eclipsing",
        "R2",
        "amp_snr",
        "n_filled",
        "combined",
    ]

    def __init__(self, num_bins=20):
        self.num_bins = num_bins

    def calc(
        self,
        times,
        mags,
        periods,
        period_dts,
        errs=None,
        output="stats",
        normalize=False,
        center=False,
        n_stats=1,
        significance_type="stdmean",
        n_peaks=N_PEAKS_DEFAULT,
        min_distance=MIN_DISTANCE_DEFAULT,
    ):
        """Runs Matched Filter periodogram on a list of light curves.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times.
        mags : list of ndarray
            List of light curve magnitudes.
        periods : ndarray
            Array of trial periods (float32).
        period_dts : ndarray
            Array of trial period time derivatives (float32).
        errs : list of ndarray or None, default=None
            List of per-point uncertainties.
            If None, uniform uncertainties of 1.0 are assumed.
        output : {'stats', 'periodogram', 'peaks'}, default='stats'
            Type of output to return.
        normalize : bool, default=False
            Unused (accepted for API consistency).
        center : bool, default=False
            Unused (accepted for API consistency).
        n_stats : int, default=1
            Number of top Statistics to return.
        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Significance metric.
        n_peaks : int, default=32
            Number of peaks to keep per light curve (used when output='peaks').
        min_distance : int, default=1
            Minimum distance between accepted peaks.

        Returns
        -------
        list of Statistics, list of Periodogram, or list of list of Statistics
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")

        if errs is None:
            errs = [np.ones(len(t), dtype=np.float32) for t in times]
        else:
            ensure_float32(errs, "errs")
            if len(errs) != len(times):
                raise ValueError(
                    f"errs must have the same number of arrays as times, "
                    f"got {len(errs)} and {len(times)}"
                )
            for i, (e, t) in enumerate(zip(errs, times)):
                if len(e) != len(t):
                    raise ValueError(
                        f"errs[{i}] and times[{i}] have different lengths: "
                        f"{len(e)} vs {len(t)}"
                    )

        if output == "peaks":
            idx, val = calc_mf_peaks_batched(
                times,
                mags,
                errs,
                periods,
                period_dts,
                self.num_bins,
                n_peaks,
                min_distance,
            )
            return _unravel_peaks(idx, val, periods, period_dts, n_peaks)

        mf_ndarr = calc_mf_batched(
            times,
            mags,
            errs,
            periods,
            period_dts,
            self.num_bins,
        )

        if output == "stats":
            all_stats = []
            for i in range(len(times)):
                stats = Statistics.statistics_from_data(
                    mf_ndarr[i],
                    [periods, period_dts],
                    True,
                    n=n_stats,
                    significance_type=significance_type,
                )
                all_stats.append(stats)
            return all_stats
        elif output == "periodogram":
            return [Periodogram(data, [periods, period_dts], True) for data in mf_ndarr]
        else:
            raise NotImplementedError(
                f'Output type "{output}" is not implemented. '
                f'Use "stats", "periodogram", or "peaks".'
            )

    def calc_features(self, times, mags, errs, periods):
        """Compute detailed MF features at given periods (one per curve).

        Parameters
        ----------
        times : list of ndarray (float32)
            List of light curve times.
        mags : list of ndarray (float32)
            List of light curve magnitudes.
        errs : list of ndarray (float32)
            List of per-point uncertainties.
        periods : ndarray (float32)
            Array of periods, one per curve.

        Returns
        -------
        ndarray of shape (n_curves, 7)
            Features: [best_sawtooth, best_sinusoidal, best_eclipsing,
                       R², amp_snr, n_filled, combined]
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")
        ensure_float32(errs, "errs")

        if len(errs) != len(times):
            raise ValueError(
                f"errs must have the same number of arrays as times, "
                f"got {len(errs)} and {len(times)}"
            )

        periods = np.asarray(periods, dtype=np.float32)
        if periods.ndim != 1 or len(periods) != len(times):
            raise ValueError(
                f"periods must be a 1D array with one entry per curve, "
                f"got shape {periods.shape} for {len(times)} curves"
            )

        return calc_mf_features_batched(times, mags, errs, periods, self.num_bins)


class MultiHarmonicFourier:
    """Multi-Harmonic Fourier periodogram (CPU backend).

    Fits Fourier models with 0..max_harmonics terms at every trial period
    and uses BIC model selection.  The score is ΔBIC = BIC_flat - BIC_best
    (higher = more periodic).  Captures non-sinusoidal shapes (sawtooth,
    eclipsing) that single-sinusoid Lomb-Scargle misses.

    Parameters
    ----------
    max_harmonics : int, default=5
        Maximum number of Fourier harmonics to try (1–5).
    """

    def __init__(self, max_harmonics=5):
        if not 1 <= max_harmonics <= 5:
            raise ValueError(
                f"max_harmonics must be between 1 and 5, got {max_harmonics}"
            )
        self.max_harmonics = max_harmonics

    def calc(
        self,
        times,
        mags,
        periods,
        period_dts,
        errs=None,
        output="stats",
        normalize=False,
        center=False,
        n_stats=1,
        significance_type="stdmean",
        n_peaks=N_PEAKS_DEFAULT,
        min_distance=MIN_DISTANCE_DEFAULT,
    ):
        """Runs Multi-Harmonic Fourier periodogram on a list of light curves.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times.
        mags : list of ndarray
            List of light curve magnitudes.
        periods : ndarray
            Array of trial periods (float32).
        period_dts : ndarray
            Array of trial period time derivatives (float32).
        errs : list of ndarray or None, default=None
            List of per-point uncertainties (standard deviations).
            If None, uniform uncertainties of 1.0 are assumed.
        output : {'stats', 'periodogram', 'peaks'}, default='stats'
            Type of output to return.
        normalize : bool, default=False
            Unused (accepted for API consistency).
        center : bool, default=False
            Unused (accepted for API consistency).
        n_stats : int, default=1
            Number of top Statistics to return.
        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Significance metric.
        n_peaks : int, default=32
            Number of peaks to keep per light curve (used when output='peaks').
        min_distance : int, default=1
            Minimum distance between accepted peaks.

        Returns
        -------
        list of Statistics, list of Periodogram, or list of list of Statistics
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")

        if errs is None:
            errs = [np.ones(len(t), dtype=np.float32) for t in times]
        else:
            ensure_float32(errs, "errs")
            if len(errs) != len(times):
                raise ValueError(
                    f"errs must have the same number of arrays as times, "
                    f"got {len(errs)} and {len(times)}"
                )
            for i, (e, t) in enumerate(zip(errs, times)):
                if len(e) != len(t):
                    raise ValueError(
                        f"errs[{i}] and times[{i}] have different lengths: "
                        f"{len(e)} vs {len(t)}"
                    )

        if output == "peaks":
            idx, val = calc_mhf_peaks_batched(
                times,
                mags,
                errs,
                periods,
                period_dts,
                self.max_harmonics,
                n_peaks,
                min_distance,
            )
            return _unravel_peaks(idx, val, periods, period_dts, n_peaks)

        mhf_ndarr = calc_mhf_batched(
            times,
            mags,
            errs,
            periods,
            period_dts,
            self.max_harmonics,
        )

        if output == "stats":
            all_stats = []
            for i in range(len(times)):
                stats = Statistics.statistics_from_data(
                    mhf_ndarr[i],
                    [periods, period_dts],
                    True,
                    n=n_stats,
                    significance_type=significance_type,
                )
                all_stats.append(stats)
            return all_stats
        elif output == "periodogram":
            return [Periodogram(data, [periods, period_dts], True) for data in mhf_ndarr]
        else:
            raise NotImplementedError(
                f'Output type "{output}" is not implemented. '
                f'Use "stats", "periodogram", or "peaks".'
            )

    def calc_per_k(self, times, mags, errs, periods, period_dts=None):
        """Compute per-harmonic-level ΔBIC at given periods (one per curve).

        Evaluates the MHF model at a single period per curve and returns
        the ΔBIC for each harmonic level K=0..max_harmonics, plus the
        BIC-optimal K.  This is useful for morphology discrimination:
        ΔBIC(K=3) >> ΔBIC(K=1) indicates non-sinusoidal shapes.

        Parameters
        ----------
        times : list of ndarray (float32)
            List of light curve times.
        mags : list of ndarray (float32)
            List of light curve magnitudes.
        errs : list of ndarray (float32)
            List of per-point uncertainties.
        periods : ndarray (float32)
            Array of periods, one per curve.
        period_dts : ndarray (float32) or None
            Array of period time derivatives, one per curve.
            If None, zeros are used.

        Returns
        -------
        ndarray of shape (n_curves, max_harmonics + 2)
            Each row: [ΔBIC_k0, ΔBIC_k1, ..., ΔBIC_kN, best_k]
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")
        ensure_float32(errs, "errs")

        if len(errs) != len(times):
            raise ValueError(
                f"errs must have the same number of arrays as times, "
                f"got {len(errs)} and {len(times)}"
            )
        for i, (e, t) in enumerate(zip(errs, times)):
            if len(e) != len(t):
                raise ValueError(
                    f"errs[{i}] and times[{i}] have different lengths: "
                    f"{len(e)} vs {len(t)}"
                )

        periods = np.asarray(periods, dtype=np.float32)
        if periods.ndim != 1 or len(periods) != len(times):
            raise ValueError(
                f"periods must be a 1D array with one entry per curve, "
                f"got shape {periods.shape} for {len(times)} curves"
            )

        if period_dts is None:
            period_dts = np.zeros(len(times), dtype=np.float32)
        else:
            period_dts = np.asarray(period_dts, dtype=np.float32)
            if period_dts.ndim != 1 or len(period_dts) != len(times):
                raise ValueError(
                    f"period_dts must be a 1D array with one entry per curve, "
                    f"got shape {period_dts.shape} for {len(times)} curves"
                )

        return calc_mhf_per_k_batched(
            times, mags, errs, periods, period_dts, self.max_harmonics
        )


class ViterbiNarrowband:
    """Viterbi Narrowband period-finding score (CPU backend).

    Builds the same 2D phase-magnitude histogram as Conditional Entropy,
    then runs a circular Viterbi algorithm to find the most likely narrow
    path through phase-mag space.  The score is the fraction of histogram
    mass concentrated within ``margin`` bins of the optimal path.

    Parameters
    ----------
    n_phase : int, default=20
        The number of phase bins in the histogram.
    n_mag : int, default=20
        The number of magnitude bins in the histogram.
    phase_bin_extent : int, default=1
        Effective width (in bins) of each phase bin (overlap/smoothing).
    mag_bin_extent : int, default=1
        Effective width (in bins) of each magnitude bin (overlap/smoothing).
    bandwidth : int, default=2
        Maximum magnitude-bin shift between adjacent phase bins.
    margin : int, default=1
        Number of bins around the Viterbi path counted for the
        concentration ratio.
    """

    def __init__(
        self,
        n_phase=20,
        n_mag=20,
        phase_bin_extent=1,
        mag_bin_extent=1,
        bandwidth=2,
        margin=1,
    ):
        self.n_phase = n_phase
        self.n_mag = n_mag
        self.phase_bin_extent = phase_bin_extent
        self.mag_bin_extent = mag_bin_extent
        self.bandwidth = bandwidth
        self.margin = margin

    def calc(
        self,
        times,
        mags,
        periods,
        period_dts,
        output="stats",
        normalize=True,
        center=False,
        n_stats=1,
        significance_type="stdmean",
        n_peaks=N_PEAKS_DEFAULT,
        min_distance=MIN_DISTANCE_DEFAULT,
    ):
        """Runs Viterbi Narrowband calculations on a list of light curves.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times.
        mags : list of ndarray
            List of light curve magnitudes.
        periods : ndarray
            Array of trial periods (float32).
        period_dts : ndarray
            Array of trial period time derivatives (float32).
        output : {'stats', 'periodogram', 'peaks'}, default='stats'
            Type of output to return.
        normalize : bool, default=True
            Whether to normalize magnitudes to (0, 1).
        center : bool, default=False
            Whether to center magnitudes to zero mean.
        n_stats : int, default=1
            Number of top Statistics to return.
        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Significance metric.
        n_peaks : int, default=32
            Number of peaks to keep per light curve (used when output='peaks').
        min_distance : int, default=1
            Minimum distance between accepted peaks.

        Returns
        -------
        list of Statistics, list of Periodogram, or list of list of Statistics
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")

        mags_use = prepare_magnitudes(mags, center, normalize)

        if output == "peaks":
            idx, val = calc_vn_peaks_batched(
                times,
                mags_use,
                periods,
                period_dts,
                self.n_phase,
                self.n_mag,
                self.phase_bin_extent,
                self.mag_bin_extent,
                self.bandwidth,
                self.margin,
                n_peaks,
                min_distance,
            )
            return _unravel_peaks(idx, val, periods, period_dts, n_peaks)

        vn_ndarr = calc_vn_batched(
            times,
            mags_use,
            periods,
            period_dts,
            self.n_phase,
            self.n_mag,
            self.phase_bin_extent,
            self.mag_bin_extent,
            self.bandwidth,
            self.margin,
        )

        if output == "stats":
            all_stats = []
            for i in range(len(times)):
                stats = Statistics.statistics_from_data(
                    vn_ndarr[i],
                    [periods, period_dts],
                    True,
                    n=n_stats,
                    significance_type=significance_type,
                )
                all_stats.append(stats)
            return all_stats
        elif output == "periodogram":
            return [Periodogram(data, [periods, period_dts], True) for data in vn_ndarr]
        else:
            raise NotImplementedError(
                f'Output type "{output}" is not implemented. '
                f'Use "stats", "periodogram", or "peaks".'
            )


class FourierDecomposition:
    """Fourier decomposition via weighted linear least-squares (CPU backend).

    Computes Fourier features for light curves given pre-determined periods.
    Uses BIC model selection over 0–5 harmonics.

    Returns 14 features per curve:
        [power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]
    """

    def calc(self, times, mags, errs, periods):
        """Run Fourier decomposition on a list of light curves.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times (float32).
        mags : list of ndarray
            List of light curve magnitudes (float32).
        errs : list of ndarray
            List of per-point uncertainties (float32).
        periods : ndarray
            Array of periods, one per curve (float32).

        Returns
        -------
        ndarray of shape (n_curves, 14)
            Fourier features for each curve.
        """
        validate_inputs(times, mags)
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")
        ensure_float32(errs, "errs")

        if len(errs) != len(times):
            raise ValueError(
                f"errs must have the same number of arrays as times, "
                f"got {len(errs)} and {len(times)}"
            )
        for i, (e, t) in enumerate(zip(errs, times)):
            if len(e) != len(t):
                raise ValueError(
                    f"errs[{i}] and times[{i}] have different lengths: {len(e)} vs {len(t)}"
                )

        periods = np.asarray(periods, dtype=np.float32)
        if periods.ndim != 1 or len(periods) != len(times):
            raise ValueError(
                f"periods must be a 1D array with one entry per curve, "
                f"got shape {periods.shape} for {len(times)} curves"
            )

        return calc_fourier_batched(times, mags, errs, periods)


class DmDt:
    """dm-dt histogram feature extractor (CPU backend).

    Computes L2-normalised 2D histograms of pairwise magnitude differences
    vs. time differences for each light curve.
    """

    def calc(self, times, mags, dt_edges, dm_edges):
        """Compute dm-dt histograms for a batch of light curves.

        Parameters
        ----------
        times : list of ndarray (float32)
            List of light curve times.
        mags : list of ndarray (float32)
            List of light curve magnitudes.
        dt_edges : ndarray (float32)
            Sorted bin edges for time differences.
        dm_edges : ndarray (float32)
            Sorted bin edges for magnitude differences.

        Returns
        -------
        ndarray of shape (n_curves, n_dm_bins, n_dt_bins)
        """
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")
        dt_edges = np.asarray(dt_edges, dtype=np.float32)
        dm_edges = np.asarray(dm_edges, dtype=np.float32)
        return compute_dmdt_batched(times, mags, dt_edges, dm_edges)


class BasicStats:
    """Light curve basic statistics extractor (CPU backend).

    Computes 22 summary statistics per light curve.
    """

    STAT_NAMES = [
        "N",
        "median",
        "wmean",
        "chi2red",
        "RoMS",
        "wstd",
        "NormPeaktoPeakamp",
        "NormExcessVar",
        "medianAbsDev",
        "iqr",
        "i60r",
        "i70r",
        "i80r",
        "i90r",
        "skew",
        "smallkurt",
        "invNeumann",
        "WelchI",
        "StetsonJ",
        "StetsonK",
        "AD",
        "SW",
    ]

    def calc(self, times, mags, errs):
        """Compute basic statistics for a batch of light curves.

        Parameters
        ----------
        times : list of ndarray (float32)
            List of light curve times.
        mags : list of ndarray (float32)
            List of light curve magnitudes.
        errs : list of ndarray (float32)
            List of per-point magnitude uncertainties.

        Returns
        -------
        ndarray of shape (n_curves, 22)
        """
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")
        ensure_float32(errs, "errs")
        return calc_basic_stats_batched(times, mags, errs)


class RemoveHighCadence:
    """High-cadence point removal (CPU backend).

    Filters light curves to keep only points separated by at least
    a given cadence.
    """

    def __init__(self, cadence_minutes=30.0):
        self.cadence_minutes = cadence_minutes

    def calc(self, times, mags, errs):
        """Remove high-cadence points from a batch of light curves.

        Parameters
        ----------
        times : list of ndarray (float32)
            List of light curve times.
        mags : list of ndarray (float32)
            List of light curve magnitudes.
        errs : list of ndarray (float32)
            List of per-point magnitude uncertainties.

        Returns
        -------
        list of (ndarray, ndarray, ndarray)
            Filtered (times, mags, errs) tuples.
        """
        ensure_float32(times, "times")
        ensure_float32(mags, "mags")
        ensure_float32(errs, "errs")
        return remove_high_cadence_batched(times, mags, errs, self.cadence_minutes)
