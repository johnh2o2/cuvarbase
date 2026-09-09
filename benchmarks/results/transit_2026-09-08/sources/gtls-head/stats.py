from __future__ import division, print_function
import numpy as np
from os import path
from .import constants as constants
from .helpers import running_median, transit_mask
from tqdm import tqdm
# from core import fold


def FAP(SDE):
    """Returns FAP (False Alarm Probability) for a given SDE"""
    data = np.genfromtxt(
        path.join(constants.resources_dir, "fap.csv"),
        dtype="f8, f8",
        delimiter=",",
        names=["FAP", "SDE"],
    )
    return data["FAP"][np.argmax(data["SDE"] > SDE)]


def rp_rs_from_depth(depth, law, params):
    """Takes the maximum transit depth, limb-darkening law and parameters
    Returns R_P / R_S (ratio of planetary to stellar radius)
    Source: Heller 2019, https://arxiv.org/abs/1901.01730"""

    # Validations:
    # - LD law must exist
    # - All parameters must be floats or ints
    # - All parameters must be given in the correct quanitity for the law

    if len(params) == 1:
        params = float(params[0])

    if not isinstance(params, (float, int)) and not all(
        isinstance(x, (float, int)) for x in params
    ):
        raise ValueError("All limb-darkening parameters must be numbers")

    laws = "linear, quadratic, squareroot, logarithmic, nonlinear"
    if law not in laws:
        raise ValueError("Please provide a supported limb-darkening law:", laws)

    if law == "linear" and not isinstance(params, float):
        raise ValueError("Please provide exactly one parameter")

    if law in "quadratic, logarithmic, squareroot" and len(params) != 2:
        raise ValueError("Please provide exactly two limb-darkening parameters")

    if law == "nonlinear" and len(params) != 4:
        raise ValueError("Please provide exactly four limb-darkening parameters")

    # Actual calculations of the return value
    if law == "linear":
        return (depth * (1 - params / 3)) ** (1 / 2)

    if law == "quadratic":
        return (depth * (1 - params[0] / 3 - params[1] / 6)) ** (1 / 2)

    if law == "squareroot":
        return (depth * (1 - params[0] / 3 - params[1] / 5)) ** (1 / 2)

    if law == "logarithmic":
        return (depth * (1 + 2 * params[1] / 9 - params[0] / 3)) ** (1 / 2)

    if law == "nonlinear":
        return (
            depth
            * (1 - params[0] / 5 - params[1] / 3 - 3 * params[2] / 7 - params[3] / 2)
        ) ** (1 / 2)

def period_uncertainty(periods, power):
    # Determine estimate for uncertainty in period
    # Method: Full width at half maximum
    try:
        # Upper limit
        index_highest_power = np.argmax(power)
        idx = index_highest_power
        while True:
            idx += 1
            if power[idx] <= 0.5 * power[index_highest_power]:
                idx_upper = idx
                break
        # Lower limit
        idx = index_highest_power
        while True:
            idx -= 1
            if power[idx] <= 0.5 * power[index_highest_power]:
                idx_lower = idx
                break
        period_uncertainty = 0.5 * (periods[idx_upper] - periods[idx_lower])
    except:
        period_uncertainty = float("inf")
    return period_uncertainty

def spectra(chi2, oversampling_factor):
    SR = np.min(chi2) / chi2
    SDE_raw = (1 - np.mean(SR)) / np.std(SR)

    # Scale SDE_power from 0 to SDE_raw
    power_raw = SR - np.mean(SR)  # shift down to the mean being zero
    scale = SDE_raw / np.max(power_raw)  # scale factor to touch max=SDE_raw
    power_raw = power_raw * scale

    # Detrended SDE, named "power"
    kernel = oversampling_factor * constants.SDE_MEDIAN_KERNEL_SIZE
    if kernel % 2 == 0:
        kernel = kernel + 1
    if len(power_raw) > 2 * kernel:
        my_median = running_median(power_raw, kernel)
        power = power_raw - my_median
        # Re-normalize to range between median = 0 and peak = SDE
        # shift down to the mean being zero
        power = power - np.mean(power)
        SDE = np.max(power / np.std(power))
        # scale factor to touch max=SDE
        scale = SDE / np.max(power)
        power = power * scale
    else:
        power = power_raw
        SDE = SDE_raw

    return SR, power_raw, power, SDE_raw, SDE


# def final_T0_fit(signal, depth, t, y, dy, period, T0_fit_margin, show_progress_bar, verbose):
#     """ After the search, we know the best period, width and duration.
#         But T0 was not preserved due to speed optimizations. 
#         Thus, iterate over T0s using the given parameters
#         Fold to all T0s so that the transit is expected at phase = 0"""

#     dur = len(signal)
#     scale = constants.SIGNAL_DEPTH / (1 - depth)
#     signal = 1 - ((1 - signal) / scale)
#     samples_per_period = np.size(y)

#     if T0_fit_margin == 0:
#         points = samples_per_period
#     else:
#         step_factor = T0_fit_margin * dur
#         points = int(samples_per_period / step_factor)
#     if points > samples_per_period:
#         points = samples_per_period

#     # Create all possible T0s from the start of [t] to [t+period] in [samples] steps
#     T0_array = np.linspace(
#         start=np.min(t), stop=np.min(t) + period, num=points
#     )

#     # Avoid showing progress bar when expected runtime is short
#     if points > constants.PROGRESSBAR_THRESHOLD and show_progress_bar:
#         show_progress_info = True
#     else:
#         show_progress_info = False

#     residuals_lowest = float("inf")
#     T0 = 0

#     if verbose:
#         print("Searching for best T0 for period", format(period, ".5f"), "days")

#     if show_progress_info:
#         pbar2 = tqdm(total=np.size(T0_array))
#     signal_ootr = np.ones(len(y[dur:]))

#     # Future speed improvement possible: Add multiprocessing. Will be slower for
#     # short data and T0_FIT_MARGIN > 0.01, but faster for large data with dense
#     # sampling (T0_FIT_MARGIN=0)
#     for Tx in T0_array:
#         phases = fold(time=t, period=period, T0=Tx)
#         sort_index = np.argsort(phases, kind="mergesort")  # 75% of CPU time
#         phases = phases[sort_index]
#         flux = y[sort_index]
#         dy = dy[sort_index]

#         # Roll so that the signal starts at index 0
#         # np roll is slow, so we replace it with less elegant concatenate
#         # flux = np.roll(flux, roll_cadences)
#         # dy = np.roll(dy, roll_cadences)
#         roll_cadences = int(dur / 2) + 1
#         flux = np.concatenate([flux[-roll_cadences:], flux[:-roll_cadences]])
#         dy = np.concatenate([flux[-roll_cadences:], flux[:-roll_cadences]])

#         residuals_intransit = np.sum((flux[:dur] - signal) ** 2 / dy[:dur] ** 2)
#         residuals_ootr = np.sum((flux[dur:] - signal_ootr) ** 2 / dy[dur:] ** 2)
#         residuals_total = residuals_intransit + residuals_ootr

#         if show_progress_info:
#             pbar2.update(1)
#         if residuals_total < residuals_lowest:
#             residuals_lowest = residuals_total
#             T0 = Tx
#     if show_progress_info:
#         pbar2.close()
#     return T0


def model_lightcurve(transit_times, period, t, model_transit_single):
    """Creates the model light curve for the full unfolded dataset"""

    # Append one more transit after and before end of nominal time series
    # to fully cover beginning and end with out of transit calculations
    earlier_tt = transit_times[0] - period
    extended_transit_times = np.append(earlier_tt, transit_times)
    next_tt = transit_times[-1] + period
    extended_transit_times = np.append(extended_transit_times, next_tt)
    full_x_array = np.array([])
    full_y_array = np.array([])
    rounds = len(extended_transit_times)
    internal_samples = (
        int(len(t) / len(transit_times))
    ) * constants.OVERSAMPLE_MODEL_LIGHT_CURVE

    # Append all periods
    for i in range(rounds):
        xmin = extended_transit_times[i] - period / 2
        xmax = extended_transit_times[i] + period / 2
        x_array = np.linspace(xmin, xmax, internal_samples)
        full_x_array = np.append(full_x_array, x_array)
        full_y_array = np.append(full_y_array, model_transit_single)

    if np.all(np.isnan(full_x_array)):
        return None, None
    else:  # Determine start and end of relevant time series, and crop it
        start_cadence = np.nanargmax(full_x_array > min(t))
        stop_cadence = np.nanargmax(full_x_array > max(t))
        full_x_array = full_x_array[start_cadence:stop_cadence]
        full_y_array = full_y_array[start_cadence:stop_cadence]
        model_lightcurve_model = full_y_array
        model_lightcurve_time = full_x_array
        return model_lightcurve_model, model_lightcurve_time


def all_transit_times(T0, t, period):
    """Return all mid-transit times within t"""

    if T0 < min(t):
        transit_times = [T0 + period]
    else:
        transit_times = [T0]
    previous_transit_time = transit_times[0]
    transit_number = 0
    while True:
        transit_number = transit_number + 1
        next_transit_time = previous_transit_time + period
        if next_transit_time < (np.min(t) + (np.max(t) - np.min(t))):
            transit_times.append(next_transit_time)
            previous_transit_time = next_transit_time
        else:
            break
    return transit_times

def calcDurationDays(times, period, StartT0, rawDuration):
# def calcDurationDays(times, period, StartT0, duration):
    """Return estimate for transit duration in days (Alternative method)"""

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def centerFold(time, period, T0):
        """Normal phase folding"""
        T0 = T0 + period/2
        return (time - T0) / period - np.floor((time - T0) / period)

    phases = centerFold(times, period, StartT0)
    phasesIndex = np.argsort(phases)
    phases_sorted = phases[phasesIndex]
    startIdx = find_nearest(phases_sorted,0.5)
    endIdx = startIdx + (np.array(rawDuration) * len(times)).astype(int)
    # endIdx = startIdx + duration
    #Seems duration can not be longer than 0.5
    endPhases = phases_sorted[endIdx]
    DurationDays = (endPhases - 0.5) * period
    return DurationDays

def calculate_transit_duration_in_days(t, period, transit_times, duration):
    """Return estimate for transit duration in days"""

    # Difference between (time series duration / period) and epochs
    transit_duration_in_days_raw = (
        duration * calculate_stretch(t, period, transit_times) * period
    )

    # Correct the duration for gaps in the data
    transit_duration_in_days = transit_duration_in_days_raw * calculate_fill_factor(t)

    return transit_duration_in_days


def calculate_stretch(t, period, transit_times):
    """Return difference between (time series duration / period) and epochs
        Example: 
        - Time series duration = 100 days
        - Period = 40 days
        - Epochs = 2 at t0s = [30, 70] days
        ==> stretch = (100 / 40) / 2 = 1.25"""

    duration_timeseries = (np.max(t) - np.min(t)) / period
    epochs = len(transit_times)
    stretch = duration_timeseries / epochs
    return stretch

def calculate_fill_factor(t):
    """Return the fraction of existing cadences, assuming constant cadences"""

    average_cadence = np.median(np.diff(t))
    span = max(t) - min(t)
    theoretical_cadences = span / average_cadence
    fill_factor = (len(t) - 1) / theoretical_cadences
    return fill_factor


def count_stats(t, y, transit_times, transit_duration_in_days):
    """Return:
    * in_transit_count:     Number of data points in transit (phase-folded)
    * after_transit_count:  Number of data points in a bin of transit duration, 
                            after transit (phase-folded)
    * before_transit_count: Number of data points in a bin of transit duration, 
                            before transit (phase-folded)
    """
    in_transit_count = 0
    after_transit_count = 0
    before_transit_count = 0

    for mid_transit in transit_times:
        T0 = (
            mid_transit - 1.5 * transit_duration_in_days
        )  # start of 1 transit dur before ingress
        T1 = mid_transit - 0.5 * transit_duration_in_days  # start of ingress
        T4 = mid_transit + 0.5 * transit_duration_in_days  # end of egress
        T5 = (
            mid_transit + 1.5 * transit_duration_in_days
        )  # end of egress + 1 transit dur

        if T0 > min(t) and T5 < max(t):  # inside time
            idx_intransit = np.where(np.logical_and(t > T1, t < T4))
            idx_before_transit = np.where(np.logical_and(t > T0, t < T1))
            idx_after_transit = np.where(np.logical_and(t > T4, t < T5))
            points_in_this_in_transit = len(y[idx_intransit])
            points_in_this_before_transit = len(y[idx_before_transit])
            points_in_this_after_transit = len(y[idx_after_transit])

            in_transit_count += points_in_this_in_transit
            before_transit_count += points_in_this_before_transit
            after_transit_count += points_in_this_after_transit

    return in_transit_count, after_transit_count, before_transit_count


def intransit_stats(t, y, transit_times, transit_duration_in_days):
    """Return all intransit odd and even flux points"""

    all_flux_intransit_odd = np.array([])
    all_flux_intransit_even = np.array([])
    all_flux_intransit = np.array([])
    all_idx_intransit = np.array([])
    per_transit_count = np.zeros([len(transit_times)])
    transit_depths = np.zeros([len(transit_times)])
    transit_depths_uncertainties = np.zeros([len(transit_times)])

    for i in range(len(transit_times)):

        depth_mean_odd = np.nan
        depth_mean_even = np.nan
        depth_mean_odd_std = np.nan
        depth_mean_even_std = np.nan

        mid_transit = transit_times[i]
        tmin = mid_transit - 0.5 * transit_duration_in_days
        tmax = mid_transit + 0.5 * transit_duration_in_days
        if np.isnan(tmin) or np.isnan(tmax):
            idx_intransit = []
            flux_intransit = []
            mean_flux = np.nan
        else:
            idx_intransit = np.where(np.logical_and(t > tmin, t < tmax))
            flux_intransit = y[idx_intransit]
            if len(y[idx_intransit]) > 0:
                mean_flux = np.mean(y[idx_intransit])
            else:
                mean_flux = np.nan
        intransit_points = np.size(y[idx_intransit])
        transit_depths[i] = mean_flux
        if len(y[idx_intransit] > 0):
            transit_depths_uncertainties[i] = np.std(y[idx_intransit]) / np.sqrt(
                intransit_points
            )
        else:
            transit_depths_uncertainties[i] = np.nan
        per_transit_count[i] = intransit_points

        # Check if transit odd/even to collect the flux for the mean calculations
        if i % 2 == 0:  # even
            all_flux_intransit_even = np.append(
                all_flux_intransit_even, flux_intransit
            )
        else:  # odd
            all_flux_intransit_odd = np.append(
                all_flux_intransit_odd, flux_intransit
            )
        if len(all_flux_intransit_odd) > 0:
            depth_mean_odd = np.mean(all_flux_intransit_odd)

            depth_mean_odd_std = np.std(all_flux_intransit_odd) / np.sum(
                len(all_flux_intransit_odd)
            ) ** (0.5)
        if len(all_flux_intransit_even) > 0:
            depth_mean_even = np.mean(all_flux_intransit_even)
            depth_mean_even_std = np.std(all_flux_intransit_even) / np.sum(
                len(all_flux_intransit_even)
            ) ** (0.5)

    return (
        depth_mean_odd,
        depth_mean_even,
        depth_mean_odd_std,
        depth_mean_even_std,
        all_flux_intransit_odd,
        all_flux_intransit_even,
        per_transit_count,
        transit_depths,
        transit_depths_uncertainties,
    )

def pink_noise(data, width):
    std = 0
    datapoints = len(data) - width + 1
    for i in range(datapoints):
        std += np.std(data[i : i + width])
    return std / (datapoints * width ** 0.5)

def snr_stats(
    t,
    y,
    period,
    duration,
    T0,
    transit_times,
    transit_duration_in_days,
    per_transit_count,
):
    """Return snr_per_transit and snr_pink_per_transit"""

    snr_per_transit = np.zeros([len(transit_times)])
    snr_pink_per_transit = np.zeros([len(transit_times)])
    intransit = transit_mask(t, period, 2 * duration, T0)
    flux_ootr = y[~intransit]

    try:
        pinknoise = pink_noise(flux_ootr, int(np.mean(per_transit_count)))
    except:
        pinknoise = np.nan

    # Estimate SNR and pink SNR
    # Second run because now the out of transit points are known
    if len(flux_ootr) > 0:
        std = np.std(flux_ootr)
    else:
        std = np.nan
    for i in range(len(transit_times)):
        mid_transit = transit_times[i]
        tmin = mid_transit - 0.5 * transit_duration_in_days
        tmax = mid_transit + 0.5 * transit_duration_in_days
        if np.isnan(tmin) or np.isnan(tmax):
            idx_intransit = []
            mean_flux = np.nan
        else:
            idx_intransit = np.where(np.logical_and(t > tmin, t < tmax))
            if len(y[idx_intransit]) > 0:
                mean_flux = np.mean(y[idx_intransit])
            else:
                mean_flux = np.nan

        intransit_points = np.size(y[idx_intransit])
        try:
            snr_pink_per_transit[i] = (1 - mean_flux) / pinknoise
            if intransit_points > 0 and not np.isnan(std):
                std_binned = std / intransit_points ** 0.5
                snr_per_transit[i] = (1 - mean_flux) / std_binned
            else:
                snr_per_transit[i] = 0
                snr_pink_per_transit[i] = 0
        except:
            snr_per_transit[i] = 0
            snr_pink_per_transit[i] = 0

    return snr_per_transit, snr_pink_per_transit
