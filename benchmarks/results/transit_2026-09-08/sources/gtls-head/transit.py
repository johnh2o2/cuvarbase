from __future__ import division, print_function
import batman  # https://www.cfa.harvard.edu/~lkreidberg/batman/
import numpy as np
from . import constants as constants
from .interpolation import interp1d


def reference_transit(samples, per, rp, a, inc, ecc, w, u, limb_dark):
    """Returns an Earth-like transit of width 1 and depth 1"""

    f = np.ones(constants.SUPERSAMPLE_SIZE)
    duration = 1  # transit duration in days. Increase for exotic cases
    t = np.linspace(-duration * 0.5, duration * 0.5, constants.SUPERSAMPLE_SIZE)
    ma = batman.TransitParams()
    ma.t0 = 0  # time of inferior conjunction
    ma.per = per  # orbital period, use Earth as a reference
    ma.rp = rp  # planet radius (in units of stellar radii)
    ma.a = a  # semi-major axis (in units of stellar radii)
    ma.inc = inc  # orbital inclination (in degrees)
    ma.ecc = ecc  # eccentricity
    ma.w = w  # longitude of periastron (in degrees)
    ma.u = u  # limb darkening coefficients
    ma.limb_dark = limb_dark  # limb darkening model
    m = batman.TransitModel(ma, t)  # initializes model
    flux = m.light_curve(ma)  # calculates light curve

    # Determine start of transit (first value < 1)
    idx_first = np.argmax(flux < 1)
    intransit_flux = flux[idx_first : -idx_first + 1]
    intransit_time = t[idx_first : -idx_first + 1]

    # Downsample (bin) to target sample size
    x_new = np.linspace(t[idx_first], t[-idx_first - 1], samples)
    f = interp1d(x_new, intransit_time)
    downsampled_intransit_flux = f(intransit_flux)

    # Rescale to height [0..1]
    rescaled = (np.min(downsampled_intransit_flux) - downsampled_intransit_flux) / (
        np.min(downsampled_intransit_flux) - 1
    )

    return rescaled


def fractional_transit(
    duration,
    maxwidth,
    depth,
    samples,
    per,
    rp,
    a,
    inc,
    ecc,
    w,
    u,
    limb_dark,
    cached_reference_transit=None,
):
    """Returns a scaled reference transit with fractional width and depth"""

    if cached_reference_transit is None:
        reference_flux = reference_transit(
            samples=samples,
            per=per,
            rp=rp,
            a=a,
            inc=inc,
            ecc=ecc,
            w=w,
            u=u,
            limb_dark=limb_dark,
        )
    else:
        reference_flux = cached_reference_transit

    # Interpolate to shorter interval - new method without scipy
    reference_time = np.linspace(-0.5, 0.5, samples)
    occupied_samples = int((duration / maxwidth) * samples)
    x_new = np.linspace(-0.5, 0.5, occupied_samples)
    f = interp1d(x_new, reference_time)
    y_new = f(reference_flux)

    # Patch ends with ones ("1")
    missing_samples = samples - occupied_samples
    emtpy_segment = np.ones(int(missing_samples * 0.5))
    result = np.append(emtpy_segment, y_new)
    result = np.append(result, emtpy_segment)
    if np.size(result) < samples:  # If odd number of samples
        result = np.append(result, np.ones(1))

    # Depth rescaling
    result = 1 - ((1 - result) * depth)

    return result


def get_cache(durations, maxwidth_in_samples, per, rp, a, inc, ecc, w, u,
              limb_dark, verbose=True):
    """Fetches (size(durations)*size(depths)) light curves of length 
        maxwidth_in_samples and returns these LCs in a 2D array, together with 
        their metadata in a separate array."""

    # if verbose:
    #     print("Creating model cache for", str(len(durations)), "durations")
    lc_arr = []
    rows = np.size(durations)
    lc_cache_overview = np.zeros(
        rows,
        dtype=[("duration", "f8"), ("width_in_samples", "i8"), ("overshoot", "f8")],
    )
    cached_reference_transit = reference_transit(
        samples=maxwidth_in_samples,
        per=per,
        rp=rp,
        a=a,
        inc=inc,
        ecc=ecc,
        w=w,
        u=u,
        limb_dark=limb_dark,
    )

    row = 0
    for duration in durations:
        scaled_transit = fractional_transit(
            duration=duration,
            maxwidth=np.max(durations),
            depth=constants.SIGNAL_DEPTH,
            samples=maxwidth_in_samples,
            per=per,
            rp=rp,
            a=a,
            inc=inc,
            ecc=ecc,
            w=w,
            u=u,
            limb_dark=limb_dark,
            cached_reference_transit=cached_reference_transit,
        )
        lc_cache_overview["duration"][row] = duration
        used_samples = int((duration / np.max(durations)) * maxwidth_in_samples)
        lc_cache_overview["width_in_samples"][row] = used_samples
        full_values = np.where(
            scaled_transit < (1 - constants.NUMERICAL_STABILITY_CUTOFF)
        )
        first_sample = np.min(full_values)
        last_sample = np.max(full_values) + 1
        signal = scaled_transit[first_sample:last_sample]
        lc_arr.append(signal)

        # Fraction of transit bottom and mean flux
        overshoot = np.mean(signal) / np.min(signal)

        # Later, we multiply the inverse fraction ==> convert to inverse percentage
        lc_cache_overview["overshoot"][row] = 1 / (2 - overshoot)
        row += +1

    lc_arr = np.array(lc_arr, dtype=object)
    return lc_cache_overview, lc_arr

def mutipleTransitFit(pointSize,depth):

    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = 0.                        #time of inferior conjunction
    params.per = 1.                       #orbital period
    params.rp = 0.1                       #planet radius (in units of stellar radii)
    params.a = 15.                        #semi-major axis (in units of stellar radii)
    params.inc = 87.                      #orbital inclination (in degrees)
    params.ecc = 0.                       #eccentricity
    params.w = 90.                        #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"        #limb darkening model
    params.u = [0.4804, 0.1867]      #limb darkening coefficients [u1, u2, u3, u4]

    t = np.linspace(-0.025, 0.025, 1000)  #times at which to calculate light curve
    m = batman.TransitModel(params, t)    #initializes model

    TargetIncMin = 86.2
    TargetIncMax = 90
    baseInc = 86
    incsLog = np.linspace(np.log(TargetIncMin - baseInc),np.log(TargetIncMax - baseInc),100)
    incs = np.exp(incsLog)+baseInc

    transitArr = []
    overshootArr = []
    params.rp  = 0.025
    for inc in incs:
        params.inc = inc
        m = batman.TransitModel(params,t)
        flux = m.light_curve(params)                    #calculates light curveradii = np.linspace(0.09, 0.11, 20)

        idx_first = np.argmax(flux < 1)
        intransit_time = t[idx_first : -idx_first + 1]

        tStart = intransit_time[0]
        tEnd = intransit_time[-1]
        tNew = np.linspace(tStart, tEnd, pointSize)
        m = batman.TransitModel(params,tNew)
        new_flux = m.light_curve(params)
        # rescaled = (np.min(new_flux) - new_flux) / (np.min(new_flux) - 1) * 0.5 + 0.5
        rescaled = (np.min(new_flux) - new_flux) / (np.min(new_flux) - 1) * depth + 1 - depth
        transitArr.append(rescaled)
        # overshootArr.append(np.mean(rescaled)/0.5)
    return np.array(transitArr)#, np.array(overshootArr)