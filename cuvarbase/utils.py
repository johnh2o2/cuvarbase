from copy import deepcopy
import os
import numpy as np


def weights(err):
    """ generate observation weights from uncertainties """
    w = np.power(err, -2)
    return w/np.sum(w)


def subtract_epoch(t):
    """
    Shift observation times so that they start near zero.

    Returns ``(t - floor(min(t)), floor(min(t)))``, with the
    subtraction performed in float64. Phase folding on the GPU happens
    in single precision, so for absolute timestamps (e.g. BJD ~
    2,455,000 days) the product ``float32(t) * freq`` loses nearly all
    phase information; times must be epoch-subtracted *before* any
    cast to float32. All phases (``phi0`` solutions) are measured
    relative to the returned epoch.

    The epoch is ``floor(min(t))`` rather than ``min(t)`` itself: a
    round-number epoch is friendlier for reconstructing absolute
    transit times, and subtracting ``min(t)`` exactly would place the
    first observation at phase exactly 0.0 for *every* trial
    frequency — a systematic bin-edge alignment that makes binned
    (GPU) and exact (CPU) box memberships disagree at wrap-around
    solutions.

    Parameters
    ----------
    t: array_like, float
        Observation times

    Returns
    -------
    t_shifted: ndarray, float64
        ``t - floor(min(t))``
    epoch: float
        ``floor(min(t))``, the epoch that was subtracted
    """
    t = np.asarray(t, dtype=np.float64)
    epoch = np.floor(t.min())
    return t - epoch, epoch


def find_kernel(name):
    # Resolve relative to this file rather than importlib.resources:
    # setuptools PEP-660 editable installs hand files("cuvarbase") a
    # MultiplexedPath that misresolves to the project root on py<3.12,
    # and the kernels must be real on-disk files for open()/nvcc anyway.
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'kernels', f'{name}.cu')


def _module_reader(fname, cpp_defs=None):
    txt = open(fname, 'r').read()

    if cpp_defs is None:
        return txt

    preamble = ['#define {key} {value}'.format(key=key,
                                               value=('' if value is None
                                                      else value))
                for key, value in cpp_defs.items()]
    txt = txt.replace('//{CPP_DEFS}', '\n'.join(preamble))

    return txt


def tophat_window(t, t0, d):
    w_window = np.zeros_like(t)
    w_window[np.absolute(t - t0) < d] += 1.
    return w_window / np.max(w_window)


def gaussian_window(t, t0, d):
    w_window = np.exp(-0.5 * np.power(t - t0, 2) / (d * d))
    return w_window / (1. if len(w_window) == 0 else np.max(w_window))


def autofrequency(t, nyquist_factor=5, samples_per_peak=5,
                  minimum_frequency=None,
                  maximum_frequency=None, **kwargs):
    """
    Determine a suitable frequency grid for data.

    Note that this assumes the peak width is driven by the observational
    baseline, which is generally a good assumption when the baseline is
    much larger than the oscillation period.
    If you are searching for periods longer than the baseline of your
    observations, this may not perform well.

    Even with a large baseline, be aware that the maximum frequency
    returned is based on the concept of "average Nyquist frequency", which
    may not be useful for irregularly-sampled data. The maximum frequency
    can be adjusted via the nyquist_factor argument, or through the
    maximum_frequency argument.

    Parameters
    ----------
    samples_per_peak : float (optional, default=5)
        The approximate number of desired samples across the typical peak
    nyquist_factor : float (optional, default=5)
        The multiple of the average nyquist frequency used to choose the
        maximum frequency if maximum_frequency is not provided.
    minimum_frequency : float (optional)
        If specified, then use this minimum frequency rather than one
        chosen based on the size of the baseline.
    maximum_frequency : float (optional)
        If specified, then use this maximum frequency rather than one
        chosen based on the average nyquist frequency.

    Returns
    -------
    frequency : ndarray or Quantity
        The heuristically-determined optimal frequency bin
    """
    baseline = np.max(t) - np.min(t)
    n_samples = len(t)

    df = 1. / (baseline * samples_per_peak)

    nf0 = 1
    if minimum_frequency is not None:
        nf0 = max([nf0, int(minimum_frequency / df)])

    if maximum_frequency is not None:
        Nf = int(maximum_frequency / df) - nf0
    else:
        Nf = int(0.5 * samples_per_peak * nyquist_factor * n_samples)

    return df * (nf0 + np.arange(Nf))


def dphase(dt, freq):
    dph = dt * freq - np.floor(dt * freq)
    dph_final = dph if dph < 0.5 else 1 - dph
    return dph_final


def get_autofreqs(t, **kwargs):
    autofreqs_kwargs = {var: value for var, value in kwargs.items()
                        if var in ['minimum_frequency', 'maximum_frequency',
                                   'nyquist_factor', 'samples_per_peak']}
    return autofrequency(t, **autofreqs_kwargs)


def normalize_light_curves(data: list[tuple[np.array, ...]]):
    """
    Normalize light curves by subtracting the mean from the magnitudes and the observation times.

    Parameters
    ----------
    data: list of tuples
        list of [(t, y, ...), ...] containing
        * ``t``: observation times
        * ``y``: observations
        * ... other columns

    Returns
    -------
    data: list of tuples
        list of [(t, y, ...), ...] containing
        * ``t``: updated observation times
        * ``y``: updated observations
        * ... other columns (preserved as in input; ``None`` entries --
          e.g. ``dy=None`` for unweighted runs -- pass through unchanged)

    """
    data = deepcopy(data)
    for i, lc in enumerate(data):
        updated_lc = []
        # Precompute means for the first two elements
        means = [np.nanmean(lc[j]) if j < 2 else None for j in range(len(lc))]
        for j in range(len(lc)):
            if j < 2:
                updated_lc.append((lc[j] - means[j]).copy())
            elif lc[j] is None:
                updated_lc.append(None)
            else:
                updated_lc.append(lc[j].copy())
        data[i] = tuple(updated_lc)

    return data
