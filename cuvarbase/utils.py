from copy import deepcopy
import os
import re
import numpy as np


def weights(err):
    """ generate observation weights from uncertainties """
    w = np.power(err, -2)
    return w/np.sum(w)


def conflict_scatter_perm(n):
    """
    Deterministic permutation that de-clusters time-ordered data for
    the shared-memory histogram kernels.

    Survey lightcurves arrive time-sorted; at nearly every trial
    frequency, consecutive samples of a dense cadence fold to the same
    phase bin, so the 32 lanes of a warp fight for one shared-memory
    atomic counter (measured on an RTX A5000: 3.1x kernel slowdown for
    a TESS-like 2-minute cadence versus randomly ordered input).
    Binning is order-independent (the histogram is a sum), so storing
    the points in a scattered order removes the conflicts without
    touching the math.

    Uses the golden-ratio stride ``p[i] = (i * k) % n`` with ``k``
    the largest integer <= 0.618 n coprime to ``n``: adjacent output
    slots come from samples ~0.618 n apart in time, for any n, with no
    RNG state involved.

    Returns ``None`` for ``n < 64`` (a warp or two; nothing to gain).
    """
    if n < 64:
        return None
    k = max(1, int(round(0.6180339887498949 * n)))
    while np.gcd(k, n) != 1:
        k -= 1
    return (np.arange(n, dtype=np.int64) * k) % n


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


# ``//{INCLUDE filename}`` directive: inlined by _module_reader at load
# time, resolved relative to the including file's directory. This lets
# shared device code live in a single source file (e.g. bls_common.cuh)
# without an nvcc include path -- pycuda's SourceModule compiles from the
# assembled string, so nvcc never sees an #include of our own files.
_INCLUDE_RE = re.compile(r'^[ \t]*//\{INCLUDE\s+([^\s}]+)\}[ \t]*$', re.M)


def _expand_includes(txt, base_dir, _seen=None):
    """Recursively inline ``//{INCLUDE filename}`` directives."""
    if _seen is None:
        _seen = set()

    def _sub(match):
        name = match.group(1)
        real = os.path.abspath(os.path.join(base_dir, name))
        if real in _seen:
            raise ValueError("circular kernel include: %s" % name)
        _seen.add(real)
        with open(real, 'r') as f:
            included = f.read()
        return _expand_includes(included, os.path.dirname(real), _seen)

    return _INCLUDE_RE.sub(_sub, txt)


def _module_reader(fname, cpp_defs=None):
    txt = open(fname, 'r').read()

    # Inline shared device code before any other substitution.
    txt = _expand_includes(txt, os.path.dirname(os.path.abspath(fname)))

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
    t : array_like
        The observation times.
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
