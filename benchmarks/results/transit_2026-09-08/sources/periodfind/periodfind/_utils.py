"""Shared utilities for periodfind Cython extensions.

Extracts the magnitude preparation, input validation, and dtype checking
logic that was previously duplicated across ce.pyx, aov.pyx, and ls.pyx.
"""

import warnings

import numpy as np


def prepare_magnitudes(mags, center, normalize):
    """Center, normalize, or pass through magnitude arrays.

    Parameters
    ----------
    mags : list of ndarray
        List of light curve magnitude arrays.
    center : bool
        Whether to center magnitudes to zero mean.
    normalize : bool
        Whether to normalize magnitudes to (0, 1) range.

    Returns
    -------
    mags_use : list of ndarray
        Processed magnitude arrays.
    """
    if center and normalize:
        warnings.warn(
            "Center and normalize are conflicting settings. Normalize will be ignored.",
            RuntimeWarning,
            stacklevel=3,
        )

    if center:
        return [mag - np.mean(mag) for mag in mags]
    elif normalize:
        result = []
        for mag in mags:
            min_v = np.min(mag)
            max_v = np.max(mag)
            scaled = ((mag - min_v) / (max_v - min_v)) * 0.999 + 5e-4
            result.append(scaled)
        return result
    else:
        return mags


def validate_inputs(times, mags):
    """Validate that times and mags lists are compatible.

    Parameters
    ----------
    times : list of ndarray
        List of light curve time arrays.
    mags : list of ndarray
        List of light curve magnitude arrays.

    Raises
    ------
    ValueError
        If the number of time and magnitude arrays differ, or if any
        paired arrays have mismatched lengths.
    """
    if len(times) != len(mags):
        raise ValueError(
            f"times and mags must have the same number of light curves, "
            f"got {len(times)} and {len(mags)}"
        )

    for i, (t, m) in enumerate(zip(times, mags)):
        if len(t) != len(m):
            raise ValueError(
                f"times[{i}] and mags[{i}] have different lengths: {len(t)} vs {len(m)}"
            )


def ensure_float32(arrays, name):
    """Check that all arrays in a list are float32.

    Parameters
    ----------
    arrays : list of ndarray
        Arrays to check.
    name : str
        Name used in error messages (e.g. 'times' or 'mags').

    Raises
    ------
    TypeError
        If any array does not have dtype float32.
    """
    for i, arr in enumerate(arrays):
        if arr.dtype != np.float32:
            raise TypeError(f"{name}[{i}] has dtype {arr.dtype}, expected float32")
