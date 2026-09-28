"""Experimental allocation/ranking helpers; unchanged math stays in the baseline module."""
import numpy as np
from .tls_reference_math import (augment_duration_grid, build_cache,
    harmonic_candidate_indices, native_spectra, preprocess_inputs)

def chunk_width_masks(widths, minima, maxima, chunk_size):
    """Native union of admissible integer widths over each period chunk."""
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    widths, minima, maxima = np.asarray(widths), np.asarray(minima), np.asarray(maxima)
    if minima.shape != maxima.shape or minima.ndim != 1:
        raise ValueError('minima and maxima must be aligned 1D arrays')
    # Only one logical group's temporary membership matrix is needed. Dense
    # M-dwarf grids can contain millions of periods, while the returned union
    # normally has only thirty rows. Do not retain the full period/width table.
    return np.array([np.any(
        (widths[None, :] >= minima[start:start + chunk_size, None]) &
        (widths[None, :] <= maxima[start:start + chunk_size, None]), axis=0)
        for start in range(0, len(minima), chunk_size)], dtype=bool)

def refinement_candidate_indices(periods, power):
    """Rank valid candidates: top100, then next100 at P>1d.

    Filtering before the stable sort fixes a native GTLS host-mask defect.
    Its masked scalars do not form a total ordering and can enter the top100
    period list as NaN, leading to undefined GPU integer conversions. Only
    finite, unmasked scores and periods represent physical first-stage trials.
    The native rank policy and tie order are unchanged on valid entries.
    """
    periods, power = np.ma.asarray(periods), np.ma.asarray(power)
    valid = (~np.ma.getmaskarray(periods) & ~np.ma.getmaskarray(power) &
             np.isfinite(np.ma.getdata(periods)) &
             np.isfinite(np.ma.getdata(power)))
    indices = np.flatnonzero(valid)
    ranked = indices[np.argsort(-power.data[indices], kind='stable')]
    # Filtering a stable ordering preserves the native second sort's tie
    # order. The first hundred rows are already excluded by this slice; no
    # Python tuple construction, repeated membership scans or second sort.
    remaining = ranked[100:]
    next_best = remaining[periods.data[remaining] > 1][:100]
    return np.concatenate((ranked[:100], next_best)).astype(np.int64, copy=False)

