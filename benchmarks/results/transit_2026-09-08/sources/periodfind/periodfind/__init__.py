import subprocess

import numpy as np

# Copyright 2020 California Institute of Technology. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
# Author: Ethan Jaszewski

"""
Provides an API for analyzing light curves using periodograms.

Supports a PyTorch-style device abstraction for transparent CPU/GPU dispatch::

    import periodfind

    periodfind.set_device('cpu')       # or 'gpu'
    ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10)

    # Per-call override
    ce_gpu = periodfind.ConditionalEntropy(n_phase=10, n_mag=10, device='gpu')
"""

# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

_default_device = None  # None means auto-detect


def _resolve_device(device=None):
    """Return ``'cpu'`` or ``'gpu'`` after resolving *device*.

    Resolution order:
    1. Explicit *device* argument (if not None).
    2. Global default set via :func:`set_device`.
    3. Auto-detect: try importing the CUDA extensions **and** running
       ``nvidia-smi``; fall back to ``'cpu'``.
    """
    if device is not None:
        device = device.lower()
        if device not in ("cpu", "gpu"):
            raise ValueError(f"Unknown device '{device}'. Choose 'cpu' or 'gpu'.")
        return device

    global _default_device
    if _default_device is not None:
        return _default_device

    # Auto-detect
    try:
        import periodfind.ce  # noqa: F401

        ret = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if ret.returncode == 0:
            return "gpu"
    except (ImportError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "cpu"


def set_device(device):
    """Set the global default device to ``'cpu'`` or ``'gpu'``."""
    device = device.lower()
    if device not in ("cpu", "gpu"):
        raise ValueError(f"Unknown device '{device}'. Choose 'cpu' or 'gpu'.")
    global _default_device
    _default_device = device


def get_device():
    """Return the current effective device (``'cpu'`` or ``'gpu'``)."""
    return _resolve_device()


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def ConditionalEntropy(**kwargs):
    """Create a Conditional Entropy algorithm on the resolved device.

    Accepts an optional ``device='cpu'|'gpu'`` keyword; all other keywords
    are forwarded to the backend class constructor.
    """
    device = _resolve_device(kwargs.pop("device", None))
    if device == "gpu":
        from periodfind.gpu import ConditionalEntropy as _Cls
    else:
        from periodfind.cpu import ConditionalEntropy as _Cls
    return _Cls(**kwargs)


def AOV(**kwargs):
    """Create an Analysis of Variance algorithm on the resolved device.

    Accepts an optional ``device='cpu'|'gpu'`` keyword; all other keywords
    are forwarded to the backend class constructor.
    """
    device = _resolve_device(kwargs.pop("device", None))
    if device == "gpu":
        from periodfind.gpu import AOV as _Cls
    else:
        from periodfind.cpu import AOV as _Cls
    return _Cls(**kwargs)


def LombScargle(**kwargs):
    """Create a Lomb-Scargle algorithm on the resolved device.

    Accepts an optional ``device='cpu'|'gpu'`` keyword; all other keywords
    are forwarded to the backend class constructor.
    """
    device = _resolve_device(kwargs.pop("device", None))
    if device == "gpu":
        from periodfind.gpu import LombScargle as _Cls
    else:
        from periodfind.cpu import LombScargle as _Cls
    return _Cls(**kwargs)


def FPW(**kwargs):
    """Create a Fast Phase-folding Weighted algorithm on the resolved device.

    Accepts an optional ``device='cpu'|'gpu'`` keyword; all other keywords
    are forwarded to the backend class constructor.
    """
    device = _resolve_device(kwargs.pop("device", None))
    if device == "gpu":
        from periodfind.gpu import FPW as _Cls
    else:
        from periodfind.cpu import FPW as _Cls
    return _Cls(**kwargs)


def BoxLeastSquares(**kwargs):
    """Create a Box Least Squares algorithm on the resolved device.

    Accepts an optional ``device='cpu'|'gpu'`` keyword; all other keywords
    are forwarded to the backend class constructor.
    """
    device = _resolve_device(kwargs.pop("device", None))
    if device == "gpu":
        from periodfind.gpu import BoxLeastSquares as _Cls
    else:
        from periodfind.cpu import BoxLeastSquares as _Cls
    return _Cls(**kwargs)


def MatchedFilter(**kwargs):
    """Create a Matched Filter morphology scorer on the resolved device.

    Accepts an optional ``device='cpu'|'gpu'`` keyword; all other keywords
    are forwarded to the backend class constructor.
    """
    device = _resolve_device(kwargs.pop("device", None))
    if device == "gpu":
        from periodfind.gpu import MatchedFilter as _Cls
    else:
        from periodfind.cpu import MatchedFilter as _Cls
    return _Cls(**kwargs)


def ViterbiNarrowband(**kwargs):
    """Create a Viterbi Narrowband scorer on the resolved device.

    Accepts an optional ``device='cpu'|'gpu'`` keyword; all other keywords
    are forwarded to the backend class constructor.
    """
    device = _resolve_device(kwargs.pop("device", None))
    if device == "gpu":
        from periodfind.gpu import ViterbiNarrowband as _Cls
    else:
        from periodfind.cpu import ViterbiNarrowband as _Cls
    return _Cls(**kwargs)


def MultiHarmonicFourier(**kwargs):
    """Create a Multi-Harmonic Fourier periodogram on the resolved device.

    Accepts an optional ``device='cpu'|'gpu'`` keyword; all other keywords
    are forwarded to the backend class constructor.
    """
    device = _resolve_device(kwargs.pop("device", None))
    if device == "gpu":
        from periodfind.gpu import MultiHarmonicFourier as _Cls
    else:
        from periodfind.cpu import MultiHarmonicFourier as _Cls
    return _Cls(**kwargs)


def FourierDecomposition(**kwargs):
    """Create a Fourier decomposition feature extractor.

    CPU-only for now.  If ``device='gpu'`` is requested, falls back to
    CPU with a warning.

    Accepts an optional ``device='cpu'|'gpu'`` keyword; all other keywords
    are forwarded to the backend class constructor.
    """
    import warnings

    device = _resolve_device(kwargs.pop("device", None))
    if device == "gpu":
        warnings.warn(
            "FourierDecomposition does not have a GPU backend; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
    from periodfind.cpu import FourierDecomposition as _Cls

    return _Cls(**kwargs)


def DmDt(**kwargs):
    """Create a dm-dt histogram feature extractor (CPU-only)."""
    kwargs.pop("device", None)
    from periodfind.cpu import DmDt as _Cls

    return _Cls(**kwargs)


def BasicStats(**kwargs):
    """Create a basic statistics extractor (CPU-only)."""
    kwargs.pop("device", None)
    from periodfind.cpu import BasicStats as _Cls

    return _Cls(**kwargs)


def remove_high_cadence(times, mags, errs, cadence_minutes=30.0):
    """Batch high-cadence removal via Rust.

    Parameters
    ----------
    times : list of ndarray (float32)
    mags : list of ndarray (float32)
    errs : list of ndarray (float32)
    cadence_minutes : float, default=30.0

    Returns
    -------
    list of (ndarray, ndarray, ndarray)
        Filtered (times, mags, errs) tuples.
    """
    from periodfind.cpu import RemoveHighCadence as _Cls

    rhc = _Cls(cadence_minutes=cadence_minutes)
    return rhc.calc(times, mags, errs)


class Statistics:
    """Stores statistics about a single set of parameters.

    Stores various periodogram statistics, as well as the parameters and
    value for a single test statistic of interest.

    Parameters
    ----------
    params : list of float
        List of paramters that produce this object's value

    value : float
        Value of the test statistic with the given params

    mean : float
        Periodogram test statistic mean

    std : float
        Periodogram test statistic std

    median : float
        Periodogram test statistic median

    mad : float
        Periodogram test statistic median absolute deviation

    significance_type : {'stdmean', 'madmedian'}, default='stdmean'
        Specifies the significance statistic that should be used. The `stdmean`
        statistic gives a rough estimate of how likely the value is. The
        `madmedian` gives a roughly analagous statistic, but is more robust.

    Attributes
    ----------
    params : list of float
        List of paramters that produce this object's value

    value : float
        Value of the test statistic with the given params

    mean : float
        Periodogram test statistic mean

    std : float
        Periodogram test statistic std

    median : float
        Periodogram test statistic median

    mad : float
        Periodogram test statistic median absolute deviation

    significance_type : {'stdmean', 'madmedian'}
        Specifies the significance statistic that should be used. The `stdmean`
        statistic gives a rough estimate of how likely the value is. The
        `madmedian` gives a roughly analagous statistic, but is more robust.

    significance : float
        Significance statistic, computed according to the significance_type
    """

    def __init__(self, params, value, mean, std, median, mad, significance_type="stdmean"):
        self.params = params
        self.value = value
        self.mean = mean
        self.std = std
        self.median = median
        self.mad = mad
        self.significance_type = significance_type

    def __repr__(self):
        return (
            f"Statistics(params={self.params}, value={self.value:.6g}, "
            f"significance_type='{self.significance_type}')"
        )

    @property
    def significance(self):
        if self.significance_type == "stdmean":
            return abs(self.value - self.mean) / self.std
        elif self.significance_type == "madmedian":
            return abs(self.value - self.median) / self.mad
        else:
            raise NotImplementedError("Statistic " + self.significance_type + " not implemented")

    @staticmethod
    def statistics_from_data(
        data,
        params,
        use_max,
        mean=None,
        std=None,
        median=None,
        mad=None,
        n=1,
        significance_type="stdmean",
    ):
        """Constructs statistics objects from a periodogram.

        Parameters
        ----------
        data : ndarray
            Periodogram data to find statistics for

        mean : float, default=None
            Periodogram test statistic mean. Calculated if not provided.

        std : float, default=None
            Periodogram test statistic std. Calculated if not provided.

        median : float, default=None
            Periodogram test statistic median. Calculated if not provided.

        mad : float, default=None
            Periodogram test statistic median absolute deviation. Calculated
            if not provided.

        n : int, default=1
            Number of `Statistics` to generate

        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Specifies the significance statistic that should be used. See class
            documentation for more information.

        Returns
        -------
        stats : `Statistics` or list of `Statistics`
            Statistics for the top `n` parameters
        """

        # Find best parameters
        if not use_max:
            partition = np.argpartition(data, n, axis=None)[:n]
        else:
            partition = np.argpartition(data, len(data) - n, axis=None)[-n:]

        idxs = np.unravel_index(partition, data.shape)
        idxs_t = []
        for i in range(n):
            idx = tuple(dim[i] for dim in idxs)
            idxs_t.append(idx)

        values = data[idxs]

        # Calculate the data-wide statistics
        if mean is None:
            mean = np.mean(data)
        if std is None:
            std = np.std(data)
        if median is None:
            median = np.median(data)
        if mad is None:
            mad = np.median(np.abs(data - median))

        best = []
        for idx, val in zip(idxs_t, values):
            param = [params[i][idx[i]] for i in range(len(params))]
            best.append(
                Statistics(
                    param,
                    val,
                    mean,
                    std,
                    median,
                    mad,
                    significance_type,
                )
            )

        # Sort by value so most significant is first
        best.sort(key=lambda s: s.value, reverse=use_max)

        if n == 1:
            return best[0]
        else:
            return best


class Periodogram:
    """Stores a full periodogram.

    Stores a full periodogram, including both the test statistic values and
    all trial parameters. Allows for plotting of the periodogram, as well as
    analysis beyond the statistics results returned by period finding
    algorithms.

    Parameters
    ----------
    periodogram : ndarray
        Periodogram test statistic values

    params : list of ndarray
        List of periodogram test parameters

    use_max : bool
        Whether likely periods are periodogram maxima

    Attributes
    ----------
    use_max : bool
        Whether likely periods are periodogram maxima

    data : ndarray
        Periodogram test statistic values

    params : list of ndarray
        List of periodogram test parameters

    mean : float
        Periodogram test statistic mean

    std : float
        Periodogram test statistic std

    median : float
        Periodogram test statistic median

    mad : float
        Periodogram test statistic median absolute deviation

    Notes
    -----
    For large periodograms, the memory usage of storing full periodograms
    can be prohibitively high, necessitating the use of the `Statistics`
    class instead.

    Computes parameters on demand, and does not cache them, so statistics
    should be stored if necessary.
    """

    def __init__(self, periodogram, params, use_max):
        self.use_max = use_max
        self.data = periodogram
        self.params = params
        self.mean = None
        self.std = None
        self.median = None
        self.mad = None

    def __repr__(self):
        shape = self.data.shape if hasattr(self.data, "shape") else "?"
        return f"Periodogram(shape={shape}, use_max={self.use_max})"

    def _calc_mean(self):
        if self.mean is None:
            self.mean = np.mean(self.data)

    def _calc_std(self):
        if self.std is None:
            self.std = np.std(self.data)

    def _calc_median(self):
        if self.median is None:
            self.median = np.median(self.data)

    def _calc_mad(self):
        if self.mad is None:
            self.mad = np.median(np.abs(self.data - self.median))

    def best_params(self, n=1, significance_type="stdmean"):
        """Returns the best parameters of the periodogram.

        Computes the best parameters of the periodogram, returning them as
        statistics objects.

        Parameters
        ----------
        n : int, default=1
            The number of top parameters to return

        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Specifies the significance statistic that should be used. See the
            documentation for the `Statistics` class for more information.

        Returns
        -------
        stats : `Statistics` or list of `Statistics`
            Statistics for the top `n` parameters
        """
        self._calc_mean()
        self._calc_std()
        self._calc_median()
        self._calc_mad()

        return Statistics.statistics_from_data(
            self.data,
            self.params,
            self.use_max,
            mean=self.mean,
            std=self.std,
            median=self.median,
            mad=self.mad,
            n=n,
            significance_type=significance_type,
        )
