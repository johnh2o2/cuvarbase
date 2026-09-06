"""
Memory management for Conditional Entropy period-finding operations.
"""
import numpy as np

import pycuda.driver as cuda  # noqa: F401  (used by transfer methods)
import pycuda.gpuarray as gpuarray

from ..base import ensure_context
from ._host import host_array


class ConditionalEntropyMemory:
    """
    Container class for managing memory allocation and data transfer
    for Conditional Entropy computations on GPU.
    
    Parameters
    ----------
    phase_bins : int, optional (default: 10)
        Number of phase bins for conditional entropy calculation
    mag_bins : int, optional (default: 5)
        Number of magnitude bins
    phase_overlap : int, optional (default: 0)
        Overlap between phase bins
    mag_overlap : int, optional (default: 0)
        Overlap between magnitude bins
    max_phi : float, optional (default: 3.0)
        Maximum phase value
    stream : pycuda.driver.Stream, optional
        CUDA stream for asynchronous operations
    weighted : bool, optional (default: False)
        Use weighted binning
    use_fast : bool, optional (default: False)
        The memory will only ever be used by the shared-memory
        (``use_fast=True``) kernels, which keep their histogram in
        shared memory: skip the ``nf * phase_bins * mag_bins`` global
        histogram (``bins_g``) they never read.  That array is 20 MB
        for a 100k-frequency 10 x 5 search, and it was allocated -- and
        zero-filled on every ``run`` -- for nothing.  The standard
        kernels need it, so they refuse a memory allocated this way.
    **kwargs : dict
        Additional parameters
    """
    
    def __init__(self, **kwargs):
        # Constructing GPU memory is a "first GPU use" -- retain the CUDA
        # primary context now (no longer created eagerly at import).
        ensure_context()
        self.phase_bins = kwargs.get('phase_bins', 10)
        self.mag_bins = kwargs.get('mag_bins', 5)
        self.phase_overlap = kwargs.get('phase_overlap', 0)
        self.mag_overlap = kwargs.get('mag_overlap', 0)

        self.max_phi = kwargs.get('max_phi', 3.)
        self.stream = kwargs.get('stream', None)
        self.weighted = kwargs.get('weighted', False)
        self.use_fast = kwargs.get('use_fast', False)
        self.widen_mag_range = kwargs.get('widen_mag_range', False)
        self.n0 = kwargs.get('n0', None)
        self.nf = kwargs.get('nf', None)

        self.compute_log_prob = kwargs.get('compute_log_prob', False)

        self.balanced_magbins = kwargs.get('balanced_magbins', False)

        # Pinned (page-locked) host buffers by default for async overlap;
        # graceful fallback to page-aligned if pinning fails.
        self.pinned = kwargs.get('pinned', True)

        if self.weighted and self.balanced_magbins:
            raise ValueError("simultaneous balanced_magbins and weighted"
                            " options is not currently supported")

        if self.weighted and self.compute_log_prob:
            raise ValueError("simultaneous compute_log_prob and weighted"
                            " options is not currently supported")

        if self.use_fast and self.compute_log_prob:
            # the fast kernels compute only the conditional entropy; a
            # memory built this way silently returned the CE instead of
            # the log-probability
            raise ValueError("use_fast must be False if compute_log_prob"
                             " is True (there is no shared-memory"
                             " log-probability kernel)")
        self.n0_buffer = kwargs.get('n0_buffer', None)
        self.buffered_transfer = kwargs.get('buffered_transfer', False)
        self.t = None
        self.y = None
        self.dy = None

        self.t_g = None
        self.y_g = None
        self.dy_g = None

        self.bins_g = None
        self.ce_c = None
        self.ce_g = None
        self.mag_bwf = None
        self.mag_bwf_g = None
        self.real_type = np.float32
        if kwargs.get('use_double', False):
            self.real_type = np.float64

        self.freqs = kwargs.get('freqs', None)
        self.freqs_g = None
        # True once ``freqs`` has been uploaded into ``freqs_g``;
        # ``allocate_freqs`` creates a zero-filled array, so a run on a
        # memory whose grid was never transferred would evaluate every
        # frequency at f = 0 (``run(memory=...)`` checks this flag)
        self._freqs_on_device = False

        self.mag_bin_fracs = None
        self.mag_bin_fracs_g = None

        self.ytype = np.uint32 if not self.weighted else self.real_type

    def allocate_buffered_data_arrays(self, **kwargs):
        """Allocate buffered CPU arrays for data transfer."""
        n0 = kwargs.get('n0', self.n0)
        if self.buffered_transfer:
            n0 = kwargs.get('n0_buffer', self.n0_buffer)
        if not (n0 is not None):
            raise RuntimeError(
                "ConditionalEntropyMemory: requirement "
                "`n0 is not None` not satisfied")

        p = self.pinned
        self.t = host_array((n0,), self.real_type, pinned=p)
        self.y = host_array((n0,), self.ytype, pinned=p)

        if self.weighted:
            self.dy = host_array((n0,), self.real_type, pinned=p)

        if self.balanced_magbins:
            self.mag_bwf = host_array((self.mag_bins,), self.real_type,
                                      pinned=p)

        if self.compute_log_prob:
            self.mag_bin_fracs = host_array((self.mag_bins,), self.real_type,
                                            pinned=p)
        return self

    def allocate_pinned_cpu(self, **kwargs):
        """Allocate the host result buffer (page-locked by default;
        falls back to page-aligned if pinning fails)."""
        nf = kwargs.get('nf', self.nf)
        if not (nf is not None):
            raise RuntimeError(
                "ConditionalEntropyMemory: requirement "
                "`nf is not None` not satisfied")

        self.ce_c = host_array((nf,), self.real_type, pinned=self.pinned)

        return self

    def allocate_data(self, **kwargs):
        """Allocate GPU memory for input data."""
        n0 = kwargs.get('n0', self.n0)
        if self.buffered_transfer:
            n0 = kwargs.get('n0_buffer', self.n0_buffer)

        if not (n0 is not None):
            raise RuntimeError(
                "ConditionalEntropyMemory: requirement "
                "`n0 is not None` not satisfied")
        self.t_g = gpuarray.zeros(n0, dtype=self.real_type)
        self.y_g = gpuarray.zeros(n0, dtype=self.ytype)
        if self.weighted:
            self.dy_g = gpuarray.zeros(n0, dtype=self.real_type)

    def allocate_bins(self, **kwargs):
        """Allocate GPU memory for histogram bins.

        The global ``bins_g`` histogram belongs to the standard kernels;
        ``ce_classical_fast``/``_faster`` build theirs in shared memory
        and never touch it, so ``use_fast=True`` skips it (``bins_g``
        stays ``None``).  The per-magnitude-bin side arrays are small
        and are still allocated when the corresponding option is on.
        """
        nf = kwargs.get('nf', self.nf)
        if not (nf is not None):
            raise RuntimeError(
                "ConditionalEntropyMemory: requirement "
                "`nf is not None` not satisfied")

        self.nbins = nf * self.phase_bins * self.mag_bins

        if self.use_fast:
            self.bins_g = None
        elif self.weighted:
            self.bins_g = gpuarray.zeros(self.nbins, dtype=self.real_type)
        else:
            self.bins_g = gpuarray.zeros(self.nbins, dtype=np.uint32)

        if self.balanced_magbins:
            self.mag_bwf_g = gpuarray.zeros(self.mag_bins,
                                            dtype=self.real_type)
        if self.compute_log_prob:
            self.mag_bin_fracs_g = gpuarray.zeros(self.mag_bins,
                                                  dtype=self.real_type)

    def allocate_freqs(self, **kwargs):
        """Allocate GPU memory for frequency array."""
        nf = kwargs.get('nf', self.nf)
        if not (nf is not None):
            raise RuntimeError(
                "ConditionalEntropyMemory: requirement "
                "`nf is not None` not satisfied")
        self.freqs_g = gpuarray.zeros(nf, dtype=self.real_type)
        self._freqs_on_device = False
        if self.ce_g is None or self.ce_g.size != nf:
            self.ce_g = gpuarray.zeros(nf, dtype=self.real_type)

    def allocate(self, **kwargs):
        """Allocate all required GPU memory."""
        self.freqs = kwargs.get('freqs', self.freqs)
        self.nf = kwargs.get('nf', len(self.freqs))

        if self.freqs is not None:
            self.freqs = np.asarray(self.freqs).astype(self.real_type)

        if not (self.nf is not None):
            raise RuntimeError(
                "ConditionalEntropyMemory: requirement "
                "`self.nf is not None` not satisfied")

        self.allocate_data(**kwargs)
        self.allocate_bins(**kwargs)
        self.allocate_freqs(**kwargs)
        self.allocate_pinned_cpu(**kwargs)

        if self.buffered_transfer:
            self.allocate_buffered_data_arrays(**kwargs)

        return self

    def transfer_data_to_gpu(self, **kwargs):
        """Transfer data from CPU to GPU asynchronously."""
        if not (not any([x is None for x in [self.t, self.y]])):
            raise RuntimeError(
                "ConditionalEntropyMemory: requirement "
                "`not any([x is None for x in [self.t, self.y]])` not satisfied")

        self.t_g.set_async(self.t, stream=self.stream)
        self.y_g.set_async(self.y, stream=self.stream)

        if self.weighted:
            if not (self.dy is not None):
                raise RuntimeError(
                    "ConditionalEntropyMemory: requirement "
                    "`self.dy is not None` not satisfied")
            self.dy_g.set_async(self.dy, stream=self.stream)

        if self.balanced_magbins:
            self.mag_bwf_g.set_async(self.mag_bwf, stream=self.stream)

        if self.compute_log_prob:
            self.mag_bin_fracs_g.set_async(self.mag_bin_fracs,
                                           stream=self.stream)

    def transfer_freqs_to_gpu(self, **kwargs):
        """Transfer frequency array to GPU.

        Uses ``freqs`` if given (it then becomes the memory's grid),
        otherwise ``self.freqs``; the grid is cast to ``real_type``.
        ``self.freqs`` is a private copy: ``run(memory=...)`` compares
        it with the grid of the next call to decide whether to upload
        again, and for a caller's float32 grid ``np.ascontiguousarray``
        returned the caller's own array, so a grid modified in place
        between two calls compared equal to itself and stayed stale on
        the device.
        """
        freqs = kwargs.get('freqs', self.freqs)
        if not (freqs is not None):
            raise ValueError(
                "ConditionalEntropyMemory: requirement "
                "`freqs is not None` not satisfied")
        freqs = np.array(freqs, dtype=self.real_type, copy=True)
        if self.freqs_g is None or self.freqs_g.size != len(freqs):
            raise ValueError(
                "ConditionalEntropyMemory: freqs_g holds %s frequencies "
                "but %d were given; call allocate(freqs=...) first"
                % (None if self.freqs_g is None else self.freqs_g.size,
                   len(freqs)))
        self.freqs = freqs
        self.freqs_g.set_async(freqs, stream=self.stream)
        self._freqs_on_device = True

    def transfer_ce_to_cpu(self, **kwargs):
        """Transfer conditional entropy results from GPU to CPU."""
        self.ce_g.get_async(stream=self.stream, ary=self.ce_c)

    def compute_mag_bin_fracs(self, y, **kwargs):
        """Compute magnitude bin fractions for probability calculations.

        ``y`` holds integer magnitude-bin indices; the fractions sum to 1.
        """
        N = float(len(y))
        yb = np.minimum(np.asarray(y).astype(np.int64), self.mag_bins - 1)
        mbf = np.bincount(yb, minlength=self.mag_bins)[:self.mag_bins] / N

        if self.mag_bin_fracs is None:
            self.mag_bin_fracs = np.zeros(self.mag_bins, dtype=self.real_type)
        self.mag_bin_fracs[:self.mag_bins] = mbf[:]

    # Lower limit on a balanced bin's width, as a fraction of the
    # (already normalized) magnitude range.  Only reached when a whole
    # bin (and the neighbouring edges) sit on one quantized magnitude
    # value; it keeps ``log(width)`` finite.
    balanced_min_width = 1e-6

    def balance_magbins(self, y, **kwargs):
        """Create balanced magnitude bins with equal number of observations.

        The ``mag_bins`` bins each hold (as nearly as possible) the same
        number of points.  Bin edges are placed at the midpoints between
        the largest value of one group and the smallest value of the next,
        so the widths ``mag_bwf`` tile the normalized magnitude range
        ``[0, 1]`` (they sum to 1).  Widths are floored at
        ``balanced_min_width`` so that quantized magnitudes (fewer distinct
        values than points) cannot produce a zero-width bin, which would
        make the conditional entropy ``-inf``.

        Parameters
        ----------
        y : array-like
            Magnitudes, normalized to ``[0, 1]``.

        Returns
        -------
        ybins : array
            Balanced bin index of each point.
        mag_bwf : array, ``real_type``
            Width of each bin (fraction of the magnitude range).
        """
        y = np.asarray(y)
        yinds = np.argsort(y, kind='stable')
        ybins = np.zeros(len(y))

        if len(y) < self.mag_bins:
            raise ValueError(
                "balanced_magbins requires at least mag_bins=%d "
                "observations; got %d" % (self.mag_bins, len(y)))

        # integer group boundaries: bounds[-1] == len(y) exactly, so every
        # sorted point belongs to a group (``int(i * (len(y) / mag_bins))``
        # could fall one short of len(y) through float rounding and leave
        # the brightest point(s) in bin 0)
        bounds = (np.arange(self.mag_bins + 1) * len(y)) // self.mag_bins
        edges = np.zeros(self.mag_bins + 1, dtype=np.float64)
        edges[0] = np.min(y)
        edges[-1] = np.max(y)
        for i in range(self.mag_bins):
            imin, imax = int(bounds[i]), int(bounds[i + 1])

            inds = yinds[imin:imax]
            ybins[inds] = i

            if i > 0:
                # midpoint between the previous group's largest value
                # and this group's smallest value
                edges[i] = 0.5 * (float(y[yinds[imin - 1]])
                                  + float(y[yinds[imin]]))

        yrange = float(edges[-1] - edges[0])
        if yrange > 0:
            mag_bwf = np.diff(edges) / yrange
        else:
            mag_bwf = np.full(self.mag_bins, 1.0 / self.mag_bins)
        mag_bwf = np.maximum(mag_bwf, self.balanced_min_width)

        return ybins, mag_bwf.astype(self.real_type)

    def setdata(self, t, y, **kwargs):
        """
        Set data for conditional entropy computation.
        
        Parameters
        ----------
        t : array-like
            Time values
        y : array-like
            Observation values
        dy : array-like, optional
            Observation uncertainties (required if weighted=True)
        **kwargs : dict
            Additional parameters
        """
        dy = kwargs.get('dy', self.dy)

        self.n0 = kwargs.get('n0', len(t))

        t = np.asarray(t).astype(self.real_type)
        y = np.asarray(y).astype(self.real_type)

        yscale = max(y[:self.n0]) - min(y[:self.n0])
        y0 = min(y[:self.n0])
        if self.weighted:
            dy = np.asarray(dy).astype(self.real_type)
            if self.widen_mag_range:
                med_sigma = np.median(dy[:self.n0])
                yscale += 2 * self.max_phi * med_sigma
                y0 -= self.max_phi * med_sigma

            dy /= yscale
        y = (y - y0) / yscale
        if not self.weighted:
            if self.balanced_magbins:
                y, self.mag_bwf = self.balance_magbins(y)
                y = y.astype(self.ytype)

            else:
                # y is normalized to [0, 1] with the brightest point at
                # exactly 1.0, so floor(y * mag_bins) would give the
                # out-of-range index mag_bins for it: clamp into the
                # last bin.
                y = np.minimum(np.floor(y * self.mag_bins),
                               self.mag_bins - 1).astype(self.ytype)

            if self.compute_log_prob:
                self.compute_mag_bin_fracs(y[:self.n0])

        if self.buffered_transfer:
            arrs = [self.t, self.y]
            if self.weighted:
                arrs.append(self.dy)

            if any([arr is None for arr in arrs]):
                if self.buffered_transfer:
                    self.allocate_buffered_data_arrays(**kwargs)

            if not (self.n0 <= len(self.t)):
                raise RuntimeError(
                    "ConditionalEntropyMemory: requirement "
                    "`self.n0 <= len(self.t)` not satisfied")

            self.t[:self.n0] = t[:self.n0]
            self.y[:self.n0] = y[:self.n0]

            if self.weighted:
                self.dy[:self.n0] = dy[:self.n0]
        else:
            self.t = t
            self.y = y
            if self.weighted:
                self.dy = dy
        return self

    def set_gpu_arrays_to_zero(self, **kwargs):
        """Zero out GPU arrays (``bins_g`` only when it exists: the
        fast kernels do not allocate it)."""
        self.t_g.fill(self.real_type(0), stream=self.stream)
        self.y_g.fill(self.ytype(0), stream=self.stream)
        if self.weighted:
            self.dy_g.fill(self.real_type(0), stream=self.stream)
        if self.bins_g is not None:
            self.bins_g.fill(self.bins_g.dtype.type(0), stream=self.stream)

    def fromdata(self, t, y, **kwargs):
        """
        Initialize memory from data arrays.
        
        Parameters
        ----------
        t : array-like
            Time values
        y : array-like
            Observation values
        allocate : bool, optional (default: True)
            Whether to allocate GPU memory
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        self : ConditionalEntropyMemory
        """
        self.setdata(t, y, **kwargs)

        if kwargs.get('allocate', True):
            self.allocate(**kwargs)

        return self
