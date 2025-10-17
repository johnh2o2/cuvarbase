"""
Memory management for Conditional Entropy period-finding operations.
"""
import resource
import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray


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
    **kwargs : dict
        Additional parameters
    """
    
    def __init__(self, **kwargs):
        self.phase_bins = kwargs.get('phase_bins', 10)
        self.mag_bins = kwargs.get('mag_bins', 5)
        self.phase_overlap = kwargs.get('phase_overlap', 0)
        self.mag_overlap = kwargs.get('mag_overlap', 0)

        self.max_phi = kwargs.get('max_phi', 3.)
        self.stream = kwargs.get('stream', None)
        self.weighted = kwargs.get('weighted', False)
        self.widen_mag_range = kwargs.get('widen_mag_range', False)
        self.n0 = kwargs.get('n0', None)
        self.nf = kwargs.get('nf', None)

        self.compute_log_prob = kwargs.get('compute_log_prob', False)

        self.balanced_magbins = kwargs.get('balanced_magbins', False)

        if self.weighted and self.balanced_magbins:
            raise Exception("simultaneous balanced_magbins and weighted"
                            " options is not currently supported")

        if self.weighted and self.compute_log_prob:
            raise Exception("simultaneous compute_log_prob and weighted"
                            " options is not currently supported")
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

        self.mag_bin_fracs = None
        self.mag_bin_fracs_g = None

        self.ytype = np.uint32 if not self.weighted else self.real_type

    def allocate_buffered_data_arrays(self, **kwargs):
        """Allocate buffered CPU arrays for data transfer."""
        n0 = kwargs.get('n0', self.n0)
        if self.buffered_transfer:
            n0 = kwargs.get('n0_buffer', self.n0_buffer)
        assert(n0 is not None)

        kw = dict(dtype=self.real_type,
                  alignment=resource.getpagesize())

        self.t = cuda.aligned_zeros(shape=(n0,), **kw)

        self.y = cuda.aligned_zeros(shape=(n0,),
                                    dtype=self.ytype,
                                    alignment=resource.getpagesize())

        if self.weighted:
            self.dy = cuda.aligned_zeros(shape=(n0,), **kw)

        if self.balanced_magbins:
            self.mag_bwf = cuda.aligned_zeros(shape=(self.mag_bins,), **kw)

        if self.compute_log_prob:
            self.mag_bin_fracs = cuda.aligned_zeros(shape=(self.mag_bins,),
                                                    **kw)
        return self

    def allocate_pinned_cpu(self, **kwargs):
        """Allocate pinned CPU memory for async transfers."""
        nf = kwargs.get('nf', self.nf)
        assert(nf is not None)

        self.ce_c = cuda.aligned_zeros(shape=(nf,), dtype=self.real_type,
                                       alignment=resource.getpagesize())

        return self

    def allocate_data(self, **kwargs):
        """Allocate GPU memory for input data."""
        n0 = kwargs.get('n0', self.n0)
        if self.buffered_transfer:
            n0 = kwargs.get('n0_buffer', self.n0_buffer)

        assert(n0 is not None)
        self.t_g = gpuarray.zeros(n0, dtype=self.real_type)
        self.y_g = gpuarray.zeros(n0, dtype=self.ytype)
        if self.weighted:
            self.dy_g = gpuarray.zeros(n0, dtype=self.real_type)

    def allocate_bins(self, **kwargs):
        """Allocate GPU memory for histogram bins."""
        nf = kwargs.get('nf', self.nf)
        assert(nf is not None)

        self.nbins = nf * self.phase_bins * self.mag_bins

        if self.weighted:
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
        assert(nf is not None)
        self.freqs_g = gpuarray.zeros(nf, dtype=self.real_type)
        if self.ce_g is None:
            self.ce_g = gpuarray.zeros(nf, dtype=self.real_type)

    def allocate(self, **kwargs):
        """Allocate all required GPU memory."""
        self.freqs = kwargs.get('freqs', self.freqs)
        self.nf = kwargs.get('nf', len(self.freqs))

        if self.freqs is not None:
            self.freqs = np.asarray(self.freqs).astype(self.real_type)

        assert(self.nf is not None)

        self.allocate_data(**kwargs)
        self.allocate_bins(**kwargs)
        self.allocate_freqs(**kwargs)
        self.allocate_pinned_cpu(**kwargs)

        if self.buffered_transfer:
            self.allocate_buffered_data_arrays(**kwargs)

        return self

    def transfer_data_to_gpu(self, **kwargs):
        """Transfer data from CPU to GPU asynchronously."""
        assert(not any([x is None for x in [self.t, self.y]]))

        self.t_g.set_async(self.t, stream=self.stream)
        self.y_g.set_async(self.y, stream=self.stream)

        if self.weighted:
            assert(self.dy is not None)
            self.dy_g.set_async(self.dy, stream=self.stream)

        if self.balanced_magbins:
            self.mag_bwf_g.set_async(self.mag_bwf, stream=self.stream)

        if self.compute_log_prob:
            self.mag_bin_fracs_g.set_async(self.mag_bin_fracs,
                                           stream=self.stream)

    def transfer_freqs_to_gpu(self, **kwargs):
        """Transfer frequency array to GPU."""
        freqs = kwargs.get('freqs', self.freqs)
        assert(freqs is not None)

        self.freqs_g.set_async(freqs, stream=self.stream)

    def transfer_ce_to_cpu(self, **kwargs):
        """Transfer conditional entropy results from GPU to CPU."""
        self.ce_g.get_async(stream=self.stream, ary=self.ce_c)

    def compute_mag_bin_fracs(self, y, **kwargs):
        """Compute magnitude bin fractions for probability calculations."""
        N = float(len(y))
        mbf = np.array([np.sum(y == i)/N for i in range(self.mag_bins)])

        if self.mag_bin_fracs is None:
            self.mag_bin_fracs = np.zeros(self.mag_bins, dtype=self.real_type)
        self.mag_bin_fracs[:self.mag_bins] = mbf[:]

    def balance_magbins(self, y, **kwargs):
        """Create balanced magnitude bins with equal number of observations."""
        yinds = np.argsort(y)
        ybins = np.zeros(len(y))

        assert len(y) >= self.mag_bins

        di = len(y) / self.mag_bins
        mag_bwf = np.zeros(self.mag_bins)
        for i in range(self.mag_bins):
            imin = max([0, int(i * di)])
            imax = min([len(y), int((i + 1) * di)])

            inds = yinds[imin:imax]
            ybins[inds] = i

            mag_bwf[i] = y[inds[-1]] - y[inds[0]]

        mag_bwf /= (max(y) - min(y))

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
                y = np.floor(y * self.mag_bins).astype(self.ytype)

            if self.compute_log_prob:
                self.compute_mag_bin_fracs(y)

        if self.buffered_transfer:
            arrs = [self.t, self.y]
            if self.weighted:
                arrs.append(self.dy)

            if any([arr is None for arr in arrs]):
                if self.buffered_transfer:
                    self.allocate_buffered_data_arrays(**kwargs)

            assert(self.n0 <= len(self.t))

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
        """Zero out GPU arrays."""
        self.t_g.fill(self.real_type(0), stream=self.stream)
        self.y_g.fill(self.ytype(0), stream=self.stream)
        if self.weighted:
            self.bins_g.fill(self.real_type(0), stream=self.stream)
            self.dy_g.fill(self.real_type(0), stream=self.stream)
        else:
            self.bins_g.fill(np.uint32(0), stream=self.stream)

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
