"""
Memory management for NFFT (Non-equispaced Fast Fourier Transform) operations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import object

import resource
import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import skcuda.fft as cufft


class NFFTMemory(object):
    """
    Container class for managing memory allocation and data transfer
    for NFFT computations on GPU.
    
    Parameters
    ----------
    sigma : float
        Oversampling factor for NFFT
    stream : pycuda.driver.Stream
        CUDA stream for asynchronous operations
    m : int
        NFFT truncation parameter
    use_double : bool, optional (default: False)
        Use double precision floating point
    precomp_psi : bool, optional (default: True)
        Precompute psi values for faster gridding
    **kwargs : dict
        Additional parameters
    """
    
    def __init__(self, sigma, stream, m, use_double=False,
                 precomp_psi=True, **kwargs):

        self.sigma = sigma
        self.stream = stream
        self.m = m
        self.use_double = use_double
        self.precomp_psi = precomp_psi

        # set datatypes
        self.real_type = np.float32 if not self.use_double \
            else np.float64
        self.complex_type = np.complex64 if not self.use_double \
            else np.complex128

        self.other_settings = {}
        self.other_settings.update(kwargs)

        self.t = kwargs.get('t', None)
        self.y = kwargs.get('y', None)
        self.f0 = kwargs.get('f0', 0.)
        self.n0 = kwargs.get('n0', None)
        self.nf = kwargs.get('nf', None)
        self.t_g = kwargs.get('t_g', None)
        self.y_g = kwargs.get('y_g', None)
        self.ghat_g = kwargs.get('ghat_g', None)
        self.ghat_c = kwargs.get('ghat_c', None)
        self.q1 = kwargs.get('q1', None)
        self.q2 = kwargs.get('q2', None)
        self.q3 = kwargs.get('q3', None)
        self.cu_plan = kwargs.get('cu_plan', None)

        D = (2 * self.sigma - 1) * np.pi
        self.b = float(2 * self.sigma * self.m) / D

    def allocate_data(self, **kwargs):
        """Allocate GPU memory for input data (times and values)."""
        self.n0 = kwargs.get('n0', self.n0)
        self.nf = kwargs.get('nf', self.nf)

        assert(self.n0 is not None)
        assert(self.nf is not None)

        self.t_g = gpuarray.zeros(self.n0, dtype=self.real_type)
        self.y_g = gpuarray.zeros(self.n0, dtype=self.real_type)

        return self

    def allocate_precomp_psi(self,  **kwargs):
        """Allocate memory for precomputed psi values."""
        self.n0 = kwargs.get('n0', self.n0)

        assert(self.n0 is not None)

        self.q1 = gpuarray.zeros(self.n0, dtype=self.real_type)
        self.q2 = gpuarray.zeros(self.n0, dtype=self.real_type)
        self.q3 = gpuarray.zeros(2 * self.m + 1, dtype=self.real_type)

        return self

    def allocate_grid(self, **kwargs):
        """Allocate GPU memory for the frequency grid."""
        self.nf = kwargs.get('nf', self.nf)

        assert(self.nf is not None)

        self.n = int(self.sigma * self.nf)
        self.ghat_g = gpuarray.zeros(self.n,
                                     dtype=self.complex_type)
        self.cu_plan = cufft.Plan(self.n, self.complex_type, self.complex_type,
                                  stream=self.stream)
        return self

    def allocate_pinned_cpu(self, **kwargs):
        """Allocate pinned CPU memory for async transfers."""
        self.nf = kwargs.get('nf', self.nf)

        assert(self.nf is not None)
        self.ghat_c = cuda.aligned_zeros(shape=(self.nf,),
                                         dtype=self.complex_type,
                                         alignment=resource.getpagesize())

        return self

    def is_ready(self):
        """Verify all required memory is allocated."""
        assert(self.n0 == len(self.t_g))
        assert(self.n0 == len(self.y_g))
        assert(self.n == len(self.ghat_g))

        if self.ghat_c is not None:
            assert(self.nf == len(self.ghat_c))

        if self.precomp_psi:
            assert(self.n0 == len(self.q1))
            assert(self.n0 == len(self.q2))
            assert(2 * self.m + 1 == len(self.q3))

    def allocate(self, **kwargs):
        """Allocate all required memory for NFFT computation."""
        self.n0 = kwargs.get('n0', self.n0)
        self.nf = kwargs.get('nf', self.nf)

        assert(self.n0 is not None)
        assert(self.nf is not None)
        self.n = int(self.sigma * self.nf)

        self.allocate_data(**kwargs)
        self.allocate_grid(**kwargs)
        self.allocate_pinned_cpu(**kwargs)
        if self.precomp_psi:
            self.allocate_precomp_psi(**kwargs)

        return self

    def transfer_data_to_gpu(self, **kwargs):
        """Transfer data from CPU to GPU asynchronously."""
        t = kwargs.get('t', self.t)
        y = kwargs.get('y', self.y)

        assert(t is not None)
        assert(y is not None)

        self.t_g.set_async(t, stream=self.stream)
        self.y_g.set_async(y, stream=self.stream)

    def transfer_nfft_to_cpu(self, **kwargs):
        """Transfer NFFT result from GPU to CPU asynchronously."""
        cuda.memcpy_dtoh_async(self.ghat_c, self.ghat_g.ptr,
                               stream=self.stream)

    def fromdata(self, t, y, allocate=True, **kwargs):
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
        self : NFFTMemory
        """
        self.tmin = min(t)
        self.tmax = max(t)

        self.t = np.asarray(t).astype(self.real_type)
        self.y = np.asarray(y).astype(self.real_type)

        self.n0 = kwargs.get('n0', len(t))
        self.nf = kwargs.get('nf', self.nf)

        if self.nf is not None and allocate:
            self.allocate(**kwargs)

        return self
