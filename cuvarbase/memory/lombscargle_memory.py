"""
Memory management for Lomb-Scargle periodogram computations.
"""
import numpy as np

import pycuda.driver as cuda  # noqa: F401  (used by transfer methods)
import pycuda.gpuarray as gpuarray

from ..base import ensure_context
from ._host import host_array
from .nfft_memory import NFFTMemory


def weights(err):
    """
    Generate observation weights from uncertainties.
    
    Note: This function is also available in cuvarbase.utils for backward compatibility.
    
    Parameters
    ----------
    err : array-like
        Observation uncertainties
        
    Returns
    -------
    weights : ndarray
        Normalized weights (inverse square of errors, normalized to sum to 1)
    """
    w = np.power(err, -2)
    return w/sum(w)


class LombScargleMemory:
    """
    Container class for allocating memory and transferring
    data between the GPU and CPU for Lomb-Scargle computations.
    
    Parameters
    ----------
    sigma : float
        The sigma parameter for the NFFT
    stream : pycuda.driver.Stream
        The CUDA stream used for calculations/data transfer
    m : int
        The m parameter for the NFFT
    **kwargs : dict
        Additional parameters
    """
    def __init__(self, sigma, stream, m, **kwargs):
        # Constructing GPU memory is a "first GPU use" -- retain the CUDA
        # primary context now (no longer created eagerly at import). This
        # __init__ allocates reg_g immediately, so the context must exist.
        ensure_context()

        self.sigma = sigma
        self.stream = stream
        self.m = m
        self.k0 = kwargs.get('k0', 0)
        self.precomp_psi = kwargs.get('precomp_psi', True)
        self.amplitude_prior = kwargs.get('amplitude_prior', None)
        self.window = kwargs.get('window', False)
        self.nharmonics = kwargs.get('nharmonics', 1)
        self.use_fft = kwargs.get('use_fft', True)
        # Pinned (page-locked) host buffers by default for async overlap;
        # graceful fallback to page-aligned if pinning fails.
        self.pinned = kwargs.get('pinned', True)

        self.other_settings = {}
        self.other_settings.update(kwargs)

        self.floating_mean = kwargs.get('floating_mean', True)
        self.use_double = kwargs.get('use_double', False)

        self.mode = 1 if self.floating_mean else 0
        if self.window:
            self.mode = 2

        self.n0 = kwargs.get('n0', None)
        self.nf = kwargs.get('nf', None)

        self.t_g = kwargs.get('t_g', None)
        self.yw_g = kwargs.get('yw_g', None)
        self.w_g = kwargs.get('w_g', None)
        self.lsp_g = kwargs.get('lsp_g', None)

        if self.use_fft:
            self.nfft_mem_yw = kwargs.get('nfft_mem_yw', None)
            self.nfft_mem_w = kwargs.get('nfft_mem_w', None)

            if self.nfft_mem_yw is None:
                self.nfft_mem_yw = NFFTMemory(self.sigma, self.stream,
                                              self.m, **kwargs)

            if self.nfft_mem_w is None:
                self.nfft_mem_w = NFFTMemory(self.sigma, self.stream,
                                             self.m, **kwargs)

            self.real_type = self.nfft_mem_yw.real_type
            self.complex_type = self.nfft_mem_yw.complex_type

        else:
            self.real_type = np.float32
            self.complex_type = np.complex64

            if self.use_double:
                self.real_type = np.float64
                self.complex_type = np.complex128

        # Set up regularization
        self.reg_g = gpuarray.zeros(2 * self.nharmonics + 1,
                                    dtype=self.real_type)
        self.reg = np.zeros(2 * self.nharmonics + 1,
                            dtype=self.real_type)

        if self.amplitude_prior is not None:
            lmbda = np.power(self.amplitude_prior, -2)
            if isinstance(lmbda, float):
                lmbda = lmbda * np.ones(self.nharmonics)

            for i, l in enumerate(lmbda):
                self.reg[2 * i] = self.real_type(l)
                self.reg[1 + 2 * i] = self.real_type(l)

            self.reg_g.set_async(self.reg, stream=self.stream)

        self.buffered_transfer = kwargs.get('buffered_transfer', False)
        self.n0_buffer = kwargs.get('n0_buffer', None)

        self.lsp_c = kwargs.get('lsp_c', None)

        self.t = kwargs.get('t', None)
        self.yw = kwargs.get('yw', None)
        self.w = kwargs.get('w', None)

    def allocate_data(self, **kwargs):
        """Allocates memory for lightcurve."""
        n0 = kwargs.get('n0', self.n0)
        if self.buffered_transfer:
            n0 = kwargs.get('n0_buffer', self.n0_buffer)

        if not (n0 is not None):
            raise RuntimeError(
                "LombScargleMemory: requirement "
                "`n0 is not None` not satisfied")
        self.t_g = gpuarray.zeros(n0, dtype=self.real_type)
        self.yw_g = gpuarray.zeros(n0, dtype=self.real_type)
        self.w_g = gpuarray.zeros(n0, dtype=self.real_type)

        if self.use_fft:
            self.nfft_mem_w.t_g = self.t_g
            self.nfft_mem_w.y_g = self.w_g

            self.nfft_mem_yw.t_g = self.t_g
            self.nfft_mem_yw.y_g = self.yw_g

            self.nfft_mem_yw.n0 = n0
            self.nfft_mem_w.n0 = n0

        return self

    def allocate_grids(self, **kwargs):
        """
        Allocates memory for NFFT grids, NFFT precomputation vectors,
        and the GPU vector for the Lomb-Scargle power.
        """
        k0 = kwargs.get('k0', self.k0)
        n0 = kwargs.get('n0', self.n0)
        if self.buffered_transfer:
            n0 = kwargs.get('n0_buffer', self.n0_buffer)
        if not (n0 is not None):
            raise RuntimeError(
                "LombScargleMemory: requirement "
                "`n0 is not None` not satisfied")

        self.nf = kwargs.get('nf', self.nf)
        if not (self.nf is not None):
            raise RuntimeError(
                "LombScargleMemory: requirement "
                "`self.nf is not None` not satisfied")

        if self.use_fft:
            if self.nfft_mem_yw.precomp_psi:
                self.nfft_mem_yw.allocate_precomp_psi(n0=n0)

            # Only one precomp psi needed
            self.nfft_mem_w.precomp_psi = False
            self.nfft_mem_w.q1 = self.nfft_mem_yw.q1
            self.nfft_mem_w.q2 = self.nfft_mem_yw.q2
            self.nfft_mem_w.q3 = self.nfft_mem_yw.q3

            fft_size = self.nharmonics * (self.nf + k0)
            self.nfft_mem_yw.allocate_grid(nf=fft_size - k0)
            self.nfft_mem_w.allocate_grid(nf=2 * fft_size - k0)

        self.lsp_g = gpuarray.zeros(self.nf, dtype=self.real_type)
        return self

    def allocate_pinned_cpu(self, **kwargs):
        """Allocate the host result buffer (page-locked by default;
        falls back to page-aligned if pinning fails)."""
        nf = kwargs.get('nf', self.nf)
        if not (nf is not None):
            raise RuntimeError(
                "LombScargleMemory: requirement "
                "`nf is not None` not satisfied")

        self.lsp_c = host_array((nf,), self.real_type, pinned=self.pinned)

        return self

    def is_ready(self):
        """Verify all required memory is allocated for a run.

        Raises RuntimeError if frequencies or device arrays are
        missing or inconsistently sized (mirrors
        ``NFFTMemory.is_ready``).
        """
        if self.nf is None:
            raise RuntimeError(
                "LombScargleMemory: nf is not set (call allocate "
                "first)")
        if self.lsp_g is None or len(self.lsp_g) < self.nf:
            raise RuntimeError(
                "LombScargleMemory: lsp_g is not allocated for "
                "nf=%d" % self.nf)
        if any(arr is None for arr in (self.t_g, self.yw_g, self.w_g)):
            raise RuntimeError(
                "LombScargleMemory: data arrays (t_g, yw_g, w_g) are "
                "not allocated (call allocate_data first)")
        if self.use_fft:
            self.nfft_mem_yw.is_ready()
            self.nfft_mem_w.is_ready()

    def allocate_buffered_data_arrays(self, **kwargs):
        """
        Allocate host memory for lightcurves if we're reusing this
        container (page-locked by default; page-aligned fallback).
        """
        n0 = kwargs.get('n0', self.n0)
        if self.buffered_transfer:
            n0 = kwargs.get('n0_buffer', self.n0_buffer)
        if not (n0 is not None):
            raise RuntimeError(
                "LombScargleMemory: requirement "
                "`n0 is not None` not satisfied")

        self.t = host_array((n0,), self.real_type, pinned=self.pinned)
        self.yw = host_array((n0,), self.real_type, pinned=self.pinned)
        self.w = host_array((n0,), self.real_type, pinned=self.pinned)

        return self

    def allocate(self, **kwargs):
        """Allocate all memory necessary."""
        self.nf = kwargs.get('nf', self.nf)
        if not (self.nf is not None):
            raise RuntimeError(
                "LombScargleMemory: requirement "
                "`self.nf is not None` not satisfied")

        self.allocate_data(**kwargs)
        self.allocate_grids(**kwargs)
        self.allocate_pinned_cpu(**kwargs)

        if self.buffered_transfer:
            self.allocate_buffered_data_arrays(**kwargs)

        return self

    def setdata(self, **kwargs):
        """Sets the value of the data arrays."""
        t = kwargs.get('t', self.t)
        yw = kwargs.get('yw', self.yw)
        w = kwargs.get('w', self.w)

        y = kwargs.get('y', None)
        dy = kwargs.get('dy', None)
        self.ybar = 0.
        self.yy = kwargs.get('yy', 1.)

        self.n0 = kwargs.get('n0', len(t))
        if dy is not None:
            if not ('w' not in kwargs):
                raise ValueError(
                    "LombScargleMemory: requirement "
                    "`'w' not in kwargs` not satisfied")
            w = weights(dy)

        if y is not None:
            if not ('yw' not in kwargs):
                raise ValueError(
                    "LombScargleMemory: requirement "
                    "`'yw' not in kwargs` not satisfied")

            self.ybar = np.dot(y, w)
            yw = np.multiply(w, y - self.ybar)
            y2 = np.power(y - self.ybar, 2)
            self.yy = np.dot(w, y2)

        t = np.asarray(t).astype(self.real_type)
        yw = np.asarray(yw).astype(self.real_type)
        w = np.asarray(w).astype(self.real_type)

        if self.buffered_transfer:
            if any([arr is None for arr in [self.t, self.yw, self.w]]):
                if self.buffered_transfer:
                    self.allocate_buffered_data_arrays(**kwargs)

            if not (self.n0 <= len(self.t)):
                raise RuntimeError(
                    "LombScargleMemory: requirement "
                    "`self.n0 <= len(self.t)` not satisfied")

            self.t[:self.n0] = t[:self.n0]
            self.yw[:self.n0] = yw[:self.n0]
            self.w[:self.n0] = w[:self.n0]
        else:
            self.t = np.asarray(t).astype(self.real_type)
            self.yw = np.asarray(yw).astype(self.real_type)
            self.w = np.asarray(w).astype(self.real_type)

        # Set minimum and maximum t values (needed to scale things
        # for the NFFT)
        self.tmin = min(t)
        self.tmax = max(t)

        if self.use_fft:
            self.nfft_mem_yw.tmin = self.tmin
            self.nfft_mem_w.tmin = self.tmin

            self.nfft_mem_yw.tmax = self.tmax
            self.nfft_mem_w.tmax = self.tmax

            self.nfft_mem_w.n0 = len(t)
            self.nfft_mem_yw.n0 = len(t)

        return self

    def transfer_data_to_gpu(self, **kwargs):
        """Transfers the lightcurve to the GPU."""
        t, yw, w = self.t, self.yw, self.w

        if not (not any([arr is None for arr in [t, yw, w]])):
            raise RuntimeError(
                "LombScargleMemory: requirement "
                "`not any([arr is None for arr in [t, yw, w]])` not satisfied")

        # Do asynchronous data transfer
        self.t_g.set_async(t, stream=self.stream)
        self.yw_g.set_async(yw, stream=self.stream)
        self.w_g.set_async(w, stream=self.stream)

    def transfer_lsp_to_cpu(self, **kwargs):
        """Asynchronous transfer of LSP result to CPU."""
        self.lsp_g.get_async(ary=self.lsp_c, stream=self.stream)

    def fromdata(self, **kwargs):
        """Sets and (optionally) allocates memory for data."""
        self.setdata(**kwargs)

        if kwargs.get('allocate', True):
            self.allocate(**kwargs)

        return self

    def set_gpu_arrays_to_zero(self, **kwargs):
        """Sets all gpu arrays to zero."""
        for x in [self.t_g, self.yw_g, self.w_g]:
            if x is not None:
                x.fill(self.real_type(0), stream=self.stream)

        for x in [self.t, self.yw, self.w]:
            if x is not None:
                x[:] = 0.

        if hasattr(self, 'nfft_mem_yw'):
            self.nfft_mem_yw.ghat_g.fill(self.complex_type(0),
                                         stream=self.stream)
        if hasattr(self, 'nfft_mem_w'):
            self.nfft_mem_w.ghat_g.fill(self.complex_type(0),
                                        stream=self.stream)
