"""
Memory management for NFFT (Non-equispaced Fast Fourier Transform) operations.
"""
import numpy as np

import pycuda.driver as cuda  # noqa: F401  (used by transfer methods)
import pycuda.gpuarray as gpuarray

from ..base import ensure_context
from ._host import host_array
from .. import _cufft as cufft


def next_fast_len(n):
    """Smallest integer ``>= n`` whose prime factors are all in
    {2, 3, 5, 7} -- the radices cuFFT has dedicated fast kernels for.

    Other lengths fall back to Bluestein's algorithm, which is several
    times slower and needs a much larger work area (the Lomb-Scargle
    grids sized by ``sigma * (nf + k0)`` are essentially never smooth
    by accident: an audit measured cuFFT 0.83 -> 0.08 ms at n ~ 2.9e6
    from padding alone). Padding a gridded NFFT to a longer grid is
    harmless -- the transform is evaluated at the same modes, on a
    finer grid, so the result moves slightly *toward* the exact DFT.

    Parameters
    ----------
    n : int
        Minimum length.

    Returns
    -------
    int
        The smallest 7-smooth number ``>= max(n, 1)``.
    """
    n = int(n)
    if n <= 1:
        return 1
    best = 1 << (n - 1).bit_length()          # power of two >= n
    p7 = 1
    while p7 < best:
        p5 = p7
        while p5 < best:
            p3 = p5
            while p3 < best:
                # smallest power of two that lifts p3 to >= n
                q = -(-n // p3)
                cand = p3 << max(0, (q - 1).bit_length())
                if cand < best:
                    best = cand
                p3 *= 3
            p5 *= 5
        p7 *= 7
    return best


class NFFTMemory:
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
        # Constructing GPU memory is a "first GPU use" -- retain the CUDA
        # primary context now (no longer created eagerly at import).
        ensure_context()

        self.sigma = sigma
        self.stream = stream
        self.m = m
        self.use_double = use_double
        self.precomp_psi = precomp_psi
        # Pinned (page-locked) host buffer by default; falls back to
        # page-aligned if pinning fails.
        self.pinned = kwargs.get('pinned', True)

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

        if not (self.n0 is not None):
            raise RuntimeError(
                "NFFTMemory: requirement "
                "`self.n0 is not None` not satisfied")
        if not (self.nf is not None):
            raise RuntimeError(
                "NFFTMemory: requirement "
                "`self.nf is not None` not satisfied")

        self.t_g = gpuarray.zeros(self.n0, dtype=self.real_type)
        self.y_g = gpuarray.zeros(self.n0, dtype=self.real_type)

        return self

    def allocate_precomp_psi(self,  **kwargs):
        """Allocate memory for precomputed psi values."""
        self.n0 = kwargs.get('n0', self.n0)

        if not (self.n0 is not None):
            raise RuntimeError(
                "NFFTMemory: requirement "
                "`self.n0 is not None` not satisfied")

        self.q1 = gpuarray.zeros(self.n0, dtype=self.real_type)
        self.q2 = gpuarray.zeros(self.n0, dtype=self.real_type)
        self.q3 = gpuarray.zeros(2 * self.m + 1, dtype=self.real_type)

        return self

    def allocate_grid(self, **kwargs):
        """Allocate the oversampled grid ``ghat_g`` and its cuFFT plan.

        Parameters
        ----------
        nf : int, optional
            Number of modes the transform is evaluated at (entries
            ``ghat_g[0:nf]`` after ``normalize``). Defaults to
            ``self.nf``.
        n : int, optional
            Grid (FFT) length. Defaults to ``int(sigma * nf)``, which
            is right for the *centred* convention (modes
            ``-nf/2 .. nf/2 - 1``). Callers that read one-sided modes
            ``k0 .. k0 + nf - 1`` (the Lomb-Scargle memory) must size
            the grid from the top mode instead, ``>= sigma * (k0 + nf)``,
            and may pad to :func:`next_fast_len`.
        """
        self.nf = kwargs.get('nf', self.nf)

        if not (self.nf is not None):
            raise RuntimeError(
                "NFFTMemory: requirement "
                "`self.nf is not None` not satisfied")

        n = kwargs.get('n', None)
        self.n = int(self.sigma * self.nf) if n is None else int(n)
        if self.n < self.nf:
            raise ValueError(
                "NFFTMemory: grid length n=%d is smaller than the number "
                "of requested modes nf=%d" % (self.n, self.nf))
        self.ghat_g = gpuarray.zeros(self.n,
                                     dtype=self.complex_type)
        self.cu_plan = cufft.Plan(self.n, self.complex_type, self.complex_type,
                                  stream=self.stream)
        return self

    def allocate_pinned_cpu(self, **kwargs):
        """Allocate the host result buffer (page-locked by default).

        With ``pinned=True`` (default) the array is page-locked so
        ``get_async`` overlaps with computation; falls back to
        page-aligned memory if pinning fails.
        """
        self.nf = kwargs.get('nf', self.nf)

        if not (self.nf is not None):
            raise RuntimeError(
                "NFFTMemory: requirement "
                "`self.nf is not None` not satisfied")
        self.ghat_c = host_array((self.nf,), self.complex_type,
                                 pinned=self.pinned)

        return self

    def is_ready(self):
        """Verify all required memory is allocated."""
        if not (self.n0 == len(self.t_g)):
            raise RuntimeError(
                "NFFTMemory: requirement "
                "`self.n0 == len(self.t_g)` not satisfied")
        if not (self.n0 == len(self.y_g)):
            raise RuntimeError(
                "NFFTMemory: requirement "
                "`self.n0 == len(self.y_g)` not satisfied")
        if not (self.n == len(self.ghat_g)):
            raise RuntimeError(
                "NFFTMemory: requirement "
                "`self.n == len(self.ghat_g)` not satisfied")

        if self.ghat_c is not None:
            if not (self.nf == len(self.ghat_c)):
                raise RuntimeError(
                    "NFFTMemory: requirement "
                    "`self.nf == len(self.ghat_c)` not satisfied")

        if self.precomp_psi:
            if not (self.n0 == len(self.q1)):
                raise RuntimeError(
                    "NFFTMemory: requirement "
                    "`self.n0 == len(self.q1)` not satisfied")
            if not (self.n0 == len(self.q2)):
                raise RuntimeError(
                    "NFFTMemory: requirement "
                    "`self.n0 == len(self.q2)` not satisfied")
            if not (2 * self.m + 1 == len(self.q3)):
                raise RuntimeError(
                    "NFFTMemory: requirement "
                    "`2 * self.m + 1 == len(self.q3)` not satisfied")

    def allocate(self, **kwargs):
        """Allocate all required memory for NFFT computation."""
        self.n0 = kwargs.get('n0', self.n0)
        self.nf = kwargs.get('nf', self.nf)

        if not (self.n0 is not None):
            raise RuntimeError(
                "NFFTMemory: requirement "
                "`self.n0 is not None` not satisfied")
        if not (self.nf is not None):
            raise RuntimeError(
                "NFFTMemory: requirement "
                "`self.nf is not None` not satisfied")
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

        if not (t is not None):
            raise ValueError(
                "NFFTMemory: requirement "
                "`t is not None` not satisfied")
        if not (y is not None):
            raise ValueError(
                "NFFTMemory: requirement "
                "`y is not None` not satisfied")

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
