"""
Lomb-Scargle periodogram implementation.

GPU-accelerated implementation of the generalized Lomb-Scargle periodogram.
"""
import resource

import numpy as np
from scipy.special import gamma, gammaln

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
# import pycuda.autoinit

from . import _cufft as cufft

from .core import GPUAsyncProcess
from .utils import find_kernel, _module_reader, normalize_light_curves
from .utils import autofrequency as utils_autofreq
from .memory import NFFTMemory, LombScargleMemory, weights
from .cunfft import NFFTAsyncProcess, nfft_adjoint_async

try:
    from .cufinufft_backend import cufinufft_nfft_adjoint, HAS_CUFINUFFT
except ImportError:
    HAS_CUFINUFFT = False



def get_k0(freqs):
    return max([1, int(round(freqs[0] / (freqs[1] - freqs[0])))])


def check_k0(freqs, k0=None, rtol=1E-2, atol=1E-7):
    k0 = k0 if k0 is not None else get_k0(freqs)
    df = freqs[1] - freqs[0]
    f0 = k0 * df
    if not (abs(f0 - freqs[0]) < rtol * df + atol):
        raise ValueError(
            "freqs[0]=%g is not k0 * df for integer k0 (df=%g): the GPU "
            "Lomb-Scargle requires freqs = df * (k0 + arange(nf))"
            % (freqs[0], df))


def mhdirect_sums(t, yw, w, freq, YY, nharms=1):
    """
    Compute the set of frequency-dependent sums
    for (multi-harmonic) Lomb Scargle at a given frequency

    Parameters
    ----------
    t: array_like
        Observation times.
    yw: array_like
        Observations multiplied by their corresponding weights.
        `sum(yw)` is the weighted mean.
    w: array_like
        Weights for each of the observations. Usually proportional
        to `1/sigma ** 2` where `sigma` is the observation uncertainties.
        Normalized so that `sum(w) = 1`.
    freq: float
        Signal frequency.
    YY: float
        Weighted variance of the observations.
    nharms: int, optional (default: 1)
        Number of harmonics to compute. This is 1 for the standard
        Lomb-Scargle

    Returns
    -------
    C, S, CC, CS, SS, YC, YS: array_like
        The set of sums with regularization added.

    See also
    --------
    :func:`mhdirect_sums`
    """
    phase = 2 * np.pi * ((t * freq) % 1.0).astype(np.float64)

    ns = np.arange(2 * nharms + 1)

    c = [np.dot(w, np.cos(n * phase)) for n in ns]
    s = [np.dot(w, np.sin(n * phase)) for n in ns]

    yc = np.asarray([np.dot(yw, np.cos(n * phase))
                     for n in ns[1:nharms+1]])
    ys = np.asarray([np.dot(yw, np.sin(n * phase))
                     for n in ns[1:nharms+1]])

    ybar = sum(yw)
    C = np.asarray(c)[1:nharms+1]
    S = np.asarray(s)[1:nharms+1]
    YC = yc - ybar * C
    YS = ys - ybar * S

    return _mh_assemble_from_centered(c, s, YC, YS, nharms)


def _mh_assemble_from_centered(c, s, YC, YS, nharms):
    """Assemble the (C, S, CC, CS, SS, YC, YS) multiharmonic GLS sums from
    raw weight moments and already-mean-subtracted YC/YS.

    Shared by :func:`mhdirect_sums` (which gets the moments from direct
    trig sums) and the GPU multiharmonic path (which reads them off the
    NFFT spectra of ``w`` and ``w*(y-ybar)``).

    Parameters
    ----------
    c, s : array_like
        Weight moments, length ``2*nharms+1``:
        ``c[m] = sum w cos(2 pi m f t)``, ``s[m] = sum w sin(2 pi m f t)``
        (so ``c[0]=sum w=1``, ``s[0]=0``). Indices up to ``2*nharms`` are
        needed for the cross-term matrices.
    YC, YS : array_like
        Already mean-subtracted ``sum w (y-ybar) cos/sin(2 pi h f t)`` for
        ``h = 1..nharms``.
    nharms : int
        Number of harmonics.
    """
    c = np.asarray(c, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    H = nharms

    def sgn(n):
        return 1 if n == 0 else np.sign(n)

    hs = range(1, H + 1)
    cc = [[0.5 * (c[n+m] + c[abs(n-m)]) for m in hs] for n in hs]
    cs = [[0.5 * (s[n+m] - sgn(n-m) * s[abs(n-m)]) for m in hs] for n in hs]
    ss = [[0.5 * (c[abs(n-m)] - c[n+m]) for m in hs] for n in hs]

    C = c[1:H+1]
    S = s[1:H+1]

    CC = np.asarray(cc) - np.outer(C, C)
    CS = np.asarray(cs) - np.outer(C, S)
    SS = np.asarray(ss) - np.outer(S, S)

    return C, S, CC, CS, SS, np.asarray(YC), np.asarray(YS)


def add_regularization(sums, amplitude_priors=None, cn0=None, sn0=None):
    """
    Add regularization to sums. See Zechmeister & Kuerster 2009
    for details about notation.

    Parameters
    ----------
    sums: tuple of array_like
        C, S, CC, CS, SS, YC, YS. See `mhdirect_sums`
        for more information.
    amplitude_priors: float or array_like, optional
        Corresponds to standard deviation of a Gaussian
        prior on the amplitudes of all harmonics (if its a `float`),
        or for each of the harmonics (if it's an `array_like`).
    cn0: array_like
        Location of the centroid of the Gaussian amplitude prior
        on each of the cosine amplitudes
    sn0: array_like
        Location of the centroid of the Gaussian amplitude prior
        on each of the sine amplitudes

    Returns
    -------
    C, S, CC, CS, SS, YC, YS: array_like
        The set of sums with regularization added.

    See also
    --------
    :func:`mhdirect_sums`
    """
    C, S, CC, CS, SS, YC, YS = sums

    D = np.zeros_like(C)
    if amplitude_priors is not None:
        D = np.ones_like(C) * np.power(amplitude_priors, -2)

    cn0 = np.zeros(len(C)) if cn0 is None else cn0
    sn0 = np.zeros(len(S)) if sn0 is None else sn0

    CCreg = CC + np.diag(D)
    SSreg = SS + np.diag(D)
    YCreg = YC + D * cn0
    YSreg = YS + D * sn0

    return C, S, CCreg, CS, SSreg, YCreg, YSreg


def mhgls_params_from_sums(sums, YY, ybar):
    """
    Compute optimal amplitudes and offset from
    set of sums. See Zechmeister & Kuerster 2009
    for details about notation.

    Parameters
    ----------
    sums: tuple of array_like
        C, S, CC, CS, SS, YC, YS. See `mhdirect_sums`
        for more information.
    YY: float
        Weighted variance of `y - ybar`, where `ybar`
        is the weighted mean and `y` are the observations
    ybar: float
        Weighted mean of the data (`np.dot(w, y)`), where
        `w` are the weights and `y` are the observations.

    Returns
    -------
    cn: array_like
        Cosine amplitudes of each harmonic
    sn: array_like
        Sine amplitudes of each harmonic
    offset: float
        Constant offset

    See also
    --------
    :func:`mhdirect_sums`
    """
    C, S, CC, CS, SS, YC, YS = sums

    nharms = len(C)

    A = np.block([[CC, CS], [CS.T, SS]])
    b = np.concatenate((YC, YS))

    theta = np.linalg.solve(A, b)

    cn = theta[:nharms]
    sn = theta[nharms:]
    offset = ybar - (np.dot(cn, C) + np.dot(sn, S))

    return cn, sn, offset


def mhgls_from_sums(sums, YY, ybar):
    """
    Compute multiharmonic periodogram power from
    set of sums. See Zechmeister & Kuerster 2009
    for details about notation.

    Parameters
    ----------
    sums: tuple of array_like
        C, S, CC, CS, SS, YC, YS. See `mhdirect_sums`
        for more information.
    YY: float
        Weighted variance of `y - ybar`, where `ybar`
        is the weighted mean and `y` are the observations
    ybar: float
        Weighted mean of the data (`np.dot(w, y)`), where
        `w` are the weights and `y` are the observations.

    Returns
    -------
    power: float
        periodogram power


    See also
    --------
    :func:`mhdirect_sums`
    """
    C, S, CC, CS, SS, YC, YS = sums

    cn, sn, offset = mhgls_params_from_sums(sums, YY, ybar)

    XX = np.outer(cn, cn) * CC
    XX += 2 * np.outer(cn, sn) * CS
    XX += np.outer(sn, sn) * SS

    YX = 2 * (np.dot(cn, YC) + np.dot(sn, YS))
    P = (YX - np.sum(XX)) / YY

    return P


def _mh_power_from_spectra(sw, syw, k0, nharms, nf, YY, reg_kwargs=None):
    """Multiharmonic GLS power from the GPU NFFT spectra.

    ``sw`` is the adjoint NFFT of the weights ``w`` and ``syw`` of
    ``w*(y-ybar)``, both laid out so that entry ``j`` holds the spectrum
    at frequency index ``(k0 + j)`` (i.e. frequency ``(k0+j)*df``). The
    value at the ``m``-th harmonic of the ``i``-th output frequency
    ``m*f_i`` is therefore at array index ``(m-1)*k0 + m*i``.

    For each output frequency the weight moments ``c[0..2H], s[0..2H]``
    are read from ``sw`` (``c[0]=1``, ``s[0]=0``) and the mean-subtracted
    ``YC, YS`` (h=1..H) from ``syw``; these are fed through the existing,
    tested :func:`_mh_assemble_from_centered` + :func:`mhgls_from_sums`
    (the small 2H x 2H solve runs in float64 on the host -- cheap, and
    numerically safer than a float32 in-kernel solve).
    """
    H = int(nharms)
    i = np.arange(nf)

    # weight moments c[m], s[m] for m = 1..2H from the w-spectrum
    cm = np.empty((2 * H + 1, nf), dtype=np.float64)
    sm = np.empty((2 * H + 1, nf), dtype=np.float64)
    cm[0] = 1.0
    sm[0] = 0.0
    for m in range(1, 2 * H + 1):
        idx = (m - 1) * k0 + m * i
        vals = sw[idx]
        cm[m] = vals.real
        sm[m] = vals.imag

    # mean-subtracted YC[h], YS[h] for h = 1..H from the w*(y-ybar) spectrum
    YC = np.empty((H, nf), dtype=np.float64)
    YS = np.empty((H, nf), dtype=np.float64)
    for h in range(1, H + 1):
        idx = (h - 1) * k0 + h * i
        vals = syw[idx]
        YC[h - 1] = vals.real
        YS[h - 1] = vals.imag

    power = np.empty(nf, dtype=np.float64)
    for j in range(nf):
        sums = _mh_assemble_from_centered(cm[:, j], sm[:, j],
                                          YC[:, j], YS[:, j], H)
        if reg_kwargs:
            sums = add_regularization(sums, **reg_kwargs)
        power[j] = mhgls_from_sums(sums, YY, 0.0)
    return power


def lomb_scargle_direct_sums(t, yw, w, freqs, YY, nharms=1, **kwargs):
    """
    Compute Lomb-Scargle periodogram using direct summations. This
    is usually only useful for debugging and/or small numbers of
    frequencies.

    Parameters
    ----------
    t: array_like
        Observation times.
    yw: array_like
        Observations multiplied by their corresponding weights.
        `sum(yw)` is the weighted mean.
    w: array_like
        Weights for each of the observations. Usually proportional
        to `1/sigma ** 2` where `sigma` is the observation uncertainties.
        Normalized so that `sum(w) = 1`.
    freqs: array_like
        Trial frequencies to evaluate the periodogram
    YY: float
        Weighted variance of the observations.
    nharms: int, optional (default: 1)
        Number of harmonics to use in the model. Lomb Scargle only uses
        1, but more harmonics allow for greater model flexibility
        at the cost of higher complexity and therefore reduced signal-
        to-noise.

    Returns
    -------
    power: array_like
        The periodogram powers at each of the trial frequencies
    """
    def sfunc(f):
        return mhdirect_sums(t, yw, w, f, YY, nharms=nharms)
    sums = [add_regularization(s, **kwargs) for s in list(map(sfunc, freqs))]

    ybar = sum(yw)
    return np.array([mhgls_from_sums(s, YY, ybar) for s in sums])


def lomb_scargle_async(memory, functions, freqs,
                       block_size=256, use_fft=True,
                       use_cufinufft=False,
                       python_dir_sums=False,
                       transfer_to_device=True,
                       transfer_to_host=True,
                       window=False, **kwargs):
    """
    Asynchronous Lomb Scargle periodogram

    Use the ``LombScargleAsyncProcess`` class and
    related subroutines when possible.

    Parameters
    ----------
    memory: ``LombScargleMemory``
        Allocated memory, must have data already set (see, e.g.,
        ``LombScargleAsyncProcess.allocate()``)
    functions: tuple (lombscargle_functions, nfft_functions)
        Tuple of compiled functions from ``SourceModule``. Must be
        prepared with their appropriate dtype.
    freqs: array_like, optional (default: 0)
        Linearly-spaced frequencies starting at an integer multiple
        of the frequency spacing (i.e. freqs = df * (k0 + np.arange(nf)))
    block_size: int, optional
        Number of CUDA threads per block
    use_fft: bool, optional (default: True)
        If False, uses direct sums.
    python_dir_sums: bool, optional (default: False)
        If True, performs direct sums with Python on the CPU
    transfer_to_device: bool, optional, (default: True)
        If the data is already on the gpu, set as False
    transfer_to_host: bool, optional, (default: True)
        If False, will not transfer the resulting periodogram to
        CPU memory
    window: bool, optional (default: False)
        If True, computes the window function for the data

    Returns
    -------
    lsp_c: ``np.array``
        The resulting periodgram (``memory.lsp_c``)
    """
    if use_cufinufft and not HAS_CUFINUFFT:
        raise ImportError(
            "use_cufinufft=True but cufinufft is not installed. "
            "Install with: pip install cufinufft>=2.2")

    (lomb, lomb_dirsum), nfft_funcs = functions

    df = freqs[1] - freqs[0]
    samples_per_peak = 1./((memory.tmax - memory.tmin) * df)
    if not (get_k0(freqs) == memory.k0):
        raise ValueError(
            "freqs does not match the grid this memory was set up for "
            "(k0 mismatch: %d != %d)" % (get_k0(freqs), memory.k0))

    stream = memory.stream

    block = (block_size, 1, 1)
    grid = (int(np.ceil(memory.nf / float(block_size))), 1)

    # lightcurve -> gpu
    if transfer_to_device:
        memory.transfer_data_to_gpu()

    # do direct summations with python on the CPU (for debugging)
    if python_dir_sums:
        t = memory.t_g.get()
        yw = memory.yw_g.get()
        w = memory.w_g.get()
        return lomb_scargle_direct_sums(t, yw, w, freqs, memory.yy)

    # Use direct sums (on GPU)
    if not use_fft:
        args = (grid, block, stream,
                memory.t_g.ptr, memory.yw_g.ptr, memory.w_g.ptr,
                memory.lsp_g.ptr, memory.reg_g.ptr,
                np.int32(memory.nf),
                np.int32(memory.n0),
                memory.real_type(memory.yy),
                memory.real_type(memory.ybar),
                memory.real_type(df),
                memory.real_type(min(freqs)),
                memory.mode)

        lomb_dirsum.prepared_async_call(*args)
        if transfer_to_host:
            memory.transfer_lsp_to_cpu()
        return memory.lsp_c
    else:
        # NFFT
        nfft_kwargs = dict(transfer_to_host=False,
                           transfer_to_device=False)

        nfft_kwargs.update(kwargs)

        nfft_kwargs['minimum_frequency'] = freqs[0]
        nfft_kwargs['samples_per_peak'] = samples_per_peak

        if use_cufinufft:
            # cuFINUFFT path: replace custom NFFT with cufinufft type-1
            cufinufft_nfft_adjoint(memory.nfft_mem_yw, **nfft_kwargs)
            cufinufft_nfft_adjoint(memory.nfft_mem_w, **nfft_kwargs)
        else:
            # Custom NFFT path (Gaussian spreading + FFT)
            # NFFT(w * (y - ybar))
            nfft_adjoint_async(memory.nfft_mem_yw, nfft_funcs,
                               **nfft_kwargs)

            # NFFT(w)
            nfft_adjoint_async(memory.nfft_mem_w, nfft_funcs,
                               **nfft_kwargs)

    nharm = getattr(memory, 'nharmonics', 1)
    if nharm > 1:
        # Multiharmonic GLS: the GPU NFFT already produced the w-spectrum
        # (to 2H harmonics) and the w*(y-ybar)-spectrum (to H); read them
        # back and do the small per-frequency 2H x 2H solve on the host
        # (see _mh_power_from_spectra). Sync the stream first so the async
        # NFFT has completed before the device->host copy.
        if stream is not None:
            stream.synchronize()
        sw = memory.nfft_mem_w.ghat_g.get()
        syw = memory.nfft_mem_yw.ghat_g.get()
        power = _mh_power_from_spectra(sw, syw, int(memory.k0), nharm,
                                       int(memory.nf), memory.yy)
        memory.lsp_c[:memory.nf] = power.astype(memory.real_type)
        return memory.lsp_c

    args = (grid, block, stream)
    args += (memory.nfft_mem_w.ghat_g.ptr, memory.nfft_mem_yw.ghat_g.ptr)
    args += (memory.lsp_g.ptr, memory.reg_g.ptr, np.int32(memory.nf))
    args += (memory.real_type(memory.yy),
             memory.real_type(memory.ybar))
    args += (np.int32(memory.k0),
             np.int32(memory.mode))
    lomb.prepared_async_call(*args)

    if transfer_to_host:
        memory.transfer_lsp_to_cpu()

    return memory.lsp_c


class LombScargleAsyncProcess(GPUAsyncProcess):
    """
    GPUAsyncProcess for the Lomb Scargle periodogram

    Parameters
    ----------
    use_cufinufft: bool, optional (default: False)
        Use the cuFINUFFT library for the NFFT instead of the custom
        Gaussian-spreading kernel. Requires ``pip install
        cufinufft>=2.2`` (raises ImportError otherwise). Provided as a
        numerical cross-check backend: in cuvarbase's benchmarks the
        custom kernel was faster end-to-end (see
        ``cuvarbase.cufinufft_backend``).
    **kwargs: passed to ``NFFTAsyncProcess``

    Example
    -------
    >>> proc = LombScargleAsyncProcess()
    >>> Ndata = 1000
    >>> t = np.sort(365 * np.random.rand(N))
    >>> y = 12 + 0.01 * np.cos(2 * np.pi * t / 5.0)
    >>> y += 0.01 * np.random.randn(len(t))
    >>> dy = 0.01 * np.ones_like(y)
    >>> freqs, powers = proc.run([(t, y, dy)])
    >>> proc.finish()
    >>> ls_freqs, ls_powers = freqs[0], powers[0]

    """
    def __init__(self, *args, **kwargs):
        super(LombScargleAsyncProcess, self).__init__(*args, **kwargs)

        self.use_cufinufft = kwargs.pop('use_cufinufft', False)

        self.nfft_proc = NFFTAsyncProcess(*args, **kwargs)
        self._cpp_defs = self.nfft_proc._cpp_defs

        self.real_type = self.nfft_proc.real_type
        self.complex_type = self.nfft_proc.complex_type

        self.block_size = self.nfft_proc.block_size
        self.module_options = self.nfft_proc.module_options
        self.use_double = self.nfft_proc.use_double
        self.memory = None

        self.nharmonics = kwargs.get('nharmonics', 1)

        if self.nharmonics < 1:
            raise ValueError("nharmonics must be >= 1, got %r"
                             % (self.nharmonics,))

        if self.use_cufinufft and not HAS_CUFINUFFT:
            raise ImportError(
                "cufinufft not found. Install with: pip install cufinufft>=2.2"
            )

    def _compile_and_prepare_functions(self, **kwargs):

        module_text = _module_reader(find_kernel('lomb'), self._cpp_defs)

        self.module = SourceModule(module_text, options=self.module_options)
        self.dtypes = dict(
            lomb=[np.intp, np.intp, np.intp, np.intp, np.int32,
                  self.real_type, self.real_type, np.int32, np.int32],
            lomb_dirsum=[np.intp, np.intp, np.intp, np.intp, np.intp,
                         np.int32, np.int32, self.real_type, self.real_type,
                         self.real_type, self.real_type, np.int32]
        )

        self.nfft_proc._compile_and_prepare_functions(**kwargs)
        for fname, dtype in self.dtypes.items():
            func = self.module.get_function(fname)
            self.prepared_functions[fname] = func.prepare(dtype)
        self.function_tuple = tuple(self.prepared_functions[fname]
                                    for fname in sorted(self.dtypes.keys()))

    def memory_requirement(self, n0, nf, k0, nbatch=1,
                           autoadjust_sigma=False, **kwargs):
        """ return an approximate GPU memory requirement in bytes """
        H = self.nharmonics
        sigma = self.nfft_proc.sigma
        m = self.nfft_proc.get_m(nf)

        if autoadjust_sigma:
            sigma = int(np.round(float(sigma * (nf + k0)) / nf))

        fft_size = H * (nf + k0)

        mem = 0

        # data
        mem += 3 * n0

        # final result
        mem += nf

        # regularization
        mem += 2 * H + 1

        rsize = self.real_type(1).nbytes
        csize = self.complex_type(1).nbytes
        c = int(np.ceil(float(csize) / rsize))

        if kwargs.get('use_fft', True):
            # yw grid / fft (doubled because complex)
            mem += c * sigma * (fft_size - k0)

            # work area size for cufft.Plan
            # double because large non-power-of-two sizes trigger Bluestein algorithm
            nx = sigma * (fft_size - k0)
            mem += 1/rsize * 2 * cufft.cufft.cufftEstimate1d(nx, cufft.cufft.CUFFT_C2C)

            # w grid / fft (doubled because complex)
            mem += c * sigma * (2 * fft_size - k0)

            # work area size for cufft.Plan
            # double because large non-power-of-two sizes trigger Bluestein algorithm
            nx = sigma * (2 * fft_size - k0)
            mem += 1/rsize * 2 * cufft.cufft.cufftEstimate1d(nx, cufft.cufft.CUFFT_C2C)

            # precomputation (q1 = n0, q2 = n0, q3 = 2m + 1)
            mem += 2 * n0 + 2 * m + 1

        # inverse of design matrix
        if H > 1:

            # sparse matrix A (block-diagonal)
            mem += (2 * H) ** 2

            # vector b (Ax = b)
            mem += 1

        mem *= nbatch

        # size of float
        mem *= rsize

        return mem

    def allocate_for_single_lc(self, t, y, dy, nf, k0=0,
                               stream=None, **kwargs):
        """
        Allocate GPU (and possibly CPU) memory for single lightcurve

        Parameters
        ----------
        t: array_like
            Observation times
        y: array_like
            Observations
        dy: array_like
            Observation uncertainties
        nf: int
            Number of frequencies
        k0: int
            The starting index for the Fourier transform. The minimum
            frequency ``f0 = k0 * df``, where ``df`` is the frequency
            spacing
        stream: pycuda.driver.Stream
            CUDA stream you want this to run on
        **kwargs

        Returns
        -------
        mem: ~cuvarbase.memory.lombscargle_memory.LombScargleMemory
            Memory object.
        """
        m = self.nfft_proc.get_m(nf)

        sigma = self.nfft_proc.sigma

        kwargs_lsmem = dict(use_double=self.use_double,
                            nharmonics=self.nharmonics)

        kwargs_lsmem.update(kwargs)
        mem = LombScargleMemory(sigma, stream, m, k0=k0,
                                **kwargs_lsmem)

        mem.fromdata(t=t, y=y, dy=dy, nf=nf, allocate=True,
                     **kwargs)

        return mem

    def preallocate(self, max_nobs, nlcs=1, nf=None, k0=None,
                    freqs=None, streams=None, **kwargs):

        if freqs is not None:
            k0 = get_k0(freqs)
            nf = len(freqs)
        if nf is not None and k0 is None:
            raise ValueError("k0 must be given when nf is specified "
                             "without freqs")

        m = self.nfft_proc.get_m(nf)

        sigma = self.nfft_proc.sigma

        self.memory = []
        for i in range(nlcs):
            stream = None if streams is None else streams[i]
            mem = LombScargleMemory(sigma, stream, m,
                                    k0=k0,
                                    buffered_transfer=True,
                                    n0_buffer=max_nobs,
                                    nf=nf,
                                    use_double=self.use_double,
                                    nharmonics=self.nharmonics,
                                    **kwargs)

            mem.allocate(**kwargs)
            self.memory.append(mem)
        return self.memory

    def autofrequency(self, *args, **kwargs):
        return utils_autofreq(*args, **kwargs)

    def _nfreqs(self, *args, **kwargs):

        return len(self.autofrequency(*args, **kwargs))

    def allocate(self, data, nfreqs=None, k0s=None, **kwargs):

        """
        Allocate GPU memory for Lomb Scargle computations

        Parameters
        ----------
        data: list of (t, y, dy) tuples
            List of data, ``[(t_1, y_1, dy_1), ...]``
            * ``t``: Observation times
            * ``y``: Observations
            * ``dy``: Observation uncertainties
        **kwargs

        Returns
        -------
        allocated_memory: list of ``LombScargleMemory``
            list of allocated memory objects for each lightcurve

        """

        if len(data) > len(self.streams):
            self._create_streams(len(data) - len(self.streams))

        allocated_memory = []

        nfrqs = nfreqs
        k0 = k0s
        if nfrqs is None:
            nfrqs = [self._nfreqs(t, **kwargs) for (t, y, dy) in data]
        elif isinstance(nfreqs, int):
            nfrqs = nfrqs * np.ones(len(data))

        if k0s is None:
            k0s = [1] * len(nfrqs)
        elif isinstance(k0s, float):
            k0s = [k0s] * len(nfrqs)
        for i, ((t, y, dy), nf, k0) in enumerate(zip(data, nfrqs, k0s)):
            mem = self.allocate_for_single_lc(t, y, dy, nf, k0=k0,
                                              stream=self.streams[i],
                                              **kwargs)
            allocated_memory.append(mem)

        return allocated_memory

    def run(self, data,
            use_fft=True, memory=None,
            freqs=None,
            **kwargs):

        """
        Run Lomb Scargle on a batch of data.

        Parameters
        ----------
        data: list of tuples
            list of [(t, y, dy), ...] containing
            * ``t``: observation times
            * ``y``: observations
            * ``dy``: observation uncertainties
        freqs: optional, list of ``np.ndarray`` frequencies
            List of custom frequencies. Right now, this has to be linearly
            spaced with ``freqs[0] / (freqs[1] - freqs[0])`` being an integer.
        memory: optional, list of ``LombScargleMemory`` objects
            List of memory objects, length of list must be ``>= len(data)``
        use_fft: optional, bool (default: True)
            Uses the NFFT, otherwise just does direct summations (which
            are quite slow...)
        floating_mean: optional, bool (default: True)
            Add a floating mean to the model (see Zechmeister & Kurster 2009)
        window: optional, bool (default: False)
            If true, computes the window function for the data instead of
            Lomb-Scargle
        amplitude_prior: optional, float (default: None)
            If not None, sets the variance of a Gaussian prior on
            the amplitude (sometimes useful for suppressing aliases)
        **kwargs

        Returns
        -------
        results: list of lists
            list of (freqs, pows) for each LS periodogram; the power
            arrays are page-locked host buffers filled asynchronously —
            call :meth:`finish` before reading them (the batched entry
            points synchronize for you)

        """

        # compile module if not compiled already
        if not hasattr(self, 'prepared_functions') or \
            not all([func in self.prepared_functions for func in
                     ['lomb', 'lomb_dirsum']]):
            self._compile_and_prepare_functions(**kwargs)

        # Prepare data
        data = normalize_light_curves(data)

        # create and/or check frequencies
        frqs = freqs
        if frqs is None:
            frqs = [self.autofrequency(d[0], **kwargs) for d in data]

        elif not isinstance(frqs, list):
            frqs = [frqs] * len(data)

        if len(frqs) != len(data):
            raise ValueError(
                "number of frequency grids (%d) does not match number of "
            "lightcurves (%d)" % (len(frqs), len(data)))

        dfs = [frq[1] - frq[0] for frq in frqs]
        k0s = [get_k0(frq) for frq in frqs]

        # make sure k0 * df is the minimum frequency
        [check_k0(frq, k0=k0) for frq, k0 in zip(frqs, k0s)]

        if memory is None:
            memory = self.memory

        if memory is None:
            nfreqs = [len(frq) for frq in frqs]
            memory = self.allocate(data, nfreqs=nfreqs, k0s=k0s,
                                   use_fft=use_fft, **kwargs)
        else:
            for i, (t, y, dy) in enumerate(data):
                memory[i].set_gpu_arrays_to_zero(**kwargs)
                memory[i].setdata(t=t, y=y, dy=dy, **kwargs)

        ls_kwargs = dict(block_size=self.block_size,
                         use_fft=use_fft,
                         use_cufinufft=self.use_cufinufft)
        ls_kwargs.update(kwargs)

        funcs = (self.function_tuple, self.nfft_proc.function_tuple)
        results = [lomb_scargle_async(memory[i], funcs, frqs[i],
                                      **ls_kwargs)
                   for i in range(len(data))]

        results = [(f, r) for f, r in zip(frqs, results)]
        return results

    def batched_run_const_nfreq(self, data, batch_size=1,
                                use_fft=True, freqs=None,
                                only_return_best_freqs=False,
                                ignore_freq_mask=None,
                                **kwargs):
        """
        Same as ``batched_run`` but is more efficient when the frequencies are
        the same for each lightcurve. Doesn't reallocate memory for each batch.

        Parameters
        ----------
        batch_size: int, optional (default: 1)
            Lightcurves processed per multi-stream batch. The default
            of 1 is the safe choice — all published survey-throughput
            numbers (e.g. 4.4 ms/LC for ZTF-scale grids) were measured
            at ``batch_size=1``. The "multi-stream overhead" that made
            larger values slower is per-call setup, diagnosed Jul 2026
            (A5000): this method builds ``batch_size`` separate
            ``LombScargleMemory`` sets — pinned host buffers, device
            arrays, and a cuFFT plan each — on *every call*, a cost
            that scales with ``batch_size``, while the GPU compute
            stages barely benefit because a single survey-scale
            Lomb-Scargle already saturates the device. When one call
            processes many lightcurves (hundreds+) that setup
            amortizes: ``batch_size=4`` measured ~10% faster per LC
            than 1 at 256 LCs/call, while 8 was net slower. Only
            increase this if your call sizes are large and you
            benchmark it on your own workload; see
            ``analysis/v1.0-gpu-batch3-jul2026/E1_E2_DIAGNOSIS.md``.

        Notes
        -----
        To get best efficiency, make sure the maximum number of observations
        is not much larger than the typical number of observations
        """

        # compile and prepare module functions if not already done
        if not hasattr(self, 'prepared_functions') or \
            not all([func in self.prepared_functions for func in
                     ['lomb', 'lomb_dirsum']]):
            self._compile_and_prepare_functions(**kwargs)

        # create streams if needed
        bsize = min([len(data), batch_size])
        if len(self.streams) < bsize:
            self._create_streams(bsize - len(self.streams))

        streams = [self.streams[i] for i in range(bsize)]
        max_ndata = max([len(t) for t, y, dy in data])

        if freqs is None:
            data_with_max_baseline = max(data,
                                         key=lambda d: np.max(d[0]) - np.min(d[0]))
            freqs = self.autofrequency(data_with_max_baseline[0], **kwargs)

            # now correct frequencies
            df = freqs[1] - freqs[0]
            k0 = get_k0(freqs)
            # nf = len(freqs)
            nf = int(round(np.max(freqs) / df)) - k0
            freqs = df * (k0 + np.arange(nf))

        df = freqs[1] - freqs[0]
        k0 = get_k0(freqs)
        nf = len(freqs)
        check_k0(freqs, k0=k0)

        lsps = []

        # make data batches
        batches = []
        while len(batches) * batch_size < len(data):
            start = len(batches) * batch_size
            finish = start + min([batch_size, len(data) - start])
            batches.append([data[i] for i in range(start, finish)])

        # set up memory containers for gpu and cpu (pinned) memory
        m = self.nfft_proc.get_m(nf)
        sigma = self.nfft_proc.sigma

        kwargs_lsmem = dict(buffered_transfer=True,
                            n0_buffer=max_ndata,
                            use_double=self.use_double,
                            nharmonics=self.nharmonics,
                            use_fft=use_fft)
        kwargs_lsmem.update(kwargs)
        memory = [LombScargleMemory(sigma, stream, m, k0=k0,
                                    **kwargs_lsmem)
                  for stream in streams]

        # allocate memory
        [mem.allocate(nf=nf, **kwargs) for mem in memory]

        funcs = (self.function_tuple, self.nfft_proc.function_tuple)
        best_freqs, best_freq_significances = [], []

        default_mask = np.array([True] * len(freqs))
        mask = default_mask if ignore_freq_mask is None else ~np.asarray(ignore_freq_mask)
        for b, batch in enumerate(batches):

            results = self.run(batch, memory=memory, freqs=freqs,
                               use_fft=use_fft,
                               **kwargs)
            self.finish()

            for i, (f, p) in enumerate(results):
                if only_return_best_freqs:
                    best_index = np.argmax(p[mask])
                    fap = fap_baluev(batch[i][0], batch[i][2], p[mask], np.max(freqs[mask]))
                    significance = 1. - fap[best_index]
                    best_freqs.append(freqs[mask][best_index])
                    best_freq_significances.append(significance)
                else:
                    lsps.append(np.copy(p))

        if only_return_best_freqs:
            return best_freqs, best_freq_significances
        else:
            return [(freqs, lsp) for lsp in lsps]


def fap_baluev(t, dy, z, fmax, d_K=3, d_H=1, use_gamma=True):
    """
    False alarm probability for periodogram peak
    based on Baluev (2008) [2008MNRAS.385.1279B]

    Parameters
    ----------
    t: array_like
        Observation times.
    dy: array_like
        Observation uncertainties.
    z: array_like or float
        Periodogram value(s)
    fmax: float
        Maximum frequency searched
    d_K: int, optional (default: 3)
        Number of degrees of fredom for periodgram model.
        2H - 1 where H is the number of harmonics
    d_H: int, optional (default: 1)
        Number of degrees of freedom for default model.
    use_gamma: bool, optional (default: True)
        Use gamma function for computation of numerical
        coefficient; computed with scipy.special.gammaln
        to avoid overflow at large N
    Returns
    -------
    fap: float
        False alarm probability

    Example
    -------
    >>> rand = np.random.RandomState(100)
    >>> t = np.sort(rand.rand(100))
    >>> y = 12 + 0.01 * np.cos(2 * np.pi * 10. * t)
    >>> dy = 0.01 * np.ones_like(y)
    >>> y += dy * rand.rand(len(t))
    >>> proc = LombScargleAsyncProcess()
    >>> results = proc.run([(t, y, dy)])
    >>> freqs, powers = results[0]
    >>> fap_baluev(t, dy, powers, max(freqs))
    """

    N = len(t)
    d = d_K - d_H

    N_K = N - d_K
    N_H = N - d_H
    g = (0.5 * N_H) ** (0.5 * (d - 1))

    if use_gamma:
        g = np.exp(gammaln(0.5 * N_H) - gammaln(0.5 * (N_K + 1)))

    w = np.power(dy, -2)

    tbar = np.dot(w, t) / sum(w)
    Dt = np.dot(w, np.power(t - tbar, 2)) / sum(w)

    Teff = np.sqrt(4 * np.pi * Dt)

    W = fmax * Teff
    A = (2 * np.pi ** 1.5) * W

    # Evaluate in log space (issue #14): for z near 1 the naive
    #     FAP = 1 - (1 - (1-z)**(0.5*N_K)) * exp(-tau)
    # underflows -- both factors round to 1.0 and the subtraction
    # cancels to exactly 0.0 for significant peaks. Rewriting as
    #     FAP = -expm1(-tau) + exp(log(1 - Psing) - tau)
    # keeps the result positive down to the float64 limit (~1e-308).
    z = np.asarray(z, dtype=np.float64)

    eZ1 = (z / np.pi) ** 0.5 * (d - 1)

    with np.errstate(divide='ignore'):
        # log(1 - z); -inf at z == 1 (exp() of it is 0, as intended)
        log1mz = np.log1p(-np.minimum(z, 1.0))
        log_eZ1 = np.log(eZ1)

    log_tau = (np.log(g * A / (2 * np.pi))
               + log_eZ1
               + 0.5 * (N_K - 1) * log1mz)
    tau = np.exp(log_tau)

    # log(1 - Psing) = log((1 - z)**(0.5 * N_K))
    log_Psing_c = 0.5 * N_K * log1mz

    return -np.expm1(-tau) + np.exp(log_Psing_c - tau)


def lomb_scargle_simple(t, y, dy, **kwargs):
    """
    Simple lomb-scargle interface for testing that
    things work on the GPU. Note: This will be
    substantially slower than working with the
    ``LombScargleAsyncProcess`` interface.
    """

    # Pass dy straight through: LombScargleMemory.setdata converts
    # uncertainties to normalized inverse-variance weights itself.
    # (Pre-normalizing here double-applied the conversion, effectively
    # weighting by dy^4 and giving the *largest*-error points the most
    # weight.)
    proc = LombScargleAsyncProcess()
    results = proc.run([(t, y, dy)], **kwargs)

    freqs, powers = results[0]

    proc.finish()

    return freqs, powers
