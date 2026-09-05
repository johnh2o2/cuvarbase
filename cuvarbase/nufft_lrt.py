"""
NUFFT-based Likelihood Ratio Test for transit detection.

Contributed by Jamila Taaki (`@xiaziyna <https://github.com/xiaziyna>`_).
This module implements a frequency-domain matched-filter / likelihood-
ratio test for box transits in correlated noise, with the noise spectrum
estimated adaptively from the data.

The data and each transit template are transformed with the GPU adjoint
NFFT (:class:`cuvarbase.cunfft.NFFTAsyncProcess`), which handles the
non-uniform (gappy / multi-season) sampling directly over the full
observational baseline. The per-template matched-filter combination
(SNR = sum_k Y_k T_k* w_k / P_s(k) / sqrt(sum_k |T_k|^2 w_k / P_s(k)))
runs on the host -- it is an O(nf) reduction, negligible next to the NFFT.

Conventions
-----------
* **Times.** :meth:`NUFFTLRTAsyncProcess.run` subtracts
  ``floor(min(t))`` in float64 (:func:`cuvarbase.utils.subtract_epoch`)
  before anything is cast to the device precision, so absolute BJD-scale
  timestamps are safe (float32 spacing at 2.457e6 is 0.25 d, wider than
  a transit). ``epochs`` passed in and the best epochs returned are in
  the caller's original time scale.
* **PSD.** ``psd[k]`` is the expected squared modulus of the noise's
  *unnormalized* adjoint NFFT at mode ``k``:
  ``P(k) = E |sum_j s_j exp(2 pi i f_k t_j)|^2`` with
  ``f_k = k / (max(t) - min(t))``, ``k = 0..nf-1`` (one-sided, all nf
  modes are physical positive-frequency coefficients). White noise of
  variance ``sigma^2`` per point has ``P(k) = n sigma^2`` at every k.
  ``psd=np.ones(nf)`` therefore gives a statistic in *data units*, not
  an SNR.
* **The statistic is not N(0, 1).** Under irregular sampling the NFFT
  modes are not orthogonal, so the frequency-diagonal whitened
  correlation is over-dispersed even with the TRUE noise PSD (null
  standard deviation 1.8-2.7 for ground-based sampling at ``nf = 2n``,
  growing with ``nf``). Detection thresholds must be calibrated
  empirically per (sampling, ``nf``, PSD estimator) configuration, e.g.
  from the null-percentile of signal-free or scrambled light curves as
  ``scripts/nufft_lrt_validation.py`` does. Raising ``nf`` inflates the
  raw value without adding information.
* **Detectors** (:meth:`NUFFTLRTAsyncProcess.run`, ``detector=``):
  ``'matched'`` (default) is the stationary whitened filter above;
  ``'marginal'`` is Detector A of Taaki, Kamalabadi & Kemball (2020),
  which marginalizes systematics coefficients under a Gaussian prior;
  ``'sequential'`` least-squares cotrends against the same basis and
  then runs the matched filter on the residual. The latter two need
  ``systematics_basis``, Detector A also ``coeff_prior_cov``.
* **Detector A's prior is effectively wider than specified.** Its
  Gram matrix is accumulated over ``nf`` (by default ``2n``)
  non-orthogonal NFFT modes, which overcounts the corresponding
  time-domain inner products by ~2.2-2.4x for the samplings measured
  in the Sep-2026 audit, so ``coeff_prior_cov`` acts as though it were
  about that much wider. The effect on the statistic is small, but
  calibrate the prior and the threshold together.
* ``dy`` is not used by any detector (a ``UserWarning`` is emitted if it
  is passed); the noise model is the PSD.
"""
import warnings

import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from .base import GPUAsyncProcess, ensure_context
from .cunfft import NFFTAsyncProcess
from .memory import NFFTMemory
from .utils import (find_kernel, _module_reader,
                    subtract_epoch, check_lightcurve)

# Emitted once per NUFFTLRTAsyncProcess construction (not at import, so
# ``from cuvarbase import *`` and the BLS/LS/PDM users never see it).
# Keep the "cuvarbase.nufft_lrt is EXPERIMENTAL" prefix: filterwarnings
# entries match on it.
_EXPERIMENTAL_MSG = (
    "cuvarbase.nufft_lrt is EXPERIMENTAL and outside the 1.x API-stability "
    "promise. The Sep-2026 correctness fixes (float64 epoch subtraction, "
    "automatic epoch grid for epochs=None, Detector A PSD from the "
    "cotrended residual, centred sequential cotrend, full-band NFFT "
    "accuracy) are awaiting injection-recovery re-validation; the "
    "statistic is not N(0, 1) and thresholds must be calibrated "
    "empirically (see https://johnh2o2.github.io/cuvarbase/nufft_lrt.html).")


def _whitened_inner(A, B, psd, weights):
    """Whitened frequency-domain inner product Re sum_k A_k B_k* w_k / P_k
    -- the metric of the stationary matched filter."""
    return float(np.real(np.sum(A * np.conj(B) * weights / psd)))


def _prior_response_matrix(G, prior_cov):
    """Return ``M = (Cov_c^{-1} + G)^{-1}`` for the Detector A Woodbury
    term without ever inverting the prior covariance:

        (C^{-1} + G)^{-1} = C (I + G C)^{-1}

    (push-through identity), so a zero prior variance along a mode
    correctly gives the "prior pinned to its mean" limit (no
    marginalization along that mode). A ``pinv`` of the prior would turn
    that same zero into an *improper flat* prior -- the opposite limit.
    ``I + G C`` has eigenvalues >= 1 for positive semidefinite ``G`` and
    ``C``, so the solve is always well posed.

    Raises ``ValueError`` if ``prior_cov`` is not a symmetric positive
    semidefinite ``(K, K)`` matrix.
    """
    G = np.asarray(G, dtype=np.float64)
    K = G.shape[0]
    C = np.atleast_2d(np.asarray(prior_cov, dtype=np.float64))
    if C.shape != (K, K):
        raise ValueError("coeff_prior_cov must be (K, K) with K = %d "
                         "basis vectors (got shape %r)" % (K, C.shape))
    if not np.all(np.isfinite(C)):
        raise ValueError("coeff_prior_cov must be finite")
    scale = max(float(np.max(np.abs(C))), 1.0)
    if not np.allclose(C, C.T, rtol=1e-8, atol=1e-12 * scale):
        raise ValueError("coeff_prior_cov must be symmetric")
    ev = np.linalg.eigvalsh(C)
    if ev.min() < -1e-10 * scale:
        raise ValueError("coeff_prior_cov must be positive semidefinite "
                         "(smallest eigenvalue %g)" % ev.min())
    A = np.eye(K) + G @ C
    M = np.linalg.solve(A.T, C.T).T          # C A^{-1}
    return 0.5 * (M + M.T)


def _marginal_precompute(Y, V_ks, psd, weights, prior_cov):
    """Template-independent part of Detector A (hoisted out of the
    template loop). Returns ``(Vw, M, w_y)`` with ``Vw = V_k w / P``
    (K, nf), ``M`` the (K, K) response matrix of
    :func:`_prior_response_matrix` and ``w_y[j] = <v_j, Y>_W``."""
    K = len(V_ks)
    Vk = np.asarray(V_ks).reshape(K, -1)
    wp = np.asarray(weights, dtype=np.float64) / np.asarray(psd, np.float64)
    Vw = Vk * wp
    G = np.real(Vw @ np.conj(Vk).T)
    G = 0.5 * (G + G.T)
    M = _prior_response_matrix(G, prior_cov)
    w_y = np.real(Vw @ np.conj(np.asarray(Y)))
    return Vw, M, w_y


def _marginal_evaluate(Yw, wp, T, Vw, M, w_y, eps_floor=1e-12):
    """Per-template part of Detector A: ``Yw = Y w / P`` and ``wp = w / P``
    are precomputed; ``T`` is the template transform."""
    T = np.asarray(T)
    w_t = np.real(Vw @ np.conj(T))
    num = float(np.real(np.sum(Yw * np.conj(T)))) - float(w_y @ M @ w_t)
    den = float(np.sum((np.abs(T) ** 2) * wp)) - float(w_t @ M @ w_t)
    if den <= eps_floor:
        return 0.0
    return float(num / np.sqrt(den))


def _marginal_statistic(Y, T, V_ks, psd, weights, prior_cov,
                        eps_floor=1e-12):
    """Taaki et al. (2020) Detector A (marginalized joint detector) in
    the whitened frequency domain, via the Woodbury identity.

    The joint model is y = t + V c + s with c ~ N(mu_c, Cov_c) and s
    stationary with PSD P(k); marginalizing c gives a matched filter
    under the combined covariance Cov_z = Cov_s + V Cov_c V^T. With
    W = Cov_s^{-1} applied diagonally in the frequency domain,

        <a, b>_z = <a, b>_W - w_a^T (Cov_c^{-1} + G)^{-1} w_b,

    where G_ij = <v_i, v_j>_W and (w_a)_j = <v_j, a>_W. The statistic is
    T_A = <y_hat, t>_z / sqrt(<t, t>_z) with y_hat = y - V mu_c
    (the mean-systematics subtraction happens in the time domain before
    the transform). The K basis transforms V_ks are computed once per
    lightcurve; per template this adds only K-dimensional algebra.

    ``(Cov_c^{-1} + G)^{-1}`` is formed as ``Cov_c (I + G Cov_c)^{-1}``
    (see :func:`_prior_response_matrix`), so singular priors are handled
    in the correct limit and non-PSD priors raise ``ValueError``.

    Parameters: Y, T = NFFTs of the (mean-subtracted) data and template;
    V_ks = list/array of K basis NFFTs; prior_cov = Cov_c (K x K).
    Returns the marginalized SNR (float).
    """
    K = len(V_ks)
    wp = np.asarray(weights, dtype=np.float64) / np.asarray(psd, np.float64)
    Y = np.asarray(Y)
    if K == 0:
        num = float(np.real(np.sum(Y * np.conj(T) * wp)))
        den = float(np.sum((np.abs(T) ** 2) * wp))
        return num / np.sqrt(den) if den > 0 else 0.0
    Vw, M, w_y = _marginal_precompute(Y, V_ks, psd, weights, prior_cov)
    return _marginal_evaluate(Y * wp, wp, T, Vw, M, w_y, eps_floor)


def _sequential_detrend(t, y, basis):
    """The papers' "standard" baseline: ordinary least-squares cotrend
    against the systematics basis (time domain, unwhitened -- as a
    pipeline would), returning the residual for the stationary matched
    filter.

    The fit includes an intercept: the basis columns and ``y`` are
    centred before the least-squares solve and the centred basis is
    subtracted, so the residual keeps the mean of ``y`` (removed later
    by the demean in :meth:`NUFFTLRTAsyncProcess.run`) and a basis
    column with a non-zero mean cannot absorb the mean flux. Without the
    intercept a column with mean ``m_v`` and std ``s_v`` biases its
    coefficient by ``ybar m_v / (m_v^2 + s_v^2)`` and leaves a
    residual systematic of amplitude ``ybar m_v / s_v`` (a 1% column
    mean on relative flux left 10x the noise; audit Sep 2026).
    """
    V = np.asarray(basis, dtype=np.float64)
    if V.ndim == 1:
        V = V[:, None]
    y = np.asarray(y, dtype=np.float64)
    Vc = V - V.mean(axis=0)
    coeff, *_ = np.linalg.lstsq(Vc, y - y.mean(), rcond=None)
    return y - Vc @ coeff


def _smoothed_periodogram(power, window):
    """Boxcar-smooth a periodogram with edge correction.

    Each output bin is the mean of the *available* neighbors inside the
    window, so the first/last ``window//2`` bins are not biased low by
    the implicit zero-padding of a plain ``np.convolve(..., 'same')``
    (which would overweight those bins by up to ~2x after the 1/P(k)
    whitening).

    The window is clamped to ``len(power)``: ``np.convolve(..., 'same')``
    returns ``max(len(power), window)`` samples, so a window wider than
    the spectrum used to lengthen the PSD and fail later with a raw
    numpy broadcast error (``nf < smooth_window``, e.g. nf = 4 with the
    default ``smooth_window=5``).
    """
    k = min(int(window), len(power))
    if k <= 1:
        return power
    kernel = np.ones(k, dtype=power.dtype)
    num = np.convolve(power, kernel, mode='same')
    den = np.convolve(np.ones_like(power), kernel, mode='same')
    return (num / den).astype(power.dtype, copy=False)


def _floor_psd(psd, eps_floor, real_type):
    """Floor a PSD at ``eps_floor`` times its positive median (once, for
    every detector path). This caps any single bin's whitening weight at
    ``1/eps_floor`` times the typical weight: a zero bin in a user PSD
    otherwise gives a statistic of ~1e6 (matched) or nan (marginal)."""
    psd = np.asarray(psd, dtype=real_type)
    pos = psd[psd > 0]
    median_ps = np.median(pos) if pos.size else real_type(1.0)
    return np.maximum(psd, real_type(eps_floor) * real_type(median_ps)
                      ).astype(real_type, copy=False)


def epoch_grid(period, duration, oversample=2.0, min_epochs=8,
               max_epochs=96):
    """Epoch grid used by :meth:`NUFFTLRTAsyncProcess.run` when
    ``epochs=None``: ``n = clip(ceil(oversample * period / duration),
    min_epochs, max_epochs)`` epochs at ``arange(n) * period / n``
    (relative to the epoch-subtracted time origin), so consecutive
    templates are misaligned by at most ``duration / oversample`` until
    the ``max_epochs`` cap is reached.
    """
    n = int(np.ceil(float(oversample) * float(period) / float(duration)))
    n = int(min(max(n, int(min_epochs)), int(max_epochs)))
    return np.arange(n, dtype=np.float64) * float(period) / n


class NUFFTLRTMemory:
    """
    Memory management for NUFFT LRT computations.

    Parameters
    ----------
    nfft_memory : NFFTMemory
        Memory for NUFFT computation
    stream : pycuda.driver.Stream
        CUDA stream for operations
    use_double : bool, optional (default: False)
        Use double precision
    """

    def __init__(self, nfft_memory, stream, use_double=False, **kwargs):
        # Direct construction is a supported entry point (exported in
        # __all__): retain the CUDA context before any GPU allocation,
        # like every other *Memory class.
        ensure_context()
        self.nfft_memory = nfft_memory
        self.stream = stream
        self.use_double = use_double

        self.real_type = np.float64 if use_double else np.float32
        self.complex_type = np.complex128 if use_double else np.complex64

        # Memory for LRT computation
        self.template_g = None
        self.power_spectrum_g = None
        self.weights_g = None
        self.results_g = None
        self.results_c = None

    def allocate(self, nf, **kwargs):
        """Allocate GPU memory for LRT computation."""
        self.nf = nf

        # Template NUFFT result
        self.template_nufft_g = gpuarray.zeros(nf, dtype=self.complex_type)

        # Power spectrum estimate
        self.power_spectrum_g = gpuarray.zeros(nf, dtype=self.real_type)

        # Per-mode weights (all ones: every mode k = 0..nf-1 is a distinct
        # positive-frequency coefficient; see NUFFTLRTAsyncProcess.run)
        self.weights_g = gpuarray.zeros(nf, dtype=self.real_type)

        # Results: [numerator, denominator]
        self.results_g = gpuarray.zeros(2, dtype=self.real_type)
        self.results_c = cuda.aligned_zeros(shape=(2,),
                                            dtype=self.real_type,
                                            alignment=4096)

        return self

    def transfer_results_to_cpu(self):
        """Transfer LRT results from GPU to CPU."""
        cuda.memcpy_dtoh_async(self.results_c, self.results_g.ptr,
                               stream=self.stream)


class NUFFTLRTAsyncProcess(GPUAsyncProcess):
    """
    GPU implementation of the NUFFT likelihood-ratio transit search.

    This implements a matched filter in the frequency domain:

    .. math::
        \\text{SNR} = \\frac{\\sum_k Y_k T_k^* w_k / P_s(k)}
        {\\sqrt{\\sum_k |T_k|^2 w_k / P_s(k)}}

    where:
    - Y_k is the NUFFT of the lightcurve
    - T_k is the NUFFT of the transit template
    - P_s(k) is the power spectrum (adaptively estimated or provided)
    - w_k are per-mode weights; they are all 1 (every returned mode
      ``k = 0..nf-1`` is a distinct positive-frequency coefficient, so
      the 1/2/1 weighting of a packed one-sided RFFT does not apply)

    .. warning:: **Experimental.** This module and the :meth:`run`
        signature are outside the 1.x API-stability promise: the
        Sep-2026 correctness fixes are pending injection-recovery
        re-validation (release-plan Phase 4), after which the API may
        change without a deprecation cycle. Constructing this class
        emits a ``UserWarning`` saying so. The class is importable as
        ``cuvarbase.nufft_lrt.NUFFTLRTAsyncProcess`` only; it is not in
        the top-level ``cuvarbase`` namespace.

    The value is a whitened correlation, not an N(0, 1) SNR: see the
    module docstring for the PSD convention and the calibration caveat.
    :meth:`run` selects between three detectors with ``detector=``:
    ``'matched'`` (default, the formula above), ``'marginal'`` (Taaki
    et al. Detector A, systematics marginalized under a Gaussian prior)
    and ``'sequential'`` (least-squares cotrend, then the filter).

    Parameters
    ----------
    sigma : float, optional (default: 4.0)
        Oversampling factor of the NFFT grid (``sigma * nf`` grid
        points). The transform returns the one-sided modes
        ``k = 0..nf-1``, so the effective oversampling at the top of the
        band is ``sigma / 2``; ``sigma = 4`` keeps every returned mode
        inside the Gaussian window's accuracy band (full-band error
        ~4e-4 in float32, ~1e-6 in float64 vs the exact adjoint DFT).
        With ``sigma = 2`` (the pre-Sep-2026 default) the modes
        ``k >= nf/2`` carried O(1) aliasing error.
    m : int, optional (default: None)
        NFFT truncation parameter. ``None`` means 8 when
        ``autoset_m=False``; ignored when ``autoset_m=True``.
    use_double : bool, optional (default: False)
        Use double precision
    use_fast_math : bool, optional (default: True)
        Use fast math in CUDA kernels
    block_size : int, optional (default: 256)
        CUDA block size
    autoset_m : bool, optional (default: True)
        Choose ``m`` from the NFFT truncation-error bound (see
        :meth:`cuvarbase.cunfft.NFFTAsyncProcess.estimate_m`); one
        ``m`` per :meth:`run` sized for the largest transformed vector.
    **kwargs : dict
        Additional parameters passed to :class:`NFFTAsyncProcess`.

    Example
    -------
    >>> import numpy as np
    >>> from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
    >>>
    >>> rng = np.random.RandomState(0)
    >>> t = np.sort(rng.uniform(0, 60, 600))           # any time origin
    >>> P, dur, t0 = 5.3, 0.22, 1.7                    # injected transit
    >>> phase = ((t - t0) / P) % 1.0
    >>> y = 1.0 - 0.01 * (np.minimum(phase, 1 - phase) < 0.5 * dur / P)
    >>> y += 0.003 * rng.randn(len(t))
    >>>
    >>> proc = NUFFTLRTAsyncProcess()
    >>> # focused search around a candidate: the period step must keep
    >>> # the box aligned over the baseline T, dP <~ dur * P / (2 T)
    >>> periods = np.arange(4.8, 5.8, 0.22 * 4.8 / (2 * 60))
    >>> # periods x durations is a full outer product, so always pass a
    >>> # short explicit duration array (the ``durations=None`` default
    >>> # is 0.1 * periods, i.e. len(periods)**2 cells)
    >>> durations = np.array([0.12, 0.25])
    >>> # epochs=None scans an automatic epoch grid per (period,
    >>> # duration) and returns the max over epochs plus the best epoch
    >>> snr, best_epoch = proc.run(t, y, periods, durations=durations)
    >>> i, j = np.unravel_index(np.argmax(snr), snr.shape)
    >>> periods[i], durations[j], best_epoch[i, j]   # ~5.3, 0.25, ~1.7 (mod P)
    """

    def __init__(self, sigma=4.0, m=None, use_double=False,
                 use_fast_math=True, block_size=256, autoset_m=True,
                 **kwargs):
        warnings.warn(_EXPERIMENTAL_MSG, UserWarning, stacklevel=2)
        super(NUFFTLRTAsyncProcess, self).__init__(**kwargs)

        self.sigma = sigma
        self.m = m
        self.use_double = use_double
        self.use_fast_math = use_fast_math
        self.block_size = block_size
        self.autoset_m = autoset_m

        self.real_type = np.float64 if use_double else np.float32
        self.complex_type = np.complex128 if use_double else np.complex64

        # NUFFT processor for computing transforms
        self.nufft_proc = NFFTAsyncProcess(
            sigma=sigma, m=(8 if m is None else m), use_double=use_double,
            use_fast_math=use_fast_math, block_size=block_size,
            autoset_m=autoset_m, **kwargs
        )

        self.function_names = [
            'nufft_matched_filter',
            'estimate_power_spectrum',
            'compute_frequency_weights',
            'demean_data',
            'compute_mean',
            'generate_transit_template'
        ]

        # Module options
        self.module_options = ['--use_fast_math'] if use_fast_math else []
        # Preprocessor defines for CUDA kernels
        self._cpp_defs = {}
        if use_double:
            self._cpp_defs['DOUBLE_PRECISION'] = None

    def _compile_and_prepare_functions(self, **kwargs):
        """Compile CUDA kernels and prepare function calls."""
        module_txt = _module_reader(find_kernel('nufft_lrt'), self._cpp_defs)

        self.module = SourceModule(module_txt, options=self.module_options)

        # Function signatures
        self.dtypes = dict(
            nufft_matched_filter=[np.intp, np.intp, np.intp, np.intp,
                                  np.intp, np.int32, self.real_type],
            estimate_power_spectrum=[np.intp, np.intp, np.int32, np.int32,
                                     self.real_type],
            compute_frequency_weights=[np.intp, np.int32, np.int32],
            demean_data=[np.intp, np.int32, self.real_type],
            compute_mean=[np.intp, np.intp, np.int32],
            generate_transit_template=[np.intp, np.intp, np.int32,
                                       self.real_type, self.real_type,
                                       self.real_type, self.real_type]
        )

        # Prepare functions
        self.prepared_functions = {}
        for func_name in self.function_names:
            func = self.module.get_function(func_name)
            func.prepare(self.dtypes[func_name])
            self.prepared_functions[func_name] = func

    def _nfft_memory(self, t, nf, l1_max, **kwargs):
        """Allocate ONE :class:`NFFTMemory` (device buffers, cuFFT plan,
        pinned host buffer) for the epoch-subtracted times ``t`` and
        ``nf`` modes, reused by :meth:`run` for the data, the basis
        vectors and every template. The truncation radius ``m`` is
        sized from ``l1_max``, an upper bound on the L1 norm of every
        vector that will be transformed (the NFFT error bound scales
        with ``||y||_1``; see :meth:`NFFTAsyncProcess.estimate_m`).
        Allocating per transform cost 2.5-12 ms per template against
        ~0.1 ms of transform (audit Sep 2026).
        """
        proc = self.nufft_proc
        if not proc.streams:
            proc._create_streams(1)
        m = proc.get_m(int(nf), y=np.array([float(l1_max)]))
        t = np.ascontiguousarray(t, dtype=self.real_type)
        mem = NFFTMemory(proc.sigma, proc.streams[0], m,
                         use_double=self.use_double, **kwargs)
        return mem.fromdata(t, np.zeros(len(t), dtype=self.real_type),
                            nf=int(nf), allocate=True, **kwargs)

    def compute_nufft(self, t, y, nf, memory=None, **kwargs):
        """
        Compute the adjoint NUFFT of data on the GPU.

        Parameters
        ----------
        t : array-like
            Time values (any origin; ``floor(min(t))`` is subtracted in
            float64 before the cast to the device precision)
        y : array-like
            Observation values
        nf : int
            Number of frequency samples
        memory : NFFTMemory, optional
            A buffer set from :meth:`_nfft_memory` already holding
            these times; only ``y`` is uploaded and the buffers are
            reused (``t`` must be the array the memory was built from).
        **kwargs : dict
            Additional parameters for NUFFT

        Returns
        -------
        nufft_result : np.ndarray, complex
            ``ghat[k] = sum_j y_j exp(2 pi i f_k (t_j - t_ref))`` at the
            modes ``f_k = k / (max(t) - min(t))``, ``k = 0..nf-1``,
            with ``t_ref = floor(min(t))``. The transform's own time
            reference is a common per-mode phase that cancels in every
            ``Re sum A B* / P`` inner product of the detectors. Every
            one of the nf modes is accurate to the Gaussian-window
            bound with the default ``sigma = 4`` (~4e-4 relative in
            float32, ~1e-6 in float64 against the exact adjoint DFT
            over the FULL band); with ``sigma = 2`` the upper half band
            ``k >= nf/2`` is aliased at O(1) -- it does not "cancel"
            between data and template.
        """
        # GPU adjoint NFFT of the (non-uniform) samples. Unlike a uniform-
        # grid RFFT, the adjoint NFFT takes the raw times directly and
        # normalizes by the true [min(t), max(t)] baseline, so it (a) runs
        # on the device -- actually exercising the compiled kernels rather
        # than computing on the host -- and (b) covers the full baseline
        # with no ``median(dt)*nf`` span limit, so multi-season / gappy
        # data is no longer silently truncated. ``ghat`` is returned at
        # Fourier modes k = 0..nf-1, i.e. frequencies k/(max(t)-min(t)),
        # with a common per-mode phase set by the transform's own time
        # reference (the kernel references t=0, not min(t)); that phase
        # cancels in every Re sum A B*/P inner product of the detectors.
        # Every one of the nf modes must be accurate: the per-mode NFFT
        # error does NOT cancel between data and template (it is the
        # l = -1 aliasing term of the Gaussian window, different for each
        # input), so the grid is oversampled with sigma = 4 (default),
        # which keeps k = 0..nf-1 inside the window's accuracy band
        # (~4e-4 relative in float32, ~1e-6 in float64 against the exact
        # adjoint DFT over the full band; with sigma = 2 the modes
        # k >= nf/2 were aliased at O(1), in double precision too).
        if len(t) < 2:
            return np.zeros(nf, dtype=self.complex_type)
        y = np.ascontiguousarray(y, dtype=self.real_type)
        if memory is not None:
            memory.y = y
            ghat = self.nufft_proc.run([(memory.t, y, int(nf))],
                                       memory=[memory], **kwargs)[0]
            # ghat_c is the memory's reused pinned buffer: copy it out
            return np.array(ghat, dtype=self.complex_type)
        # float64 epoch subtraction BEFORE the cast: float32 spacing at
        # BJD ~ 2.457e6 is 0.25 d (wider than a transit), so gridding
        # absolute times in float32 returned a different transform.
        t64, _ = subtract_epoch(np.asarray(t, dtype=np.float64))
        t32 = np.ascontiguousarray(t64, dtype=self.real_type)
        ghat = self.nufft_proc.run([(t32, y, int(nf))], **kwargs)[0]
        return np.array(ghat, dtype=self.complex_type)

    def run(self, t, y, periods, durations=None, epochs=None,
            depth=1.0, nf=None, estimate_psd=True, psd=None,
            smooth_window=5, eps_floor=1e-3,
            detector='matched', systematics_basis=None,
            coeff_prior_mean=None, coeff_prior_cov=None, dy=None,
            epoch_oversample=2.0, min_epochs=8, max_epochs=96,
            **kwargs):
        """
        Run NUFFT LRT for transit detection.

        Parameters
        ----------
        t : array-like
            Observation times, any origin (absolute BJD is fine):
            ``floor(min(t))`` is subtracted in float64 before any cast
            to the device precision.
        y : array-like
            Observation values (lightcurve)
        periods : array-like
            Trial periods to test (same units as ``t``)
        durations : array-like, optional
            Trial transit durations, searched as a full outer product
            with ``periods``: every (period, duration) pair is
            evaluated, not the elementwise pairing.

            ``None`` (the default) sets ``durations = 0.1 * periods``,
            i.e. ``len(periods)`` durations, so the default call costs
            ``len(periods)**2`` cells -- quadratic in the size of the
            period grid, and with the automatic epoch grid
            (``epochs=None``) up to ``max_epochs`` transforms per cell
            (154 periods is already ~2.3 million templates at ~0.2 ms
            each). **Pass an explicit, short duration array** (a
            handful of physically motivated durations, or
            ``0.1 * P`` for one representative ``P``) for anything but
            a toy grid.
        epochs : array-like, optional
            Trial epochs (transit mid-times) in the caller's time scale.
            ``None`` (default) scans an automatic epoch grid per
            (period, duration) cell -- see :func:`epoch_grid`:
            ``clip(ceil(epoch_oversample * P / duration), min_epochs,
            max_epochs)`` epochs spaced ``P / n`` apart -- and reduces
            by the maximum over epochs. Cost: that many NFFTs per
            (period, duration) cell (~2P/duration transforms at the
            default oversampling; 0.2-0.4 ms each on an A40). An
            explicit array is used as given for every cell.
        depth : float, optional (default: 1.0)
            Transit depth of the template (the statistic is
            normalized, so this only sets the template's scale)
        nf : int, optional
            Number of frequency samples for NUFFT. If None, uses 2 * len(t)
        estimate_psd : bool, optional (default: True)
            Estimate power spectrum from data. If False, must provide psd.
            The estimate is the ``smooth_window``-bin boxcar-smoothed
            periodogram ``|Y_k|^2`` of the demeaned data (for
            ``detector='marginal'``: of the basis-projected residual
            ``y - V c_ols``, since the mean-subtracted data ``y - V mu``
            still contain the realized systematics ``V (c - mu)``, whose
            power the spectral window spreads over the whole band; the
            PSD from ``y - V mu`` inflated the estimate ~36x and whitened
            the transit away -- audit Sep 2026).
        psd : array-like, optional
            Pre-computed power spectrum of length ``nf`` in the
            convention of the module docstring (``E|S_k|^2`` of the
            noise's unnormalized adjoint NFFT; white noise: ``n sigma^2``).
            Required if ``estimate_psd=False``. Floored at
            ``eps_floor * median`` like the estimate.
        smooth_window : int, optional (default: 5)
            Window size (in frequency bins) for smoothing the power
            spectrum estimate; clamped to ``nf`` when the grid is
            shorter than the window.
        eps_floor : float, optional (default: 1e-3)
            The PSD (estimated or supplied) is floored at ``eps_floor``
            times its positive median once, for every detector, capping
            any bin's whitening weight at ``1/eps_floor`` of typical.
        detector : str, optional (default: 'matched')
            Which detector of Taaki, Kamalabadi & Kemball (2020) to run:

            * ``'matched'`` -- the stationary PSD-whitened matched
              filter (no systematics model). The pre-2026 behavior.
            * ``'marginal'`` -- Detector A: the joint detector with the
              Gaussian prior on systematics coefficients marginalized
              in closed form (Woodbury, in the whitened frequency
              domain). Requires ``systematics_basis`` and
              ``coeff_prior_cov``.
            * ``'sequential'`` -- the papers' "standard" baseline:
              ordinary least-squares cotrend (with intercept) against
              ``systematics_basis`` in the time domain, then the
              stationary matched filter on the residual.

            The papers' Detector B (joint MAP plug-in over a depth
            grid) is intentionally not implemented: the 2020 paper
            found it comparable to Detector A ("exploratory"), and the
            closed-form marginalization supersedes the plug-in.
        systematics_basis : array-like (n, K), optional
            K systematics basis vectors sampled at the observation
            times (e.g. instrument cotrending vectors, or PCA modes of
            a lightcurve population). Columns need not be zero-mean.
        coeff_prior_mean : array-like (K,), optional
            Prior mean of the systematics coefficients (default: zeros).
        coeff_prior_cov : array-like (K, K), optional
            Prior covariance of the coefficients (required for
            ``detector='marginal'``; estimate it from population fits
            as in the papers). Must be symmetric positive semidefinite;
            a zero variance pins that mode to its prior mean (drop the
            mode from the basis if that is not intended). The Gram
            matrix that meets this prior is accumulated over ``nf``
            non-orthogonal NFFT modes and overcounts the corresponding
            time-domain inner products by ~2.2-2.4x (audit Sep 2026),
            so the prior acts as if it were about that much wider than
            what you supply.
        dy : array-like, optional
            Not used by any detector (the noise model is the PSD); a
            ``UserWarning`` is emitted if it is passed.
        epoch_oversample, min_epochs, max_epochs : float, int, int
            Automatic epoch grid parameters (``epochs=None`` only):
            ``n = clip(ceil(epoch_oversample * P / duration),
            min_epochs, max_epochs)``. Defaults 2.0, 8, 96 (the
            validation harness's). At long periods the cap makes the
            epoch step ``P / max_epochs`` exceed the duration; raise
            ``max_epochs`` if those periods matter.
        **kwargs : dict
            Additional parameters passed to the NFFT.

        Returns
        -------
        ``epochs=None`` (default): a tuple ``(snr, best_epoch)`` of two
        float64 arrays of shape ``(len(periods), len(durations))``;
        ``snr[i, j]`` is the maximum of the statistic over the automatic
        epoch grid of cell ``(periods[i], durations[j])`` and
        ``best_epoch[i, j]`` the epoch (transit mid-time, in the
        caller's time scale, within one period of ``floor(min(t))``)
        that attains it.

        ``epochs`` given: one float64 array of shape ``(len(periods),
        len(durations), len(epochs))`` with the statistic at every
        template.

        In both cases the value is the whitened correlation of the
        module docstring: not N(0, 1), calibrate thresholds empirically.
        """
        # ---- validate and epoch-subtract (float64) before ANY cast
        t = np.asarray(t, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        # Shared validator, so the message reads the same as every
        # other entry point's. min_n = 3: the detrending and PSD
        # estimate need more than a two-point series (Detector A's
        # marginal statistic raises a broadcast error at N <= 2).
        # ``dy`` is deliberately not passed: no detector uses it (the
        # noise model is the PSD) and it is warned about below.
        check_lightcurve(t, y, min_n=3, name='NUFFTLRTAsyncProcess.run')
        if dy is not None:
            warnings.warn("NUFFTLRTAsyncProcess.run: dy is not used by any "
                          "detector (the noise model is the PSD); it is "
                          "ignored", UserWarning, stacklevel=2)
        t, t0 = subtract_epoch(t)
        n = len(t)

        periods = np.atleast_1d(np.asarray(periods, dtype=np.float64))
        if periods.ndim != 1 or len(periods) == 0 or np.any(periods <= 0) \
                or not np.all(np.isfinite(periods)):
            raise ValueError("periods must be a non-empty 1-D array of "
                             "positive finite values")

        if detector not in ('matched', 'marginal', 'sequential'):
            raise ValueError("detector must be 'matched', 'marginal' or "
                             "'sequential' (got %r)" % (detector,))
        V = None
        if detector in ('marginal', 'sequential'):
            if systematics_basis is None:
                raise ValueError("detector=%r requires systematics_basis"
                                 % (detector,))
            V = np.atleast_2d(np.asarray(systematics_basis,
                                         dtype=np.float64))
            if V.shape[0] != n:
                V = V.T
            if V.shape[0] != n:
                raise ValueError("systematics_basis must be (n, K) with "
                                 "n = len(t)")
            if not np.all(np.isfinite(V)):
                raise ValueError("systematics_basis must be finite")
        if detector == 'marginal' and coeff_prior_cov is None:
            raise ValueError("detector='marginal' requires "
                             "coeff_prior_cov (estimate it from "
                             "population fits, as in Taaki et al. 2020)")

        # Durations: default to 10% of period if not provided
        if durations is None:
            durations = 0.1 * periods
        durations = np.atleast_1d(np.asarray(durations, dtype=np.float64))
        if durations.ndim != 1 or len(durations) == 0 \
                or np.any(durations <= 0) \
                or not np.all(np.isfinite(durations)):
            raise ValueError("durations must be a non-empty 1-D array of "
                             "positive finite values")

        # Epochs: None -> automatic per-cell grid (max over epochs, best
        # epoch returned); explicit -> shifted into the epoch-subtracted
        # frame and used for every cell (epoch axis in the output).
        auto_epochs = epochs is None
        if not auto_epochs:
            epochs_arr = np.atleast_1d(np.asarray(epochs, dtype=np.float64))
            if epochs_arr.ndim != 1 or len(epochs_arr) == 0 \
                    or not np.all(np.isfinite(epochs_arr)):
                raise ValueError("epochs must be a non-empty 1-D finite "
                                 "array (or None)")
            epochs_arr = epochs_arr - t0

        if nf is None:
            nf = 2 * n
        nf = int(nf)
        if nf < 1:
            raise ValueError("nf must be a positive integer")

        # NOTE: the matched-filter combination runs on the host (an O(nf)
        # reduction, negligible next to the per-template NFFT), so the
        # nufft_lrt.cu kernels are not compiled here. The only GPU work is
        # the adjoint NFFT inside compute_nufft (compiled by nufft_proc).

        # ---- detector-specific data vector (float64 host algebra)
        resid = None
        if detector == 'sequential':
            y_work = _sequential_detrend(t, y, V)
        elif detector == 'marginal':
            K = V.shape[1]
            mu = (np.zeros(K) if coeff_prior_mean is None
                  else np.asarray(coeff_prior_mean, dtype=np.float64).ravel())
            if mu.shape != (K,):
                raise ValueError("coeff_prior_mean must have length K = %d"
                                 % K)
            y_work = y - V @ mu
            if estimate_psd:
                # PSD source: the basis-projected residual, NOT y - V mu
                # (which still holds V (c - mu); with gappy sampling the
                # spectral window spreads that power over the whole band
                # and the whitening then removes the transit too)
                resid = _sequential_detrend(t, y, V)
                resid = resid - resid.mean()
        else:
            y_work = y
        y_demeaned = y_work - np.mean(y_work)
        Vc = None
        if detector == 'marginal':
            Vc = V - V.mean(axis=0)

        # ---- one NFFT buffer set for everything transformed in this run
        l1 = [float(np.sum(np.abs(y_demeaned))), float(n * abs(depth))]
        if resid is not None:
            l1.append(float(np.sum(np.abs(resid))))
        if Vc is not None:
            l1.extend(float(np.sum(np.abs(Vc[:, j])))
                      for j in range(Vc.shape[1]))
        mem = self._nfft_memory(t, nf, max(l1), **kwargs)

        # Compute NUFFT of lightcurve
        Y_nufft = self.compute_nufft(t, y_demeaned, nf, memory=mem,
                                     **kwargs)

        # ---- power spectrum: estimated or supplied, floored ONCE here.
        # The adjoint NFFT returns a physical Fourier coefficient at every
        # one of the nf modes (no rfft-style zero-padded upper half), so
        # the PSD spans all nf bins.
        if estimate_psd:
            if resid is not None:
                src = self.compute_nufft(t, resid, nf, memory=mem, **kwargs)
            else:
                src = Y_nufft
            psd = (np.abs(src) ** 2).astype(self.real_type, copy=False)
            if smooth_window and smooth_window > 1:
                psd = _smoothed_periodogram(psd, smooth_window)
        else:
            if psd is None:
                raise ValueError("Must provide psd if estimate_psd=False")
            psd = np.asarray(psd, dtype=np.float64).ravel()
            if len(psd) != nf:
                raise ValueError("psd must have length nf = %d (got %d); "
                                 "see the module docstring for the PSD "
                                 "convention" % (nf, len(psd)))
            if not np.all(np.isfinite(psd)) or np.any(psd < 0):
                raise ValueError("psd must be finite and non-negative")
        psd = _floor_psd(psd, eps_floor, self.real_type)

        # Every NFFT mode is a physical positive-frequency coefficient, so
        # all bins are weighted equally (the old rfft one-sided 1/2/1
        # weighting was tied to the now-removed uniform-grid RFFT packing).
        weights = np.ones(nf, dtype=self.real_type)
        wp = np.asarray(weights, dtype=np.float64) / np.asarray(psd,
                                                                np.float64)
        Yw = np.asarray(Y_nufft) * wp

        # Detector A: transform the (demeaned) systematics basis once and
        # hoist the template-independent algebra (G, M, w_y) out of the
        # template loop; per template only K inner products remain.
        if detector == 'marginal':
            V_ks = [self.compute_nufft(t, Vc[:, j], nf, memory=mem,
                                       **kwargs)
                    for j in range(Vc.shape[1])]
            Vw, M, w_y = _marginal_precompute(Y_nufft, V_ks, psd, weights,
                                              coeff_prior_cov)

            def _statistic(T_nufft):
                return _marginal_evaluate(Yw, wp, T_nufft, Vw, M, w_y)
        else:
            def _statistic(T_nufft):
                T_nufft = np.asarray(T_nufft)
                num = float(np.real(np.sum(Yw * np.conj(T_nufft))))
                den = float(np.sum((np.abs(T_nufft) ** 2) * wp))
                return num / np.sqrt(den) if den > 0 else 0.0

        def _template_statistic(period, epoch, duration):
            template = self._generate_template(t, period, epoch, duration,
                                               depth)
            template = template - np.mean(template)
            T_nufft = self.compute_nufft(t, template, nf, memory=mem,
                                         **kwargs)
            return _statistic(T_nufft)

        # ---- template loop
        if auto_epochs:
            snr_results = np.zeros((len(periods), len(durations)))
            best_epochs = np.zeros((len(periods), len(durations)))
            for i, period in enumerate(periods):
                for j, duration in enumerate(durations):
                    grid = epoch_grid(period, duration, epoch_oversample,
                                      min_epochs, max_epochs)
                    vals = np.array([_template_statistic(period, e, duration)
                                     for e in grid])
                    k = int(np.argmax(vals))
                    snr_results[i, j] = vals[k]
                    best_epochs[i, j] = grid[k] + t0
            return snr_results, best_epochs

        snr_results = np.zeros((len(periods), len(durations),
                                len(epochs_arr)))
        for i, period in enumerate(periods):
            for j, duration in enumerate(durations):
                for k, epoch in enumerate(epochs_arr):
                    snr_results[i, j, k] = _template_statistic(
                        period, epoch, duration)
        return snr_results

    def _generate_template(self, t, period, epoch, duration, depth):
        """
        Generate simple box transit template.

        Parameters
        ----------
        t : array-like
            Time values
        period : float
            Orbital period
        epoch : float
            Transit mid-time, in the same frame as ``t``
        duration : float
            Transit duration
        depth : float
            Transit depth

        Returns
        -------
        template : np.ndarray
            Transit template (``-depth`` in transit, 0 elsewhere)
        """
        t = np.asarray(t, dtype=np.float64)
        # Phase fold
        phase = np.fmod(t - epoch, period) / period
        phase[phase < 0] += 1.0

        # Center phase around 0.5
        phase[phase > 0.5] -= 1.0

        # Generate box template
        template = np.zeros_like(t)
        phase_width = duration / (2.0 * period)
        in_transit = np.abs(phase) <= phase_width
        template[in_transit] = -depth

        return template

    def _compute_matched_filter_snr(self, Y, T, P_s, weights, eps_floor):
        """
        Compute matched filter SNR.

        Parameters
        ----------
        Y : np.ndarray
            NUFFT of lightcurve
        T : np.ndarray
            NUFFT of template
        P_s : np.ndarray
            Power spectrum
        weights : np.ndarray
            Per-mode weights (``run`` passes all ones)
        eps_floor : float
            Floor for power spectrum

        Returns
        -------
        snr : float
            Signal-to-noise ratio
        """
        # Ensure proper types
        Y = np.asarray(Y, dtype=self.complex_type)
        T = np.asarray(T, dtype=self.complex_type)
        P_s = np.asarray(P_s, dtype=self.real_type)
        weights = np.asarray(weights, dtype=self.real_type)

        # Apply floor to power spectrum
        P_s = _floor_psd(P_s, eps_floor, self.real_type)

        # Compute numerator: sum(Y * conj(T) * weights / P_s)
        numerator = np.real(np.sum((Y * np.conj(T)) * weights / P_s))

        # Compute denominator: sqrt(sum(|T|^2 * weights / P_s))
        denominator = np.sqrt(np.real(np.sum((np.abs(T) ** 2)
                                             * weights / P_s)))

        # Return SNR
        if denominator > 0:
            return numerator / denominator
        else:
            return 0.0
