#!/usr/bin/env python
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
* ``dy`` is not used by any detector (a ``UserWarning`` is emitted if it
  is passed); the noise model is the PSD.
"""
import warnings

import numpy as np

warnings.warn(
    "cuvarbase.nufft_lrt is EXPERIMENTAL. The Sep-2026 correctness fixes "
    "(float64 epoch subtraction, automatic epoch grid for epochs=None, "
    "Detector A PSD from the cotrended residual, centred sequential "
    "cotrend, full-band NFFT accuracy) are awaiting injection-recovery "
    "re-validation; the statistic is not N(0, 1) and thresholds must be "
    "calibrated empirically (see docs/NUFFT_LRT_README.md).",
    UserWarning)

import pycuda.driver as cuda  # noqa: E402
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from .base import GPUAsyncProcess, ensure_context
from .cunfft import NFFTAsyncProcess
from .utils import find_kernel, _module_reader, subtract_epoch


def _whitened_inner(A, B, psd, weights):
    """Whitened frequency-domain inner product Re sum_k A_k B_k* w_k / P_k
    -- the metric of the stationary matched filter."""
    return float(np.real(np.sum(A * np.conj(B) * weights / psd)))


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

    Parameters: Y, T = NFFTs of the (mean-subtracted) data and template;
    V_ks = list/array of K basis NFFTs; prior_cov = Cov_c (K x K).
    Returns the marginalized SNR (float).
    """
    K = len(V_ks)
    if K == 0:
        num = _whitened_inner(Y, T, psd, weights)
        den = _whitened_inner(T, T, psd, weights)
        return num / np.sqrt(den) if den > 0 else 0.0

    G = np.empty((K, K))
    for i in range(K):
        for j in range(i, K):
            G[i, j] = G[j, i] = _whitened_inner(V_ks[i], V_ks[j],
                                                psd, weights)
    prior_cov = np.atleast_2d(np.asarray(prior_cov, dtype=np.float64))
    M = np.linalg.pinv(np.linalg.pinv(prior_cov) + G)

    w_y = np.array([_whitened_inner(V_ks[j], Y, psd, weights)
                    for j in range(K)])
    w_t = np.array([_whitened_inner(V_ks[j], T, psd, weights)
                    for j in range(K)])

    num = _whitened_inner(Y, T, psd, weights) - w_y @ M @ w_t
    den = _whitened_inner(T, T, psd, weights) - w_t @ M @ w_t
    if den <= eps_floor:
        return 0.0
    return float(num / np.sqrt(den))


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
    """
    k = int(window)
    if k <= 1:
        return power
    kernel = np.ones(k, dtype=power.dtype)
    num = np.convolve(power, kernel, mode='same')
    den = np.convolve(np.ones_like(power), kernel, mode='same')
    return (num / den).astype(power.dtype, copy=False)


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
        
        # Frequency weights for one-sided spectrum
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
    GPU implementation of NUFFT-based Likelihood Ratio Test for transit detection.
    
    This implements a matched filter in the frequency domain:
    
    .. math::
        \\text{SNR} = \\frac{\\sum_k Y_k T_k^* w_k / P_s(k)}{\\sqrt{\\sum_k |T_k|^2 w_k / P_s(k)}}
    
    where:
    - Y_k is the NUFFT of the lightcurve
    - T_k is the NUFFT of the transit template
    - P_s(k) is the power spectrum (adaptively estimated or provided)
    - w_k are frequency weights for one-sided spectrum
    
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
        :meth:`cuvarbase.cunfft.NFFTAsyncProcess.estimate_m`).
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
            nufft_matched_filter=[np.intp, np.intp, np.intp, np.intp, np.intp,
                                 np.int32, self.real_type],
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
            
    def compute_nufft(self, t, y, nf, **kwargs):
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
        # float64 epoch subtraction BEFORE the cast: float32 spacing at
        # BJD ~ 2.457e6 is 0.25 d (wider than a transit), so gridding
        # absolute times in float32 returned a different transform.
        t64, _ = subtract_epoch(np.asarray(t, dtype=np.float64))
        t32 = np.ascontiguousarray(t64, dtype=self.real_type)
        ghat = self.nufft_proc.run([(t32, y, int(nf))], **kwargs)[0]
        return np.array(ghat, dtype=self.complex_type)
        
    def run(self, t, y, periods, durations=None, epochs=None,
            depth=1.0, nf=None, estimate_psd=True, psd=None,
            smooth_window=5, eps_floor=1e-12,
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
            Trial transit durations. If None, uses 0.1 * periods
        epochs : array-like, optional
            Trial epochs (transit mid-times) in the caller's time scale.
            ``None`` (default) scans an automatic epoch grid per
            (period, duration) cell -- see :func:`epoch_grid`:
            ``clip(ceil(epoch_oversample * P / duration), min_epochs,
            max_epochs)`` epochs spaced ``P / n`` apart -- and reduces
            by the maximum over epochs. Cost: that many NFFTs per
            (period, duration) cell (~2P/duration transforms at the
            default oversampling). An explicit array is used as given
            for every cell.
        depth : float, optional (default: 1.0)
            Transit depth for template (not critical for normalized matched filter)
        nf : int, optional
            Number of frequency samples for NUFFT. If None, uses 2 * len(t)
        estimate_psd : bool, optional (default: True)
            Estimate power spectrum from data. If False, must provide psd
        psd : array-like, optional
            Pre-computed power spectrum of length ``nf`` in the
            convention of the module docstring (``E|S_k|^2`` of the
            noise's unnormalized adjoint NFFT; white noise: ``n sigma^2``).
            Required if ``estimate_psd=False``.
        smooth_window : int, optional (default: 5)
            Window size for smoothing power spectrum estimate
        eps_floor : float, optional (default: 1e-12)
            Floor for power spectrum to avoid division by zero
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
            as in the papers).
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
        if t.shape != y.shape:
            raise ValueError("t and y must have the same length (got %d "
                             "and %d)" % (len(t), len(y)))
        if len(t) < 3:
            raise ValueError("need at least 3 observations (got %d)"
                             % len(t))
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
            raise ValueError("t and y must be finite")
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
        else:
            y_work = y
        y_demeaned = y_work - np.mean(y_work)
        
        # Compute NUFFT of lightcurve
        Y_nufft = self.compute_nufft(t, y_demeaned, nf, **kwargs)
        
        # Estimate or use provided power spectrum. The adjoint NFFT returns
        # a physical Fourier coefficient at every one of the nf modes (no
        # rfft-style zero-padded upper half), so the PSD spans all nf bins.
        if estimate_psd:
            psd = (np.abs(Y_nufft) ** 2).astype(self.real_type, copy=False)
            if smooth_window and smooth_window > 1:
                psd = _smoothed_periodogram(psd, smooth_window)
            # Floor to avoid division issues
            median_ps = np.median(psd[psd > 0]) if np.any(psd > 0) else self.real_type(1.0)
            psd = np.maximum(psd, self.real_type(eps_floor) * self.real_type(median_ps)).astype(self.real_type, copy=False)
        else:
            if psd is None:
                raise ValueError("Must provide psd if estimate_psd=False")
            psd = np.asarray(psd, dtype=self.real_type)

        # Every NFFT mode is a physical positive-frequency coefficient, so
        # all bins are weighted equally (the old rfft one-sided 1/2/1
        # weighting was tied to the now-removed uniform-grid RFFT packing).
        weights = np.ones(nf, dtype=self.real_type)

        # Detector A: transform the (demeaned) systematics basis once;
        # per template the marginalization is K-dimensional algebra.
        V_ks = None
        if detector == 'marginal':
            V_ks = [self.compute_nufft(t, V[:, j] - V[:, j].mean(), nf,
                                       **kwargs)
                    for j in range(V.shape[1])]

        def _statistic(T_nufft):
            if detector == 'marginal':
                return _marginal_statistic(Y_nufft, T_nufft, V_ks, psd,
                                           weights, coeff_prior_cov,
                                           eps_floor)
            return self._compute_matched_filter_snr(
                Y_nufft, T_nufft, psd, weights, eps_floor)

        def _template_statistic(period, epoch, duration):
            template = self._generate_template(t, period, epoch, duration,
                                               depth)
            template = template - np.mean(template)
            T_nufft = self.compute_nufft(t, template, nf, **kwargs)
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
            Frequency weights
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
        P_s = np.maximum(P_s, eps_floor * np.median(P_s[P_s > 0]))
        
        # Compute numerator: sum(Y * conj(T) * weights / P_s)
        numerator = np.real(np.sum((Y * np.conj(T)) * weights / P_s))
        
        # Compute denominator: sqrt(sum(|T|^2 * weights / P_s))
        denominator = np.sqrt(np.real(np.sum((np.abs(T) ** 2) * weights / P_s)))
        
        # Return SNR
        if denominator > 0:
            return numerator / denominator
        else:
            return 0.0
