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
"""
import warnings

import numpy as np

warnings.warn(
    "cuvarbase.nufft_lrt is EXPERIMENTAL and not yet validated against a "
    "reference transit search; use with care. The NFFT transforms now run "
    "on the GPU over the full baseline (the earlier CPU-rfft / median(dt)*nf "
    "grid-truncation issues are fixed), but the matched-filter combination "
    "is still computed on the host and the method has not had a full "
    "injection-recovery validation.",
    UserWarning)

import pycuda.driver as cuda  # noqa: E402
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from .base import GPUAsyncProcess, ensure_context
from .cunfft import NFFTAsyncProcess
from .utils import find_kernel, _module_reader


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
    sigma : float, optional (default: 2.0)
        Oversampling factor for NFFT
    m : int, optional (default: None)
        NFFT truncation parameter (auto-estimated if None)
    use_double : bool, optional (default: False)
        Use double precision
    use_fast_math : bool, optional (default: True)
        Use fast math in CUDA kernels
    block_size : int, optional (default: 256)
        CUDA block size
    autoset_m : bool, optional (default: True)
        Automatically estimate m parameter
    **kwargs : dict
        Additional parameters
        
    Example
    -------
    >>> import numpy as np
    >>> from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
    >>> 
    >>> # Generate sample data
    >>> t = np.sort(np.random.uniform(0, 10, 100))
    >>> y = np.sin(2 * np.pi * t / 2.0) + 0.1 * np.random.randn(len(t))
    >>> 
    >>> # Run NUFFT LRT
    >>> proc = NUFFTLRTAsyncProcess()
    >>> periods = np.linspace(1.5, 3.0, 50)
    >>> durations = np.linspace(0.1, 0.5, 10)
    >>> snr = proc.run(t, y, periods, durations)
    """
    
    def __init__(self, sigma=2.0, m=None, use_double=False,
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
            sigma=sigma, m=m, use_double=use_double,
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
        Compute NUFFT of data.
        
        Parameters
        ----------
        t : array-like
            Time values
        y : array-like
            Observation values
        nf : int
            Number of frequency samples
        **kwargs : dict
            Additional parameters for NUFFT
            
        Returns
        -------
        nufft_result : np.ndarray
            NUFFT of the data
        """
        # GPU adjoint NFFT of the (non-uniform) samples. Unlike a uniform-
        # grid RFFT, the adjoint NFFT takes the raw times directly and
        # normalizes by the true [min(t), max(t)] baseline, so it (a) runs
        # on the device -- actually exercising the compiled kernels rather
        # than computing on the host -- and (b) covers the full baseline
        # with no ``median(dt)*nf`` span limit, so multi-season / gappy
        # data is no longer silently truncated. ``ghat`` is returned at
        # Fourier modes k = 0..nf-1, i.e. frequencies k/(max(t)-min(t)),
        # with ABSOLUTE-t phases: ghat[k] = sum_j y_j exp(2 pi i f_k t_j)
        # (the kernel re-references to t=0, NOT to min(t); verified
        # against the exact adjoint DFT on device, batch 3 Jul 2026).
        # Only modes k < nf/2 lie inside the sigma=2 Gaussian window's
        # guaranteed-accuracy band; the upper half band carries growing
        # deconvolution error. The matched filter uses the same transform
        # for data and template, so the common phase and per-mode error
        # largely cancel in the whitened correlation.
        t = np.asarray(t, dtype=self.real_type)
        y = np.asarray(y, dtype=self.real_type)
        if len(t) < 2:
            return np.zeros(nf, dtype=self.complex_type)

        ghat = self.nufft_proc.run([(t, y, int(nf))], **kwargs)[0]
        return np.asarray(ghat, dtype=self.complex_type)
        
    def run(self, t, y, periods, durations=None, epochs=None,
            depth=1.0, nf=None, estimate_psd=True, psd=None,
            smooth_window=5, eps_floor=1e-12, **kwargs):
        """
        Run NUFFT LRT for transit detection.
        
        Parameters
        ----------
        t : array-like
            Time values (observation times)
        y : array-like
            Observation values (lightcurve)
        periods : array-like
            Trial periods to test
        durations : array-like, optional
            Trial transit durations. If None, uses 0.1 * periods
        epochs : array-like, optional
            Trial epochs. If None, uses 0.0 for all
        depth : float, optional (default: 1.0)
            Transit depth for template (not critical for normalized matched filter)
        nf : int, optional
            Number of frequency samples for NUFFT. If None, uses 2 * len(t)
        estimate_psd : bool, optional (default: True)
            Estimate power spectrum from data. If False, must provide psd
        psd : array-like, optional
            Pre-computed power spectrum. Required if estimate_psd=False
        smooth_window : int, optional (default: 5)
            Window size for smoothing power spectrum estimate
        eps_floor : float, optional (default: 1e-12)
            Floor for power spectrum to avoid division by zero
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        snr : np.ndarray
            SNR values, shape (len(periods), len(durations), len(epochs))
        """
        # Validate inputs
        t = np.asarray(t, dtype=self.real_type)
        y = np.asarray(y, dtype=self.real_type)
        periods = np.atleast_1d(np.asarray(periods, dtype=self.real_type))
        
        # Durations: default to 10% of period if not provided
        if durations is None:
            durations = 0.1 * periods
        durations = np.atleast_1d(np.asarray(durations, dtype=self.real_type))
        
        # Epochs: if None, treat as single-epoch search (no epoch axis in output)
        return_epoch_axis = epochs is not None
        if epochs is None:
            epochs_arr = np.array([0.0], dtype=self.real_type)
        else:
            epochs_arr = np.atleast_1d(np.asarray(epochs, dtype=self.real_type))
        
        if nf is None:
            nf = 2 * len(t)

        # NOTE: the matched-filter combination runs on the host (an O(nf)
        # reduction, negligible next to the per-template NFFT), so the
        # nufft_lrt.cu kernels are not compiled here. The only GPU work is
        # the adjoint NFFT inside compute_nufft (compiled by nufft_proc).
        # A future pass may wire a batched matched-filter kernel.


        # Demean data
        y_mean = np.mean(y)
        y_demeaned = y - y_mean
        
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
        
        # Prepare results array
        if return_epoch_axis:
            snr_results = np.zeros((len(periods), len(durations), len(epochs_arr)))
        else:
            snr_results = np.zeros((len(periods), len(durations)))
        
        # Loop over periods, durations, and epochs
        for i, period in enumerate(periods):
            # If epochs were requested to span [0, P], allow callers to pass epochs in [0, P]
            # Tests already pass absolute epochs in [0, period], so use epochs_arr directly
            for j, duration in enumerate(durations):
                if return_epoch_axis:
                    for k, epoch in enumerate(epochs_arr):
                        template = self._generate_template(t, period, epoch, duration, depth)
                        template = template - np.mean(template)
                        T_nufft = self.compute_nufft(t, template, nf, **kwargs)
                        snr = self._compute_matched_filter_snr(
                            Y_nufft, T_nufft, psd, weights, eps_floor
                        )
                        snr_results[i, j, k] = snr
                else:
                    template = self._generate_template(t, period, 0.0, duration, depth)
                    template = template - np.mean(template)
                    T_nufft = self.compute_nufft(t, template, nf, **kwargs)
                    snr = self._compute_matched_filter_snr(
                        Y_nufft, T_nufft, psd, weights, eps_floor
                    )
                    snr_results[i, j] = snr
        
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
            Transit epoch
        duration : float
            Transit duration
        depth : float
            Transit depth
            
        Returns
        -------
        template : np.ndarray
            Transit template
        """
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
