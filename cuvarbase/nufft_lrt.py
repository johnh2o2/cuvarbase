#!/usr/bin/env python
"""
NUFFT-based Likelihood Ratio Test for transit detection.

This module implements the matched filter approach described in:
"Wavelet-based matched filter for detection of known up to parameters signals 
in unknown correlated Gaussian noise" (IEEE paper)

The method uses NUFFT for gappy data and adaptive noise estimation via power spectrum.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import object

import sys
import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from .base import GPUAsyncProcess
from .cunfft import NFFTAsyncProcess
from .memory import NFFTMemory
from .utils import find_kernel, _module_reader


class NUFFTLRTMemory(object):
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
        self._cpp_defs = '#define DOUBLE_PRECISION\n' if use_double else ''
        
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
        data = [(t, y, nf)]
        memory = self.nufft_proc.allocate(data, **kwargs)
        results = self.nufft_proc.run(data, memory=memory, **kwargs)
        self.nufft_proc.finish()
        
        return results[0]
        
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
        
        if durations is None:
            durations = 0.1 * periods
        durations = np.atleast_1d(np.asarray(durations, dtype=self.real_type))
        
        if epochs is None:
            epochs = np.array([0.0], dtype=self.real_type)
        epochs = np.atleast_1d(np.asarray(epochs, dtype=self.real_type))
        
        if nf is None:
            nf = 2 * len(t)
            
        # Compile kernels if needed
        if not hasattr(self, 'prepared_functions') or \
           not all([func in self.prepared_functions 
                   for func in self.function_names]):
            self._compile_and_prepare_functions(**kwargs)
            
        # Demean data
        y_mean = np.mean(y)
        y_demeaned = y - y_mean
        
        # Compute NUFFT of lightcurve
        Y_nufft = self.compute_nufft(t, y_demeaned, nf, **kwargs)
        
        # Estimate or use provided power spectrum
        if estimate_psd:
            # Transfer Y_nufft to GPU for PSD estimation
            if len(self.streams) == 0:
                self._create_streams(1)
            stream = self.streams[0]
            
            Y_g = gpuarray.to_gpu_async(Y_nufft, stream=stream)
            P_s_g = gpuarray.zeros(nf, dtype=self.real_type)
            
            # Estimate power spectrum
            block = (self.block_size, 1, 1)
            grid = (int(np.ceil(nf / self.block_size)), 1)
            
            func = self.prepared_functions['estimate_power_spectrum']
            func.prepared_async_call(
                grid, block, stream,
                Y_g.ptr, P_s_g.ptr,
                np.int32(nf), np.int32(smooth_window),
                self.real_type(eps_floor)
            )
            
            psd = P_s_g.get()
            stream.synchronize()
        else:
            if psd is None:
                raise ValueError("Must provide psd if estimate_psd=False")
            psd = np.asarray(psd, dtype=self.real_type)
            
        # Compute frequency weights
        if len(self.streams) == 0:
            self._create_streams(1)
        stream = self.streams[0]
        
        weights_g = gpuarray.zeros(nf, dtype=self.real_type)
        block = (self.block_size, 1, 1)
        grid = (int(np.ceil(nf / self.block_size)), 1)
        
        func = self.prepared_functions['compute_frequency_weights']
        func.prepared_async_call(
            grid, block, stream,
            weights_g.ptr, np.int32(nf), np.int32(len(t))
        )
        stream.synchronize()
        
        # Prepare results array
        snr_results = np.zeros((len(periods), len(durations), len(epochs)))
        
        # Loop over periods, durations, and epochs
        for i, period in enumerate(periods):
            for j, duration in enumerate(durations):
                for k, epoch in enumerate(epochs):
                    # Generate transit template
                    template = self._generate_template(
                        t, period, epoch, duration, depth
                    )
                    
                    # Demean template
                    template = template - np.mean(template)
                    
                    # Compute NUFFT of template
                    T_nufft = self.compute_nufft(t, template, nf, **kwargs)
                    
                    # Compute matched filter SNR
                    snr = self._compute_matched_filter_snr(
                        Y_nufft, T_nufft, psd, 
                        weights_g.get(), eps_floor
                    )
                    
                    snr_results[i, j, k] = snr
                    
        return np.squeeze(snr_results)
        
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
