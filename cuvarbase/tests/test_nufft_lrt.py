"""
Tests for NUFFT-based Likelihood Ratio Test (LRT) for transit detection.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
from numpy.testing import assert_allclose
from pycuda.tools import mark_cuda_test

try:
    from ..nufft_lrt import NUFFTLRTAsyncProcess
    NUFFT_LRT_AVAILABLE = True
except ImportError:
    NUFFT_LRT_AVAILABLE = False


@pytest.mark.skipif(not NUFFT_LRT_AVAILABLE, 
                   reason="NUFFT LRT not available")
class TestNUFFTLRT:
    """Test NUFFT LRT functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.n_data = 100
        self.t = np.sort(np.random.uniform(0, 10, self.n_data))
        
    def generate_transit_signal(self, t, period, epoch, duration, depth):
        """Generate a simple transit signal"""
        phase = np.fmod(t - epoch, period) / period
        phase[phase < 0] += 1.0
        phase[phase > 0.5] -= 1.0
        
        signal = np.zeros_like(t)
        phase_width = duration / (2.0 * period)
        in_transit = np.abs(phase) <= phase_width
        signal[in_transit] = -depth
        
        return signal
        
    @mark_cuda_test
    def test_basic_initialization(self):
        """Test that NUFFTLRTAsyncProcess can be initialized"""
        proc = NUFFTLRTAsyncProcess()
        assert proc is not None
        assert proc.sigma == 2.0
        assert proc.use_double is False
        
    @mark_cuda_test
    def test_template_generation(self):
        """Test transit template generation"""
        proc = NUFFTLRTAsyncProcess()
        
        period = 2.0
        epoch = 0.0
        duration = 0.2
        depth = 1.0
        
        template = proc._generate_template(
            self.t, period, epoch, duration, depth
        )
        
        # Check template properties
        assert len(template) == len(self.t)
        assert np.min(template) == -depth
        assert np.max(template) == 0.0
        
        # Check that some points are in transit
        in_transit = template < 0
        assert np.sum(in_transit) > 0
        assert np.sum(in_transit) < len(template)
        
    @mark_cuda_test
    def test_nufft_computation(self):
        """Test NUFFT computation"""
        proc = NUFFTLRTAsyncProcess()
        
        # Generate simple sinusoidal signal
        y = np.sin(2 * np.pi * self.t / 2.0)
        
        nf = 2 * len(self.t)
        Y_nufft = proc.compute_nufft(self.t, y, nf)
        
        # Check output properties
        assert len(Y_nufft) == nf
        assert Y_nufft.dtype in [np.complex64, np.complex128]
        
        # Peak should be near the signal frequency
        freqs = np.fft.rfftfreq(nf, d=np.median(np.diff(self.t)))
        power = np.abs(Y_nufft) ** 2
        peak_freq_idx = np.argmax(power[1:]) + 1  # Skip DC
        peak_freq = freqs[peak_freq_idx]
        
        # Should be close to 0.5 Hz (period 2.0)
        assert np.abs(peak_freq - 0.5) < 0.1
        
    @mark_cuda_test
    def test_matched_filter_snr_computation(self):
        """Test matched filter SNR computation"""
        proc = NUFFTLRTAsyncProcess()
        
        # Generate signals
        nf = 200
        Y = np.random.randn(nf) + 1j * np.random.randn(nf)
        T = np.random.randn(nf) + 1j * np.random.randn(nf)
        P_s = np.ones(nf)
        weights = np.ones(nf)
        
        snr = proc._compute_matched_filter_snr(
            Y, T, P_s, weights, eps_floor=1e-12
        )
        
        # SNR should be a finite scalar
        assert np.isfinite(snr)
        assert isinstance(snr, (float, np.floating))
        
    @mark_cuda_test
    def test_detection_of_known_transit(self):
        """Test detection of a known transit signal"""
        proc = NUFFTLRTAsyncProcess()
        
        # Generate transit signal
        true_period = 2.5
        true_duration = 0.2
        true_epoch = 0.0
        depth = 0.5
        noise_level = 0.1
        
        signal = self.generate_transit_signal(
            self.t, true_period, true_epoch, true_duration, depth
        )
        noise = noise_level * np.random.randn(len(self.t))
        y = signal + noise
        
        # Search over periods
        periods = np.linspace(2.0, 3.0, 20)
        durations = np.array([true_duration])
        
        snr = proc.run(self.t, y, periods, durations=durations)
        
        # Check output shape
        assert snr.shape == (len(periods), len(durations))
        
        # Peak should be near true period
        best_period_idx = np.argmax(snr[:, 0])
        best_period = periods[best_period_idx]
        
        # Allow for some tolerance
        assert np.abs(best_period - true_period) < 0.3
        
    @mark_cuda_test
    def test_white_noise_gives_low_snr(self):
        """Test that white noise gives low SNR"""
        proc = NUFFTLRTAsyncProcess()
        
        # Pure white noise
        y = np.random.randn(len(self.t))
        
        periods = np.array([2.0, 3.0, 4.0])
        durations = np.array([0.2])
        
        snr = proc.run(self.t, y, periods, durations=durations)
        
        # SNR should be relatively low for pure noise
        assert np.all(np.abs(snr) < 5.0)
        
    @mark_cuda_test
    def test_custom_psd(self):
        """Test using a custom power spectrum"""
        proc = NUFFTLRTAsyncProcess()
        
        # Generate simple signal
        y = np.sin(2 * np.pi * self.t / 2.0) + 0.1 * np.random.randn(len(self.t))
        
        periods = np.array([2.0])
        durations = np.array([0.2])
        nf = 2 * len(self.t)
        
        # Create custom PSD (flat spectrum)
        custom_psd = np.ones(nf)
        
        snr = proc.run(
            self.t, y, periods, durations=durations,
            nf=nf, estimate_psd=False, psd=custom_psd
        )
        
        # Should run without error
        assert snr.shape == (1, 1)
        assert np.isfinite(snr[0, 0])
        
    @mark_cuda_test
    def test_double_precision(self):
        """Test double precision mode"""
        proc = NUFFTLRTAsyncProcess(use_double=True)
        
        y = np.sin(2 * np.pi * self.t / 2.0)
        periods = np.array([2.0])
        durations = np.array([0.2])
        
        snr = proc.run(self.t, y, periods, durations=durations)
        
        assert snr.shape == (1, 1)
        assert np.isfinite(snr[0, 0])
        
    @mark_cuda_test
    def test_multiple_epochs(self):
        """Test searching over multiple epochs"""
        proc = NUFFTLRTAsyncProcess()
        
        # Generate transit signal
        true_period = 2.5
        true_duration = 0.2
        true_epoch = 0.5
        depth = 0.5
        
        signal = self.generate_transit_signal(
            self.t, true_period, true_epoch, true_duration, depth
        )
        y = signal + 0.1 * np.random.randn(len(self.t))
        
        periods = np.array([true_period])
        durations = np.array([true_duration])
        epochs = np.linspace(0, true_period, 10)
        
        snr = proc.run(
            self.t, y, periods, durations=durations, epochs=epochs
        )
        
        # Check output shape
        assert snr.shape == (1, 1, len(epochs))
        
        # Best epoch should be close to true epoch
        best_epoch_idx = np.argmax(snr[0, 0, :])
        best_epoch = epochs[best_epoch_idx]
        
        # Allow for periodicity and tolerance
        epoch_diff = np.abs(best_epoch - true_epoch)
        epoch_diff = min(epoch_diff, true_period - epoch_diff)
        assert epoch_diff < 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
