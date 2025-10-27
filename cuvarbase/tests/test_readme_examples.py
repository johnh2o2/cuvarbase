"""
Test code examples from README.md to ensure they work correctly.
"""
import pytest
import numpy as np
from pycuda.tools import mark_cuda_test


@mark_cuda_test
class TestReadmeExamples:
    """Test that README.md code examples work correctly"""

    def test_quick_start_example(self):
        """Test the Quick Start example from README"""
        from cuvarbase import bls

        # Generate some sample time series data (same as README)
        np.random.seed(42)  # For reproducibility
        t = np.sort(np.random.uniform(0, 10, 1000)).astype(np.float32)
        y = np.sin(2 * np.pi * t / 2.5) + np.random.normal(0, 0.1, len(t))
        dy = np.ones_like(y) * 0.1  # uncertainties

        # Box Least Squares (BLS) - Transit detection
        # Define frequency grid
        freqs = np.linspace(0.1, 2.0, 5000).astype(np.float32)

        # Standard BLS
        power = bls.eebls_gpu(t, y, dy, freqs)
        best_freq = freqs[np.argmax(power)]
        best_period = 1 / best_freq

        # Check that we got reasonable results
        assert power.shape == freqs.shape
        assert len(power) == 5000
        assert np.max(power) > 0.0

        # Period should be close to true period (2.5 days)
        # Allow generous tolerance since this is a simple test
        assert 2.0 < best_period < 3.0, f"Best period {best_period} not near expected 2.5"

    def test_adaptive_bls_example(self):
        """Test the adaptive BLS example from README"""
        from cuvarbase import bls

        # Generate test data
        np.random.seed(42)
        t = np.sort(np.random.uniform(0, 10, 1000)).astype(np.float32)
        y = np.sin(2 * np.pi * t / 2.5) + np.random.normal(0, 0.1, len(t))
        dy = np.ones_like(y) * 0.1

        freqs = np.linspace(0.1, 2.0, 5000).astype(np.float32)

        # Use adaptive BLS for automatic optimization (5-90x faster!)
        power_adaptive = bls.eebls_gpu_fast_adaptive(t, y, dy, freqs)
        best_freq_adaptive = freqs[np.argmax(power_adaptive)]
        best_period_adaptive = 1 / best_freq_adaptive

        # Check results
        assert power_adaptive.shape == freqs.shape
        assert np.max(power_adaptive) > 0.0
        assert 2.0 < best_period_adaptive < 3.0

    def test_standard_vs_adaptive_consistency(self):
        """Verify standard and adaptive BLS give similar results"""
        from cuvarbase import bls

        # Generate test data
        np.random.seed(42)
        t = np.sort(np.random.uniform(0, 10, 500)).astype(np.float32)
        y = np.sin(2 * np.pi * t / 2.5) + np.random.normal(0, 0.1, len(t))
        dy = np.ones_like(y) * 0.1

        freqs = np.linspace(0.1, 2.0, 1000).astype(np.float32)

        # Run both versions
        power_standard = bls.eebls_gpu(t, y, dy, freqs)
        power_adaptive = bls.eebls_gpu_fast_adaptive(t, y, dy, freqs)

        # Should give very similar results
        max_diff = np.max(np.abs(power_standard - power_adaptive))
        assert max_diff < 1e-5, f"Standard and adaptive differ by {max_diff}"

        # Best frequency should be the same
        best_freq_standard = freqs[np.argmax(power_standard)]
        best_freq_adaptive = freqs[np.argmax(power_adaptive)]
        assert best_freq_standard == best_freq_adaptive
