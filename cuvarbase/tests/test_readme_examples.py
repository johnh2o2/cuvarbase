"""
Test code examples from README.md to ensure they work correctly.

These require a GPU; on CPU-only machines the root conftest converts
them to skips. (An earlier version of this file was silently never
collected — @mark_cuda_test on the class turned it into a plain
function — and unpacked eebls_gpu's tuple return incorrectly.)
"""
import numpy as np
import pytest


class TestReadmeExamples:
    """Test that README.md code examples work correctly"""

    def _data(self, ndata=1000):
        np.random.seed(42)  # For reproducibility
        t = np.sort(np.random.uniform(0, 10, ndata)).astype(np.float32)
        y = np.sin(2 * np.pi * t / 2.5) + np.random.normal(0, 0.1, len(t))
        dy = np.ones_like(y) * 0.1  # uncertainties
        return t, y, dy

    def test_quick_start_example(self):
        """Test the Quick Start example from README"""
        from cuvarbase import bls

        t, y, dy = self._data()
        freqs = np.linspace(0.1, 2.0, 5000).astype(np.float32)

        # Standard BLS returns (power, solutions) — as in the README
        power, solutions = bls.eebls_gpu(t, y, dy, freqs)
        best_freq = freqs[np.argmax(power)]
        best_period = 1 / best_freq

        assert power.shape == freqs.shape
        assert len(solutions) == len(freqs)
        assert np.max(power) > 0.0

        # Period should be close to true period (2.5 days); BLS on a
        # sinusoid typically locks onto P or P/2.
        assert (2.0 < best_period < 3.0) or (1.0 < best_period < 1.5), \
            "Best period %s not near 2.5 or 1.25" % best_period

    def test_adaptive_bls_example(self):
        """Test the adaptive BLS example from README"""
        from cuvarbase import bls

        t, y, dy = self._data()
        freqs = np.linspace(0.1, 2.0, 5000).astype(np.float32)

        power_adaptive = bls.eebls_gpu_fast_adaptive(t, y, dy, freqs)

        assert power_adaptive.shape == freqs.shape
        assert np.max(power_adaptive) > 0.0

    def test_standard_vs_adaptive_consistency(self):
        """Standard and adaptive BLS should agree on the periodogram.

        They use different binning strategies (eebls_gpu bins per-(q,phi)
        solution; the fast/adaptive kernel scans a binned histogram), so
        exact equality is not expected — require strong correlation and
        matching peak, the same criteria used for the GPU/GPU checks in
        scripts/benchmark_new_features.py.
        """
        from cuvarbase import bls

        t, y, dy = self._data(ndata=500)
        freqs = np.linspace(0.1, 2.0, 1000).astype(np.float32)

        power_standard, _ = bls.eebls_gpu(t, y, dy, freqs)
        power_adaptive = bls.eebls_gpu_fast_adaptive(t, y, dy, freqs)

        corr = np.corrcoef(power_standard, power_adaptive)[0, 1]
        assert corr > 0.95, "standard/adaptive correlation %.4f" % corr

        # Peak frequencies should agree to within a few grid points
        ipeak_standard = np.argmax(power_standard)
        ipeak_adaptive = np.argmax(power_adaptive)
        assert abs(int(ipeak_standard) - int(ipeak_adaptive)) <= 3
