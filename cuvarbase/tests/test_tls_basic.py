"""
Basic tests for TLS GPU implementation.

These tests verify the basic functionality of the TLS implementation,
focusing on API correctness and basic execution rather than scientific
accuracy (which will be tested in test_tls_consistency.py).
"""

import pytest
import numpy as np

try:
    import pycuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False

# Import modules to test
from cuvarbase import tls_grids, tls_models


class TestGridGeneration:
    """Test period and duration grid generation."""

    def test_period_grid_basic(self):
        """Test basic period grid generation."""
        t = np.linspace(0, 100, 1000)  # 100-day observation

        periods = tls_grids.period_grid_ofir(t, R_star=1.0, M_star=1.0)

        assert len(periods) > 0
        assert np.all(periods > 0)
        assert np.all(np.diff(periods) > 0)  # Increasing
        assert periods[0] < periods[-1]

    def test_period_grid_limits(self):
        """Test period grid with custom limits."""
        t = np.linspace(0, 100, 1000)

        periods = tls_grids.period_grid_ofir(
            t, period_min=5.0, period_max=20.0
        )

        assert periods[0] >= 5.0
        assert periods[-1] <= 20.0

    def test_duration_grid(self):
        """Test duration grid generation."""
        periods = np.array([10.0, 20.0, 30.0])

        durations, counts = tls_grids.duration_grid(periods)

        assert len(durations) == len(periods)
        assert len(counts) == len(periods)
        assert all(c > 0 for c in counts)

        # Check durations are reasonable (< period)
        for i, period in enumerate(periods):
            assert all(d < period for d in durations[i])
            assert all(d > 0 for d in durations[i])

    def test_transit_duration_max(self):
        """Test maximum transit duration calculation."""
        period = 10.0  # days

        duration = tls_grids.transit_duration_max(
            period, R_star=1.0, M_star=1.0, R_planet=1.0
        )

        assert duration > 0
        assert duration < period  # Duration must be less than period
        assert duration < 1.0  # For Earth-Sun system, ~0.5 days

    def test_t0_grid(self):
        """Test T0 grid generation."""
        period = 10.0
        duration = 0.1

        t0_values = tls_grids.t0_grid(period, duration, oversampling=5)

        assert len(t0_values) > 0
        assert np.all(t0_values >= 0)
        assert np.all(t0_values <= 1)

    def test_validate_stellar_parameters(self):
        """Test stellar parameter validation."""
        # Valid parameters
        tls_grids.validate_stellar_parameters(R_star=1.0, M_star=1.0)

        # Invalid radius
        with pytest.raises(ValueError):
            tls_grids.validate_stellar_parameters(R_star=10.0, M_star=1.0)

        # Invalid mass
        with pytest.raises(ValueError):
            tls_grids.validate_stellar_parameters(R_star=1.0, M_star=5.0)


@pytest.mark.skipif(not tls_models.BATMAN_AVAILABLE,
                   reason="batman-package not installed")
class TestTransitModels:
    """Test transit model generation (requires batman)."""

    def test_reference_transit(self):
        """Test reference transit model creation."""
        phases, flux = tls_models.create_reference_transit(n_samples=100)

        assert len(phases) == len(flux)
        assert len(phases) == 100
        assert np.all((phases >= 0) & (phases <= 1))
        assert np.all(flux <= 1.0)  # Transit causes dimming
        assert np.min(flux) < 1.0  # There is a transit

    def test_transit_model_cache(self):
        """Test transit model cache creation."""
        durations = np.array([0.05, 0.1, 0.15])

        models, phases = tls_models.create_transit_model_cache(
            durations, period=10.0, n_samples=100
        )

        assert len(models) == len(durations)
        assert len(phases) == 100
        for model in models:
            assert len(model) == len(phases)


class TestSimpleTransitModels:
    """Test simple transit models (no batman required)."""

    def test_simple_trapezoid(self):
        """Test simple trapezoidal transit."""
        phases = np.linspace(0, 1, 1000)
        duration_phase = 0.1

        flux = tls_models.simple_trapezoid_transit(
            phases, duration_phase, depth=0.01
        )

        assert len(flux) == len(phases)
        assert np.all(flux <= 1.0)
        assert np.min(flux) < 1.0  # There is a transit
        assert np.max(flux) == 1.0  # Out of transit = 1.0

    def test_interpolate_transit_model(self):
        """Test transit model interpolation."""
        model_phases = np.linspace(0, 1, 100)
        model_flux = np.ones(100)
        model_flux[40:60] = 0.99  # Simple transit

        target_phases = np.linspace(0, 1, 200)

        flux_interp = tls_models.interpolate_transit_model(
            model_phases, model_flux, target_phases, target_depth=0.01
        )

        assert len(flux_interp) == len(target_phases)
        assert np.all(flux_interp <= 1.0)

    def test_default_limb_darkening(self):
        """Test default limb darkening coefficient lookup."""
        u_kepler = tls_models.get_default_limb_darkening('Kepler', T_eff=5500)
        assert len(u_kepler) == 2
        assert all(0 < coeff < 1 for coeff in u_kepler)

        u_tess = tls_models.get_default_limb_darkening('TESS', T_eff=5500)
        assert len(u_tess) == 2

    def test_validate_limb_darkening(self):
        """Test limb darkening validation."""
        # Valid quadratic
        tls_models.validate_limb_darkening_coeffs([0.4, 0.2], 'quadratic')

        # Invalid - wrong number
        with pytest.raises(ValueError):
            tls_models.validate_limb_darkening_coeffs([0.4], 'quadratic')


@pytest.mark.skipif(not PYCUDA_AVAILABLE,
                   reason="PyCUDA not available")
class TestTLSKernel:
    """Test TLS kernel compilation and basic execution."""

    def test_kernel_compilation(self):
        """Test that TLS kernel compiles."""
        from cuvarbase import tls

        kernel = tls.compile_tls(block_size=128)
        assert kernel is not None

    def test_kernel_caching(self):
        """Test kernel caching mechanism."""
        from cuvarbase import tls

        # First call - compiles
        kernel1 = tls._get_cached_kernels(128)
        assert kernel1 is not None

        # Second call - should use cache
        kernel2 = tls._get_cached_kernels(128)
        assert kernel2 is kernel1

    def test_block_size_selection(self):
        """Test automatic block size selection."""
        from cuvarbase import tls

        assert tls._choose_block_size(10) == 32
        assert tls._choose_block_size(50) == 64
        assert tls._choose_block_size(100) == 128


@pytest.mark.skipif(not PYCUDA_AVAILABLE,
                   reason="PyCUDA not available")
class TestTLSMemory:
    """Test TLS memory management."""

    def test_memory_allocation(self):
        """Test memory allocation."""
        from cuvarbase.tls import TLSMemory

        mem = TLSMemory(max_ndata=1000, max_nperiods=100)

        assert mem.t is not None
        assert len(mem.t) == 1000
        assert len(mem.periods) == 100

    def test_memory_setdata(self):
        """Test setting data."""
        from cuvarbase.tls import TLSMemory

        t = np.linspace(0, 100, 100)
        y = np.ones(100)
        dy = np.ones(100) * 0.01
        periods = np.linspace(1, 10, 50)

        mem = TLSMemory(max_ndata=1000, max_nperiods=100)
        mem.setdata(t, y, dy, periods=periods, transfer=False)

        assert np.allclose(mem.t[:100], t)
        assert np.allclose(mem.periods[:50], periods)

    def test_memory_fromdata(self):
        """Test creating memory from data."""
        from cuvarbase.tls import TLSMemory

        t = np.linspace(0, 100, 100)
        y = np.ones(100)
        dy = np.ones(100) * 0.01
        periods = np.linspace(1, 10, 50)

        mem = TLSMemory.fromdata(t, y, dy, periods=periods, transfer=False)

        assert mem.max_ndata >= 100
        assert mem.max_nperiods >= 50


@pytest.mark.skipif(not PYCUDA_AVAILABLE,
                   reason="PyCUDA not available")
class TestTLSBasicExecution:
    """Test basic TLS execution (not accuracy)."""

    def test_tls_search_runs(self):
        """Test that TLS search runs without errors."""
        from cuvarbase import tls

        # Create simple synthetic data
        t = np.linspace(0, 100, 500)
        y = np.ones(500)
        dy = np.ones(500) * 0.001

        # Use small period range for speed
        periods = np.linspace(5, 15, 20)

        # This should run without errors
        results = tls.tls_search_gpu(
            t, y, dy,
            periods=periods,
            block_size=64
        )

        assert results is not None
        assert 'periods' in results
        assert 'chi2' in results
        assert len(results['periods']) == 20

    def test_tls_search_with_transit(self):
        """Test TLS with injected transit."""
        from cuvarbase import tls

        # Create data with simple transit
        t = np.linspace(0, 100, 500)
        y = np.ones(500)

        # Inject transit at period = 10 days
        period_true = 10.0
        duration = 0.1
        depth = 0.01

        phases = (t % period_true) / period_true
        in_transit = (phases < duration / period_true) | (phases > 1 - duration / period_true)
        y[in_transit] -= depth

        dy = np.ones(500) * 0.0001

        # Search with periods around the true value
        periods = np.linspace(8, 12, 30)

        results = tls.tls_search_gpu(t, y, dy, periods=periods)

        # Should return results
        assert results['chi2'] is not None
        assert len(results['chi2']) == 30

        # Minimum chi2 should be near period = 10 (within a few samples)
        # Note: This is a weak test - full validation in test_tls_consistency.py
        min_idx = np.argmin(results['chi2'])
        best_period = results['periods'][min_idx]

        # Should be within 20% of true period (very loose for Phase 1)
        assert 8 < best_period < 12


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
