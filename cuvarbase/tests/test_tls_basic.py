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
from cuvarbase import tls_grids, tls_models, tls_stats


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


class TestTransitTemplate:
    """Test transit template generation for GPU kernel."""

    def test_trapezoid_template_shape(self):
        """Test trapezoidal fallback template has correct shape."""
        template = tls_models._trapezoid_template(n_template=500)

        assert template.shape == (500,)
        assert template.dtype == np.float32

    def test_trapezoid_template_normalization(self):
        """Test trapezoidal template values are in [0, 1]."""
        template = tls_models._trapezoid_template(n_template=1000)

        assert np.all(template >= 0.0)
        assert np.all(template <= 1.0)
        # Center should be at max depth
        assert template[500] == pytest.approx(1.0)
        # Edges should be near zero
        assert template[0] == pytest.approx(0.0, abs=0.01)
        assert template[-1] == pytest.approx(0.0, abs=0.01)

    def test_trapezoid_template_symmetric(self):
        """Test trapezoidal template is symmetric."""
        template = tls_models._trapezoid_template(n_template=1001)
        np.testing.assert_allclose(template, template[::-1], atol=1e-6)

    @pytest.mark.skipif(not tls_models.BATMAN_AVAILABLE,
                       reason="batman-package not installed")
    def test_batman_template_shape(self):
        """Test batman template has correct shape and dtype."""
        template = tls_models.generate_transit_template(n_template=1000)

        assert template.shape == (1000,)
        assert template.dtype == np.float32

    @pytest.mark.skipif(not tls_models.BATMAN_AVAILABLE,
                       reason="batman-package not installed")
    def test_batman_template_normalization(self):
        """Test batman template values are in [0, 1] with max = 1."""
        template = tls_models.generate_transit_template(n_template=1000)

        assert np.all(template >= 0.0)
        assert np.all(template <= 1.0)
        assert np.max(template) == pytest.approx(1.0, abs=0.01)
        # Edges should be near zero
        assert template[0] < 0.1
        assert template[-1] < 0.1

    @pytest.mark.skipif(not tls_models.BATMAN_AVAILABLE,
                       reason="batman-package not installed")
    def test_batman_template_limb_darkened(self):
        """Test batman template shows limb darkening (not a box)."""
        template = tls_models.generate_transit_template(n_template=1000)

        # The template should NOT be a perfect box (all 0 or 1).
        # With limb darkening, there should be intermediate values.
        n_intermediate = np.sum((template > 0.1) & (template < 0.9))
        assert n_intermediate > 10, "Template should have limb-darkened shape, not a box"

    def test_generate_fallback_without_batman(self):
        """Test generate_transit_template falls back to trapezoid."""
        # Force fallback by testing _trapezoid_template directly
        template = tls_models._trapezoid_template(n_template=500)

        assert template.shape == (500,)
        assert np.max(template) == pytest.approx(1.0)
        assert np.min(template) == pytest.approx(0.0, abs=0.01)


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


class TestStatistics:
    """Test TLS statistics calculations."""

    def test_signal_residue_with_signal(self):
        """Test SR is positive for a signal."""
        # Simulate chi2 values where one period has much lower chi2
        chi2 = np.ones(100) * 1000.0
        chi2[50] = 500.0  # Signal at index 50

        SR = tls_stats.signal_residue(chi2)

        # SR at signal should be highest
        assert SR[50] > SR[0]
        assert SR[50] > 0

    def test_sde_positive_for_signal(self):
        """Test SDE > 0 for an injected signal (regression test)."""
        # Simulate chi2 values with a clear signal
        np.random.seed(42)
        chi2 = np.random.normal(1000, 10, size=200)
        chi2[100] = 500.0  # Strong signal

        SDE, SDE_raw, power = tls_stats.signal_detection_efficiency(
            chi2, detrend=False
        )

        assert SDE > 0, "SDE should be > 0 for injected signal"
        assert SDE_raw > 0

    def test_snr_with_chi2(self):
        """Test SNR estimation from chi2 values."""
        snr = tls_stats.signal_to_noise(
            0.01, chi2_null=1000.0, chi2_best=500.0
        )
        assert snr > 0

    def test_snr_returns_zero_without_info(self):
        """Test SNR returns 0 when no depth_err or chi2 provided."""
        snr = tls_stats.signal_to_noise(0.01)
        assert snr == 0.0


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
        min_idx = np.argmin(results['chi2'])
        best_period = results['periods'][min_idx]

        # Should be within 20% of true period (very loose for Phase 1)
        assert 8 < best_period < 12

    def test_sde_positive_with_transit(self):
        """Test SDE > 0 when a transit is present (regression test)."""
        from cuvarbase import tls

        # Create data with obvious transit
        t = np.linspace(0, 100, 500)
        y = np.ones(500)

        period_true = 10.0
        depth = 0.02
        phases = (t % period_true) / period_true
        in_transit = phases < 0.02
        y[in_transit] -= depth

        dy = np.ones(500) * 0.0001

        periods = np.linspace(8, 12, 50)
        results = tls.tls_search_gpu(t, y, dy, periods=periods)

        assert results['SDE'] > 0, (
            "SDE should be > 0 for a clear transit signal"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestSharedMemoryGuard:
    """tls_search_gpu must fail loudly (before touching the GPU) when
    the shared-memory layout exceeds the 48 KB per-block budget."""

    def test_large_ndata_raises_value_error(self):
        from cuvarbase.tls import tls_search_gpu
        rand = np.random.RandomState(3)
        ndata = 20000  # TESS-like; needs ~245 KB of shared memory
        t = np.sort(27 * rand.rand(ndata))
        y = 1 + 0.001 * rand.randn(ndata)
        dy = 0.001 * np.ones(ndata)
        with pytest.raises(ValueError, match="shared memory"):
            tls_search_gpu(t, y, dy, periods=np.array([1.0, 2.0]))

    def test_guard_accounts_for_template_size(self):
        from cuvarbase.tls import tls_search_gpu
        rand = np.random.RandomState(3)
        # ndata below the default cap, but a huge template pushes the
        # layout over the budget
        ndata = 3000
        t = np.sort(27 * rand.rand(ndata))
        y = 1 + 0.001 * rand.randn(ndata)
        dy = 0.001 * np.ones(ndata)
        with pytest.raises(ValueError, match="shared memory"):
            tls_search_gpu(t, y, dy, periods=np.array([1.0, 2.0]),
                           n_template=4000)


class TestFailedPeriodMasking:
    """chi2 == 1e30 sentinels (failed periods) must be masked before
    computing argmin/SDE/FAP."""

    def _chi2_with_dip(self, nperiods=200, dip_idx=100):
        chi2 = np.full(nperiods, 1000.0) + np.random.RandomState(5).randn(nperiods)
        chi2[dip_idx] = 900.0  # clear transit signal
        return chi2

    def test_mask_warns_and_excludes_sentinels(self):
        from cuvarbase.tls import _mask_failed_periods, TLS_CHI2_SENTINEL
        chi2 = self._chi2_with_dip()
        chi2[[3, 50, 150]] = TLS_CHI2_SENTINEL
        with pytest.warns(UserWarning, match="3 of 200"):
            valid = _mask_failed_periods(chi2)
        assert valid.sum() == 197
        assert not valid[3] and not valid[50] and not valid[150]

    def test_all_failed_raises(self):
        from cuvarbase.tls import _mask_failed_periods, TLS_CHI2_SENTINEL
        chi2 = np.full(20, TLS_CHI2_SENTINEL)
        with pytest.raises(RuntimeError, match="no valid solution"):
            _mask_failed_periods(chi2)

    def test_no_failures_no_warning(self):
        import warnings as _warnings
        from cuvarbase.tls import _mask_failed_periods
        chi2 = self._chi2_with_dip()
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            valid = _mask_failed_periods(chi2)
        assert valid.all()

    def test_sde_survives_sentinels_when_masked(self):
        # The audit reproduced SDE collapsing 15.3 -> 0.06 when 1e30
        # sentinels entered the statistics; masking must prevent that.
        from cuvarbase.tls import _mask_failed_periods, TLS_CHI2_SENTINEL
        chi2 = self._chi2_with_dip()
        sde_clean, _, _ = tls_stats.signal_detection_efficiency(chi2)

        chi2_corrupt = chi2.copy()
        chi2_corrupt[::7] = TLS_CHI2_SENTINEL  # 29 failed periods
        sde_corrupt, _, _ = tls_stats.signal_detection_efficiency(
            chi2_corrupt)

        with pytest.warns(UserWarning):
            valid = _mask_failed_periods(chi2_corrupt)
        sde_masked, _, _ = tls_stats.signal_detection_efficiency(
            chi2_corrupt[valid])

        assert sde_corrupt < 0.5 * sde_clean  # corruption is real
        assert sde_masked > 0.8 * sde_clean   # masking restores it


class TestSnrNotInflated:
    """signal_to_noise must not multiply by sqrt(n_transits): the
    chi2-based depth_err already includes every in-transit point."""

    def test_n_transits_does_not_inflate(self):
        snr1 = tls_stats.signal_to_noise(
            0.01, chi2_null=200.0, chi2_best=100.0, n_transits=1)
        snr9 = tls_stats.signal_to_noise(
            0.01, chi2_null=200.0, chi2_best=100.0, n_transits=9)
        assert snr1 == pytest.approx(np.sqrt(100.0))
        assert snr9 == pytest.approx(snr1)

    def test_explicit_depth_err(self):
        snr = tls_stats.signal_to_noise(0.01, depth_err=0.002,
                                        n_transits=16)
        assert snr == pytest.approx(5.0)


class TestTemplateFallbackWarns:
    """generate_transit_template must warn (not silently degrade) when
    batman fails at call time."""

    def test_batman_exception_warns(self, monkeypatch):
        monkeypatch.setattr(tls_models, 'BATMAN_AVAILABLE', True)

        def _boom(**kwargs):
            raise RuntimeError("batman exploded")

        monkeypatch.setattr(tls_models, 'create_reference_transit',
                            _boom)
        with pytest.warns(UserWarning, match="trapezoid"):
            template = tls_models.generate_transit_template(
                n_template=100)
        assert len(template) == 100
        assert template.max() == pytest.approx(1.0)


class TestT0GridDurationScaled:
    """The epoch (t0) grid must scale with transit duration: the old
    fixed 30-point grid missed transits narrower than 1/30 of the
    period (audit: 8/8 injected epochs missed at P=100 d)."""

    def test_grid_size_scales_with_duration(self):
        assert tls_grids.t0_grid_size(0.2) == 30      # wide: floor
        assert tls_grids.t0_grid_size(0.01) == 300
        assert tls_grids.t0_grid_size(0.001) == 3000
        assert tls_grids.t0_grid_size(1e-6) == 20000  # capped

    def test_coverage_guarantee(self):
        # Every possible transit epoch must lie within half a transit
        # duration of a tested t0 (with margin: stride <= q/3).
        rand = np.random.RandomState(11)
        for q in (0.05, 0.008, 0.003):
            n = tls_grids.t0_grid_size(q)
            grid = np.arange(n) / n
            epochs = rand.rand(500)
            # circular distance to the nearest tested t0
            dist = np.abs((epochs[:, None] - grid[None, :] + 0.5) % 1.0
                          - 0.5).min(axis=1)
            assert dist.max() <= 0.5 / n + 1e-12
            assert 1.0 / n <= q / 3 + 1e-12

    def test_kernel_source_uses_duration_scaled_grid(self):
        # Both CUDA kernels must derive n_t0 from the duration; the
        # GPU-side recovery test runs in the pod batch.
        from cuvarbase.utils import find_kernel
        src = open(find_kernel('tls')).read()
        assert 'int n_t0 = 30;' not in src
        assert src.count('t0_grid_size(duration_phase)') == 2
        assert 'T0_OVERSAMPLE' in src


class TestNoBitonicSort:
    """The bitonic sort was provably incomplete for non-power-of-2
    sizes AND its output order was never consumed (the depth/chi2
    accumulations are order-independent) — pure wasted GPU work with
    misleading naming. It must stay removed."""

    def test_kernel_has_no_sort(self):
        from cuvarbase.utils import find_kernel
        src = open(find_kernel('tls')).read()
        assert 'bitonic_sort_phases' not in src
        assert 'y_sorted' not in src
