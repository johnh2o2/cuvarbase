"""
Basic tests for TLS GPU implementation.

These tests verify the basic functionality of the TLS implementation,
focusing on API correctness and basic execution rather than scientific
accuracy (the golden-reference comparisons against transitleastsquares
and batman live in test_tls_golden.py; the fast-path behaviour in
test_tls_fast.py).
"""

import pytest
import numpy as np

try:
    # NOT pycuda.autoinit: it creates its own (non-primary) CUDA context
    # at import time, while cuvarbase lazily retains the PRIMARY context
    # (cuvarbase.base.ensure_context). pytest imports every test module
    # during collection, so the stray autoinit context outlived this file
    # and left two contexts on the stack for the whole session; pycuda's
    # context-dependent kernel cache then handed out handles from the
    # wrong one and unrelated tests died with
    # "cuFuncSetBlockShape failed: invalid resource handle"
    # (23 failures in test_nfft.py / the cuFINUFFT tests, Sep 2026).
    import pycuda.driver  # noqa: F401
    PYCUDA_AVAILABLE = True
except Exception:
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


class TestSDEKernelSize:
    """SDE median-detrend kernel selection (fast-TLS survey rework):
    auto kernel = min(len//10 forced odd, 91), even values round up,
    and short series skip detrending like the reference package."""

    @staticmethod
    def _trended_chi2(n, seed=0):
        # slow trend + one sharp dip so different medfilt windows give
        # measurably different detrended spectra
        rng = np.random.RandomState(seed)
        chi2 = 1000.0 - 30.0 * np.sin(np.linspace(0, 3, n)) \
            + rng.normal(0, 1.0, n)
        chi2[int(0.7 * n)] -= 200.0
        return chi2

    def test_auto_kernel_capped_at_91(self):
        chi2 = self._trended_chi2(5000)
        auto = tls_stats.signal_detection_efficiency(chi2, detrend=True)
        capped = tls_stats.signal_detection_efficiency(
            chi2, detrend=True, kernel_size=91)
        uncapped = tls_stats.signal_detection_efficiency(
            chi2, detrend=True, kernel_size=501)
        assert auto[0] == capped[0]
        np.testing.assert_array_equal(auto[2], capped[2])
        assert auto[0] != uncapped[0]

    def test_small_grids_keep_length_scaled_kernel(self):
        chi2 = self._trended_chi2(400)  # len//10 = 40 -> odd 41 < 91
        auto = tls_stats.signal_detection_efficiency(chi2, detrend=True)
        k41 = tls_stats.signal_detection_efficiency(
            chi2, detrend=True, kernel_size=41)
        assert auto[0] == k41[0]

    def test_even_kernel_rounds_up_to_odd(self):
        chi2 = self._trended_chi2(2000)
        k10 = tls_stats.signal_detection_efficiency(
            chi2, detrend=True, kernel_size=10)
        k11 = tls_stats.signal_detection_efficiency(
            chi2, detrend=True, kernel_size=11)
        assert k10[0] == k11[0]
        np.testing.assert_array_equal(k10[2], k11[2])

    def test_short_series_skips_detrending(self):
        chi2 = self._trended_chi2(100)
        # len(SR) <= 2 * kernel_size -> reference behavior: raw SDE
        sde, sde_raw, power = tls_stats.signal_detection_efficiency(
            chi2, detrend=True, kernel_size=51)
        assert sde == sde_raw
        np.testing.assert_array_equal(
            power, tls_stats.signal_residue(chi2))


class TestBatchPreprocessValidation:
    """CPU-side validation in the fast path's batch preprocessing."""

    class _FakeArr(object):
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

    def test_int32_point_count_guard(self):
        from cuvarbase import tls
        fake = self._FakeArr(2 ** 31)
        with pytest.raises(ValueError, match="int32"):
            tls._preprocess_batch([(fake, fake, fake)])

    def test_durations_param_removed(self):
        # The never-released ``durations=`` no-op was removed in the
        # Sep-2026 API freeze: it is rejected by the signature (keyword-
        # only parameters after ``periods``) before any validation or
        # GPU work.
        from cuvarbase import tls
        t = np.linspace(0, 10, 100)
        y = np.ones(100)
        dy = np.full(100, 1e-3)
        with pytest.raises(TypeError, match="durations"):
            tls.tls_search_gpu(t, y, dy, periods=np.array([1.0]),
                               durations=np.array([0.1]))


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

    def test_memory_setdata_subtracts_epoch(self):
        """Defect 11 (tls-T0): the legacy kernel folds relative to
        floor(min t), subtracted in float64 BEFORE the float32 cast, so
        BJD-scale times keep their phase and 't0_phase' means the same
        thing on both paths."""
        from cuvarbase.tls import TLSMemory

        t = 2457000.3 + np.linspace(0, 100, 100)
        y = np.ones(100)
        dy = np.ones(100) * 0.01

        mem = TLSMemory(max_ndata=1000, max_nperiods=100)
        mem.setdata(t, y, dy, periods=np.linspace(1, 10, 50),
                    transfer=False)

        assert mem.epoch == 2457000.0
        np.testing.assert_allclose(mem.t[:100], t - 2457000.0,
                                   rtol=0, atol=1e-5)


@pytest.mark.skipif(not PYCUDA_AVAILABLE,
                   reason="PyCUDA not available")
class TestTLSBasicExecution:
    """Test basic TLS execution (not accuracy)."""

    def test_tls_search_runs(self):
        """Test that TLS search runs without errors."""
        from cuvarbase import tls

        # Create simple synthetic data. Note: y needs (tiny) noise —
        # a perfectly flat lightcurve gives depth == 0 for every
        # (t0, duration) candidate, so no trial period records a
        # solution and tls_search_gpu raises RuntimeError (all
        # periods masked as failed).
        rand = np.random.RandomState(99)
        t = np.linspace(0, 100, 500)
        y = np.ones(500) + 0.001 * rand.randn(500)
        dy = np.ones(500) * 0.001

        # Use small period range for speed
        periods = np.linspace(5, 15, 20)

        # This should run without errors
        results = tls.tls_search_gpu(
            t, y, dy,
            periods=periods,
            block_size=64,
            method='binned',
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

        results = tls.tls_search_gpu(t, y, dy, periods=periods, method='binned')

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
        results = tls.tls_search_gpu(t, y, dy, periods=periods, method='binned')

        assert results['SDE'] > 0, (
            "SDE should be > 0 for a clear transit signal"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestSharedMemoryGuard:
    """The legacy kernel (method='legacy') must fail loudly (before
    touching the GPU) when its shared-memory layout exceeds the 48 KB
    per-block budget. The binned engine has no such cap."""

    def test_large_ndata_raises_value_error(self):
        from cuvarbase.tls import tls_search_gpu
        rand = np.random.RandomState(3)
        ndata = 20000  # TESS-like; needs ~245 KB of shared memory
        t = np.sort(27 * rand.rand(ndata))
        y = 1 + 0.001 * rand.randn(ndata)
        dy = 0.001 * np.ones(ndata)
        with pytest.raises(ValueError, match="shared memory"):
            tls_search_gpu(t, y, dy, periods=np.array([1.0, 2.0]),
                           method='legacy')

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
                           n_template=4000, method='legacy')

    def test_fast_path_has_no_ndata_cap(self):
        # regression for the removed cap: the binned path must
        # accept TESS-length lightcurves outright
        from cuvarbase.tls import tls_search_gpu
        rand = np.random.RandomState(3)
        ndata = 20000
        t = np.sort(27 * rand.rand(ndata))
        y = 1 + 0.001 * rand.randn(ndata)
        dy = 0.001 * np.ones(ndata)
        results = tls_search_gpu(t, y, dy,
                                 periods=np.linspace(2.0, 5.0, 50), method='binned')
        assert np.isfinite(results['chi2_min'])


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

    def test_all_failed_warns_and_masks_everything(self):
        # id 89: a flat/noiseless light curve fails every trial period;
        # since 1.0 that is a warning + all-False mask (the wrappers then
        # return SDE = 0), not a RuntimeError
        from cuvarbase.tls import _mask_failed_periods, TLS_CHI2_SENTINEL
        chi2 = np.full(20, TLS_CHI2_SENTINEL)
        with pytest.warns(UserWarning, match="no valid solution"):
            valid = _mask_failed_periods(chi2)
        assert valid.dtype == bool and valid.shape == (20,)
        assert not valid.any()

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
    """signal_to_noise is the chi2-based delta-chi-squared significance:
    the depth_err already includes every in-transit point, so there is
    no per-transit inflation (the pre-1.0 ``n_transits`` factor, which
    multiplied by ``sqrt(n_transits)``, was removed in the Sep-2026 API
    freeze together with the ignored parameter)."""

    def test_chi2_based_value(self):
        snr = tls_stats.signal_to_noise(
            0.01, chi2_null=200.0, chi2_best=100.0)
        assert snr == pytest.approx(np.sqrt(100.0))

    def test_n_transits_parameter_is_gone(self):
        with pytest.raises(TypeError, match="n_transits"):
            tls_stats.signal_to_noise(
                0.01, chi2_null=200.0, chi2_best=100.0, n_transits=9)

    def test_explicit_depth_err(self):
        snr = tls_stats.signal_to_noise(0.01, depth_err=0.002)
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


@pytest.mark.skipif(not PYCUDA_AVAILABLE,
                    reason="PyCUDA not available")
class TestTLSStreamParity:
    """TLSMemory.transfer_from_gpu enqueues async copies into
    page-locked buffers; tls_search_gpu used to synchronize BEFORE
    enqueueing them and then read the host arrays immediately, so
    stream runs could return stale/zero chi2. Results on a user
    stream must match the default-stream results."""

    def test_stream_matches_default(self):
        import pycuda.driver as cuda
        from cuvarbase import tls
        from cuvarbase.base import ensure_context

        rand = np.random.RandomState(7)
        t = np.linspace(0, 100, 400)
        y = np.ones(400) + 0.001 * rand.randn(400)
        dy = np.ones(400) * 0.001
        periods = np.linspace(5, 15, 10)

        # method='legacy' on both sides: this is a regression test for
        # the LEGACY kernel's async D2H sequencing (the fast path does
        # not take a user stream and would silently fall back to the
        # legacy kernel anyway when one is passed)
        r_default = tls.tls_search_gpu(t, y, dy, periods=periods,
                                       block_size=64, method='legacy')
        ensure_context()
        r_stream = tls.tls_search_gpu(t, y, dy, periods=periods,
                                      block_size=64,
                                      stream=cuda.Stream(), method='legacy')
        np.testing.assert_allclose(r_stream['chi2'], r_default['chi2'],
                                   rtol=1e-3)


# ---------------------------------------------------------------------
# 1.0 correctness fixes (Sep 2026 audit): CPU-runnable regression tests
# ---------------------------------------------------------------------

import warnings as _w
import inspect as _inspect


class TestDefaultDurationWindow:
    """Preserved binned engine duration window (audit id 9): tls_search_gpu /
    tls_search without qmin/qmax used a constant q window [0.005, 0.15]
    at every period while the default Ofir grid runs to span/2; beyond
    P ~ 60 d (Sun-like) no trial duration was physical and a P = 365 d
    transit on a 1400-d baseline came back at 182.5 d with half the
    depth. The default window is now the Keplerian one that
    tls_search_batch/tls_transit always used; the constant window is an
    opt-in that warns."""

    def test_keplerian_window_equals_q_transit_window(self):
        periods = np.array([0.5, 1.0, 10.0, 100.0, 365.0, 700.0])
        for R, M, Rp in ((1.0, 1.0, 1.0), (0.3, 0.3, 2.0), (1.5, 1.2, 1.0)):
            qmin, qmax = tls_grids.duration_window(
                periods, R_star=R, M_star=M, R_planet=Rp)
            q = tls_grids.q_transit(periods, R, M, Rp)
            np.testing.assert_allclose(qmin, 0.5 * q, rtol=1e-12)
            np.testing.assert_allclose(qmax, 2.0 * q, rtol=1e-12)
            # physical at every period, inside the kernels' (0, 1) bounds
            assert np.all(qmin < q) and np.all(q < qmax)
            assert np.all(qmin > 0) and np.all(qmax < 1)

    def test_window_factors_honoured(self):
        periods = np.array([3.0, 30.0])
        qmin, qmax = tls_grids.duration_window(periods, qmin_fac=0.25,
                                               qmax_fac=4.0)
        q = tls_grids.q_transit(periods)
        np.testing.assert_allclose(qmin, 0.25 * q)
        np.testing.assert_allclose(qmax, 4.0 * q)

    def test_fixed_window_crossover_near_60d(self):
        # the documented crossover: q_kep(Sun, 1 R_earth) drops below the
        # old constant qmin = 0.005 between P = 50 and 70 d
        assert tls_grids.q_transit(50.0) > tls_grids.FIXED_QMIN
        assert tls_grids.q_transit(70.0) < tls_grids.FIXED_QMIN
        assert tls_grids.q_transit(365.0) < 0.5 * tls_grids.FIXED_QMIN

    def test_fixed_window_warns_when_unphysical(self):
        periods = np.array([1.0, 10.0, 120.0, 365.0])
        with pytest.warns(UserWarning, match="excludes the Keplerian"):
            qmin, qmax = tls_grids.duration_window(periods, window='fixed')
        assert np.all(qmin == tls_grids.FIXED_QMIN)
        assert np.all(qmax == tls_grids.FIXED_QMAX)

    def test_fixed_window_silent_when_physical(self):
        periods = np.array([1.0, 3.0, 10.0, 30.0])
        with _w.catch_warnings():
            _w.simplefilter("error")
            qmin, qmax = tls_grids.duration_window(periods, window='fixed')
        assert np.all(qmin == 0.005) and np.all(qmax == 0.15)

    def test_unknown_window_rejected(self):
        with pytest.raises(ValueError, match="window"):
            tls_grids.duration_window(np.array([1.0]), window='boxy')

    @staticmethod
    def _capture_batch(monkeypatch):
        """Intercept the fast path's tls_search_batch call (no GPU)."""
        from cuvarbase import tls
        captured = {}

        def fake_batch(lightcurves, **kw):
            captured.update(kw)
            n = len(kw['periods'])
            return [tls._null_result(n, 1.0, 'intercepted',
                                     periods=kw['periods'], arrays=True)]

        monkeypatch.setattr(tls, '_tls_search_batch_binned', fake_batch)
        return captured

    def test_search_gpu_default_passes_keplerian_window(self, monkeypatch):
        from cuvarbase import tls
        captured = self._capture_batch(monkeypatch)
        t = np.linspace(0, 1400, 2000)
        y = np.ones(2000)
        dy = np.full(2000, 3e-4)
        periods = np.array([10.0, 100.0, 365.0])
        r = tls.tls_search_gpu(t, y, dy, periods=periods,
                               R_star=0.8, M_star=0.9, method='binned')
        q = tls_grids.q_transit(periods, 0.8, 0.9, 1.0)
        np.testing.assert_allclose(captured['qmin'], 0.5 * q, rtol=1e-6)
        np.testing.assert_allclose(captured['qmax'], 2.0 * q, rtol=1e-6)
        assert captured['n_durations'] == 15
        assert 'FAP' not in r

    def test_search_gpu_window_kwargs_reach_the_batch(self, monkeypatch):
        from cuvarbase import tls
        captured = self._capture_batch(monkeypatch)
        t = np.linspace(0, 100, 500)
        y = np.ones(500)
        dy = np.full(500, 1e-3)
        periods = np.array([3.0, 30.0])
        tls.tls_search_gpu(t, y, dy, periods=periods, R_planet=3.0,
                           qmin_fac=0.3, qmax_fac=3.0, n_durations=7, method='binned')
        q = tls_grids.q_transit(periods, 1.0, 1.0, 3.0)
        np.testing.assert_allclose(captured['qmin'], 0.3 * q, rtol=1e-6)
        np.testing.assert_allclose(captured['qmax'], 3.0 * q, rtol=1e-6)
        assert captured['n_durations'] == 7

    def test_search_gpu_fixed_window_optin_warns(self, monkeypatch):
        from cuvarbase import tls
        captured = self._capture_batch(monkeypatch)
        t = np.linspace(0, 1400, 2000)
        y = np.ones(2000)
        dy = np.full(2000, 3e-4)
        with pytest.warns(UserWarning, match="excludes the Keplerian"):
            tls.tls_search_gpu(t, y, dy, periods=np.array([10.0, 365.0]),
                               duration_window='fixed', method='binned')
        assert np.all(captured['qmin'] == 0.005)
        assert np.all(captured['qmax'] == 0.15)

    def test_search_gpu_explicit_q_conflicts_with_window(self):
        from cuvarbase import tls
        t = np.linspace(0, 100, 500)
        y = np.ones(500)
        dy = np.full(500, 1e-3)
        periods = np.array([3.0, 30.0])
        with pytest.raises(ValueError, match="duration_window"):
            tls.tls_search_gpu(t, y, dy, periods=periods,
                               qmin=np.full(2, 0.01), qmax=np.full(2, 0.05),
                               duration_window='fixed')
        with pytest.raises(ValueError, match="both qmin and qmax"):
            tls.tls_search_gpu(t, y, dy, periods=periods,
                               qmin=np.full(2, 0.01))
        with pytest.raises(ValueError, match="aligned with periods"):
            tls.tls_search_gpu(t, y, dy, periods=periods,
                               qmin=np.full(3, 0.01), qmax=np.full(3, 0.05))
        with pytest.raises(ValueError, match="0 < qmin <= qmax < 1"):
            tls.tls_search_gpu(t, y, dy, periods=periods,
                               qmin=np.full(2, 0.05), qmax=np.full(2, 0.01))

    def test_tls_cu_standard_kernel_is_marked_retired(self):
        # the legacy path must not launch the kernel with the hard-coded
        # [0.005, 0.15] window
        from cuvarbase.utils import find_kernel
        src = open(find_kernel('tls')).read()
        assert 'RETAINED FOR API COMPATIBILITY ONLY' in src
        from cuvarbase import tls
        body = _inspect.getsource(tls._tls_search_gpu_binned)
        assert "kernels['standard']" not in body
        assert "kernels['keplerian']" in body


class TestTransitDurationWindowBounds:
    """Phase 2 TLS-1 (audit section 5, id 52): tls_transit built the
    whole (nperiods x n_durations) Keplerian duration table with
    duration_grid_keplerian and then threw it away -- only the q_values
    it also returns were used. It now calls tls_grids.duration_window,
    the shared window helper the other entry points use, which returns
    exactly the same bounds. Bit-neutral: these tests pin the bounds
    handed to tls_search_gpu to the legacy expression, bitwise."""

    @staticmethod
    def _capture_search(monkeypatch):
        """Intercept tls_transit's tls_search_gpu call (no GPU)."""
        from cuvarbase import tls
        captured = {}

        def fake_search(t, y, dy, **kw):
            captured.update(kw)
            n = len(kw['periods'])
            return tls._null_result(n, 1.0, 'intercepted',
                                    periods=kw['periods'], arrays=True)

        monkeypatch.setattr(tls, '_tls_search_gpu_binned', fake_search)
        return captured

    PARAMS = [dict(), dict(R_star=0.7, M_star=0.65, R_planet=2.3,
                           qmin_fac=0.4, qmax_fac=2.5, n_durations=9),
              dict(R_star=2.2, M_star=1.9, R_planet=11.0,
                   qmin_fac=0.25, qmax_fac=3.0),
              dict(R_star=0.3, M_star=0.3)]

    def test_bounds_bitwise_match_duration_grid_keplerian(self, monkeypatch):
        from cuvarbase import tls
        t = np.linspace(0, 90.0, 1200)
        y = np.ones(1200)
        dy = np.full(1200, 1e-3)
        for kw in self.PARAMS:
            captured = self._capture_search(monkeypatch)
            tls._tls_transit_binned(t, y, dy, period_min=0.5, period_max=30.0, **kw)
            periods = captured['periods']
            # the pre-1.0 expression, verbatim
            _, _, q_values = tls_grids.duration_grid_keplerian(
                periods, R_star=kw.get('R_star', 1.0),
                M_star=kw.get('M_star', 1.0),
                R_planet=kw.get('R_planet', 1.0),
                qmin_fac=kw.get('qmin_fac', 0.5),
                qmax_fac=kw.get('qmax_fac', 2.0),
                n_durations=kw.get('n_durations', 15))
            assert len(periods) > 100
            assert np.array_equal(captured['qmin'],
                                  q_values * kw.get('qmin_fac', 0.5))
            assert np.array_equal(captured['qmax'],
                                  q_values * kw.get('qmax_fac', 2.0))
            assert captured['n_durations'] == kw.get('n_durations', 15)

    def test_duration_table_is_not_built(self, monkeypatch):
        """The (nperiods x n_durations) table nothing reads: 59 ms of a
        237 ms Kepler-4yr call (A40, shared)."""
        from cuvarbase import tls
        self._capture_search(monkeypatch)
        calls = []
        real = tls_grids.duration_grid_keplerian

        def counting(*a, **kw):
            calls.append(1)
            return real(*a, **kw)

        monkeypatch.setattr(tls_grids, 'duration_grid_keplerian', counting)
        t = np.linspace(0, 90.0, 1200)
        tls._tls_transit_binned(t, np.ones(1200), np.full(1200, 1e-3),
                        period_min=0.5, period_max=30.0)
        assert calls == []

    def test_bounds_match_the_other_entry_points(self, monkeypatch):
        """tls_transit and tls_search_gpu must agree on the window."""
        from cuvarbase import tls
        captured = self._capture_search(monkeypatch)
        t = np.linspace(0, 90.0, 1200)
        tls._tls_transit_binned(t, np.ones(1200), np.full(1200, 1e-3),
                        R_star=0.8, M_star=0.9, period_min=0.5,
                        period_max=30.0)
        qmin, qmax = tls_grids.duration_window(
            captured['periods'], R_star=0.8, M_star=0.9)
        assert np.array_equal(captured['qmin'], qmin)
        assert np.array_equal(captured['qmax'], qmax)


class TestTemplateTableMemoization:
    """Phase 2 TLS-3 (audit section 5, ids 95/152): every
    single-lightcurve search rebuilt the batman reference model behind
    the fast kernel's template tables. generate_template_tables now
    memoizes on (n_table, limb_dark, u, oversample); it still returns
    fresh, writable arrays, and a degraded (trapezoid-fallback) result
    is never cached so its warning keeps firing."""

    def setup_method(self):
        tls_models._clear_template_table_cache()

    teardown_method = setup_method

    def test_repeat_call_is_bitwise_identical(self):
        first = tls_models.generate_template_tables(n_table=128)
        second = tls_models.generate_template_tables(n_table=128)
        for a, b in zip(first, second):
            assert np.array_equal(a, b)
            assert a.dtype == np.float32

    def test_returns_independent_arrays(self):
        """A caller that writes to the tables must not poison the cache."""
        first = tls_models.generate_template_tables(n_table=128)
        for a in first:
            a[:] = -12345.0
        second = tls_models.generate_template_tables(n_table=128)
        assert all(a is not b for a, b in zip(first, second))
        assert not np.any(second[0] == -12345.0)
        third = tls_models.generate_template_tables(n_table=128)
        for b, c in zip(second, third):
            assert np.array_equal(b, c)

    def test_underlying_model_is_built_once_per_key(self):
        calls = []
        real = tls_models.generate_transit_template

        def counting(**kw):
            calls.append(kw.get('n_template'))
            return real(**kw)

        try:
            tls_models.generate_transit_template = counting
            tls_models.generate_template_tables(n_table=128)
            tls_models.generate_template_tables(n_table=128)
            tls_models.generate_template_tables(n_table=128)
            assert len(calls) == 1
        finally:
            tls_models.generate_transit_template = real

    def test_cache_does_not_leak_across_parameters(self):
        base = tls_models.generate_template_tables(n_table=128)
        variants = [
            dict(n_table=128, limb_dark='linear', u=[0.5]),
            dict(n_table=128, u=[0.1, 0.05]),
            dict(n_table=128, oversample=4),
            dict(n_table=256),
        ]
        for kw in variants:
            got = tls_models.generate_template_tables(**kw)
            assert len(got[0]) == kw.get('n_table', 128) + 1
            if len(got[0]) == len(base[0]):
                if tls_models.BATMAN_AVAILABLE or 'oversample' in kw:
                    assert not np.array_equal(got[0], base[0]) or \
                        not np.array_equal(got[1], base[1])
        # the original key still returns the original tables
        again = tls_models.generate_template_tables(n_table=128)
        for a, b in zip(base, again):
            assert np.array_equal(a, b)

    def test_cache_is_bounded(self):
        for i in range(2 * tls_models._TEMPLATE_TABLE_CACHE_MAX + 3):
            tls_models.generate_template_tables(n_table=32 + i)
        assert (len(tls_models._template_table_cache)
                <= tls_models._TEMPLATE_TABLE_CACHE_MAX)

    def test_fallback_result_is_not_cached(self, monkeypatch):
        """The trapezoid fallback warns on every call, so it must not be
        memoized away."""
        monkeypatch.setattr(tls_models, 'BATMAN_AVAILABLE', True)

        def _boom(**kwargs):
            raise RuntimeError("batman exploded")

        monkeypatch.setattr(tls_models, 'create_reference_transit', _boom)
        for _ in range(2):
            with pytest.warns(UserWarning, match="trapezoid"):
                tls_models.generate_template_tables(n_table=128)
        assert tls_models._template_table_cache == {}

    def test_missing_batman_tables_are_cached_under_their_own_key(
            self, monkeypatch):
        """Release review (findings 0/12): with batman not installed
        the trapezoid IS the template -- deterministic, warned about
        once at import rather than per call -- so its tables are
        memoized like any other, under a key that records batman's
        absence so a batman-backed table can never collide with it."""
        monkeypatch.setattr(tls_models, 'BATMAN_AVAILABLE', False)
        with _w.catch_warnings():
            _w.simplefilter("error")
            first = tls_models.generate_template_tables(n_table=128)
            second = tls_models.generate_template_tables(n_table=128)
        for a, b in zip(first, second):
            assert np.array_equal(a, b)
        keys = list(tls_models._template_table_cache)
        assert len(keys) == 1 and keys[0][-1] is False
        assert keys[0] == tls_models._template_table_key(
            128, 'quadratic', [0.4804, 0.1867], 8)
        # the cached tables are the trapezoid's
        expect = tls_models._trapezoid_template(128 * 8 + 1)[::8]
        assert np.array_equal(first[0], expect.astype(np.float32))
        # a batman-backed table lives under a different key
        monkeypatch.setattr(tls_models, 'BATMAN_AVAILABLE', True)
        assert tls_models._template_table_key(
            128, 'quadratic', [0.4804, 0.1867], 8) not in \
            tls_models._template_table_cache


class TestBatchHasNoThreadPool:
    """Phase 2 TLS-2 (audit section 5, id 53): tls_search_batch must not
    reintroduce the per-light-curve ThreadPoolExecutor -- the work is
    GIL-bound numpy/scipy and the pool made it 1.2-2.2x slower (A40,
    shared) while randomizing the order of per-light-curve warnings."""

    def test_module_does_not_import_a_thread_pool(self):
        from cuvarbase import tls
        assert not hasattr(tls, 'ThreadPoolExecutor')
        body = _inspect.getsource(tls._tls_search_batch_binned)
        assert 'ThreadPoolExecutor(' not in body
        assert 'cpu_count' not in body


class TestReferenceSRDefinition:
    """ids 81/146: SR was 1 - chi2/max(chi2); the reference package uses
    chi2_min/chi2. Identical under the null but ~2x lower SDE for strong
    signals, so published thresholds did not transfer. 1.0 adopts the
    reference definition on every path."""

    @staticmethod
    def _ref_running_median(data, kernel):
        # literal transcription of transitleastsquares.stats.running_median
        idx = np.arange(kernel) + np.arange(len(data) - kernel + 1)[:, None]
        med = np.median(data[idx], axis=1)
        missing = len(data) - len(med)
        front = int(missing * 0.5)
        end = missing - front
        med = np.append(np.full(front, med[0]), med)
        med = np.append(med, np.full(end, med[-1]))
        return med

    @classmethod
    def _ref_spectra(cls, chi2, kernel):
        # literal transcription of transitleastsquares.stats.spectra
        SR = np.min(chi2) / chi2
        SDE_raw = (1 - np.mean(SR)) / np.std(SR)
        power_raw = SR - np.mean(SR)
        scale = SDE_raw / np.max(power_raw)
        power_raw = power_raw * scale
        if kernel % 2 == 0:
            kernel = kernel + 1
        if len(power_raw) > 2 * kernel:
            my_median = cls._ref_running_median(power_raw, kernel)
            power = power_raw - my_median
            power = power - np.mean(power)
            SDE = np.max(power / np.std(power))
        else:
            SDE = SDE_raw
        return SDE_raw, SDE

    @staticmethod
    def _spectrum(n, dip_frac, seed=0):
        rng = np.random.RandomState(seed)
        chi2 = 1000.0 - 30.0 * np.sin(np.linspace(0, 3, n)) \
            + rng.normal(0, 1.0, n)
        chi2[int(0.7 * n)] -= dip_frac * 1000.0
        return chi2

    def test_signal_residue_is_chi2min_over_chi2(self):
        chi2 = self._spectrum(500, 0.1)
        SR = tls_stats.signal_residue(chi2)
        np.testing.assert_allclose(SR, chi2.min() / chi2, rtol=1e-14)
        assert SR.max() == 1.0
        assert SR[int(0.7 * 500)] == 1.0

    @pytest.mark.parametrize("n,kernel", [(2000, 91), (500, 51), (5000, 91)])
    def test_sde_matches_reference_spectra(self, n, kernel):
        chi2 = self._spectrum(n, 0.1)
        SDE, SDE_raw, power = tls_stats.signal_detection_efficiency(
            chi2, kernel_size=kernel)
        ref_raw, ref = self._ref_spectra(chi2, kernel)
        assert SDE_raw == pytest.approx(ref_raw, rel=1e-10)
        assert SDE == pytest.approx(ref, rel=1e-10)

    def test_auto_kernel_matches_reference_at_91(self):
        # grids with >= 910 periods use the reference's 91-point kernel
        chi2 = self._spectrum(3000, 0.05)
        SDE = tls_stats.signal_detection_efficiency(chi2)[0]
        assert SDE == pytest.approx(self._ref_spectra(chi2, 91)[1],
                                    rel=1e-10)

    def test_strong_signal_no_longer_halved(self):
        # A dip that removes 80% of chi2 on a noisy background (audit:
        # score/chi2_0 = 0.79 gave SDE 21.8 old vs 45.4 reference). The
        # old SR = 1 - chi2/max(chi2) keeps the background noise at
        # sigma_chi2/chi2_bg while chi2_min/chi2 shrinks it by
        # chi2_min/chi2_bg, so the peak's z-score roughly doubles.
        rng = np.random.RandomState(0)
        n = 20000
        chi2 = 1000.0 + rng.normal(0, 10.0, n)
        chi2[int(0.7 * n)] = 200.0
        SDE_new = tls_stats.signal_detection_efficiency(chi2)[0]
        SR_old = 1.0 - chi2 / chi2.max()
        SDE_old = (SR_old.max() - SR_old.mean()) / SR_old.std()
        assert SDE_new > 1.7 * SDE_old
        # ...while a weak signal is essentially unchanged (both linear in
        # delta-chi2 when the dip is small relative to chi2)
        chi2w = 1000.0 + rng.normal(0, 10.0, n)
        chi2w[int(0.7 * n)] = 980.0
        SDE_new_w = tls_stats.signal_detection_efficiency(
            chi2w, detrend=False)[0]
        SR_old_w = 1.0 - chi2w / chi2w.max()
        SDE_old_w = (SR_old_w.max() - SR_old_w.mean()) / SR_old_w.std()
        assert SDE_new_w == pytest.approx(SDE_old_w, rel=0.05)

    def test_chi2_null_argument_deprecated(self):
        chi2 = self._spectrum(300, 0.1)
        with pytest.warns(DeprecationWarning, match="chi2_null"):
            SR = tls_stats.signal_residue(chi2, chi2_null=5000.0)
        np.testing.assert_allclose(SR, chi2.min() / chi2)

    def test_perfect_fit_and_flat_spectra(self):
        chi2 = np.array([10.0, 0.0, 5.0])   # noiseless perfect fit
        SR = tls_stats.signal_residue(chi2)
        assert np.all(np.isfinite(SR)) and SR[1] == 1.0 and SR[0] == 0.0
        sde, sde_raw, power = tls_stats.signal_detection_efficiency(
            np.full(50, 100.0))
        assert sde == 0.0 and sde_raw == 0.0

    def test_compute_all_statistics_uses_reference_sr(self):
        chi2 = self._spectrum(400, 0.1)
        stats = tls_stats.compute_all_statistics(
            chi2, np.arange(400.0) + 1, int(0.7 * 400), 0.01, 0.1, 5)
        np.testing.assert_allclose(stats['SR'], chi2.min() / chi2)
        assert stats['SDE'] == pytest.approx(
            tls_stats.signal_detection_efficiency(chi2)[0])


class TestRunningMedianEdges:
    """id 83: scipy.signal.medfilt zero-pads and drags the SR trend to
    zero over the outer kernel//2 points (edge power inflated; null
    peaks within 45 points of an edge 2.3x more often than uniform).
    The trend must match the reference's edge-extended running median."""

    def test_matches_reference_running_median_exactly(self):
        rng = np.random.RandomState(4)
        for n, kernel in ((300, 3), (300, 21), (300, 91), (300, 299),
                          (1000, 91), (7, 5)):
            x = rng.randn(n).cumsum()
            got = tls_stats.running_median(x, kernel)
            ref = TestReferenceSRDefinition._ref_running_median(x, kernel)
            np.testing.assert_array_equal(got, ref)

    def test_no_zero_padding_bias(self):
        # a ramp: the reference trend at the ends is the first/last
        # full-window median (x[45], x[-46]); zero-padded medfilt drags
        # the first value down to x[0]
        x = 100.0 + np.arange(200.0)
        trend = tls_stats.running_median(x, 91)
        assert trend[0] == x[45] and trend[44] == x[45]
        assert trend[-1] == x[-46] and trend[-45] == x[-46]
        np.testing.assert_array_equal(trend[45:-45], x[45:-45])
        from scipy import signal
        assert signal.medfilt(x, 91)[0] == x[0] < trend[0]
        # a constant series has a constant trend
        np.testing.assert_array_equal(
            tls_stats.running_median(np.full(200, 5.0), 91), 5.0)

    def test_detrended_power_flat_at_edges(self):
        # an SR spectrum that is pure trend + noise must not get raised
        # power at the ends
        rng = np.random.RandomState(1)
        chi2 = 1000.0 + 50.0 * np.linspace(0, 1, 2000) + rng.normal(0, 1, 2000)
        _, _, power = tls_stats.signal_detection_efficiency(chi2)
        edge = np.r_[power[:45], power[-45:]]
        interior = power[45:-45]
        assert abs(edge.mean() - interior.mean()) < 3 * interior.std() / np.sqrt(90)

    def test_even_kernel_rounds_up_and_too_long_raises(self):
        x = np.arange(20.0)
        np.testing.assert_array_equal(tls_stats.running_median(x, 4),
                                      tls_stats.running_median(x, 5))
        with pytest.raises(ValueError, match="kernel"):
            tls_stats.running_median(x, 21)
        np.testing.assert_array_equal(tls_stats.running_median(x, 1), x)


class TestFAPRemoved:
    """Defect 10 (tls-fap, audit id 10): the returned 'FAP' was a fixed
    piecewise function of the SDE (discontinuous at SDE = 7) unrelated
    to the null; 23% of pure-noise light curves got FAP < 0.01. No
    result dict carries a FAP unless a null bootstrap was requested."""

    def test_compute_all_statistics_has_no_fap_key(self):
        rng = np.random.RandomState(0)
        chi2 = 1000 + rng.randn(300)
        chi2[100] = 900
        stats = tls_stats.compute_all_statistics(
            chi2, np.arange(300.0) + 1, 100, 0.01, 0.1, 5)
        assert 'FAP' not in stats
        for k in ('SDE', 'SDE_raw', 'SNR', 'power', 'SR'):
            assert k in stats

    def test_null_result_has_no_fap(self):
        from cuvarbase import tls
        r = tls._null_result(10, 123.0, 'msg', periods=np.arange(10.0),
                             arrays=True)
        assert 'FAP' not in r
        assert r['SDE'] == 0.0 and r['SDE_raw'] == 0.0 and r['SNR'] == 0.0
        assert np.isnan(r['period']) and np.isnan(r['T0'])
        assert r['chi2_min'] == 123.0 and r['error'] == 'msg'
        assert r['n_failed_periods'] == 10
        assert not r['valid_periods'].any()
        assert np.all(np.isnan(r['chi2'])) and np.all(np.isnan(r['power']))

    def test_heuristic_helper_still_works_but_warns(self):
        with pytest.warns(UserWarning, match="uncalibrated"):
            assert tls_stats.false_alarm_probability(9.0) == pytest.approx(1e-4)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            # the documented discontinuity at SDE = 7
            assert tls_stats.false_alarm_probability(6.999) == pytest.approx(0.1, rel=2e-3)
            assert tls_stats.false_alarm_probability(7.0) == pytest.approx(0.01)
            assert tls_stats.false_alarm_probability(4.0) == 1.0
        with _w.catch_warnings():
            _w.simplefilter("error")
            g = tls_stats.false_alarm_probability(3.0, method='gaussian')
        assert 0 < g < 0.01

    def test_docs_do_not_claim_a_calibration(self):
        import os
        import cuvarbase
        doc = tls_stats.signal_detection_efficiency.__doc__
        assert '1% false alarm' not in doc
        assert 'SDE > 7 for' not in doc
        rst = os.path.join(os.path.dirname(cuvarbase.__file__), '..',
                           'docs', 'source', 'tls.rst')
        if os.path.exists(rst):
            txt = open(rst).read()
            assert 'preserves the false-alarm calibration' not in txt
            assert 'fap_null_draws' in txt
        # the wrong inline comment ("~10% at SDE=5, ~1% at SDE=7") is gone
        src = _inspect.getsource(tls_stats.false_alarm_probability)
        assert '~10% at SDE=5' not in src

    def test_fast_path_result_has_no_fap(self, monkeypatch):
        from cuvarbase import tls

        def fake_batch(lightcurves, **kw):
            n = len(kw['periods'])
            r = tls._null_result(n, 1.0, 'x', periods=kw['periods'],
                                 arrays=True)
            r['FAP'] = 0.5   # even if a batch result carried one...
            return [r]

        monkeypatch.setattr(tls, '_tls_search_batch_binned', fake_batch)
        t = np.linspace(0, 100, 500)
        r = tls.tls_search_gpu(t, np.ones(500), np.full(500, 1e-3),
                               periods=np.array([3.0, 4.0]), method='binned')
        assert 'FAP' not in r        # ...tls_search_gpu never forwards it
        assert 't0_phase' in r and 'T0' in r


class TestSortedPeriodGrid:
    """id 82: descending/shuffled user grids gave negative
    period_uncertainty and a changed SDE (running median and neighbour
    walk assume period order). Grids are sorted on entry and per-period
    outputs scattered back to the caller's order."""

    def test_sort_helper(self):
        from cuvarbase import tls
        asc = np.array([1.0, 2.0, 3.0])
        p, order = tls._sort_period_grid(asc)
        assert order is None and p is asc
        desc = asc[::-1].copy()
        p, order = tls._sort_period_grid(desc)
        np.testing.assert_array_equal(p, asc)
        np.testing.assert_array_equal(desc[order], p)
        vals = np.array([10.0, 20.0, 30.0])   # aligned with ascending p
        back = tls._to_caller_order(vals, order)
        # caller order is descending: caller[i] = value of desc[i]
        np.testing.assert_array_equal(back, [30.0, 20.0, 10.0])
        assert tls._to_caller_order(vals, None) is vals
        # bool arrays round-trip too
        flags = np.array([True, False, True])
        np.testing.assert_array_equal(tls._to_caller_order(flags, order),
                                      flags[::-1])

    def test_shuffled_round_trip(self):
        from cuvarbase import tls
        rng = np.random.RandomState(2)
        grid = rng.uniform(1, 10, 50)
        p, order = tls._sort_period_grid(grid)
        assert np.all(np.diff(p) >= 0)
        vals = p * 2
        np.testing.assert_array_equal(tls._to_caller_order(vals, order),
                                      grid * 2)

    def test_validate_periods(self):
        from cuvarbase import tls
        for bad in (np.array([]), np.array([1.0, np.nan]),
                    np.array([0.0, 1.0]), np.array([[1.0, 2.0]]),
                    np.array([-1.0, 2.0])):
            with pytest.raises(ValueError):
                tls._validate_periods(bad)
        np.testing.assert_array_equal(tls._validate_periods([3.0, 1.0]),
                                      [3.0, 1.0])

    def test_caller_staged_memory_requires_ascending_grid(self):
        # legacy path with transfer_to_device=False: the caller staged
        # the periods on the device, so a grid we would have to reorder
        # is refused before any GPU work
        from cuvarbase import tls
        t = np.linspace(0, 100, 500)
        with pytest.raises(ValueError, match="ascending"):
            tls.tls_search_gpu(t, np.ones(500), np.full(500, 1e-3),
                               periods=np.array([5.0, 3.0, 4.0]),
                               method='legacy', memory=object(),
                               transfer_to_device=False)

    def test_period_uncertainty_positive_on_sorted_input(self):
        rng = np.random.RandomState(0)
        periods = np.linspace(2, 4, 200)
        chi2 = 1000 + rng.randn(200)
        chi2[95:106] -= 50 * np.exp(-0.5 * ((np.arange(95, 106) - 100) / 2.0) ** 2)
        best = int(np.argmin(chi2))
        unc = tls_stats.compute_period_uncertainty(periods, chi2, best)
        assert unc > 0


class TestT0Convention:
    """Defect 11 (tls-T0, audit ids 11/145): 'T0' was a fold phase on the
    fast path (relative to floor(min t)), a phase relative to t = 0 on
    the legacy path, and an absolute time that could precede the first
    observation on the batch path. 1.0: 'T0' is the absolute time of the
    first mid-transit at or after min(t) everywhere, plus 't0_phase'."""

    def test_first_transit_at_or_after(self):
        from cuvarbase.tls import _first_transit_at_or_after as f
        P = 3.0
        # audit case: t_start = 100.9, epoch 100, phase 0.1 -> 100.3
        # precedes min(t); shift up one period
        assert f(100.0 + 0.1009 * P, P, 100.9) == pytest.approx(100.3027 + P)
        # already inside [tmin, tmin + P): unchanged
        assert f(101.41, P, 100.3) == pytest.approx(101.41)
        # many periods early or late: wrapped into range
        assert f(101.41 - 5 * P, P, 100.3) == pytest.approx(101.41)
        assert f(101.41 + 7 * P, P, 100.3) == pytest.approx(101.41)
        # boundary: exactly tmin stays tmin
        assert f(100.3, P, 100.3) == 100.3
        # NaN propagates (null results)
        assert np.isnan(f(np.nan, P, 100.3))

    def test_docstrings_state_the_convention(self):
        from cuvarbase import tls
        for fn in (tls.tls_search_gpu, tls.tls_search_batch, tls.tls_transit):
            assert 'at or after' in fn.__doc__, fn.__name__
            assert 't0_phase' in fn.__doc__, fn.__name__


class TestTlsSearchDispatch:
    """``tls.tls_search`` is a thin, validated forward to
    ``tls_search_gpu`` (release finding 71/149: the documented "main
    user-facing function" had no test). CPU: the GPU function is
    replaced by a recorder."""

    def test_forwards_all_kwargs_to_tls_search_gpu(self, monkeypatch):
        from cuvarbase import tls
        seen = {}
        sentinel = object()

        def fake_search_gpu(t, y, dy, **kwargs):
            seen['args'] = (t, y, dy)
            seen['kwargs'] = kwargs
            return sentinel

        monkeypatch.setattr(tls, 'tls_search_gpu', fake_search_gpu)
        t = np.linspace(0, 30.0, 400)
        y = np.ones(400)
        dy = np.full(400, 1e-3)
        periods = np.linspace(2.0, 5.0, 50)
        out = tls.tls_search(t, y, dy, periods=periods, n_durations=7,
                             use_fast=False, refine_top_k=3, R_star=0.8)
        assert out is sentinel
        assert seen['args'][0] is t and seen['args'][1] is y
        assert seen['args'][2] is dy
        assert seen['kwargs'] == dict(periods=periods, n_durations=7,
                                      use_fast=False, refine_top_k=3,
                                      R_star=0.8)

    def test_validates_before_forwarding(self, monkeypatch):
        from cuvarbase import tls
        calls = []
        monkeypatch.setattr(tls, 'tls_search_gpu',
                            lambda *a, **k: calls.append(1))
        t = np.linspace(0, 30.0, 400)
        y = np.ones(400)
        dy = np.full(400, 1e-3)
        with pytest.raises(ValueError, match='tls_search'):
            tls.tls_search(t[:-1], y, dy)
        with pytest.raises(ValueError, match='tls_search'):
            tls.tls_search(t, np.r_[y[:-1], np.nan], dy)
        assert calls == []


class TestDurationGridKeplerian:
    """CPU tests for ``tls_grids.duration_grid_keplerian`` (release
    finding 71/149: untested). Shapes, bounds consistent with
    ``q_transit`` and the qmin_fac/qmax_fac factors, log spacing, and
    monotonicity in period."""

    PERIODS = np.array([1.0, 2.5, 5.0, 10.0, 30.0, 100.0])

    def test_shapes_and_counts(self):
        durations, counts, q = tls_grids.duration_grid_keplerian(
            self.PERIODS, n_durations=11)
        assert len(durations) == len(self.PERIODS)
        assert all(d.shape == (11,) for d in durations)
        assert all(d.dtype == np.float32 for d in durations)
        assert counts.shape == (len(self.PERIODS),)
        assert counts.dtype == np.int32 and np.all(counts == 11)
        assert q.shape == (len(self.PERIODS),)

    @pytest.mark.parametrize("kw", [
        dict(), dict(R_star=0.7, M_star=0.65, R_planet=2.3,
                     qmin_fac=0.4, qmax_fac=2.5, n_durations=9),
        dict(R_star=2.2, M_star=1.9, R_planet=11.0,
             qmin_fac=0.25, qmax_fac=3.0)])
    def test_bounds_follow_q_transit_and_the_factors(self, kw):
        stellar = {k: kw[k] for k in ('R_star', 'M_star', 'R_planet')
                   if k in kw}
        qmin_fac = kw.get('qmin_fac', 0.5)
        qmax_fac = kw.get('qmax_fac', 2.0)
        durations, _, q = tls_grids.duration_grid_keplerian(
            self.PERIODS, **kw)
        np.testing.assert_array_equal(
            q, tls_grids.q_transit(self.PERIODS, **stellar))
        dur = np.stack(durations)
        # first/last duration = (qmin_fac, qmax_fac) * q * P (absolute
        # days), to float32 rounding
        np.testing.assert_allclose(dur[:, 0], qmin_fac * q * self.PERIODS,
                                   rtol=1e-6)
        np.testing.assert_allclose(dur[:, -1], qmax_fac * q * self.PERIODS,
                                   rtol=1e-6)
        assert np.all(dur[:, 0] <= dur[:, -1])
        # every duration is inside the window, as a fraction of period
        frac = dur / self.PERIODS[:, None]
        assert np.all(frac >= qmin_fac * q[:, None] * (1 - 1e-6))
        assert np.all(frac <= qmax_fac * q[:, None] * (1 + 1e-6))

    def test_log_spaced_within_a_period_and_monotonic_in_period(self):
        durations, _, q = tls_grids.duration_grid_keplerian(
            self.PERIODS, n_durations=15)
        dur = np.stack(durations).astype(np.float64)
        # strictly increasing along the duration axis, constant ratio
        assert np.all(np.diff(dur, axis=1) > 0)
        ratios = dur[:, 1:] / dur[:, :-1]
        np.testing.assert_allclose(
            ratios, np.broadcast_to(ratios[:, :1], ratios.shape), rtol=1e-5)
        # a Keplerian duration grows with period (~ P^(1/3)) while the
        # fractional duration q shrinks (~ P^(-2/3))
        assert np.all(np.diff(dur, axis=0) > 0)
        assert np.all(np.diff(q) < 0)

    def test_single_period_and_scalar_input(self):
        durations, counts, q = tls_grids.duration_grid_keplerian(
            [7.5], n_durations=3)
        assert len(durations) == 1 and counts.tolist() == [3]
        assert q.shape == (1,)
        assert q[0] == pytest.approx(float(tls_grids.q_transit(7.5)))


class TestTlsInputGuards:
    """Sep-2026 release review (findings 2, 11, 14): ``dy=None`` passed
    the shared validator and the fast path built NaN weights and
    returned the flat-light-curve null result; the legacy path forwarded
    ``n_durations`` to the kernel unchecked (``n_durations <= 1`` is a
    0/0 duration step, again a null result for a good light curve); and
    ``tls_search``/``tls_search_gpu``/``tls_transit`` accepted and
    dropped any unknown keyword, so ``fap_null_draws=`` silently
    produced a result with no 'FAP' key. All three are host-side
    errors raised before any GPU work (CPU tests)."""

    def _lc(self, n=400):
        rand = np.random.RandomState(11)
        t = np.sort(30.0 * rand.rand(n))
        y = 1.0 + 1e-3 * rand.randn(n)
        dy = np.full(n, 1e-3)
        return t, y, dy

    def test_dy_none_rejected_on_every_entry_point(self):
        from cuvarbase import tls
        t, y, _ = self._lc()
        periods = np.array([2.0, 3.0])
        with pytest.raises(ValueError, match="tls_search_gpu: dy is required"):
            tls.tls_search_gpu(t, y, None, periods=periods)
        with pytest.raises(ValueError, match="tls_search_gpu: dy is required"):
            tls.tls_search_gpu(t, y, None, periods=periods, use_fast=False)
        with pytest.raises(ValueError, match="tls_search: dy is required"):
            tls.tls_search(t, y, None, periods=periods)
        with pytest.raises(ValueError, match="tls_transit: dy is required"):
            tls.tls_transit(t, y, None)
        with pytest.raises(ValueError,
                           match="tls_search_batch lightcurve 1: dy is required"):
            tls.tls_search_batch([(t, y, np.full(len(t), 1e-3)), (t, y, None)],
                                 periods=periods)
        with pytest.raises(ValueError, match="lightcurve 0: dy is required"):
            tls._preprocess_batch([(t, y, None)])

    @pytest.mark.parametrize("use_fast", [True, False])
    @pytest.mark.parametrize("n_durations", [1, 0, -3])
    def test_n_durations_below_two_rejected_on_both_paths(self, use_fast,
                                                          n_durations):
        from cuvarbase import tls
        t, y, dy = self._lc()
        with pytest.raises(ValueError, match="n_durations must be >= 2"):
            tls.tls_search_gpu(t, y, dy, periods=np.array([2.0, 3.0]),
                               n_durations=n_durations, use_fast=use_fast)

    def test_n_durations_must_be_an_integer(self):
        from cuvarbase import tls
        t, y, dy = self._lc()
        with pytest.raises(ValueError, match="n_durations must be an integer"):
            tls.tls_search_gpu(t, y, dy, periods=np.array([2.0, 3.0]),
                               n_durations=2.5, use_fast=False)
        with pytest.raises(ValueError, match="n_durations"):
            tls.tls_search_batch([(t, y, dy)], periods=np.array([2.0, 3.0]),
                                 n_durations=1)
        # numpy integers are integers
        assert tls._validate_n_durations(np.int64(7)) == 7

    def test_unknown_keywords_are_rejected_with_a_fap_hint(self):
        from cuvarbase import tls
        t, y, dy = self._lc()
        periods = np.array([2.0, 3.0])
        with pytest.raises(TypeError, match="fap_null_draws.*tls_search_batch"):
            tls.tls_search(t, y, dy, periods=periods, fap_null_draws=100)
        with pytest.raises(TypeError, match="fap_seed.*tls_search_batch"):
            tls.tls_search_gpu(t, y, dy, periods=periods, fap_seed=1)
        with pytest.raises(TypeError, match="tls_search_batch"):
            tls.tls_transit(t, y, dy, fap_null_draws=10, fap_seed=1)
        with pytest.raises(TypeError, match="'bogus_kwarg'") as excinfo:
            tls.tls_search_gpu(t, y, dy, periods=periods, bogus_kwarg=42)
        assert 'tls_search_batch' not in str(excinfo.value)
        # (n_template, the one legitimate extra keyword, is exercised by
        # the legacy-path GPU tests)
