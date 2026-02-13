"""
Tests for FFA-BLS (Fast Folding Algorithm for Box-Least Squares).

CPU-only tests validate the algorithm logic without GPU.
GPU tests (marked with pycuda import) validate GPU kernels.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
import importlib.util
import os


# --- Load ffa_bls module ---
# Try real import first (works on GPU machines); fall back to stub loading
# for CPU-only environments where pycuda is unavailable.
import sys as _sys

try:
    from cuvarbase.ffa_bls import (
        _next_power_of_2, _build_shift_vectors,
        _compute_section_boundaries, eebls_ffa_cpu,
    )
    _HAS_GPU = True
except (ImportError, Exception):
    _HAS_GPU = False
    import types as _types

    # Stub loading for CPU-only environments
    def _load_ffa_bls_module():
        """Load ffa_bls.py directly, stubbing pycuda to avoid GPU deps."""
        pycuda_stub = _types.ModuleType('pycuda')
        pycuda_stub.autoprimaryctx = _types.ModuleType('pycuda.autoprimaryctx')
        pycuda_stub.driver = _types.ModuleType('pycuda.driver')
        pycuda_stub.gpuarray = _types.ModuleType('pycuda.gpuarray')
        pycuda_compiler = _types.ModuleType('pycuda.compiler')
        pycuda_compiler.SourceModule = None

        stubs = {
            'pycuda': pycuda_stub,
            'pycuda.autoprimaryctx': pycuda_stub.autoprimaryctx,
            'pycuda.driver': pycuda_stub.driver,
            'pycuda.gpuarray': pycuda_stub.gpuarray,
            'pycuda.compiler': pycuda_compiler,
        }
        for k, v in stubs.items():
            _sys.modules[k] = v

        cuvarbase_dir = os.path.dirname(os.path.dirname(__file__))

        # Load utils first
        utils_path = os.path.join(cuvarbase_dir, 'utils.py')
        utils_spec = importlib.util.spec_from_file_location(
            'cuvarbase.utils', utils_path)
        utils_mod = importlib.util.module_from_spec(utils_spec)
        utils_spec.loader.exec_module(utils_mod)
        _sys.modules['cuvarbase.utils'] = utils_mod

        # Create cuvarbase package stub
        cuvarbase_stub = _types.ModuleType('cuvarbase')
        cuvarbase_stub.utils = utils_mod
        _sys.modules['cuvarbase'] = cuvarbase_stub

        # Load ffa_bls
        ffa_path = os.path.join(cuvarbase_dir, 'ffa_bls.py')
        spec = importlib.util.spec_from_file_location(
            'cuvarbase.ffa_bls', ffa_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = 'cuvarbase'
        spec.loader.exec_module(mod)
        return mod

    _ffa = _load_ffa_bls_module()
    _next_power_of_2 = _ffa._next_power_of_2
    _build_shift_vectors = _ffa._build_shift_vectors
    _compute_section_boundaries = _ffa._compute_section_boundaries
    eebls_ffa_cpu = _ffa.eebls_ffa_cpu


# ======== Helper functions ========

def make_transit_lightcurve(ndata=500, freq=1.0, q=0.05, phi0=0.3,
                            snr=20, sigma=0.1, baseline=365., seed=42):
    """Generate a synthetic transit light curve."""
    rng = np.random.RandomState(seed)

    delta = snr * sigma / np.sqrt(ndata * q * (1 - q))

    t = baseline * np.sort(rng.rand(ndata))
    phi = (t * freq) % 1.0
    y = np.zeros(ndata)
    y[phi < q] -= delta  # negative delta = transit dip
    y += sigma * rng.randn(ndata)
    dy = sigma * np.ones(ndata)

    return t, y, dy


def brute_force_binned_bls(t, y, dy, period, m_bins, qmin=0.01, qmax=0.15,
                           dlogq=0.2, ignore_negative_delta_sols=True):
    """
    Direct binned BLS for a single period (CPU reference).
    Bins observations, then does box scan. No FFA.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)

    w = np.power(dy, -2)
    w /= w.sum()
    ybar = np.dot(w, y)
    yy = np.dot(w, (y - ybar) ** 2)
    yw = (y - ybar) * w

    freq = 1.0 / period
    phases = (t * freq) % 1.0
    bins = np.clip(np.floor(m_bins * phases).astype(int), 0, m_bins - 1)

    yw_bins = np.zeros(m_bins, dtype=np.float64)
    w_bins = np.zeros(m_bins, dtype=np.float64)
    for k in range(len(t)):
        yw_bins[bins[k]] += yw[k]
        w_bins[bins[k]] += w[k]

    nbins0 = max(1, int(np.ceil(m_bins * qmin)))
    nbinsf = min(m_bins, max(nbins0, int(np.floor(m_bins * qmax))))

    max_bls = 0.0
    for start_bin in range(m_bins):
        acc_yw = 0.0
        acc_w = 0.0
        width = 0
        next_check = nbins0

        for k in range(nbinsf):
            bin_idx = (start_bin + k) % m_bins
            acc_yw += yw_bins[bin_idx]
            acc_w += w_bins[bin_idx]
            width += 1

            if width >= next_check:
                if acc_w > 1e-10 and acc_w < 1.0 - 1e-10:
                    bls_val = acc_yw ** 2 / (acc_w * (1.0 - acc_w))
                else:
                    bls_val = 0.0
                if ignore_negative_delta_sols and acc_yw > 0:
                    bls_val = 0.0
                if bls_val > max_bls:
                    max_bls = bls_val

                if dlogq > 0:
                    step = max(1, int(np.floor(dlogq * next_check)))
                    next_check += step
                else:
                    next_check += 1
                if next_check > nbinsf:
                    break

    return max_bls / yy if yy > 0 else 0.0


# ======== Unit tests: FFA primitives ========

class TestFFAPrimitives:

    def test_next_power_of_2(self):
        assert _next_power_of_2(1) == 1
        assert _next_power_of_2(2) == 2
        assert _next_power_of_2(3) == 4
        assert _next_power_of_2(4) == 4
        assert _next_power_of_2(5) == 8
        assert _next_power_of_2(1000) == 1024
        assert _next_power_of_2(1024) == 1024
        assert _next_power_of_2(1025) == 2048

    def test_shift_vector_base(self):
        """S_1 = (0, 1)"""
        vectors = _build_shift_vectors(1)
        assert len(vectors) == 1
        assert_allclose(vectors[0], [0, 1])

    def test_shift_vector_level2(self):
        """S_2 = (0, 1, 1, 2) = concat(S_1, S_1 + 1)"""
        vectors = _build_shift_vectors(2)
        assert len(vectors) == 2
        assert_allclose(vectors[0], [0, 1])
        assert_allclose(vectors[1], [0, 1, 1, 2])

    def test_shift_vector_level3(self):
        """S_3 = (0, 1, 1, 2, 2, 3, 3, 4)"""
        vectors = _build_shift_vectors(3)
        assert len(vectors) == 3
        assert_allclose(vectors[0], [0, 1])
        assert_allclose(vectors[1], [0, 1, 1, 2])
        assert_allclose(vectors[2], [0, 1, 1, 2, 2, 3, 3, 4])

    def test_shift_vector_length(self):
        """Shift vector at level l has 2^(l+1) elements."""
        for n_levels in range(1, 7):
            vectors = _build_shift_vectors(n_levels)
            for l, v in enumerate(vectors):
                assert len(v) == 2 ** (l + 1), \
                    f"Level {l}: expected {2**(l+1)} elements, got {len(v)}"

    def test_section_boundaries_uniform(self):
        """For uniformly sampled data, sections should be evenly divided."""
        t = np.arange(0, 100, 0.1)
        P0 = 10.0
        N_p = 8  # fewer than actual sections -- padding handles the rest

        starts, ends = _compute_section_boundaries(t, P0, N_p)
        assert len(starts) == N_p
        assert len(ends) == N_p

        # First section should start at index 0
        assert starts[0] == 0

        # Sections shouldn't overlap
        for i in range(N_p - 1):
            assert starts[i + 1] >= ends[i] or starts[i + 1] == ends[i]

    def test_section_boundaries_empty_sections(self):
        """Sections beyond the data should be empty."""
        t = np.arange(0, 10, 0.1)
        P0 = 5.0
        N_p = 8  # only ~2 sections have data

        starts, ends = _compute_section_boundaries(t, P0, N_p)
        # Later sections should be empty
        for i in range(3, N_p):
            assert starts[i] == ends[i] or starts[i] >= len(t)


# ======== Integration tests: CPU FFA-BLS ========

class TestFFABLSCPU:

    @pytest.mark.parametrize("freq", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("q", [0.05, 0.1])
    def test_ffa_finds_injected_transit(self, freq, q):
        """FFA-BLS should find an injected transit at approximately the right period."""
        period_true = 1.0 / freq
        t, y, dy = make_transit_lightcurve(
            ndata=500, freq=freq, q=q, snr=30, baseline=365., seed=42)

        # Search around the true period
        pmin = period_true * 0.8
        pmax = period_true * 1.2

        periods, power = eebls_ffa_cpu(
            t, y, dy, pmin, pmax,
            qmin=q * 0.5, qmax=q * 3, dlogq=0.2, m_bins=100,
            ignore_negative_delta_sols=True)

        if len(periods) == 0:
            pytest.skip("No FFA periods generated for this configuration")

        # Best period should be close to true period
        best_period = periods[np.argmax(power)]
        rel_err = abs(best_period - period_true) / period_true
        assert rel_err < 0.05, \
            f"Best period {best_period:.4f} too far from true {period_true:.4f} (rel_err={rel_err:.4f})"

    def test_ffa_no_signal_low_power(self):
        """FFA-BLS on pure noise should have low power everywhere."""
        rng = np.random.RandomState(42)
        ndata = 500
        t = 365. * np.sort(rng.rand(ndata))
        y = 0.1 * rng.randn(ndata)
        dy = 0.1 * np.ones(ndata)

        periods, power = eebls_ffa_cpu(
            t, y, dy, 1.0, 3.0,
            qmin=0.01, qmax=0.15, m_bins=50)

        if len(power) > 0:
            # Power should be modest (no strong signal)
            assert np.max(power) < 0.3, \
                f"Max power {np.max(power):.4f} too high for noise-only data"

    def test_ffa_cpu_output_shape(self):
        """Verify output shapes are consistent."""
        rng = np.random.RandomState(42)
        ndata = 200
        t = 100. * np.sort(rng.rand(ndata))
        y = rng.randn(ndata)
        dy = np.ones(ndata)

        periods, power = eebls_ffa_cpu(t, y, dy, 1.0, 10.0, m_bins=50)

        assert len(periods) == len(power)
        if len(periods) > 0:
            # Periods should be sorted
            assert np.all(np.diff(periods) >= 0), "Periods not sorted"
            # Periods should be within search range (approximately)
            assert periods[0] >= 0.5  # allow some slack
            assert periods[-1] <= 15.0

    def test_ffa_cpu_power_nonnegative(self):
        """BLS SR should always be non-negative."""
        rng = np.random.RandomState(42)
        ndata = 300
        t = 200. * np.sort(rng.rand(ndata))
        y = rng.randn(ndata)
        dy = np.ones(ndata)

        periods, power = eebls_ffa_cpu(t, y, dy, 1.0, 5.0, m_bins=50)

        if len(power) > 0:
            assert np.all(power >= 0), f"Negative power found: min={np.min(power)}"

    @pytest.mark.parametrize("period", [2.0, 5.0, 10.0])
    def test_ffa_vs_brute_force_single_period(self, period):
        """
        FFA fold at d=0 (zero drift) should exactly match direct BLS at P0.

        We choose the search range so that the true period falls near
        a base period P0 = m_oct * dt, where d=0 gives P=P0 exactly.
        This avoids FFA quantization error from non-zero drift.
        """
        t, y, dy = make_transit_lightcurve(
            ndata=500, freq=1.0/period, q=0.05, snr=30,
            baseline=365., seed=42)

        m_bins = 100
        qmin, qmax, dlogq = 0.01, 0.15, 0.2

        # Choose search range so that period falls near d=0 of an octave.
        # dt = pmin/m_bins, P0 = m_oct*dt. We need m_oct*dt ≈ period.
        # With pmin = period * (m_bins/(m_bins+5)), m_oct near m_bins+5
        # gives P0 ≈ period.
        pmin = period * 0.8
        pmax = period * 1.2

        dt = pmin / m_bins
        # Find the m_oct whose P0 is closest to period
        m_oct_best = int(round(period / dt))
        P0_best = m_oct_best * dt

        # Direct BLS at P0_best (same bins as FFA will use)
        direct_bls = brute_force_binned_bls(
            t, y, dy, P0_best, m_oct_best,
            qmin=qmin, qmax=qmax, dlogq=dlogq)

        # FFA-BLS
        periods, power = eebls_ffa_cpu(
            t, y, dy, pmin, pmax, m_bins=m_bins,
            qmin=qmin, qmax=qmax, dlogq=dlogq)

        if len(periods) == 0:
            pytest.skip("No FFA periods near target")

        # Find the FFA period closest to P0_best (should be d=0 of that octave)
        closest_idx = np.argmin(np.abs(periods - P0_best))
        ffa_bls = power[closest_idx]

        # At d=0, FFA is exact (just direct summation), so should match closely
        if direct_bls > 0.01:
            rel_diff = abs(ffa_bls - direct_bls) / direct_bls
            assert rel_diff < 0.05, \
                f"FFA BLS ({ffa_bls:.6f}) differs from direct ({direct_bls:.6f}), " \
                f"rel_diff={rel_diff:.4f} at P0={P0_best:.6f}"

    def test_ffa_butterfly_trivial(self):
        """Test FFA with a simple evenly-sampled signal."""
        # Create evenly-sampled data with a clear periodic signal
        freq = 0.1  # period = 10 days
        period = 1.0 / freq
        ndata = 1000
        t = np.linspace(0, 100, ndata)
        q = 0.1
        phi0 = 0.3

        # Create transit signal
        phases = (t * freq) % 1.0
        y = np.zeros(ndata)
        y[phases < q] = -1.0  # transit dip
        y += 0.01 * np.random.RandomState(42).randn(ndata)
        dy = 0.01 * np.ones(ndata)

        periods, power = eebls_ffa_cpu(
            t, y, dy, 8.0, 12.0, m_bins=50,
            qmin=0.05, qmax=0.2, dlogq=0.2)

        if len(periods) == 0:
            pytest.skip("No periods generated")

        # Should find the signal
        best_period = periods[np.argmax(power)]
        assert abs(best_period - period) / period < 0.1, \
            f"Best period {best_period:.4f} != true {period:.4f}"

    def test_ffa_empty_period_range(self):
        """Gracefully handle period range that produces no octaves."""
        rng = np.random.RandomState(42)
        t = np.sort(rng.rand(100))
        y = rng.randn(100)
        dy = np.ones(100)

        # Very narrow range that might produce 0 octaves
        periods, power = eebls_ffa_cpu(t, y, dy, 0.001, 0.002, m_bins=50)
        assert len(periods) == len(power)


# ======== GPU tests (require pycuda) ========

class TestFFABLSGPU:
    """GPU tests - will be skipped if pycuda is not available."""

    @pytest.fixture(autouse=True)
    def check_gpu(self):
        """Skip all GPU tests if pycuda is not available."""
        if not _HAS_GPU:
            pytest.skip("pycuda not available")

    @pytest.mark.parametrize("freq", [0.5, 1.0])
    @pytest.mark.parametrize("q", [0.05, 0.1])
    def test_gpu_finds_transit(self, freq, q):
        """GPU fBLS should find injected transit."""
        from cuvarbase.ffa_bls import eebls_ffa_gpu

        period_true = 1.0 / freq
        t, y, dy = make_transit_lightcurve(
            ndata=500, freq=freq, q=q, snr=30, baseline=365., seed=42)

        pmin = period_true * 0.8
        pmax = period_true * 1.2

        periods, power = eebls_ffa_gpu(
            t, y, dy, pmin, pmax,
            qmin=q * 0.5, qmax=q * 3, dlogq=0.2, m_bins=100)

        if len(periods) == 0:
            pytest.skip("No FFA periods generated")

        best_period = periods[np.argmax(power)]
        rel_err = abs(best_period - period_true) / period_true
        assert rel_err < 0.05

    def test_gpu_vs_cpu(self):
        """GPU and CPU implementations should give identical results."""
        from cuvarbase.ffa_bls import eebls_ffa_gpu

        t, y, dy = make_transit_lightcurve(
            ndata=300, freq=1.0, q=0.05, snr=20, baseline=100., seed=42)

        kwargs = dict(period_min=0.5, period_max=5.0,
                      m_bins=50, qmin=0.01, qmax=0.15, dlogq=0.2)

        periods_cpu, power_cpu = eebls_ffa_cpu(t, y, dy, **kwargs)
        periods_gpu, power_gpu = eebls_ffa_gpu(t, y, dy, **kwargs)

        assert_allclose(periods_cpu, periods_gpu, rtol=1e-5)
        # GPU uses float32 atomicAdd which introduces small precision
        # differences vs CPU numpy accumulation; check correlation
        # and that >99.5% of values are close
        corr = np.corrcoef(power_cpu, power_gpu)[0, 1]
        assert corr > 0.999, f"GPU/CPU power correlation {corr:.6f} too low"
        close_frac = np.mean(np.isclose(power_cpu, power_gpu, rtol=0.05))
        assert close_frac > 0.995, \
            f"Only {close_frac*100:.1f}% of values close (need >99.5%)"

    def test_gpu_batch_vs_sequential(self):
        """Batch (Phase 2) and sequential (Phase 1) GPU paths should match."""
        from cuvarbase.ffa_bls import eebls_ffa_gpu

        t, y, dy = make_transit_lightcurve(
            ndata=500, freq=1.0, q=0.05, snr=20, baseline=200., seed=42)

        kwargs = dict(period_min=0.5, period_max=10.0,
                      m_bins=50, qmin=0.01, qmax=0.15, dlogq=0.2)

        periods_seq, power_seq = eebls_ffa_gpu(
            t, y, dy, use_batch=False, **kwargs)
        periods_bat, power_bat = eebls_ffa_gpu(
            t, y, dy, use_batch=True, **kwargs)

        assert len(periods_seq) == len(periods_bat), \
            f"Period count mismatch: seq={len(periods_seq)}, batch={len(periods_bat)}"
        assert_allclose(periods_seq, periods_bat, rtol=1e-5)

        corr = np.corrcoef(power_seq, power_bat)[0, 1]
        assert corr > 0.999, \
            f"Batch vs sequential correlation {corr:.6f} too low"

        # Peak should agree
        peak_seq = periods_seq[np.argmax(power_seq)]
        peak_bat = periods_bat[np.argmax(power_bat)]
        assert abs(peak_seq - peak_bat) / peak_seq < 0.01, \
            f"Peak mismatch: seq={peak_seq:.6f}, batch={peak_bat:.6f}"

    def test_gpu_vs_standard_bls(self):
        """GPU fBLS should produce similar peak power to standard GPU BLS."""
        from cuvarbase.ffa_bls import eebls_ffa_gpu
        from cuvarbase.bls import eebls_gpu_fast_adaptive

        freq_true = 1.0
        period_true = 1.0 / freq_true
        q = 0.05

        t, y, dy = make_transit_lightcurve(
            ndata=500, freq=freq_true, q=q, snr=30, baseline=365., seed=42)

        # Standard BLS
        df = q / (10 * (max(t) - min(t)))
        freqs = np.linspace(0.95 * freq_true, 1.05 * freq_true, 100)
        power_std = eebls_gpu_fast_adaptive(t, y, dy, freqs, qmin=0.01, qmax=0.15)
        best_std = np.max(power_std)

        # FFA BLS
        periods_ffa, power_ffa = eebls_ffa_gpu(
            t, y, dy, 0.95 * period_true, 1.05 * period_true,
            qmin=0.01, qmax=0.15, m_bins=100)

        if len(power_ffa) == 0:
            pytest.skip("No FFA periods generated")

        best_ffa = np.max(power_ffa)

        # Both should find similar power
        # (not exact due to different grids and binning)
        if best_std > 0.01:
            ratio = best_ffa / best_std
            assert 0.3 < ratio < 3.0, \
                f"Power ratio FFA/standard = {ratio:.4f} (ffa={best_ffa:.6f}, std={best_std:.6f})"

    def test_gpu_power_nonnegative(self):
        """GPU BLS SR should always be non-negative."""
        from cuvarbase.ffa_bls import eebls_ffa_gpu

        rng = np.random.RandomState(42)
        t = 200. * np.sort(rng.rand(300))
        y = rng.randn(300)
        dy = np.ones(300)

        periods, power = eebls_ffa_gpu(t, y, dy, 1.0, 20.0, m_bins=50)

        if len(power) > 0:
            assert np.all(power >= 0)
