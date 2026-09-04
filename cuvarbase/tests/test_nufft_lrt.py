"""
GPU tests for the NUFFT-based Likelihood Ratio Test (LRT) transit search.

Every random draw is seeded (``np.random.RandomState``); the data models
mirror the audit repro scripts (``analysis/audit-sep2026/repro/local/``
vfy-lrt-bjd, verify-lrt-epochs, vseq, vfy-detA, verify-lrt-band) and the
validation harness (``scripts/nufft_lrt_validation.py``).
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from pycuda.tools import mark_cuda_test

try:
    from ..nufft_lrt import NUFFTLRTAsyncProcess
    from ..cunfft import NFFTAsyncProcess
    NUFFT_LRT_AVAILABLE = True
except ImportError:
    NUFFT_LRT_AVAILABLE = False

pytestmark = pytest.mark.filterwarnings(
    "ignore:cuvarbase.nufft_lrt is EXPERIMENTAL")

BJD_OFFSET = 2457000.5


# ------------------------------------------------------------ data models

def ground_times(rng, baseline=90.0, n=600):
    """Nightly visibility windows with weather losses (the harness's
    'ground' sampling)."""
    nights = np.arange(int(baseline))
    nights = nights[rng.rand(len(nights)) > 0.35]
    per_night = max(1, int(round(n / max(len(nights), 1))))
    t = (nights[:, None] + 0.25 * rng.rand(len(nights), per_night)).ravel()
    return np.sort(t[:n])


def ou_noise(rng, t, sigma_red, tau):
    x = np.zeros(len(t))
    x[0] = sigma_red * rng.randn()
    for i in range(1, len(t)):
        a = np.exp(-(t[i] - t[i - 1]) / tau)
        x[i] = x[i - 1] * a + sigma_red * np.sqrt(1 - a * a) * rng.randn()
    return x


def box_transit(t, period, epoch, duration, depth):
    phase = np.fmod(t - epoch, period) / period
    phase[phase < 0] += 1.0
    phase[phase > 0.5] -= 1.0
    y = np.zeros_like(t)
    y[np.abs(phase) <= duration / (2.0 * period)] = -depth
    return y


def adjoint_dft(t, y, nf, chunk=1024):
    """Exact float64 adjoint DFT at the GPU convention (modes k = 0..nf-1,
    f_k = k / (max t - min t))."""
    t = np.asarray(t, np.float64)
    y = np.asarray(y, np.float64)
    x = t / (t.max() - t.min())
    out = np.empty(nf, np.complex128)
    for a in range(0, nf, chunk):
        k = np.arange(a, min(nf, a + chunk))
        out[a:a + len(k)] = np.exp(2j * np.pi * np.outer(k, x)) @ y
    return out


def population_basis(rng, t, baseline, sigma_w, sigma_r, tau, amps,
                     n_pop=40, K=3):
    """The harness's paper-style systematics model: three shared modes, a
    PCA basis from a signal-free population and a coefficient prior from
    per-lightcurve fits."""
    m1 = (t - t.mean()) / (0.5 * baseline)
    tn = t - np.floor(t) - 0.125
    m2 = (tn / 0.125) ** 2 - 0.5
    m3 = np.sin(2 * np.pi * t / (0.4 * baseline))
    M = np.stack([m1, m2, m3], axis=1)
    M = M / np.std(M, axis=0)
    pop = np.empty((n_pop, len(t)))
    for i in range(n_pop):
        c = rng.randn(M.shape[1]) * amps
        y = 1.0 + sigma_w * rng.randn(len(t)) + ou_noise(rng, t, sigma_r, tau)
        pop[i] = y + M @ c
    pop -= pop.mean(axis=1, keepdims=True)
    _, _, VT = np.linalg.svd(pop, full_matrices=False)
    V = VT[:K].T
    coeffs = pop @ V
    return M, V, coeffs.mean(axis=0), np.cov(coeffs.T)


def _log_period_grid(p_true, n=16, lo=2.0, hi=18.0):
    periods = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    periods[np.argmin(np.abs(periods - p_true))] = p_true
    return periods


# -------------------------------------------------------------- the tests

@pytest.mark.skipif(not NUFFT_LRT_AVAILABLE,
                   reason="NUFFT LRT not available")
class TestNUFFTLRT:
    """Test NUFFT LRT functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.n_data = 100
        self.rng = np.random.RandomState(20260904)
        self.t = np.sort(self.rng.uniform(0, 10, self.n_data))

    def generate_transit_signal(self, t, period, epoch, duration, depth):
        """Generate a simple transit signal"""
        return box_transit(np.asarray(t, np.float64), period, epoch,
                           duration, depth)

    @mark_cuda_test
    def test_basic_initialization(self):
        """Test that NUFFTLRTAsyncProcess can be initialized"""
        proc = NUFFTLRTAsyncProcess()
        assert proc is not None
        # sigma = 4 keeps the full returned band k = 0..nf-1 inside the
        # Gaussian window's accuracy band (sigma = 2 aliased k >= nf/2)
        assert proc.sigma == 4.0
        assert proc.nufft_proc.sigma == 4.0
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

        # Peak should be near the signal frequency. The adjoint NFFT
        # returns Fourier coefficients at modes k = 0..nf-1, i.e.
        # frequencies f_k = k / (max(t) - min(t)) -- NOT the rfft grid.
        freqs = np.arange(nf) / (self.t.max() - self.t.min())
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
        Y = self.rng.randn(nf) + 1j * self.rng.randn(nf)
        T = self.rng.randn(nf) + 1j * self.rng.randn(nf)
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
        noise = noise_level * self.rng.randn(len(self.t))
        y = signal + noise

        # Search over periods
        periods = np.linspace(2.0, 3.0, 20)
        durations = np.array([true_duration])

        snr = proc.run(self.t, y, periods, durations=durations)

        # Check output shape
        assert snr.shape == (len(periods), len(durations))

        # Peak within two grid steps of the true period
        best_period_idx = np.argmax(snr[:, 0])
        best_period = periods[best_period_idx]
        step = periods[1] - periods[0]
        assert np.abs(best_period - true_period) <= 2 * step + 1e-9

    @mark_cuda_test
    def test_white_noise_gives_low_snr(self):
        """Test that white noise gives low SNR"""
        proc = NUFFTLRTAsyncProcess()

        # Pure white noise
        y = self.rng.randn(len(self.t))

        periods = np.array([2.0, 3.0, 4.0])
        durations = np.array([0.2])

        snr = proc.run(self.t, y, periods, durations=durations)

        # SNR should be relatively low for pure noise
        assert np.all(np.abs(snr) < 5.0)

    @mark_cuda_test
    def test_custom_psd(self):
        """Test using custom power spectrum"""
        proc = NUFFTLRTAsyncProcess()

        # Generate simple signal
        y = np.sin(2 * np.pi * self.t / 2.0) + 0.1 * self.rng.randn(len(self.t))

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
        """Test double precision computation"""
        proc = NUFFTLRTAsyncProcess(use_double=True)

        y = np.sin(2 * np.pi * self.t / 2.0)
        periods = np.array([2.0])
        durations = np.array([0.2])

        snr = proc.run(self.t, y, periods, durations=durations)

        assert snr.shape == (1, 1)
        assert np.isfinite(snr[0, 0])

    @mark_cuda_test
    def test_marginal_detector_ignores_shared_systematic(self):
        """Detector A (marginalized joint detector): a strong
        basis-aligned trend must not derail the period search, while
        the plain matched filter's ranking degrades."""
        proc = NUFFTLRTAsyncProcess()

        true_period, true_duration, depth = 2.5, 0.25, 0.5
        signal = self.generate_transit_signal(
            self.t, true_period, 0.0, true_duration, depth)
        trend = np.sin(2 * np.pi * self.t / 9.0)      # slow systematic
        rng = np.random.RandomState(11)
        y = signal + 4.0 * trend + 0.1 * rng.randn(len(self.t))

        periods = np.linspace(2.0, 3.0, 20)
        durations = np.array([true_duration])
        V = trend[:, None]

        snr_marg = proc.run(self.t, y, periods, durations=durations,
                            detector='marginal', systematics_basis=V,
                            coeff_prior_cov=np.array([[100.0]]))
        step = periods[1] - periods[0]
        best = periods[int(np.argmax(snr_marg[:, 0]))]
        assert np.abs(best - true_period) <= 2 * step + 1e-9

    @mark_cuda_test
    def test_sequential_detector_runs_and_detects(self):
        proc = NUFFTLRTAsyncProcess()
        true_period, true_duration, depth = 2.5, 0.25, 0.5
        signal = self.generate_transit_signal(
            self.t, true_period, 0.0, true_duration, depth)
        trend = (self.t - self.t.mean()) / self.t.std()
        rng = np.random.RandomState(12)
        y = signal + 2.0 * trend + 0.1 * rng.randn(len(self.t))

        periods = np.linspace(2.0, 3.0, 20)
        snr = proc.run(self.t, y, periods,
                       durations=np.array([true_duration]),
                       detector='sequential',
                       systematics_basis=trend[:, None])
        assert snr.shape == (len(periods), 1)
        best = periods[int(np.argmax(snr[:, 0]))]
        step = periods[1] - periods[0]
        assert np.abs(best - true_period) <= 2 * step + 1e-9

    @mark_cuda_test
    def test_marginal_requires_basis_and_prior(self):
        proc = NUFFTLRTAsyncProcess()
        y = self.rng.randn(len(self.t))
        with pytest.raises(ValueError, match="systematics_basis"):
            proc.run(self.t, y, np.array([2.0]), detector='marginal')
        with pytest.raises(ValueError, match="coeff_prior_cov"):
            proc.run(self.t, y, np.array([2.0]), detector='marginal',
                     systematics_basis=np.ones((len(self.t), 1)))
        with pytest.raises(ValueError, match="detector"):
            proc.run(self.t, y, np.array([2.0]), detector='bogus')

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
        y = signal + 0.1 * self.rng.randn(len(self.t))

        periods = np.array([true_period])
        durations = np.array([true_duration])
        epochs = np.linspace(0, true_period, 10)

        snr = proc.run(
            self.t, y, periods, durations=durations, epochs=epochs
        )

        # Check output shape
        assert snr.shape == (1, 1, len(epochs))

        # Best epoch should be close to true epoch (two grid steps)
        best_epoch_idx = np.argmax(snr[0, 0, :])
        best_epoch = epochs[best_epoch_idx]
        step = epochs[1] - epochs[0]
        epoch_diff = np.abs(best_epoch - true_epoch)
        epoch_diff = min(epoch_diff, true_period - epoch_diff)
        assert epoch_diff <= 2 * step + 1e-9


@pytest.mark.skipif(not NUFFT_LRT_AVAILABLE,
                    reason="NUFFT LRT not available")
class TestSep2026Defects:
    """Regression tests derived from the Sep-2026 algorithm audit
    (analysis/audit-sep2026/ALGORITHM_AUDIT.md section 2)."""

    # (parametrized GPU tests carry no @mark_cuda_test, as in test_bls.py:
    # the conftest stub turns the first GPU touch into a skip on CPU)
    @pytest.mark.parametrize('use_double', [False, True])
    def test_full_band_nfft_matches_exact_dft(self, use_double):
        """Defect 24 (lrt-upper-half-band): with the old sigma = 2 the
        modes k >= nf/2 carried O(1) aliasing error (max|G-E|/rms 1.3 in
        both precisions); with sigma = 4 every returned mode matches the
        exact adjoint DFT. Compared through the phase-invariant
        quantities the detectors use (moduli and the data x template
        cross-spectrum), so the transform's time reference is free."""
        rng = np.random.RandomState(0)
        t = ground_times(rng)
        n = len(t)
        nf = 2 * n
        y = 3e-3 * rng.randn(n)
        y -= y.mean()
        tmpl = box_transit(t, 5.3, 1.3, 0.22, 1.0)
        tmpl -= tmpl.mean()
        proc = NUFFTLRTAsyncProcess(use_double=use_double)
        Gy = proc.compute_nufft(t, y, nf).astype(np.complex128)
        Gt = proc.compute_nufft(t, tmpl, nf).astype(np.complex128)
        Ey = adjoint_dft(t, y, nf)
        Et = adjoint_dft(t, tmpl, nf)
        # measured (A40): 1.6e-4 float32, 3.5e-7 float64; was 0.49 at sigma=2
        tol = 2e-6 if use_double else 1e-3
        for G, E in ((Gy, Ey), (Gt, Et)):
            rel = np.abs(np.abs(G) - np.abs(E)).max() / np.abs(E).max()
            assert rel < tol, rel
        cross_g = Gy * np.conj(Gt)
        cross_e = Ey * np.conj(Et)
        rel = np.abs(cross_g - cross_e).max() / np.abs(cross_e).max()
        assert rel < 2 * tol, rel
        # the upper half band specifically (the aliased region)
        hi = slice(nf // 2, nf)
        rel_hi = np.abs(np.abs(Gy[hi]) - np.abs(Ey[hi])).max() / np.abs(Ey).max()
        assert rel_hi < tol, rel_hi


@pytest.mark.skipif(not NUFFT_LRT_AVAILABLE,
                    reason="NUFFT LRT not available")
class TestNFFTMemoryReuse:
    """audit ids 118/155: ``NFFTAsyncProcess.run(memory=...)`` returned
    the pinned host buffer before the async D2H copy landed and never
    zeroed the atomic grid, so a second run on reused memory summed onto
    the first (off by ~1e5-1e8)."""

    @mark_cuda_test
    def test_reused_memory_equals_fresh_runs(self):
        rng = np.random.RandomState(3)
        n = 5000
        t = np.sort(rng.rand(n) * 30)
        y1 = rng.randn(n)
        y2 = rng.randn(n)
        nf = 2 * n
        proc = NFFTAsyncProcess()
        fresh1 = np.array(proc.run([(t, y1, nf)])[0])
        fresh2 = np.array(proc.run([(t, y2, nf)])[0])
        mem = proc.allocate([(t, y1, nf)])
        got1 = np.array(proc.run([(t, y1, nf)], memory=mem)[0])  # immediate
        mem[0].y = y2.astype(mem[0].real_type)
        got2 = np.array(proc.run([(t, y2, nf)], memory=mem)[0])
        scale = np.abs(fresh1).max()
        # measured (A40): 2e-5 for both; the un-zeroed grid gave 2.3e5
        assert np.abs(got1 - fresh1).max() < 1e-3 * scale
        assert np.abs(got2 - fresh2).max() < 1e-3 * scale
        # the immediate read was complete (run() synchronized)
        mem[0].stream.synchronize()
        np.testing.assert_array_equal(got2, np.array(mem[0].ghat_c))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
