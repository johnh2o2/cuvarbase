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
    from ..nufft_lrt import NUFFTLRTAsyncProcess, epoch_grid
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
        """Detection of a known transit at a NON-zero epoch with the
        default ``epochs=None`` (automatic epoch grid). Before the
        Sep-2026 fix ``epochs=None`` evaluated a single phase-0 template,
        and this test passed only because it injected at epoch 0."""
        proc = NUFFTLRTAsyncProcess()

        # Generate transit signal
        true_period = 2.5
        true_duration = 0.2
        true_epoch = 0.7
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

        snr, best_epoch = proc.run(self.t, y, periods, durations=durations)

        # Check output shape
        assert snr.shape == (len(periods), len(durations))
        assert best_epoch.shape == snr.shape

        # Peak within two grid steps of the true period, best epoch
        # within a transit duration of the truth (mod P)
        best_period_idx = np.argmax(snr[:, 0])
        best_period = periods[best_period_idx]
        step = periods[1] - periods[0]
        assert np.abs(best_period - true_period) <= 2 * step + 1e-9
        d = np.abs(best_epoch[best_period_idx, 0] - true_epoch) % true_period
        d = min(d, true_period - d)
        assert d < true_duration

    @mark_cuda_test
    def test_white_noise_gives_low_snr(self):
        """Test that white noise gives low SNR"""
        proc = NUFFTLRTAsyncProcess()

        # Pure white noise
        y = self.rng.randn(len(self.t))

        periods = np.array([2.0, 3.0, 4.0])
        durations = np.array([0.2])

        snr = proc.run(self.t, y, periods, durations=durations,
                       epochs=np.array([0.0]))

        # SNR should be relatively low for pure noise (single template)
        assert np.all(np.abs(snr) < 5.0)
        # and bounded after the max over the automatic epoch grid
        # (measured 3.6)
        snr_max, _ = proc.run(self.t, y, periods, durations=durations)
        assert np.all(snr_max < 8.0)

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

        snr, best_epoch = proc.run(
            self.t, y, periods, durations=durations,
            nf=nf, estimate_psd=False, psd=custom_psd
        )

        # Should run without error
        assert snr.shape == (1, 1)
        assert np.isfinite(snr[0, 0])

    @mark_cuda_test
    def test_user_psd_zero_bin_is_floored(self):
        """audit id 121: a zero bin in a user PSD gave SNR ~1e6 (matched;
        measured -998957 on the base tree) or nan (marginal); the PSD is
        now floored at eps_floor * median once in run() for every
        detector -- the result equals a run with the bin explicitly set
        to that floor -- and its length is validated."""
        proc = NUFFTLRTAsyncProcess()
        y = self.rng.randn(len(self.t))
        nf = 2 * len(self.t)
        psd = np.ones(nf)
        psd[37] = 0.0
        psd_floored = psd.copy()
        psd_floored[37] = 1e-3                  # eps_floor * median
        kw = dict(durations=np.array([0.2]), epochs=np.array([0.3]), nf=nf,
                  estimate_psd=False)
        got = proc.run(self.t, y, np.array([2.0]), psd=psd, **kw)
        want = proc.run(self.t, y, np.array([2.0]), psd=psd_floored, **kw)
        assert np.all(np.isfinite(got))
        assert_allclose(got, want, rtol=1e-6)
        V = np.sin(self.t)[:, None]
        mkw = dict(detector='marginal', systematics_basis=V,
                   coeff_prior_cov=np.array([[1.0]]))
        marg = proc.run(self.t, y, np.array([2.0]), psd=psd, **mkw, **kw)
        marg_want = proc.run(self.t, y, np.array([2.0]), psd=psd_floored,
                             **mkw, **kw)
        assert np.all(np.isfinite(marg))
        assert_allclose(marg, marg_want, rtol=1e-6)
        with pytest.raises(ValueError, match="length nf"):
            proc.run(self.t, y, np.array([2.0]), psd=np.ones(nf + 5), **kw)

    @mark_cuda_test
    def test_double_precision(self):
        """Test double precision computation"""
        proc = NUFFTLRTAsyncProcess(use_double=True)

        y = np.sin(2 * np.pi * self.t / 2.0)
        periods = np.array([2.0])
        durations = np.array([0.2])

        snr, best_epoch = proc.run(self.t, y, periods, durations=durations)

        assert snr.shape == (1, 1)
        assert np.isfinite(snr[0, 0])

    @mark_cuda_test
    def test_marginal_detector_ignores_shared_systematic(self):
        """Detector A (marginalized joint detector): a strong
        basis-aligned trend must not derail the period search, and the
        detector must rank the true period more sharply than the plain
        matched filter on the same data (the contrast this test always
        promised; release finding 105)."""
        proc = NUFFTLRTAsyncProcess()

        true_period, true_duration, depth = 2.5, 0.25, 0.5
        signal = self.generate_transit_signal(
            self.t, true_period, 0.0, true_duration, depth)
        trend = np.sin(2 * np.pi * self.t / 9.0)      # slow systematic
        rng = np.random.RandomState(11)
        y = signal + 4.0 * trend + 0.1 * rng.randn(len(self.t))

        periods = np.linspace(2.0, 3.0, 20)
        durations = np.array([true_duration])
        epochs = np.array([0.0])
        V = trend[:, None]

        snr_marg = proc.run(self.t, y, periods, durations=durations,
                            epochs=epochs,
                            detector='marginal', systematics_basis=V,
                            coeff_prior_cov=np.array([[100.0]]))[:, 0, 0]
        snr_matched = proc.run(self.t, y, periods, durations=durations,
                               epochs=epochs)[:, 0, 0]
        step = periods[1] - periods[0]
        i_true = int(np.argmin(np.abs(periods - true_period)))
        best = periods[int(np.argmax(snr_marg))]
        assert np.abs(best - true_period) <= 2 * step + 1e-9

        def contrast(s):
            # peak height at the true period over the off-peak spread
            off = np.delete(s, i_true)
            return (s[i_true] - np.median(off)) / (np.std(off) + 1e-12)

        # measured: marginal 2.5, matched -0.45
        assert contrast(snr_marg) > contrast(snr_matched)

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
                       epochs=np.array([0.0]),
                       detector='sequential',
                       systematics_basis=trend[:, None])
        assert snr.shape == (len(periods), 1, 1)
        best = periods[int(np.argmax(snr[:, 0, 0]))]
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
        # audit ids 122/156: a non-PSD prior is rejected instead of
        # being pinv'ed into a flat prior
        with pytest.raises(ValueError, match="positive semidefinite"):
            proc.run(self.t, y, np.array([2.0]), detector='marginal',
                     systematics_basis=np.sin(self.t)[:, None],
                     coeff_prior_cov=np.array([[-1.0]]))

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

    @staticmethod
    def _bjd_data():
        """The verifier's BJD data model (repro/local/vfy-lrt-bjd): 600
        points over 60 d, a 1% box transit at P = 5.3 d, two small
        systematics that the basis detectors get as V."""
        rng = np.random.RandomState(1)
        N, T = 600, 60.0
        t = np.sort(rng.uniform(0, T, N))
        P0, dur, depth, sig, e0 = 5.3, 0.22, 0.01, 0.003, 1.2
        y = 1.0 + box_transit(t, P0, e0, dur, depth) + sig * rng.randn(N)
        V = np.stack([np.sin(2 * np.pi * t / T), (t - T / 2) / T], 1)
        y = y + 0.002 * V[:, 0] + 0.003 * V[:, 1]
        periods = np.array([4.1, 4.7, 5.3, 5.9, 6.5])
        durations = np.array([0.22])
        epochs = np.linspace(0, P0, 24, endpoint=False)
        return t, y, V, periods, durations, epochs

    # (parametrized GPU tests carry no @mark_cuda_test, as in test_bls.py:
    # the conftest stub turns the first GPU touch into a skip on CPU)
    @pytest.mark.parametrize('detector', ['matched', 'marginal',
                                          'sequential'])
    def test_bjd_invariance(self, detector):
        """Defect 5 (lrt-bjd-float32): times must be epoch-subtracted in
        float64 before the float32 cast. With BJD-scale input the three
        detectors returned a different statistic (corr 0.47-0.51, argmax
        moved, max 7.4 -> 13.9); with the fix the shifted run matches
        to float32 NFFT noise (audit: rel <= 4e-4 at sigma = 2; measured
        3e-6 at sigma = 4) with the same argmax, and the transit is seen
        at the true period."""
        t, y, V, periods, durations, epochs = self._bjd_data()
        kw = {}
        if detector == 'marginal':
            kw = dict(systematics_basis=V, coeff_prior_cov=np.eye(2) * 1e-4)
        elif detector == 'sequential':
            kw = dict(systematics_basis=V)
        proc = NUFFTLRTAsyncProcess()
        base = proc.run(t, y, periods, durations, epochs=epochs,
                        detector=detector, **kw)
        shifted = proc.run(t + BJD_OFFSET, y, periods, durations,
                           epochs=epochs + BJD_OFFSET, detector=detector,
                           **kw)
        rel = np.abs(shifted - base).max() / np.abs(base).max()
        assert rel < 1e-4
        assert np.argmax(shifted) == np.argmax(base)
        # the transit is seen (measured max 11.7-12.3) at the true period
        assert base.max() > 5.0
        assert np.unravel_index(np.argmax(base), base.shape)[0] == 2
        # the automatic epoch grid reports epochs in the caller's scale;
        # it is anchored at floor(min t), so an INTEGER offset reproduces
        # the same templates (as test_bls.py's integer bjd_offset)
        off = np.floor(BJD_OFFSET)
        s0, e0 = proc.run(t, y, periods[2:3], durations, detector=detector,
                          **kw)
        s1, e1 = proc.run(t + off, y, periods[2:3], durations,
                          detector=detector, **kw)
        assert_allclose(e1 - e0, off, atol=1e-6)
        assert abs(s1[0, 0] - s0[0, 0]) < 1e-4 * abs(s0[0, 0])

    @mark_cuda_test
    def test_epochs_none_recovers_random_epoch(self):
        """Defect 6 (lrt-epochs-none): the default ``epochs=None`` used
        to evaluate one phase-0 template per (period, duration) and
        recovered 0/12 transits injected at random epochs (0/6 on the
        base tree with this data); it now scans an automatic epoch grid
        and returns (max over epochs, best epoch)."""
        rng = np.random.RandomState(7)
        t = ground_times(rng)
        P, dur, depth = 5.3, 0.22, 0.01
        periods = _log_period_grid(P, n=16)
        ip = int(np.argmin(np.abs(periods - P)))
        proc = NUFFTLRTAsyncProcess()
        for trial in range(2):
            epoch = rng.uniform(0.2 * P, 0.9 * P)     # never phase 0
            y = 1 + 3e-3 * rng.randn(len(t)) + box_transit(t, P, epoch,
                                                            dur, depth)
            snr, best_epoch = proc.run(t, y, periods,
                                       durations=np.array([dur]))
            assert snr.shape == (len(periods), 1)
            assert int(np.argmax(snr[:, 0])) == ip
            assert snr[ip, 0] > 8.0                # measured 21-23
            d = np.abs(best_epoch[ip, 0] - epoch) % P
            d = min(d, P - d)
            assert d < 0.75 * dur                   # grid step P/n < dur/2
            # the best epoch lies on the documented grid
            grid = epoch_grid(P, dur) + np.floor(t.min())
            assert np.min(np.abs(grid - best_epoch[ip, 0])) < 1e-9

    @mark_cuda_test
    def test_marginal_psd_from_residual_matches_sequential(self):
        """Defect 22 (lrt-detectorA-defeated): with ``estimate_psd=True``
        the marginal detector's PSD came from ``y - V mu``, which still
        holds the realized systematics (median inflation ~36x across the
        band), whitening the transit away: SNR at the true template 2.5
        vs 8.3 for the sequential baseline (harness data model, depth
        0.008, base tree). With the PSD from the basis-projected
        residual the two agree (audit: 8.89 vs 8.89; measured 10.3 vs
        10.2)."""
        rng = np.random.RandomState(11)
        t = ground_times(rng)
        n = len(t)
        nf = 2 * n
        sigma_w = 3e-3
        sigma_r, tau = sigma_w, 0.8
        amps = np.array([6.0, 3.0, 6.0]) * sigma_w
        M, V, mu_c, cov_c = population_basis(rng, t, 90.0, sigma_w, sigma_r,
                                             tau, amps)
        proc = NUFFTLRTAsyncProcess()
        P, dur, ep = 5.3, 0.22, 1.0
        kw = dict(durations=np.array([dur]), epochs=np.array([ep]), nf=nf)
        marg, seq = [], []
        for r in range(4):
            noise = sigma_w * rng.randn(n) + ou_noise(rng, t, sigma_r, tau)
            y = 1.0 + noise + M @ (rng.randn(3) * amps) \
                + box_transit(t, P, ep, dur, 0.008)
            marg.append(float(proc.run(
                t, y, np.array([P]), detector='marginal',
                systematics_basis=V, coeff_prior_mean=mu_c,
                coeff_prior_cov=cov_c, **kw).max()))
            seq.append(float(proc.run(
                t, y, np.array([P]), detector='sequential',
                systematics_basis=V, **kw).max()))
        marg, seq = np.mean(marg), np.mean(seq)
        assert seq > 5.0
        assert marg > 0.8 * seq                      # was 0.26-0.29
        assert marg < 1.25 * seq

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

    @pytest.mark.parametrize('use_double', [False, True])
    def test_reused_memory_parity(self, use_double):
        """LRT-1: run() allocates one NFFT buffer set and reuses it for
        the data, the basis vectors and every template. The result must
        equal the per-template path (a fresh transform per call) to
        run-to-run NFFT noise (audit: 1.4e-6 on the transform)."""
        rng = np.random.RandomState(3)
        t = ground_times(rng, n=300)
        n = len(t)
        nf = 2 * n
        P, dur = 5.3, 0.22
        y = 1 + 3e-3 * rng.randn(n) + box_transit(t, P, 1.3, dur, 0.01)
        periods = np.array([4.0, P, 7.0])
        epochs = np.linspace(0, P, 6, endpoint=False)
        proc = NUFFTLRTAsyncProcess(use_double=use_double)
        got = proc.run(t, y, periods, np.array([dur]), epochs=epochs,
                       eps_floor=1e-12)
        # independent per-template evaluation with fresh memory per call
        from ..nufft_lrt import _smoothed_periodogram
        y0 = y - y.mean()
        Y = proc.compute_nufft(t, y0, nf)
        psd = _smoothed_periodogram((np.abs(Y) ** 2).astype(proc.real_type), 5)
        w = np.ones(nf)
        want = np.zeros_like(got)
        for i, p in enumerate(periods):
            for k, e in enumerate(epochs):
                tm = proc._generate_template(t, p, e, dur, 1.0)
                tm -= tm.mean()
                T = proc.compute_nufft(t, tm, nf)
                want[i, 0, k] = proc._compute_matched_filter_snr(
                    Y, T, psd, w, 1e-12)
        rel = np.abs(got - want).max() / np.abs(want).max()
        # measured (A40, bit-reproducible over 3 repeats): 3.64e-6
        # float32, 4.10e-8 float64 -- the residual is the different
        # NFFT truncation radius m (the reused memory is sized from an
        # L1 bound over all vectors, the per-call path from each y)
        assert rel < (1e-6 if use_double else 3e-5), rel

    def test_small_nf_does_not_break_the_psd_smoother(self):
        """nf < smooth_window used to die inside numpy with 'operands
        could not be broadcast together with shapes (4,) (5,)': the
        boxcar 'same' convolution returns max(nf, window) samples. The
        window is now clamped to nf."""
        rng = np.random.RandomState(5)
        n = 40
        t = np.sort(rng.rand(n) * 12.0)
        y = 1.0 + 0.004 * rng.randn(n)
        periods = np.array([2.0, 3.0])
        durations = np.array([0.2])
        proc = NUFFTLRTAsyncProcess()
        for nf in (1, 2, 3, 4, 5, 6, 9):
            s = proc.run(t, y, periods, durations,
                         epochs=np.array([0.0, 0.5]), nf=nf)
            assert s.shape == (2, 1, 2)
            assert np.all(np.isfinite(s))
        # a huge window is equally harmless
        s = proc.run(t, y, periods, durations, epochs=np.array([0.0]),
                     nf=8, smooth_window=1000)
        assert np.all(np.isfinite(s))

    @mark_cuda_test
    def test_sequential_nonzero_mean_basis(self):
        """Defect 21 (lrt-sequential-intercept): a basis column with a 1%
        mean on relative flux dropped the sequential detector's SNR at
        the true period from ~25 to ~5 (no intercept in the OLS); the
        centred fit is insensitive to the column mean."""
        rng = np.random.RandomState(7)
        n, T = 2000, 90.0
        t = np.sort(rng.rand(n) * T)
        P, dur, depth, e0, sig = 3.3, 0.15, 0.006, 1.1, 0.003
        V0 = np.stack([np.sin(2 * np.pi * t / 30.), np.cos(2 * np.pi * t / 17.)],
                      axis=1)
        V0 = (V0 - V0.mean(axis=0)) / V0.std(axis=0)
        c = np.array([0.004, -0.004])
        noise = sig * rng.randn(n)
        transit = box_transit(t, P, e0, dur, depth)
        periods = np.array([P, 2.9, 3.1, 3.5, 3.7, 4.1])
        epochs = np.arange(0, P, dur / 2)
        proc = NUFFTLRTAsyncProcess()
        got = {}
        for mean_off in (0.0, 1e-2):
            V = V0 + mean_off
            y = 1.0 + transit + V @ c + noise
            s = proc.run(t, y, periods, np.array([dur]), epochs=epochs,
                         detector='sequential', systematics_basis=V)
            got[mean_off] = s.max(axis=(1, 2))
        for mean_off, m in got.items():
            assert int(np.argmax(m)) == 0, mean_off
            assert m[0] > 2.0 * m[1:].max(), mean_off
        # the column mean must not change the statistic (measured 4e-9)
        assert_allclose(got[1e-2], got[0.0], rtol=5e-3)


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
