"""
Test NUFFT LRT algorithm logic without requiring GPU.

These tests exercise the *shipped* template-generation and matched-filter
code in cuvarbase.nufft_lrt (both are pure numpy). An earlier version of
this file defined local copies of the algorithms and tested those, which
validated nothing about the package.

``_compute_matched_filter_snr`` is not called by ``run()``; it is a
single-template reference wrapper over the same ``_floor_psd`` /
``_matched_filter_statistic`` helpers that ``run()`` evaluates per
template, so testing it tests the shipped arithmetic.
``TestRunHostPipeline`` closes the loop by running ``run()`` itself with
the GPU transform replaced by an exact host adjoint DFT.
"""
import numpy as np
import pytest

from ..nufft_lrt import NUFFTLRTAsyncProcess

pytestmark = pytest.mark.filterwarnings(
    "ignore:cuvarbase.nufft_lrt is EXPERIMENTAL")


@pytest.fixture(scope='module')
def proc():
    return NUFFTLRTAsyncProcess()


def test_smoothed_periodogram_clamps_the_window():
    # CPU-only: np.convolve(..., 'same') returns max(len, window)
    # samples, so an unclamped window > nf lengthened the PSD and blew
    # up downstream with a raw numpy broadcast error.
    from ..nufft_lrt import _smoothed_periodogram
    for n in (1, 2, 3, 4, 5, 6, 7, 33):
        p = np.arange(1.0, n + 1.0)
        for window in (1, 2, 5, 1000):
            out = _smoothed_periodogram(p, window)
            assert len(out) == n, (n, window)
            assert np.all(np.isfinite(out))
        # any window >= n gives the same (fully clamped) result
        assert np.allclose(_smoothed_periodogram(p, 1000),
                           _smoothed_periodogram(p, n))
        # ... and smoothing preserves the total (edge-corrected mean of
        # the available neighbours, never zero-padded)
        assert _smoothed_periodogram(p, 3).min() >= p.min()
        assert _smoothed_periodogram(p, 3).max() <= p.max()


def test_run_docstring_warns_about_the_duration_outer_product():
    doc = ' '.join(NUFFTLRTAsyncProcess.run.__doc__.split())
    assert 'outer product' in doc
    assert 'len(periods)**2' in doc
    assert '0.1 * periods' in doc


class TestNUFFTLRTAlgorithm:
    """Test NUFFT LRT algorithm logic (CPU-only, real implementation)"""

    def test_template_generation(self, proc):
        """Test transit template generation"""
        t = np.linspace(0, 10, 100)
        period = 2.0
        epoch = 0.0
        duration = 0.2
        depth = 1.0

        template = proc._generate_template(t, period, epoch, duration, depth)

        # Check properties
        assert len(template) == len(t)
        assert np.min(template) == -depth
        assert np.max(template) == 0.0

        # Check that some points are in transit
        in_transit = template < 0
        assert np.sum(in_transit) > 0
        assert np.sum(in_transit) < len(template)

        # Check expected number of points in transit
        expected_fraction = duration / period
        actual_fraction = np.sum(in_transit) / len(template)

        # Should be roughly correct (within factor of 2)
        assert 0.5 * expected_fraction < actual_fraction < 2.0 * expected_fraction

    def test_matched_filter_perfect_match(self, proc):
        """Test matched filter with perfect match gives high SNR"""
        nf = 100

        # Perfect match should give high SNR
        rng = np.random.RandomState(0)
        T = rng.randn(nf) + 1j * rng.randn(nf)
        Y = T.copy()  # Perfect match
        P_s = np.ones(nf)
        weights = np.ones(nf)

        snr = proc._compute_matched_filter_snr(Y, T, P_s, weights, 1e-12)

        # Perfect match should give SNR ~ sqrt(sum(|T|^2))
        expected_snr = np.sqrt(np.sum(np.abs(T) ** 2))
        assert np.abs(snr - expected_snr) / expected_snr < 0.01

    def test_matched_filter_orthogonal_signals(self, proc):
        """Test matched filter with orthogonal signals gives low SNR"""
        nf = 100

        rng = np.random.RandomState(1)
        T = rng.randn(nf) + 1j * rng.randn(nf)
        Y = rng.randn(nf) + 1j * rng.randn(nf)
        Y = Y - np.vdot(Y, T) * T / np.vdot(T, T)  # Make orthogonal

        P_s = np.ones(nf)
        weights = np.ones(nf)

        snr = proc._compute_matched_filter_snr(Y, T, P_s, weights, 1e-12)

        # Orthogonal signals should give SNR ~ 0
        assert np.abs(snr) < 1.0

    def test_matched_filter_scale_invariance(self, proc):
        """Test matched filter is invariant to template scaling"""
        nf = 100

        rng = np.random.RandomState(2)
        T = rng.randn(nf) + 1j * rng.randn(nf)
        Y = 2.0 * T  # Scaled version
        P_s = np.ones(nf)
        weights = np.ones(nf)

        snr1 = proc._compute_matched_filter_snr(Y, T, P_s, weights, 1e-12)
        snr2 = proc._compute_matched_filter_snr(Y, 0.5 * T, P_s, weights,
                                                1e-12)

        # SNR should be invariant to template scaling
        assert np.abs(snr1 - snr2) < 0.01

    def test_matched_filter_noise_distribution(self, proc):
        """Test matched filter gives reasonable SNR distribution for noise"""
        nf = 100
        P_s = np.ones(nf)
        weights = np.ones(nf)

        snrs = []
        rng = np.random.RandomState(42)
        for _ in range(50):
            Y = rng.randn(nf) + 1j * rng.randn(nf)
            T = rng.randn(nf) + 1j * rng.randn(nf)
            snr = proc._compute_matched_filter_snr(Y, T, P_s, weights, 1e-12)
            snrs.append(snr)

        mean_snr = np.mean(snrs)
        std_snr = np.std(snrs)

        # Mean should be close to 0, std should be reasonable
        assert np.abs(mean_snr) < 2.0
        assert std_snr > 0

    def test_power_spectrum_floor_prevents_blowup(self, proc):
        """Zero entries in the power spectrum must not produce inf/nan"""
        nf = 100
        rng = np.random.RandomState(3)
        T = rng.randn(nf) + 1j * rng.randn(nf)
        Y = T.copy()
        weights = np.ones(nf)

        P_s = np.ones(nf)
        P_s[::7] = 0.0  # exact zeros, would divide-by-zero without floor

        snr = proc._compute_matched_filter_snr(Y, T, P_s, weights, 1e-6)
        assert np.isfinite(snr)
        assert snr > 0

    def test_matched_filter_with_colored_noise(self, proc):
        """Test matched filter with non-uniform power spectrum"""
        nf = 100

        rng = np.random.RandomState(4)
        # Create frequency-dependent noise (colored noise)
        P_s = np.linspace(0.5, 2.0, nf)  # Varying power
        weights = np.ones(nf)

        T = rng.randn(nf) + 1j * rng.randn(nf)
        Y = T + np.sqrt(P_s) * (rng.randn(nf) + 1j * rng.randn(nf))

        snr = proc._compute_matched_filter_snr(Y, T, P_s, weights, 1e-12)

        # SNR should be positive and finite
        assert snr > 0
        assert np.isfinite(snr)


def test_empty_basis_is_rejected_before_device_work(proc):
    """A (n, 0) systematics basis with detector='marginal' used to fall
    through to the plain matched filter; after the Detector A
    precompute was hoisted out of the template loop it raised a raw
    numpy 'cannot reshape array of size 0' AFTER the data transforms
    had run. Both basis detectors now reject K = 0 with a ValueError
    before touching the device (so this runs under the CPU stub)."""
    from ..nufft_lrt import _marginal_precompute
    rng = np.random.RandomState(0)
    n = 60
    t = np.sort(rng.rand(n) * 20.0)
    y = 1.0 + 1e-3 * rng.randn(n)
    empty = np.zeros((n, 0))
    with pytest.raises(ValueError, match="at least one column"):
        proc.run(t, y, np.array([3.0]), durations=np.array([0.2]),
                 epochs=np.array([0.0]), detector='marginal',
                 systematics_basis=empty, coeff_prior_cov=np.zeros((0, 0)))
    with pytest.raises(ValueError, match="at least one column"):
        proc.run(t, y, np.array([3.0]), durations=np.array([0.2]),
                 epochs=np.array([0.0]), detector='sequential',
                 systematics_basis=empty)
    # the hoisted precompute mirrors the guard
    nf = 16
    Y = rng.randn(nf) + 1j * rng.randn(nf)
    with pytest.raises(ValueError, match="K >= 1"):
        _marginal_precompute(Y, [], np.ones(nf), np.ones(nf),
                             np.zeros((0, 0)))


def _adjoint_dft(t, y, nf):
    """Exact float64 adjoint DFT at the GPU convention (modes k = 0..nf-1,
    f_k = k / (max t - min t))."""
    t = np.asarray(t, np.float64)
    y = np.asarray(y, np.float64)
    x = t / (t.max() - t.min())
    k = np.arange(nf)
    return np.exp(2j * np.pi * np.outer(k, x)) @ y


class TestRunHostPipeline:
    """``run()`` end to end on the CPU: the adjoint NFFT (the only GPU
    work) is replaced by the exact adjoint DFT, so everything else --
    validation, epoch subtraction, PSD estimate and floor, the whitening
    weights and the per-template reduction ``run()`` actually executes --
    is exercised under the stub. The earlier CPU tests only reached the
    helper ``_compute_matched_filter_snr``, which ``run()`` no longer
    calls (finding 32 of the Sep-2026 review)."""

    @pytest.fixture
    def cpu_proc(self, proc, monkeypatch):
        monkeypatch.setattr(proc, '_nfft_memory',
                            lambda t, nf, l1_max, **kw: None)
        monkeypatch.setattr(proc, 'compute_nufft',
                            lambda t, y, nf, memory=None, **kw:
                            _adjoint_dft(t, y, nf))
        return proc

    @staticmethod
    def _data(rng, n=80):
        t = np.sort(rng.rand(n) * 30.0) + 2457000.0    # absolute BJD
        P, e, d = 4.3, 2457001.1, 0.25
        phase = np.fmod(t - e, P) / P
        phase[phase > 0.5] -= 1.0
        y = 1.0 + 2e-3 * rng.randn(n)
        y[np.abs(phase) <= d / (2 * P)] -= 0.02
        return t, y, P, e, d

    def test_matched_path_equals_reference_wrapper(self, cpu_proc):
        rng = np.random.RandomState(11)
        t, y, P, e, d = self._data(rng)
        periods = np.array([3.0, P, 6.0])
        epochs = np.array([0.0, e - np.floor(t.min()), 1.7])
        got = cpu_proc.run(t, y, periods, durations=np.array([d]),
                           epochs=epochs + np.floor(t.min()))
        assert got.shape == (3, 1, 3)
        # independent per-template evaluation through the wrapper, on
        # the same host transforms run() saw
        from ..nufft_lrt import _smoothed_periodogram
        t0 = t - np.floor(t.min())
        nf = 2 * len(t)
        Y = _adjoint_dft(t0, y - y.mean(), nf)
        psd = _smoothed_periodogram(
            (np.abs(Y) ** 2).astype(cpu_proc.real_type), 5)
        want = np.zeros_like(got)
        for i, p in enumerate(periods):
            for k, ep in enumerate(epochs):
                tm = cpu_proc._generate_template(t0, p, ep, d, 1.0)
                tm -= tm.mean()
                T = _adjoint_dft(t0, tm, nf)
                want[i, 0, k] = cpu_proc._compute_matched_filter_snr(
                    Y, T, psd, np.ones(nf), 1e-3)
        np.testing.assert_allclose(got, want, rtol=1e-10)
        # and the injected template is the maximum
        i, j, k = np.unravel_index(np.argmax(got), got.shape)
        assert (i, k) == (1, 1)

    def test_marginal_and_sequential_run_on_the_host(self, cpu_proc):
        rng = np.random.RandomState(5)
        t, y, P, e, d = self._data(rng)
        v = np.sin(2 * np.pi * (t - t.min()) / 11.0)
        y_sys = y + 0.05 * v
        periods = np.array([3.0, P, 6.0])
        for detector, kw in (('sequential', {}),
                             ('marginal',
                              dict(coeff_prior_cov=np.array([[1.0]])))):
            snr, best = cpu_proc.run(t, y_sys, periods,
                                     durations=np.array([d]),
                                     detector=detector,
                                     systematics_basis=v[:, None], **kw)
            assert snr.shape == best.shape == (3, 1)
            assert np.all(np.isfinite(snr))
            assert int(np.argmax(snr[:, 0])) == 1, detector
            # best epoch is returned in the caller's (BJD) time scale
            assert best[1, 0] > 2457000.0
