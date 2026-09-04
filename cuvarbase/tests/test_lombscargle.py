import numpy as np
import pytest

from numpy.testing import assert_allclose
from astropy.timeseries import LombScargle

from ..lombscargle import LombScargleAsyncProcess
from pycuda.tools import mark_cuda_test
#import pycuda.autoinit
import pycuda.autoprimaryctx
spp = 3
nfac = 3
# Tolerances vs astropy / between GPU paths. Before the Sep-2026 NFFT
# fixes (psi tables shared between differently-sized grids, grids sized
# without k0) these had to be 1e-2 -- the default path carried a
# 3e-3..2e-2 bias. The fixed float32 path is at ~3e-6 on the problems in
# this file (measured on an A40; float32 ~2e-4 at survey-scale f*T, see
# TestLombScargleAccuracy), so 1e-4 is a 30x margin here and would have
# failed on the old code.
lsrtol = 1E-4
lsatol = 1E-4
nfft_sigma = 5

rand = np.random.RandomState(100)


def data(seed=100, sigma=0.1, ndata=100, freq=3.):
    t = np.sort(rand.rand(ndata))
    y = np.cos(2 * np.pi * freq * t)

    y += sigma * rand.randn(len(t))

    err = sigma * np.ones_like(y)

    return t, y, err


def assert_similar(pdg0, pdg, top=5):
    inds = (np.argsort(pdg0)[::-1])[:top]

    p0 = np.asarray(pdg0)[inds]
    p = np.asarray(pdg)[inds]
    diff = np.absolute(p - p0)

    res = sorted(zip(p0, p, diff), key=lambda x: -x[2])

    for p0v, pv, dv in res:
        if dv > 1e-3:
            print(p0v, pv, dv)

    assert_allclose(p, p0, atol=lsatol, rtol=lsrtol)
    assert(all(diff < lsrtol * 0.5 * (p + p0) + lsatol))


class TestLombScargle(object):
    def test_against_astropy_double(self):
        t, y, err = data()
        ls_proc = LombScargleAsyncProcess(use_double=True,
                                          sigma=nfft_sigma)

        results = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                              use_fft=True,
                              samples_per_peak=spp)
        ls_proc.finish()

        fgpu, pgpu = results[0]

        power = LombScargle(t, y, err).power(fgpu)

        assert_similar(power, pgpu)

    def test_against_astropy_single(self):
        t, y, err = data()
        ls_proc = LombScargleAsyncProcess(use_double=False,
                                          sigma=nfft_sigma)

        results = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                              samples_per_peak=spp)
        ls_proc.finish()
        fgpu, pgpu = results[0]

        power = LombScargle(t, y, err).power(fgpu)

        assert_similar(power, pgpu)

    def test_ls_kernel(self):
        t, y, err = data()
        ls_proc = LombScargleAsyncProcess(use_double=False,
                                          sigma=nfft_sigma)

        results = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                              samples_per_peak=spp)
        ls_proc.finish()
        fgpu, pgpu = results[0]

        ls = LombScargle(t, y, err, fit_mean=True, center_data=False)
        power = ls.power(fgpu)

        assert_similar(power, pgpu)

    def test_ls_kernel_direct_sums(self):
        t, y, err = data()
        ls_proc = LombScargleAsyncProcess(use_double=True,
                                          sigma=nfft_sigma)

        results = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                              samples_per_peak=spp, use_fft=False)
        ls_proc.finish()
        fgpu, pgpu = results[0]

        ls = LombScargle(t, y, err, fit_mean=True, center_data=True)
        power = ls.power(fgpu)

        assert_similar(power, pgpu)

    def test_ls_kernel_direct_sums_double_pi(self):
        """Regression test for the float32 PI literal in lomb.cu
        (Jul 2026 kernel-hygiene pass; same defect class as the
        cunfft.cu A3 fix). With a float32 PI, the double-precision
        direct-sums kernels evaluate the periodogram on a frequency
        axis stretched by 1 + 2.8e-8; at f*T ~ 3000 (f ~ 100 c/d,
        T = 30 d) that shows up as ~1e-4 absolute power errors against
        a float64 CPU port of the kernel. The fixed kernel matches the
        port to float64 roundoff (measured 1.1e-10 on an A5000; the
        buggy kernel measured 1.2e-4)."""
        T, n, f0 = 30.0, 200, 97.0
        rng = np.random.RandomState(7)
        t = np.sort(rng.rand(n)) * T + 4.5
        y = 0.3 * np.cos(2 * np.pi * f0 * t) + 12.0
        y += 0.1 * rng.randn(n)
        err = 0.1 * (0.8 + 0.4 * rng.rand(n))

        df = 1.0 / (5 * T)
        k0 = int(round(95.0 / df))
        freqs = df * (k0 + np.arange(600))

        ls_proc = LombScargleAsyncProcess(use_double=True,
                                          sigma=nfft_sigma)
        results = ls_proc.run([(t, y, err)], freqs=freqs, use_fft=False)
        ls_proc.finish()
        pgpu = np.asarray(results[0][1][:len(freqs)], dtype=np.float64)

        # float64 CPU port of lomb_dirsum (FLOATING_MEAN mode, exact
        # np.pi, same phase convention: phi = (t + 0.5) * f * 2 * pi).
        # run() mean-centers t and y in float64 first; mirror that.
        tc = np.asarray(t, dtype=np.float64) - np.nanmean(t)
        yc = np.asarray(y, dtype=np.float64) - np.nanmean(y)
        w = np.power(np.asarray(err, dtype=np.float64), -2)
        w /= np.sum(w)
        ybar = np.dot(w, yc)
        yw = w * (yc - ybar)
        YY = np.dot(w, (yc - ybar) ** 2)

        tp = tc + 0.5
        pref = np.empty(len(freqs))
        for i, f in enumerate(freqs):
            arg1 = tp * f * 2.0 * np.pi
            arg2 = tp * (2.0 * f) * 2.0 * np.pi
            C, S = np.dot(w, np.cos(arg1)), np.dot(w, np.sin(arg1))
            C2, S2 = np.dot(w, np.cos(arg2)), np.dot(w, np.sin(arg2))
            YCh, YSh = np.dot(yw, np.cos(arg1)), np.dot(yw, np.sin(arg1))
            tan2wt = (S2 - 2 * S * C) / (C2 - (C * C - S * S))
            C2w = 1.0 / np.sqrt(1.0 + tan2wt ** 2)
            S2w = tan2wt * C2w
            Cw = np.sqrt(0.5 * (1.0 + C2w))
            Sw = np.sqrt(0.5 * (1.0 - C2w)) * (-1.0 if S2w < 0 else 1.0)
            Cshft, Sshft = C * Cw + S * Sw, S * Cw - C * Sw
            CC = 0.5 * (1.0 + C2 * C2w + S2 * S2w) - Cshft ** 2
            SS = 0.5 * (1.0 - C2 * C2w - S2 * S2w) - Sshft ** 2
            YC, YS = YCh * Cw + YSh * Sw, YSh * Cw - YCh * Sw
            pref[i] = (YC * YC / CC + YS * YS / SS) / YY

        assert np.max(np.abs(pgpu - pref)) < 1e-7

    def test_ls_kernel_direct_sums_is_consistent(self):
        t, y, err = data()
        ls_proc = LombScargleAsyncProcess(use_double=False,
                                          sigma=nfft_sigma)

        results_ds = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                                 samples_per_peak=spp, use_fft=False)
        ls_proc.finish()

        fgpu_ds, pgpu_ds = results_ds[0]

        results_reg = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                                  samples_per_peak=spp, use_cpu_nfft=True)
        ls_proc.finish()

        fgpu_reg, pgpu_reg = results_reg[0]

        assert_similar(pgpu_reg, pgpu_ds)

    def test_ls_kernel_direct_sums_against_python(self):

        t, y, err = data()
        ls_proc = LombScargleAsyncProcess(use_double=False, sigma=nfft_sigma)

        result_ds = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                                samples_per_peak=spp, use_fft=False)
        ls_proc.finish()

        fgpu_ds, pgpu_ds = result_ds[0]

        result_reg = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                                 samples_per_peak=spp,
                                 use_fft=False,
                                 python_dir_sums=True)
        ls_proc.finish()
        fgpu_reg, pgpu_reg = result_reg[0]

        assert_similar(pgpu_reg, pgpu_ds)

    def test_multiple_datasets(self, ndatas=5):
        datas = [data() for i in range(ndatas)]
        ls_proc = LombScargleAsyncProcess(sigma=nfft_sigma)

        mult_results = ls_proc.run(datas, nyquist_factor=nfac,
                                   samples_per_peak=spp)
        ls_proc.finish()

        sing_results = []

        for d in datas:
            sing_results.extend(ls_proc.run([d], nyquist_factor=nfac,
                                samples_per_peak=spp))
            ls_proc.finish()

        for rb, rnb in zip(mult_results, sing_results):
            fb, pb = rb
            fnb, pnb = rnb

            assert_allclose(pnb, pb, rtol=lsrtol, atol=lsatol)
            assert_allclose(fnb, fb, rtol=lsrtol, atol=lsatol)

    def test_batched_run(self, ndatas=5, batch_size=5, sigma=nfft_sigma,
                         samples_per_peak=spp, nyquist_factor=nfac,
                         **kwargs):

        datas = [data(ndata=rand.randint(50, 100))
                 for i in range(ndatas)]
        ls_proc = LombScargleAsyncProcess(sigma=sigma, **kwargs)

        kw = dict(nyquist_factor=nyquist_factor,
                  samples_per_peak=samples_per_peak)

        batched_results = ls_proc.batched_run(datas, **kw)
        ls_proc.finish()

        non_batched_results = []
        for d in datas:
            r = ls_proc.run([d], nyquist_factor=nyquist_factor,
                            samples_per_peak=samples_per_peak)
            ls_proc.finish()
            non_batched_results.extend(r)

        for rb, rnb in zip(batched_results, non_batched_results):
            fb, pb = rb
            fnb, pnb = rnb

            assert_allclose(pnb, pb, rtol=lsrtol, atol=lsatol)
            assert_allclose(fnb, fb, rtol=lsrtol, atol=lsatol)

    def test_batched_run_const_nfreq(self, make_plot=False, ndatas=27,
                                     batch_size=5, sigma=nfft_sigma,
                                     samples_per_peak=spp,
                                     nyquist_factor=nfac,
                                     **kwargs):

        frequencies = 10 + rand.rand(ndatas) * 100.
        datas = [data(ndata=rand.randint(50, 100),
                      freq=freq)
                 for i, freq in enumerate(frequencies)]
        ls_proc = LombScargleAsyncProcess(sigma=sigma, **kwargs)

        kw = dict(samples_per_peak=spp,
                  batch_size=batch_size)
        kw.update(kwargs)
        batched_results = ls_proc.batched_run_const_nfreq(datas, **kw)
        ls_proc.finish()

        ls_procnb = LombScargleAsyncProcess(sigma=nfft_sigma,
                                            use_double=False, **kwargs)

        non_batched_results = []
        for d, (frq, p) in zip(datas, batched_results):
            r = ls_procnb.run([d], freqs=frq, **kwargs)
            ls_procnb.finish()
            non_batched_results.extend(r)

        # for f0, (fb, pb), (fnb, pnb) in zip(frequencies, batched_results,
        #                                    non_batched_results):
        #    print f0, fb[np.argmax(pb)], fnb[np.argmax(pnb)]

        for f0, (fb, pb), (fnb, pnb) in zip(frequencies, batched_results,
                                            non_batched_results):

            if make_plot:
                import matplotlib.pyplot as plt
                plt.plot(fnb, pnb, color='k', lw=3)
                plt.plot(fb, pb, color='r')
                plt.axvline(f0)
                plt.show()

            assert_allclose(pnb, pb, rtol=lsrtol, atol=lsatol)
            assert_allclose(fnb, fb, rtol=lsrtol, atol=lsatol)


def _realistic_lc(N=300, T=365.0, f0=3.1, seed=1):
    """Ground-based-like lightcurve: N points over T days, mag-scale
    y, heteroscedastic dy, one sinusoid at f0 (cycles/day)."""
    rng = np.random.RandomState(seed)
    t = np.sort(rng.rand(N)) * T
    dy = 0.1 * np.exp(0.5 * rng.randn(N))
    y = 12.0 + 0.3 * np.cos(2 * np.pi * f0 * t - 0.3) + dy * rng.randn(N)
    return t, y, dy


def _uniform_grid(fmin, fmax, T, samples_per_peak=5):
    """freqs = df * (k0 + arange(nf)) with df = 1 / (spp * T)."""
    df = 1.0 / (samples_per_peak * T)
    k0 = int(round(fmin / df))
    nf = int(round((fmax - fmin) / df))
    return df * (k0 + np.arange(nf))


def _exact_dft(t, c, freqs, chunk=4000):
    """sum_j c_j exp(2 pi i f t_j) in float64 (the adjoint NFFT's target)."""
    out = np.empty(len(freqs), dtype=complex)
    for a in range(0, len(freqs), chunk):
        ph = 2 * np.pi * np.outer(freqs[a:a + chunk], t)
        out[a:a + chunk] = (np.cos(ph) + 1j * np.sin(ph)) @ c
    return out


def _run_gpu(proc, t, y, dy, freqs, **kwargs):
    r = proc.run([(t, y, dy)], freqs=freqs, **kwargs)
    proc.finish()
    return np.asarray(r[0][1][:len(freqs)], dtype=np.float64)


class TestLombScargleAccuracy(object):
    """Accuracy of the default (NFFT) path against astropy's float64
    generalized Lomb-Scargle on realistic problem sizes.

    Regression tests for the Sep-2026 NFFT defects: (a) the w-spectrum
    grid reused the psi tables precomputed for the (2x smaller) yw grid,
    displacing every point's window by a fraction of a cell -- 3e-3 to
    2.4e-2 power bias on every default call, in float32 AND float64,
    independent of m (defect 3, ``nfft-psi-table``); (b) ``floorf()`` on
    the double-precision grid coordinate misplaced ~n0*ng/2^24 points by
    one cell, making ``use_double=True`` *less* accurate than float32 on
    dense grids (``nfft-floorf-double``). Measured on an A40 after the
    fixes: float32 1.9e-4 / double 4.3e-8 on the k0=1 grid below (both
    7.7e-3 before); double 2.0e-8 on the long-baseline grid (2.3e-3
    before, float32 1.8e-3 -> 1.6e-4).
    """

    @pytest.mark.parametrize("use_double,tol", [(False, 6e-4),
                                                (True, 1e-6)])
    def test_default_grid_vs_astropy(self, use_double, tol):
        t, y, dy = _realistic_lc()
        freqs = _uniform_grid(1.0 / (5 * 365.0), 20.0, 365.0)
        ref = LombScargle(t, y, dy).power(freqs)

        proc = LombScargleAsyncProcess(use_double=use_double, sigma=4,
                                       m=8, autoset_m=False)
        p = _run_gpu(proc, t, y, dy, freqs)

        assert np.max(np.abs(p - ref)) < tol
        assert np.argmax(p) == np.argmax(ref)

    @pytest.mark.parametrize("use_double,tol", [(False, 6e-4),
                                                (True, 1e-6)])
    def test_long_baseline_dense_grid_vs_astropy(self, use_double, tol):
        # 1000 points over 3 yr, 109,499 frequencies: the double path
        # hit the floorf() cell misplacement here (2.3e-3 before the
        # fix, i.e. worse than float32).
        t, y, dy = _realistic_lc(N=1000, T=1095.0, f0=2.7, seed=3)
        freqs = _uniform_grid(1.0 / (5 * 1095.0), 20.0, 1095.0)
        ref = LombScargle(t, y, dy).power(freqs, method='cython')

        proc = LombScargleAsyncProcess(use_double=use_double, sigma=4,
                                       m=8, autoset_m=False)
        p = _run_gpu(proc, t, y, dy, freqs)

        assert np.max(np.abs(p - ref)) < tol
        assert np.argmax(p) == np.argmax(ref)

    @pytest.mark.parametrize("use_double,tol", [(False, 5e-3),
                                                (True, 1e-6)])
    def test_device_spectra_match_exact_dft(self, use_double, tol):
        # Read the two NFFT spectra straight off the device memory and
        # compare with the exact float64 adjoint DFT. The w-spectrum
        # (nfft_mem_w.ghat_g, modes k0 .. 2 nf + k0 - 1) was off by 0.15
        # with the shared psi tables; the yw-spectrum was always fine.
        from ..lombscargle import get_k0
        from ..utils import normalize_light_curves

        t, y, dy = _realistic_lc()
        freqs = _uniform_grid(1.0 / (5 * 365.0), 20.0, 365.0)
        nf, k0, df = len(freqs), get_k0(freqs), freqs[1] - freqs[0]

        # run() centres t and y on the host; mirror that for the DFT
        (tn, yn, dyn), = normalize_light_curves([(t, y, dy)])
        w = dyn ** -2
        w /= np.sum(w)
        yw = w * (yn - np.dot(w, yn))

        proc = LombScargleAsyncProcess(use_double=use_double, sigma=4,
                                       m=8, autoset_m=False)
        mem = proc.allocate([(tn, yn, dyn)], nfreqs=[nf], k0s=[k0])
        _run_gpu(proc, t, y, dy, freqs, memory=mem)

        sw = mem[0].nfft_mem_w.ghat_g.get()
        syw = mem[0].nfft_mem_yw.ghat_g.get()
        n_w = 2 * nf + k0
        assert len(sw) >= n_w and len(syw) >= nf

        sw_exact = _exact_dft(tn, w, (k0 + np.arange(n_w)) * df)
        syw_exact = _exact_dft(tn, yw, (k0 + np.arange(nf)) * df)
        # sum(w) == 1, so these are absolute errors on a unit scale
        assert np.max(np.abs(sw[:n_w] - sw_exact)) < tol
        assert np.max(np.abs(syw[:nf] - syw_exact)) < tol


class TestLombScargleSimpleWeights(object):
    """Regression tests for lomb_scargle_simple's weight handling.

    The function used to pre-normalize dy**-2 and pass the result in the
    dy slot of run(); LombScargleMemory.setdata then applied the
    inverse-variance conversion AGAIN, producing effective weights
    proportional to dy^4 -- the largest-error points got the MOST weight.
    lomb_scargle_simple must pass raw dy straight through.
    """

    def test_weights_helper_is_inverse_variance(self):
        from ..memory.lombscargle_memory import weights
        dy = np.array([0.1, 0.2, 0.4])
        w = weights(dy)
        expected = (dy ** -2) / np.sum(dy ** -2)
        assert_allclose(w, expected, rtol=1e-6)
        assert_allclose(w, [0.76190476, 0.19047619, 0.04761905], rtol=1e-5)
        # double application inverts the ordering (the old bug)
        w2 = weights(weights(dy))
        assert np.argmax(w2) == np.argmax(dy)  # largest error dominates
        assert np.argmax(w) == np.argmin(dy)   # correct: smallest error

    def test_lomb_scargle_simple_passes_raw_dy(self, monkeypatch):
        from .. import lombscargle as ls
        dy = np.array([0.1, 0.2, 0.4])
        t = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        captured = {}

        def fake_run(self, data, **kwargs):
            captured['data'] = data
            return [(np.array([1.0]), np.array([0.5]))]

        monkeypatch.setattr(ls.LombScargleAsyncProcess, 'run', fake_run)
        ls.lomb_scargle_simple(t, y, dy)

        passed_dy = captured['data'][0][2]
        assert_allclose(passed_dy, dy)  # raw uncertainties, not weights


class TestFapBaluev(object):
    """fap_baluev must not underflow to exactly 0 for significant
    peaks (issue #14): for z near 1, both (1 - z)**(0.5 * N_K) and
    exp(-tau) round to 1.0 and the final subtraction cancels
    catastrophically.
    """

    def setup_method(self):
        rand = np.random.RandomState(42)
        self.t = np.sort(365 * rand.rand(100))
        self.dy = 0.01 * (1 + 0.1 * rand.rand(100))
        self.fmax = 10.0

    def _fap_naive(self, t, dy, z, fmax, d_K=3, d_H=1):
        # Direct evaluation of Baluev (2008); valid away from the
        # z -> 1 underflow regime. Mirrors the pre-fix implementation.
        from scipy.special import gammaln
        N = len(t)
        d = d_K - d_H
        N_K = N - d_K
        N_H = N - d_H
        g = np.exp(gammaln(0.5 * N_H) - gammaln(0.5 * (N_K + 1)))
        w = np.power(dy, -2)
        tbar = np.dot(w, t) / sum(w)
        Dt = np.dot(w, np.power(t - tbar, 2)) / sum(w)
        Teff = np.sqrt(4 * np.pi * Dt)
        A = (2 * np.pi ** 1.5) * fmax * Teff
        eZ1 = (z / np.pi) ** 0.5 * (d - 1)
        eZ2 = (1 - z) ** (0.5 * (N_K - 1))
        tau = (g * A / (2 * np.pi)) * eZ1 * eZ2
        Psing = 1 - (1 - z) ** (0.5 * N_K)
        return 1 - Psing * np.exp(-tau)

    def test_matches_naive_formula_at_moderate_z(self):
        from ..lombscargle import fap_baluev
        # The naive formula computes FAP as 1 - (1 - tiny), so its own
        # precision is only ~1e-16/FAP relative; compare strictly where
        # the reference itself is accurate, loosely at FAP ~ 1e-11.
        z = np.array([0.05, 0.1, 0.2, 0.3])
        fap = fap_baluev(self.t, self.dy, z, self.fmax)
        ref = self._fap_naive(self.t, self.dy, z, self.fmax)
        assert_allclose(fap, ref, rtol=1e-8)

        z = np.array([0.5])
        fap = fap_baluev(self.t, self.dy, z, self.fmax)
        ref = self._fap_naive(self.t, self.dy, z, self.fmax)
        assert_allclose(fap, ref, rtol=1e-4)

    def test_no_underflow_to_zero_for_significant_peaks(self):
        from ..lombscargle import fap_baluev
        # N=100 -> N_K=97; z=0.95 gives FAP ~ 1e-59: representable in
        # float64, but the naive formula returns exactly 0.0
        fap = fap_baluev(self.t, self.dy, np.array([0.95, 0.99]),
                         self.fmax)
        assert np.all(fap > 0)
        assert np.all(fap < 1e-20)

    def test_monotonically_decreasing_in_z(self):
        from ..lombscargle import fap_baluev
        z = np.linspace(0.01, 0.995, 200)
        fap = fap_baluev(self.t, self.dy, z, self.fmax)
        assert np.all(np.diff(fap) <= 0)
        assert np.all(fap > 0)

    def test_z_edge_cases(self):
        from ..lombscargle import fap_baluev
        fap = fap_baluev(self.t, self.dy, np.array([0.0, 1.0]),
                         self.fmax)
        assert fap[0] == pytest.approx(1.0)
        assert fap[1] >= 0.0


class _FakePtr(object):
    ptr = 0


class _FakeKernel(object):
    def __init__(self):
        self.calls = []

    def prepared_async_call(self, *args):
        self.calls.append(args)


class _FakeLSMemory(object):
    """Minimal stand-in for LombScargleMemory: just enough attributes
    for the use_fft=False (direct sums) branch of lomb_scargle_async."""

    def __init__(self, freqs):
        from ..lombscargle import get_k0
        self.tmin, self.tmax = 0.0, 100.0
        self.k0 = get_k0(freqs)
        self.stream = None
        self.nf = len(freqs)
        self.n0 = 50
        self.real_type = np.float32
        self.yy = 1.0
        self.ybar = 0.0
        self.mode = np.int32(0)
        self.t_g = _FakePtr()
        self.yw_g = _FakePtr()
        self.w_g = _FakePtr()
        self.lsp_g = _FakePtr()
        self.reg_g = _FakePtr()
        self.lsp_c = np.zeros(len(freqs), dtype=np.float32)
        self.n_gpu_transfers = 0
        self.n_lsp_transfers = 0

    def transfer_data_to_gpu(self):
        self.n_gpu_transfers += 1

    def transfer_lsp_to_cpu(self):
        self.n_lsp_transfers += 1


class TestLombScargleAsyncGating(object):
    """Argument-gating bugs in the module-level lomb_scargle_async:
    the direct-sums branch used to key the host transfer on
    transfer_to_device, and use_cufinufft=True was silently ignored
    when cufinufft was missing."""

    def _setup(self):
        df = 0.01
        freqs = df * (1 + np.arange(64))
        memory = _FakeLSMemory(freqs)
        functions = ((_FakeKernel(), _FakeKernel()), None)
        return freqs, memory, functions

    def test_dirsums_transfer_to_host_true_copies(self):
        from ..lombscargle import lomb_scargle_async
        freqs, memory, functions = self._setup()
        lomb_scargle_async(memory, functions, freqs, use_fft=False,
                           transfer_to_device=False,
                           transfer_to_host=True)
        assert memory.n_gpu_transfers == 0
        assert memory.n_lsp_transfers == 1

    def test_dirsums_transfer_to_host_false_suppresses_copy(self):
        from ..lombscargle import lomb_scargle_async
        freqs, memory, functions = self._setup()
        lomb_scargle_async(memory, functions, freqs, use_fft=False,
                           transfer_to_device=True,
                           transfer_to_host=False)
        assert memory.n_gpu_transfers == 1
        assert memory.n_lsp_transfers == 0

    def test_use_cufinufft_without_cufinufft_raises(self, monkeypatch):
        from .. import lombscargle as ls
        monkeypatch.setattr(ls, 'HAS_CUFINUFFT', False)
        freqs, memory, functions = self._setup()
        with pytest.raises(ImportError, match="cufinufft"):
            ls.lomb_scargle_async(memory, functions, freqs,
                                  use_fft=False, use_cufinufft=True)


class TestCufinufftPlanCache(object):
    """cufinufft Plans were created (and never destroyed) on every
    call — the dominant cost that made the backend slower than the
    custom NFFT. Plans must be cached per problem shape."""

    class _FakePlan(object):
        instances = []

        def __init__(self, **kwargs):
            type(self).instances.append(kwargs)
            self.setpts_calls = 0

        def setpts(self, x):
            self.setpts_calls += 1

        def execute(self, c, out):
            out[:] = 0

    class _FakeNFFTMemory(object):
        def __init__(self, ndata=64, nf=32):
            rand = np.random.RandomState(2)
            self.t_g = np.sort(rand.rand(ndata)).astype(np.float32)
            self.y_g = rand.randn(ndata).astype(np.float32)
            self.tmin = float(self.t_g.min())
            self.tmax = float(self.t_g.max())
            self.nf = nf
            self.ghat_g = np.zeros(nf, dtype=np.complex64)
            self.ghat_c = np.zeros(nf, dtype=np.complex64)

        def transfer_data_to_gpu(self):
            pass

        def transfer_nfft_to_cpu(self):
            pass

    def _patched_backend(self, monkeypatch):
        import types
        from .. import cufinufft_backend as cb
        self._FakePlan.instances = []
        monkeypatch.setattr(cb, 'HAS_CUFINUFFT', True)
        monkeypatch.setattr(cb, 'cufinufft',
                            types.SimpleNamespace(Plan=self._FakePlan),
                            raising=False)
        monkeypatch.setattr(cb, 'gpuarray',
                            types.SimpleNamespace(zeros=np.zeros))
        cb.free_plan_cache()
        return cb

    def test_plan_reused_for_same_shape(self, monkeypatch):
        cb = self._patched_backend(monkeypatch)
        mem = self._FakeNFFTMemory()
        cb.cufinufft_nfft_adjoint(mem, transfer_to_device=False,
                                  transfer_to_host=False)
        cb.cufinufft_nfft_adjoint(mem, transfer_to_device=False,
                                  transfer_to_host=False)
        assert len(self._FakePlan.instances) == 1
        cb.free_plan_cache()

    def test_new_plan_for_different_shape(self, monkeypatch):
        cb = self._patched_backend(monkeypatch)
        cb.cufinufft_nfft_adjoint(self._FakeNFFTMemory(nf=32),
                                  transfer_to_device=False,
                                  transfer_to_host=False)
        cb.cufinufft_nfft_adjoint(self._FakeNFFTMemory(nf=64),
                                  transfer_to_device=False,
                                  transfer_to_host=False)
        assert len(self._FakePlan.instances) == 2
        cb.free_plan_cache()

    def test_cache_eviction_bounded(self, monkeypatch):
        cb = self._patched_backend(monkeypatch)
        for nf in 16 * (1 + np.arange(cb._PLAN_CACHE_MAX_SIZE + 3)):
            cb.cufinufft_nfft_adjoint(self._FakeNFFTMemory(nf=int(nf)),
                                      transfer_to_device=False,
                                      transfer_to_host=False)
        assert len(cb._plan_cache) == cb._PLAN_CACHE_MAX_SIZE
        cb.free_plan_cache()
        assert len(cb._plan_cache) == 0
