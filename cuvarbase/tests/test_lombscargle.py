import numpy as np
import pytest

from numpy.testing import assert_allclose
from astropy.timeseries import LombScargle

from ..lombscargle import LombScargleAsyncProcess
# NOT `import pycuda.autoprimaryctx`/`autoinit` here: cuvarbase retains
# the primary context itself, lazily (cuvarbase.base.ensure_context).
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


class TestLombScargleNarrowBands(object):
    """Frequency grids that do not start near zero (defect 4,
    ``nfft-k0-size`` / ``ls-grid-near-zero``, Sep 2026).

    The NFFT grids were sized ``sigma * nf`` while the ``lomb`` kernel
    reads modes ``k0 .. k0 + nf - 1`` (and ``2 k0 .. 2 (k0 + nf - 1)``
    from the w-spectrum), so the top mode sat at fraction
    ``(k0 + nf) / (sigma nf)`` of the grid and crossed the Gaussian
    window's alias-free limit at ``k0 = nf``: any band with
    ``fmin >= ~fmax / 2`` -- ``run(minimum_frequency=20,
    maximum_frequency=30)``, say -- returned powers of 1e4..1e36 with a
    wrong best frequency, through every public entry point. Grids are
    now sized from the top mode; measured on an A40 after the fix the
    bands below agree with astropy to <= 8.7e-4 (float32) and <= 1.4e-7
    (double), the ordinary m = 8 truncation level.
    """

    T = 365.0

    # (fmin, fmax) with k0 / nf = 1.2, 2, 4
    bands = [(1.2, 2.2), (2.0, 3.0), (4.0, 5.0)]

    def _case(self, fmin, fmax):
        f0 = fmin + 0.9 * (fmax - fmin)     # signal near the top of the band
        t, y, dy = _realistic_lc(N=300, T=self.T, f0=f0, seed=3)
        freqs = _uniform_grid(fmin, fmax, self.T)
        ref = LombScargle(t, y, dy).power(freqs)
        return t, y, dy, freqs, ref

    @pytest.mark.parametrize("band", bands)
    @pytest.mark.parametrize("use_double,tol", [(False, 1e-3),
                                                (True, 1e-6)])
    def test_band_vs_astropy(self, band, use_double, tol):
        from ..lombscargle import get_k0
        t, y, dy, freqs, ref = self._case(*band)
        k0, nf = get_k0(freqs), len(freqs)
        assert k0 >= 1.1 * nf      # this really is a narrow band

        proc = LombScargleAsyncProcess(use_double=use_double)
        p = _run_gpu(proc, t, y, dy, freqs)

        assert np.max(np.abs(p - ref)) < tol
        assert np.argmax(p) == np.argmax(ref)

    def test_run_with_minimum_maximum_frequency(self):
        # documented kwargs path -> autofrequency grid, k0/nf = 2
        t, y, dy, _, _ = self._case(20.0, 30.0)
        proc = LombScargleAsyncProcess()
        r = proc.run([(t, y, dy)], minimum_frequency=20.0,
                     maximum_frequency=30.0)
        proc.finish()
        freqs, p = r[0]
        p = np.asarray(p[:len(freqs)], dtype=np.float64)
        ref = LombScargle(t, y, dy).power(freqs)
        assert np.max(np.abs(p - ref)) < 2e-3
        assert np.argmax(p) == np.argmax(ref)

    def test_lomb_scargle_simple_on_band(self):
        from ..lombscargle import lomb_scargle_simple
        t, y, dy, freqs, ref = self._case(20.0, 30.0)
        f, p = lomb_scargle_simple(t, y, dy, freqs=freqs)
        p = np.asarray(p[:len(freqs)], dtype=np.float64)
        assert np.max(np.abs(p - ref)) < 2e-3
        assert np.argmax(p) == np.argmax(ref)

    def test_batched_best_freq_on_band(self):
        t, y, dy, freqs, ref = self._case(20.0, 30.0)
        proc = LombScargleAsyncProcess()
        best_freqs, _ = proc.batched_run_const_nfreq(
            [(t, y, dy)], freqs=freqs, only_return_best_freqs=True)
        assert best_freqs[0] == pytest.approx(freqs[np.argmax(ref)])

    def test_small_grid_far_from_zero(self):
        # nf = 8 at k0 = 50 used to return the -1 sentinel everywhere
        t, y, dy, _, _ = self._case(20.0, 30.0)
        df = 1.0 / (5 * self.T)
        freqs = df * (50 + np.arange(8))
        ref = LombScargle(t, y, dy).power(freqs)
        proc = LombScargleAsyncProcess()
        p = _run_gpu(proc, t, y, dy, freqs)
        assert np.all(p >= 0)
        assert np.max(np.abs(p - ref)) < 1e-3

    def test_sigma_below_3_raises(self):
        # sigma = 2 leaves the top of every band aliased even with the
        # grids sized from the top mode
        t, y, dy, freqs, _ = self._case(2.0, 3.0)
        proc = LombScargleAsyncProcess(sigma=2)
        with pytest.raises(ValueError, match="sigma"):
            proc.run([(t, y, dy)], freqs=freqs)


class TestNFFTGridChecks(object):
    """CPU tests of the grid-sizing helper and the hard check that
    ``lomb_scargle_async`` applies before touching the NFFT memories."""

    def test_nfft_grid_sizes_cover_top_mode(self):
        from ..memory.lombscargle_memory import nfft_grid_sizes
        from ..memory.nfft_memory import next_fast_len
        for H in (1, 2, 3):
            for k0, nf in [(1, 100), (50, 8), (1000, 500), (36038, 18020)]:
                for sigma in (3, 4, 5):
                    nf_yw, n_yw, nf_w, n_w = nfft_grid_sizes(
                        nf, k0, nharmonics=H, sigma=sigma)
                    # every spectrum entry the kernels read exists
                    assert (H - 1) * k0 + H * (nf - 1) < nf_yw
                    assert (2 * H - 1) * k0 + 2 * H * (nf - 1) < nf_w
                    # grids sized from the top mode, 7-smooth
                    assert n_yw >= sigma * (k0 + nf_yw)
                    assert n_w >= sigma * (k0 + nf_w)
                    assert next_fast_len(n_yw) == n_yw
                    assert next_fast_len(n_w) == n_w

    def _fake_memory(self, nf_yw, n_yw, nf_w, n_w, sigma=4):
        import types
        mk = lambda nf, n: types.SimpleNamespace(nf=nf, n=n, sigma=sigma,
                                                 ghat_g=np.zeros(n))
        return types.SimpleNamespace(nfft_mem_yw=mk(nf_yw, n_yw),
                                     nfft_mem_w=mk(nf_w, n_w))

    def test_check_passes_for_correctly_sized_grids(self):
        from ..lombscargle import _check_nfft_grids
        from ..memory.lombscargle_memory import nfft_grid_sizes
        nf, k0, H = 500, 1000, 2
        mem = self._fake_memory(*nfft_grid_sizes(nf, k0, H, 4))
        _check_nfft_grids(mem, nf, k0, H)

    def test_check_rejects_old_sizing(self):
        # the pre-1.0 allocation: sigma * count, k0 shaved off
        from ..lombscargle import _check_nfft_grids
        nf, k0, sigma = 500, 1000, 4
        fft_size = nf + k0
        mem = self._fake_memory(fft_size - k0, sigma * (fft_size - k0),
                                2 * fft_size - k0,
                                sigma * (2 * fft_size - k0))
        with pytest.raises(ValueError, match="too short"):
            _check_nfft_grids(mem, nf, k0, 1)

    def test_check_rejects_memory_for_smaller_grid(self):
        from ..lombscargle import _check_nfft_grids
        from ..memory.lombscargle_memory import nfft_grid_sizes
        mem = self._fake_memory(*nfft_grid_sizes(100, 10, 1, 4))
        with pytest.raises(ValueError, match="different frequency grid"):
            _check_nfft_grids(mem, 200, 10, 1)


def _two_harmonic_lc(seed=7, N=250, T=80.0, f0=0.9):
    """Strongly non-sinusoidal signal so that H = 1 and H = 2, 3 differ."""
    rng = np.random.RandomState(seed)
    t = np.sort(rng.rand(N)) * T
    y = (10 + 0.4 * np.sin(2 * np.pi * f0 * t)
         + 0.4 * np.sin(2 * np.pi * 2 * f0 * t + 1.0) + 0.05 * rng.randn(N))
    dy = 0.05 * np.ones(N)
    df = 1.0 / (5 * (t.max() - t.min()))
    freqs = df * (5 + np.arange(600))
    return t, y, dy, freqs


def _mh_reference(t, y, dy, freqs, H, **kwargs):
    from ..lombscargle import lomb_scargle_direct_sums
    w = dy ** -2
    w /= np.sum(w)
    ybar = np.dot(w, y)
    YY = np.dot(w, (y - ybar) ** 2)
    return lomb_scargle_direct_sums(t, w * y, w, freqs, YY, nharms=H,
                                    **kwargs)


class TestMultiharmonicDirectSums(object):
    """``nharmonics > 1`` with ``use_fft=False`` / ``python_dir_sums=True``
    (defect 13, ``ls-nharmonics-nofft``, Sep 2026): both returned the
    H = 1 periodogram (equal to the H = 1 reference to 1e-14) because
    the direct-sum kernel forms only the H = 1 moments and the host
    solve sat on the NFFT branch. They now run the float64 host
    multiharmonic direct sums."""

    @pytest.mark.parametrize("H", [2, 3])
    @pytest.mark.parametrize("use_double,tol", [(False, 2e-3),
                                                (True, 1e-10)])
    @pytest.mark.parametrize("python_dir_sums", [False, True])
    def test_matches_multiharmonic_reference(self, H, use_double, tol,
                                             python_dir_sums):
        t, y, dy, freqs = _two_harmonic_lc()
        ref_H = _mh_reference(t, y, dy, freqs, H)
        ref_1 = _mh_reference(t, y, dy, freqs, 1)
        assert np.max(np.abs(ref_H - ref_1)) > 0.3     # the signal is not a sinusoid

        proc = LombScargleAsyncProcess(use_double=use_double, nharmonics=H)
        p = _run_gpu(proc, t, y, dy, freqs, use_fft=False,
                     python_dir_sums=python_dir_sums)

        assert np.max(np.abs(p - ref_H)) < tol
        assert np.max(np.abs(p - ref_1)) > 0.3

    def test_batched_const_nfreq_direct_sums(self):
        t, y, dy, freqs = _two_harmonic_lc()
        ref_2 = _mh_reference(t, y, dy, freqs, 2)
        proc = LombScargleAsyncProcess(use_double=True, nharmonics=2)
        (f, p), = proc.batched_run_const_nfreq([(t, y, dy)], freqs=freqs,
                                               use_fft=False)
        assert np.max(np.abs(np.asarray(p, dtype=np.float64) - ref_2)) < 1e-10

    @pytest.mark.parametrize("kwargs", [dict(window=True),
                                        dict(floating_mean=False)])
    def test_non_floating_mean_raises_for_H_gt_1(self, kwargs):
        t, y, dy, freqs = _two_harmonic_lc()
        proc = LombScargleAsyncProcess(nharmonics=2)
        with pytest.raises(ValueError, match="floating-mean"):
            proc.run([(t, y, dy)], freqs=freqs, **kwargs)
        with pytest.raises(ValueError, match="floating-mean"):
            proc.run([(t, y, dy)], freqs=freqs, use_fft=False, **kwargs)


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
        # >= _LS_MIN_NDATA points: lomb_scargle_simple validates the
        # light curve before forwarding it (Sep 2026 audit, defect 23)
        dy = np.array([0.1, 0.2, 0.4, 0.3, 0.15])
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
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


class TestCufinufftBackendOnDevice(object):
    """The real cuFINUFFT backend against the built-in NFFT backend on
    a device (release finding 74: only the fake-Plan tests above ran in
    the suite; the cross-check lived in
    ``scripts/benchmark_new_features.py --tests-only``). Guarded by
    ``importorskip('cufinufft')`` so the zero-skip gate policy covers
    it; on CPU-only hosts it skips at the import."""

    @staticmethod
    def _sinusoid(ndata, baseline, period, seed, amplitude=0.01,
                  noise=0.002):
        rng = np.random.RandomState(seed)
        t = np.sort(rng.uniform(0, baseline, ndata)).astype(np.float32)
        y = amplitude * np.cos(2 * np.pi * t / period).astype(np.float32)
        y += rng.randn(ndata).astype(np.float32) * noise
        dy = np.full(ndata, noise, dtype=np.float32)
        return t, y, dy

    @pytest.mark.parametrize("ndata,nfreq,period", [
        (1000, 5000, 5.0), (5000, 10000, 3.0)])
    def test_cufinufft_matches_builtin_nfft(self, ndata, nfreq, period):
        pytest.importorskip('cufinufft')
        from ..cufinufft_backend import HAS_CUFINUFFT
        assert HAS_CUFINUFFT
        fmax = 2.0
        df = fmax / nfreq
        freqs = (np.arange(1, nfreq + 1) * df).astype(np.float32)
        for seed in (100, 101):
            t, y, dy = self._sinusoid(ndata, 365.0, period, seed)
            proc = LombScargleAsyncProcess(use_cufinufft=False)
            _, p_builtin = proc.run([(t, y, dy)], freqs=[freqs])[0]
            proc.finish()
            proc = LombScargleAsyncProcess(use_cufinufft=True)
            _, p_cufi = proc.run([(t, y, dy)], freqs=[freqs])[0]
            proc.finish()
            p_builtin = np.asarray(p_builtin, dtype=np.float64)
            p_cufi = np.asarray(p_cufi, dtype=np.float64)
            assert p_cufi.shape == p_builtin.shape == freqs.shape
            assert np.all(np.isfinite(p_cufi))
            # the benchmark script's acceptance: corr > 0.9999, max abs
            # diff < 0.01 (power is in [0, 1]), peaks within 2 df
            assert np.corrcoef(p_builtin, p_cufi)[0, 1] > 0.9999
            assert np.max(np.abs(p_builtin - p_cufi)) < 0.01
            peak_b = freqs[np.argmax(p_builtin)]
            peak_c = freqs[np.argmax(p_cufi)]
            assert abs(peak_b - peak_c) < 2 * df
            assert abs(peak_b - 1.0 / period) < 2 * df


class TestAmplitudePrior(object):
    """``amplitude_prior`` on the multiharmonic NFFT path (defect 14,
    ``ls-amplitude-prior``, Sep 2026): ``_mh_power_from_spectra`` was
    called without ``reg_kwargs``, so H > 1 silently returned the
    UNregularized power (0.9 away from the ridge reference on this
    data). The prior is the standard deviation of a Gaussian prior on
    the amplitudes, i.e. a ridge term 1 / s**2 (``add_regularization``).
    Measured on an A40 after the fix: 1.1e-5 (float32) / 1.8e-8
    (double) vs the float64 regularized direct sums."""

    s = 0.3

    @pytest.mark.parametrize("H", [2, 3])
    @pytest.mark.parametrize("use_double,tol", [(False, 1e-4),
                                                (True, 1e-7)])
    def test_nfft_path_matches_regularized_reference(self, H, use_double,
                                                     tol):
        t, y, dy, freqs = _two_harmonic_lc()
        ref_reg = _mh_reference(t, y, dy, freqs, H, amplitude_priors=self.s)
        ref_unreg = _mh_reference(t, y, dy, freqs, H)
        assert np.max(np.abs(ref_reg - ref_unreg)) > 0.5

        proc = LombScargleAsyncProcess(use_double=use_double, nharmonics=H)
        p = _run_gpu(proc, t, y, dy, freqs, amplitude_prior=self.s)

        assert np.max(np.abs(p - ref_reg)) < tol
        assert np.max(np.abs(p - ref_unreg)) > 0.5

    def test_single_harmonic_kernel_path(self):
        # H = 1 goes through reg_g in the lomb kernel (was already right)
        t, y, dy, freqs = _two_harmonic_lc()
        ref_reg = _mh_reference(t, y, dy, freqs, 1, amplitude_priors=self.s)
        proc = LombScargleAsyncProcess(nharmonics=1)
        p = _run_gpu(proc, t, y, dy, freqs, amplitude_prior=self.s)
        assert np.max(np.abs(p - ref_reg)) < 1e-5

    def test_direct_sums_path(self):
        t, y, dy, freqs = _two_harmonic_lc()
        ref_reg = _mh_reference(t, y, dy, freqs, 2, amplitude_priors=self.s)
        proc = LombScargleAsyncProcess(use_double=True, nharmonics=2)
        p = _run_gpu(proc, t, y, dy, freqs, amplitude_prior=self.s,
                     use_fft=False)
        assert np.max(np.abs(p - ref_reg)) < 1e-10


class TestCheckK0(object):
    """``check_k0`` must reject every grid the kernels cannot evaluate
    (defect 15, ``ls-nonuniform-grid``, Sep 2026): before 1.0 only
    ``freqs[0:2]`` were inspected, so concatenated / thinned grids
    passed and were silently evaluated on the implied uniform grid
    (corr 0.009 with astropy at the user's labels). CPU-only."""

    @staticmethod
    def _grid(k0=50, nf=600, T=100.0, spp=5):
        df = 1.0 / (spp * T)
        return df * (k0 + np.arange(nf))

    def test_concatenated_segments_raise_naming_the_junction(self):
        from ..lombscargle import check_k0
        fu = np.concatenate([np.arange(0.1, 1.0, 0.002),
                             np.arange(1.0, 5.0, 0.01)])
        with pytest.raises(ValueError,
                           match=r"not uniformly spaced.*freqs\[451\] - "
                                 r"freqs\[450\]"):
            check_k0(fu)

    def test_deleted_points_raise_naming_the_gap(self):
        from ..lombscargle import check_k0
        fdel = np.delete(self._grid(), np.arange(100, 120))
        with pytest.raises(ValueError,
                           match=r"freqs\[100\] - freqs\[99\]"):
            check_k0(fdel)

    def test_auditor_grid_uniform_for_two_points_raises(self):
        from ..lombscargle import check_k0
        f = self._grid()
        fb = f.copy()
        fb[2:] = f[2] + 3 * (f[2:] - f[2])
        with pytest.raises(ValueError, match="not uniformly spaced"):
            check_k0(fb)

    def test_fractional_first_mode_raises(self):
        from ..lombscargle import check_k0
        # linspace(0.1, 10, 50001): df = 1.98e-4, freqs[0] / df = 505.05
        with pytest.raises(ValueError, match="not an integer multiple"):
            check_k0(np.linspace(0.1, 10.0, 50001))
        f = self._grid()
        df = f[1] - f[0]
        with pytest.raises(ValueError, match="not an integer multiple"):
            check_k0(f + 0.3 * df)
        # 1e-3 of a mode used to pass the old 1 % tolerance
        with pytest.raises(ValueError, match="not an integer multiple"):
            check_k0(f + 1e-3 * df)

    def test_descending_short_and_geomspace_raise(self):
        from ..lombscargle import check_k0, get_k0
        f = self._grid()
        with pytest.raises(ValueError, match="strictly increasing"):
            check_k0(f[::-1])
        with pytest.raises(ValueError, match="at least two"):
            check_k0(f[:1])
        with pytest.raises(ValueError, match="at least two"):
            get_k0(f[:1])
        with pytest.raises(ValueError):
            check_k0(np.geomspace(0.1, 10.0, 1000))

    def test_valid_grids_pass(self):
        from ..lombscargle import check_k0, get_k0
        from ..utils import autofrequency
        rng = np.random.RandomState(1)
        t = np.sort(rng.uniform(0, 100.0, 600))
        for f, k0 in [(autofrequency(t), 1),
                      (autofrequency(t, minimum_frequency=2.0,
                                     maximum_frequency=3.0), None),
                      (np.linspace(0.1, 10.0, 991), 10),
                      (self._grid(), 50),
                      # float64 rounding at large k0 must not trip the
                      # k0 test (k0**2 eps = 3e-5 modes with the naive
                      # f[1] - f[0] spacing)
                      ((1.0 / (5 * 3650.0)) * (365000 + np.arange(10)),
                       365000),
                      ((1.0 / (5 * 3650.0)) * (365000 + np.arange(1000)),
                       365000),
                      # float32 grids of moderate size
                      (self._grid().astype(np.float32), 50),
                      (self._grid(k0=1000, nf=10000).astype(np.float32),
                       1000),
                      (list(self._grid()), 50)]:
            check_k0(f)
            if k0 is not None:
                assert get_k0(f) == k0

    def test_float32_survey_grid_fractional_first_mode_raises(self):
        # The freqs[0] term of the tolerance must use the rounding of
        # freqs[0] itself (eps * |f[0]|), not eps * max|f|: with the
        # latter a float32 survey-scale grid (10 yr baseline, 5 samples
        # per peak, nf ~ 9e5) accepted a first mode fractional by up to
        # ~0.43 df, and the kernels then evaluated a band shifted off
        # the user's labels (0.52 relative power error at 0.1 df).
        from ..lombscargle import check_k0
        df = 1.0 / (5 * 3650.0)
        f64 = df * (1 + np.arange(912500))
        for offset in (0.02, 0.05, 0.1, 0.3, 0.5):
            for dtype in (np.float32, np.float64):
                grid = (f64 + offset * df).astype(dtype)
                with pytest.raises(ValueError,
                                   match="not an integer multiple"):
                    check_k0(grid)

    def test_survey_scale_grids_of_two_million_points_pass(self):
        # ... and the tightened bound must not reject any grid a user
        # would actually build, in either precision.
        from ..lombscargle import check_k0, get_k0
        from ..utils import autofrequency
        nf = 2000000
        rng = np.random.RandomState(7)
        t = np.sort(rng.uniform(0, 3650.0, 4000))
        auto = autofrequency(t, maximum_frequency=120.0)
        assert len(auto) > nf
        grids = [('autofrequency', auto, None)]
        for df, k0 in ((1.0 / (5 * 3650.0), 1),
                       (1.0 / (5 * 365.0), 100)):
            f0, f1 = df * k0, df * (k0 + nf - 1)
            grids += [('arange*df', df * (k0 + np.arange(nf)), k0),
                      ('arange', np.arange(k0, k0 + nf) * df, k0),
                      ('linspace', np.linspace(f0, f1, nf), k0)]
        for name, grid, k0 in grids:
            for dtype in (np.float64, np.float32):
                g = grid.astype(dtype)
                check_k0(g)                      # must not raise
                if k0 is not None:
                    assert get_k0(g) == k0, name

    def test_float32_grid_that_is_really_nonuniform_raises(self):
        from ..lombscargle import check_k0
        f = self._grid(k0=50, nf=600).astype(np.float32)
        f[300:] += np.float32(0.05 * (f[1] - f[0]))
        with pytest.raises(ValueError, match="not uniformly spaced"):
            check_k0(f)


class TestRunGridValidation(object):
    """The public entry points must reject non-uniform grids before any
    GPU work and echo valid grids untouched."""

    def _lc(self):
        rng = np.random.RandomState(1)
        N, T = 200, 100.0
        t = np.sort(rng.uniform(0, T, N))
        y = 1 + 0.01 * np.sin(2 * np.pi * t / 0.7) + 0.005 * rng.randn(N)
        dy = 0.005 * np.ones(N)
        return t, y, dy

    def test_run_and_batched_reject_nonuniform_grid(self):
        t, y, dy = self._lc()
        fu = np.concatenate([np.arange(0.1, 1.0, 0.002),
                             np.arange(1.0, 5.0, 0.01)])
        proc = LombScargleAsyncProcess()
        with pytest.raises(ValueError, match="not uniformly spaced"):
            proc.run([(t, y, dy)], freqs=fu)
        with pytest.raises(ValueError, match="not uniformly spaced"):
            proc.run([(t, y, dy)], freqs=fu, use_fft=False)
        with pytest.raises(ValueError, match="not uniformly spaced"):
            proc.batched_run_const_nfreq([(t, y, dy)], freqs=fu)
        with pytest.raises(ValueError, match="not uniformly spaced"):
            proc.preallocate(max_nobs=len(t), freqs=fu)

    def test_single_frequency_raises_clearly(self):
        # nf = 1 used to die with IndexError (id 100)
        t, y, dy = self._lc()
        proc = LombScargleAsyncProcess()
        with pytest.raises(ValueError, match="at least two"):
            proc.run([(t, y, dy)], freqs=np.array([1.0]))

    def test_dy_none_means_unit_weights(self):
        # documented pass-through that raised TypeError before 1.0 (id 100)
        t, y, dy = self._lc()
        freqs = 0.001 * (50 + np.arange(3000))
        proc = LombScargleAsyncProcess()
        p_none = _run_gpu(proc, t, y, None, freqs)
        p_ones = _run_gpu(proc, t, y, np.ones_like(t), freqs)
        p_const = _run_gpu(proc, t, y, 0.3 * np.ones_like(t), freqs)
        ref = LombScargle(t, y).power(freqs)
        assert_allclose(p_none, p_ones, rtol=1e-6, atol=1e-6)
        assert_allclose(p_none, p_const, rtol=1e-6, atol=1e-6)
        assert np.max(np.abs(p_none - ref)) < 1e-4


class TestPreallocate(object):
    """``preallocate`` left ``memory.stream = None`` (the null stream),
    so ``finish()`` -- which synchronizes ``self.streams`` only -- did
    not wait for the asynchronous result copy and ``run()`` after
    ``preallocate()`` returned stale powers (29 of 30 reads on an A40
    with the 7-smooth grids; the audit saw 14/30)."""

    @staticmethod
    def _lc(N, seed):
        r = np.random.RandomState(seed)
        t = np.sort(r.uniform(0, 100.0, N))
        y = 0.3 * np.sin(2 * np.pi * t / 1.7) + 0.05 * r.randn(N)
        return t, y, 0.05 * np.ones(N)

    def test_run_after_preallocate_matches_fresh_runs(self):
        f = 0.001 * (50 + np.arange(3000))
        B, C = self._lc(900, 2), self._lc(300, 5)
        proc = LombScargleAsyncProcess()
        fresh = {}
        for name, d in (('B', B), ('C', C)):
            fresh[name] = _run_gpu(proc, *d, f)

        proc.preallocate(max_nobs=900, nlcs=1, freqs=f)
        mem = proc.memory[0]
        assert mem.stream is not None
        assert any(mem.stream is s for s in proc.streams)

        for k in range(10):
            for name, d in (('B', B), ('C', C)):
                r = proc.run([d], freqs=[f])
                proc.finish()
                p = np.asarray(r[0][1][:len(f)], dtype=np.float64)
                assert_allclose(p, fresh[name], rtol=1e-5, atol=1e-6)

    def test_user_streams_are_synchronized_by_finish(self):
        import pycuda.driver as cuda
        f = 0.001 * (50 + np.arange(3000))
        B = self._lc(900, 2)
        proc = LombScargleAsyncProcess()
        ref = _run_gpu(proc, *B, f)
        s = cuda.Stream()
        proc.preallocate(max_nobs=900, nlcs=1, freqs=f, streams=[s])
        assert any(s is s0 for s0 in proc.streams)
        for k in range(5):
            r = proc.run([B], freqs=[f])
            proc.finish()
            assert_allclose(np.asarray(r[0][1][:len(f)], dtype=np.float64),
                            ref, rtol=1e-5, atol=1e-6)


class TestBatchedBestFreqs(object):
    """``batched_run_const_nfreq(only_return_best_freqs=True)`` returns
    the false-alarm probability of the best peak (``fap_baluev`` with
    ``d_K = 2 H + 1``) -- before 1.0 it returned ``1 - FAP``, exactly
    1.0 for every FAP below 1e-16, with ``d_K = 3`` for any H (ids 96,
    129, 144)."""

    @staticmethod
    def _lc(N=100, T=100.0, amp=0.06, seed=3):
        r = np.random.RandomState(seed)
        t = np.sort(r.uniform(0, T, N))
        y = amp * np.sin(2 * np.pi * t / 1.7) + 0.05 * r.randn(N)
        return t, y, 0.05 * np.ones(N)

    @pytest.mark.parametrize("H", [1, 2])
    def test_returns_fap_of_best_peak(self, H):
        from ..lombscargle import fap_baluev
        t, y, dy = self._lc()
        freqs = 0.002 * (50 + np.arange(1500))
        proc = LombScargleAsyncProcess(nharmonics=H)
        (f, p), = proc.batched_run_const_nfreq([(t, y, dy)], freqs=freqs)
        p = np.asarray(p[:len(freqs)], dtype=np.float64)
        i = int(np.argmax(p))
        expected = float(fap_baluev(t, dy, p[i], freqs.max(),
                                    d_K=2 * H + 1))
        wrong_dK = float(fap_baluev(t, dy, p[i], freqs.max(), d_K=3))

        bf, faps = proc.batched_run_const_nfreq(
            [(t, y, dy)], freqs=freqs, only_return_best_freqs=True)
        assert bf[0] == freqs[i]
        assert faps[0] == pytest.approx(expected, rel=1e-6)
        # a real FAP: representable, small, and not the old 1 - FAP
        assert 0.0 < faps[0] < 1e-2
        if H > 1:
            assert wrong_dK != pytest.approx(expected, rel=1e-3)

    def test_mask_is_honoured(self):
        t, y, dy = self._lc()
        freqs = 0.002 * (50 + np.arange(1500))
        proc = LombScargleAsyncProcess()
        (f, p), = proc.batched_run_const_nfreq([(t, y, dy)], freqs=freqs)
        p = np.asarray(p[:len(freqs)], dtype=np.float64)
        i = int(np.argmax(p))
        ignore = np.zeros(len(freqs), dtype=bool)
        ignore[max(0, i - 5):i + 6] = True
        bf, faps = proc.batched_run_const_nfreq(
            [(t, y, dy)], freqs=freqs, only_return_best_freqs=True,
            ignore_freq_mask=ignore)
        assert not ignore[np.flatnonzero(freqs == bf[0])[0]]
        assert bf[0] == freqs[~ignore][np.argmax(p[~ignore])]

    def test_dy_none_through_the_fap_path(self):
        t, y, dy = self._lc()
        freqs = 0.002 * (50 + np.arange(1500))
        proc = LombScargleAsyncProcess()
        ref = LombScargle(t, y).power(freqs)
        bf, faps = proc.batched_run_const_nfreq(
            [(t, y, None)], freqs=freqs, only_return_best_freqs=True)
        assert bf[0] == freqs[np.argmax(ref)]
        assert 0.0 <= faps[0] < 1.0

    def test_freqs_none_keeps_every_autofrequency_point(self):
        # the rebuilt grid dropped the last point (id 147)
        from ..utils import autofrequency
        t, y, dy = self._lc()
        proc = LombScargleAsyncProcess()
        (f, p), = proc.batched_run_const_nfreq([(t, y, dy)])
        fa = autofrequency(t)
        assert len(f) == len(fa)
        assert_allclose(f, fa, rtol=1e-12)
        r = proc.run([(t, y, dy)])
        proc.finish()
        assert len(r[0][0]) == len(fa)


class TestFapBaluevInputs(object):
    def test_dy_none_is_unit_weights(self):
        from ..lombscargle import fap_baluev
        rng = np.random.RandomState(4)
        t = np.sort(rng.rand(80)) * 50.0
        z = np.array([0.1, 0.3, 0.5])
        assert_allclose(fap_baluev(t, None, z, 5.0),
                        fap_baluev(t, np.ones_like(t), z, 5.0), rtol=1e-12)
        assert_allclose(fap_baluev(t, None, z, 5.0),
                        fap_baluev(t, 0.2 * np.ones_like(t), z, 5.0),
                        rtol=1e-12)


class TestCufinufftBackend(object):
    """The cufinufft backend was complex64 only and raised TypeError
    for ``use_double=True`` (id 97); the precision now follows the
    memory."""

    def _proc(self, use_double):
        from ..cufinufft_backend import HAS_CUFINUFFT
        if not HAS_CUFINUFFT:
            pytest.skip("cufinufft not installed")
        return LombScargleAsyncProcess(use_cufinufft=True,
                                       use_double=use_double)

    @pytest.mark.parametrize("use_double,tol", [(False, 2e-3),
                                                (True, 1e-6)])
    def test_matches_astropy(self, use_double, tol):
        t, y, dy = _realistic_lc()
        freqs = _uniform_grid(1.0 / (5 * 365.0), 20.0, 365.0)
        ref = LombScargle(t, y, dy).power(freqs)
        proc = self._proc(use_double)
        p = _run_gpu(proc, t, y, dy, freqs)
        assert np.max(np.abs(p - ref)) < tol
        assert np.argmax(p) == np.argmax(ref)

    def test_narrow_band_double(self):
        t, y, dy = _realistic_lc(N=300, T=365.0, f0=29.0, seed=3)
        freqs = _uniform_grid(20.0, 30.0, 365.0)
        ref = LombScargle(t, y, dy).power(freqs)
        proc = self._proc(True)
        p = _run_gpu(proc, t, y, dy, freqs)
        assert np.max(np.abs(p - ref)) < 1e-6


class TestWeightsUseNumpyReductions(object):
    """``weights()`` and ``LombScargleMemory.setdata`` used the Python
    builtins ``sum``/``min``/``max`` on numpy arrays, which iterate the
    array element by element: 11.4 ms per lightcurve at N = 65,000 and
    150 ms at N = 1e6 of pure interpreter time for the same values
    (Sep-2026 algorithm audit, LS-1). They now use ``np.sum`` /
    ``np.min`` / ``np.max``.

    The weight normalization moves by the last ulp (``np.sum`` is
    pairwise, the builtin is a left-to-right accumulation), which is
    also what makes the copy here agree with the canonical
    ``cuvarbase.utils.weights`` bit for bit -- it already used
    ``np.sum``, so the two disagreed before.
    """

    @staticmethod
    def _dy(n, seed=5):
        r = np.random.RandomState(seed)
        return 0.01 * (1.0 + r.rand(n))

    @pytest.mark.parametrize("n", [7, 300, 4096])
    def test_matches_the_canonical_utils_weights_bitwise(self, n):
        from ..memory.lombscargle_memory import weights as mem_weights
        from ..utils import weights as utils_weights
        dy = self._dy(n)
        w = mem_weights(dy)
        assert np.array_equal(w, utils_weights(dy))
        assert np.array_equal(w, np.power(dy, -2) / np.sum(np.power(dy, -2)))
        assert_allclose(np.sum(w), 1.0, rtol=1e-14)

    @pytest.mark.parametrize("n", [7, 300, 4096])
    def test_agrees_with_the_builtin_sum_to_the_last_ulp(self, n):
        """Guards the direction of the change: the values are the same
        to a few ulps, so nothing but rounding moved."""
        from ..memory.lombscargle_memory import weights as mem_weights
        dy = self._dy(n)
        w = np.power(dy, -2)
        assert_allclose(mem_weights(dy), w / sum(w), rtol=1e-14, atol=0.0)

    @pytest.mark.parametrize("use_double", [False, True])
    def test_setdata_tmin_tmax_are_the_array_extremes(self, use_double):
        from ..memory.lombscargle_memory import LombScargleMemory
        r = np.random.RandomState(11)
        n = 500
        t = np.sort(2455000.0 + 30.0 * r.rand(n))
        y = 12 + 0.01 * r.randn(n)
        dy = 0.01 * np.ones(n)
        proc = LombScargleAsyncProcess(use_double=use_double)
        freqs = 0.01 * (5 + np.arange(400))
        mem = proc.allocate([(t, y, dy)], nfreqs=[len(freqs)],
                            k0s=[5])[0]
        mem.setdata(t=t, y=y, dy=dy)
        tc = np.asarray(t).astype(mem.real_type)
        assert mem.tmin == np.min(tc)
        assert mem.tmax == np.max(tc)
        # ... and the same values the Python builtins produced
        assert mem.tmin == min(tc)
        assert mem.tmax == max(tc)
        assert isinstance(mem, LombScargleMemory)


class TestBatchedMemoryReuse(object):
    """``batched_run_const_nfreq`` rebuilt its ``LombScargleMemory``
    set -- pinned host buffers, device arrays and two cuFFT plans -- on
    every call, and built an ``np.array([True] * nf)`` mask whether or
    not one was asked for (16 ms at nf = 365,000). It now reuses a
    fitting memory set (``preallocate``'s first, then the one it built
    last) and skips the mask entirely when ``ignore_freq_mask`` is None
    (Sep-2026 algorithm audit, LS-4).
    """

    @staticmethod
    def _lc(N=400, T=90.0, seed=2):
        r = np.random.RandomState(seed)
        t = np.sort(r.uniform(0, T, N))
        y = 0.2 * np.sin(2 * np.pi * t / 1.9) + 0.05 * r.randn(N)
        return t, y, 0.05 * np.ones(N)

    @staticmethod
    def _counting_memory(monkeypatch):
        from .. import lombscargle as lsmod
        built = []
        original = lsmod.LombScargleMemory

        class Counting(original):
            def __init__(self, *args, **kwargs):
                built.append(1)
                super(Counting, self).__init__(*args, **kwargs)

        monkeypatch.setattr(lsmod, 'LombScargleMemory', Counting)
        return built

    def test_memory_is_built_once_for_many_calls(self, monkeypatch):
        freqs = 0.002 * (30 + np.arange(4000))
        d = [self._lc()]
        proc = LombScargleAsyncProcess()
        built = self._counting_memory(monkeypatch)
        proc.batched_run_const_nfreq(d, freqs=freqs)
        assert sum(built) == 1
        del built[:]
        for _ in range(4):
            proc.batched_run_const_nfreq(d, freqs=freqs)
        assert sum(built) == 0

    def test_reused_memory_gives_identical_powers(self):
        freqs = 0.002 * (30 + np.arange(4000))
        d = [self._lc()]
        proc = LombScargleAsyncProcess()
        proc.batched_run_const_nfreq(d, freqs=freqs)      # warm/compile
        proc._batch_memory = None                          # force a rebuild
        fresh = np.copy(proc.batched_run_const_nfreq(d, freqs=freqs)[0][1])
        for _ in range(3):
            again = np.copy(proc.batched_run_const_nfreq(d,
                                                         freqs=freqs)[0][1])
            assert np.array_equal(fresh, again)

    def test_a_different_grid_is_not_reused(self, monkeypatch):
        f1 = 0.002 * (30 + np.arange(4000))
        f2 = 0.002 * (30 + np.arange(2500))
        d = [self._lc()]
        proc = LombScargleAsyncProcess()
        proc.batched_run_const_nfreq(d, freqs=f1)
        built = self._counting_memory(monkeypatch)
        p2 = np.copy(proc.batched_run_const_nfreq(d, freqs=f2)[0][1])
        assert sum(built) == 1
        del built[:]
        proc.batched_run_const_nfreq(d, freqs=f2)
        assert sum(built) == 0
        # and the shorter grid's powers are the head of the longer one
        # (only to float32 NFFT accuracy: the two grids are padded to
        # different 7-smooth lengths, so the spreading differs by ~2e-4
        # relative near the top of the band)
        p1 = np.copy(proc.batched_run_const_nfreq(d, freqs=f1)[0][1])
        assert_allclose(np.asarray(p2[:len(f2)], dtype=np.float64),
                        np.asarray(p1[:len(f2)], dtype=np.float64),
                        rtol=1e-3, atol=1e-5)

    def test_a_longer_lightcurve_forces_a_rebuild(self, monkeypatch):
        freqs = 0.002 * (30 + np.arange(4000))
        short, long_ = [self._lc(N=200, seed=3)], [self._lc(N=900, seed=4)]
        proc = LombScargleAsyncProcess()
        proc.batched_run_const_nfreq(short, freqs=freqs)
        built = self._counting_memory(monkeypatch)
        proc.batched_run_const_nfreq(long_, freqs=freqs)
        assert sum(built) == 1
        del built[:]
        # the bigger buffers serve the short lightcurve too
        proc.batched_run_const_nfreq(short, freqs=freqs)
        assert sum(built) == 0

    def test_padded_buffers_do_not_change_the_result(self):
        # Two different device allocations, so this is the ~1e-8 float32
        # tolerance of the NFFT gridding atomics, not bitwise (measured
        # on the A40: same buffer 15/15 bitwise, fresh allocations up to
        # 1.1e-8 on powers of order 1 -- true of the pre-1.0 code too).
        freqs = 0.002 * (30 + np.arange(4000))
        short = [self._lc(N=200, seed=3)]
        proc = LombScargleAsyncProcess()
        exact = np.asarray(proc.batched_run_const_nfreq(short,
                                                        freqs=freqs)[0][1],
                           dtype=np.float64)
        # a run through buffers sized for 900 points
        proc.batched_run_const_nfreq([self._lc(N=900, seed=4)], freqs=freqs)
        padded = np.asarray(proc.batched_run_const_nfreq(short,
                                                         freqs=freqs)[0][1],
                            dtype=np.float64)
        assert_allclose(padded, exact, rtol=1e-6, atol=1e-7)

    def test_preallocated_memory_is_used(self, monkeypatch):
        freqs = 0.002 * (30 + np.arange(4000))
        d = [self._lc(N=400)]
        proc = LombScargleAsyncProcess()
        ref = np.copy(proc.batched_run_const_nfreq(d, freqs=freqs)[0][1])
        proc._batch_memory = None
        proc.preallocate(max_nobs=400, nlcs=1, freqs=freqs)
        built = self._counting_memory(monkeypatch)
        p = np.copy(proc.batched_run_const_nfreq(d, freqs=freqs)[0][1])
        assert sum(built) == 0
        assert proc._batch_memory is None      # preallocate's set was used
        assert_allclose(np.asarray(p, dtype=np.float64),
                        np.asarray(ref, dtype=np.float64),
                        rtol=1e-6, atol=1e-7)

    def test_amplitude_prior_change_is_not_reused(self, monkeypatch):
        freqs = 0.002 * (30 + np.arange(2000))
        d = [self._lc()]
        proc = LombScargleAsyncProcess()
        p0 = np.copy(proc.batched_run_const_nfreq(d, freqs=freqs)[0][1])
        built = self._counting_memory(monkeypatch)
        p1 = np.copy(proc.batched_run_const_nfreq(
            d, freqs=freqs, amplitude_prior=0.05)[0][1])
        assert sum(built) == 1
        # the prior really was applied (it is not the unregularized run)
        assert not np.allclose(np.asarray(p0[:len(freqs)], dtype=np.float64),
                               np.asarray(p1[:len(freqs)], dtype=np.float64))
        del built[:]
        p2 = np.copy(proc.batched_run_const_nfreq(d, freqs=freqs)[0][1])
        assert sum(built) == 1                 # back to no prior: rebuild
        assert_allclose(np.asarray(p2, dtype=np.float64),
                        np.asarray(p0, dtype=np.float64),
                        rtol=1e-6, atol=1e-7)

    def test_no_mask_matches_an_all_true_mask(self):
        freqs = 0.002 * (30 + np.arange(3000))
        d = [self._lc()]
        proc = LombScargleAsyncProcess()
        bf0, fap0 = proc.batched_run_const_nfreq(
            d, freqs=freqs, only_return_best_freqs=True)
        bf1, fap1 = proc.batched_run_const_nfreq(
            d, freqs=freqs, only_return_best_freqs=True,
            ignore_freq_mask=np.zeros(len(freqs), dtype=bool))
        assert bf0[0] == bf1[0]
        assert fap0[0] == fap1[0]

    def test_per_call_nharmonics_is_not_reused(self, monkeypatch):
        """``nharmonics`` is read off the memory object
        (``lomb_scargle_async``), so a per-call ``nharmonics=`` must key
        and build its own memory set. Matching a cached H = 1 set
        against a request for H = 2 silently returned the
        single-harmonic periodogram (found reviewing LS-4)."""
        freqs = 0.002 * (30 + np.arange(1500))
        d = [self._lc()]
        proc = LombScargleAsyncProcess()
        p1 = np.copy(proc.batched_run_const_nfreq(d, freqs=freqs)[0][1])

        ref = LombScargleAsyncProcess(nharmonics=2)
        p2ref = np.copy(ref.batched_run_const_nfreq(d, freqs=freqs)[0][1])

        built = self._counting_memory(monkeypatch)
        p2 = np.copy(proc.batched_run_const_nfreq(d, freqs=freqs,
                                                  nharmonics=2)[0][1])
        assert sum(built) == 1                 # not the cached H = 1 set
        assert_allclose(np.asarray(p2, dtype=np.float64),
                        np.asarray(p2ref, dtype=np.float64),
                        rtol=1e-6, atol=1e-7)
        # it really is a different periodogram from the H = 1 one
        assert not np.allclose(np.asarray(p2[:len(freqs)], dtype=np.float64),
                               np.asarray(p1[:len(freqs)], dtype=np.float64))

        del built[:]
        proc.batched_run_const_nfreq(d, freqs=freqs, nharmonics=2)
        assert sum(built) == 0                 # the H = 2 set IS reused

        del built[:]
        p3 = np.copy(proc.batched_run_const_nfreq(d, freqs=freqs)[0][1])
        assert sum(built) == 1                 # back to H = 1: rebuild
        assert_allclose(np.asarray(p3, dtype=np.float64),
                        np.asarray(p1, dtype=np.float64),
                        rtol=1e-6, atol=1e-7)

    def test_per_call_use_double_is_not_reused(self, monkeypatch):
        """Same as above for ``use_double=``: the memory's precision
        sets the dtype of the returned periodogram, so a cached
        single-precision set must not answer a ``use_double=True``
        request. (Passing ``use_double`` per call only changes the
        buffers -- the kernels keep the precision the process was
        constructed with -- but that is pre-1.0 behaviour this must not
        change silently; construct the process with ``use_double=True``
        for a genuine double-precision run.)"""
        freqs = 0.002 * (30 + np.arange(1500))
        d = [self._lc()]
        proc = LombScargleAsyncProcess()
        p1 = proc.batched_run_const_nfreq(d, freqs=freqs)[0][1]
        assert np.asarray(p1).dtype == np.float32

        built = self._counting_memory(monkeypatch)
        p2 = proc.batched_run_const_nfreq(d, freqs=freqs,
                                          use_double=True)[0][1]
        assert sum(built) == 1                 # not the float32 set
        assert np.asarray(p2).dtype == np.float64

    def test_a_buffer_sizing_kwarg_opts_out_of_the_cache(self, monkeypatch):
        """``n0_buffer`` (like every other key that hands the memory a
        buffer or its size) opts the call out of the cache entirely, so
        it allocates its own set exactly as it did before 1.0."""
        freqs = 0.002 * (30 + np.arange(1500))
        d = [self._lc(N=400)]
        proc = LombScargleAsyncProcess()
        proc.batched_run_const_nfreq(d, freqs=freqs)
        cached = proc._batch_memory
        assert cached is not None

        built = self._counting_memory(monkeypatch)
        for _ in range(2):
            proc.batched_run_const_nfreq(d, freqs=freqs, n0_buffer=1000)
        assert sum(built) == 2                 # never reused, never cached
        assert proc._batch_memory is cached

        del built[:]
        proc.batched_run_const_nfreq(d, freqs=freqs)
        assert sum(built) == 0                 # the plain cache survived

    @pytest.mark.parametrize("kw", ['sigma', 'm', 'stream'])
    def test_constructor_positional_kwargs_raise_cold_and_warm(self, kw):
        """``LombScargleMemory`` takes sigma/m/stream positionally, so
        passing them as keywords has always raised TypeError. They must
        opt out of the memory cache too: on a cache hit the constructor
        is never called, so the call would otherwise succeed silently
        and IGNORE the keyword, returning the process-default result."""
        freqs = 0.002 * (30 + np.arange(1500))
        d = [self._lc(N=400)]
        value = {'sigma': 4, 'm': 10, 'stream': None}[kw]
        proc = LombScargleAsyncProcess()

        # cold cache
        with pytest.raises(TypeError):
            proc.batched_run_const_nfreq(d, freqs=freqs, **{kw: value})

        # warm the cache with a plain call, then the same request must
        # still raise rather than quietly returning the default
        proc.batched_run_const_nfreq(d, freqs=freqs)
        assert proc._batch_memory is not None
        with pytest.raises(TypeError):
            proc.batched_run_const_nfreq(d, freqs=freqs, **{kw: value})

    def test_grid_validation_cannot_be_switched_off_from_run(self):
        """``_grid_prechecked`` is private to the batched path and may
        suppress only the O(nf) uniformity check. ``check_freqs`` (the
        defect-23 guard against non-finite / non-positive grids) runs
        unconditionally, so no keyword reachable from a public entry
        point can turn it off."""
        proc = LombScargleAsyncProcess()
        d = [self._lc(N=400)]
        bad = 0.002 * (30 + np.arange(1500))
        bad[7] = np.nan
        with pytest.raises(ValueError):
            proc.run(d, freqs=[bad], _grid_prechecked=True)

        negative = np.linspace(-1.0, 5.0, 500)
        with pytest.raises(ValueError):
            proc.run(d, freqs=[negative], _grid_prechecked=True)

    def test_grid_validation_still_rejects_a_bad_grid(self):
        """The batched path validates the shared grid once and tells
        run() to skip the repeat; the error must survive."""
        d = [self._lc()]
        proc = LombScargleAsyncProcess()
        bad = np.concatenate([0.002 * (30 + np.arange(500)),
                              0.002 * (600 + np.arange(500))])
        with pytest.raises(ValueError):
            proc.batched_run_const_nfreq(d, freqs=bad)
        with pytest.raises(ValueError):
            proc.batched_run_const_nfreq(d, freqs=np.geomspace(0.1, 5.0, 500))
        with pytest.raises(ValueError):
            proc.run(d, freqs=[np.geomspace(0.1, 5.0, 500)])


class TestBaluevDKUsesEffectiveNharmonics(object):
    """``batched_run_const_nfreq(only_return_best_freqs=True)`` computed
    the Baluev ``d_K`` from the *process* attribute even when the call
    overrode ``nharmonics=`` (which the memory settings and the
    periodogram do honour): a 2-harmonic peak got a ``d_K=3`` FAP
    (Sep-2026 readiness audit; Phase 2 verification carry-over). The
    choice is now a pure helper fed the effective per-call value."""

    def test_helper_values(self):
        from ..lombscargle import _baluev_d_K
        assert _baluev_d_K(1) == 3
        assert _baluev_d_K(2) == 5
        assert _baluev_d_K(3) == 7
        assert _baluev_d_K(np.int64(2)) == 5
        with pytest.raises(ValueError):
            _baluev_d_K(0)

    def test_helper_tracks_the_memory_settings(self):
        # the same resolution the memory settings use: a per-call
        # nharmonics= written over the process default
        from ..lombscargle import _baluev_d_K, _ls_memory_settings
        kwargs_lsmem = dict(use_double=False, nharmonics=1, use_fft=True)
        kwargs_lsmem.update(dict(nharmonics=2))
        settings = _ls_memory_settings(1500, 50, 8, 5, False, 1, True,
                                       kwargs_lsmem)
        assert settings['nharmonics'] == 2
        assert _baluev_d_K(kwargs_lsmem['nharmonics']) == 5
        assert _baluev_d_K(settings['nharmonics']) == 5

    def test_per_call_nharmonics_sets_d_K_on_device(self, monkeypatch):
        from .. import lombscargle as lsmod
        seen = []
        real = lsmod.fap_baluev

        def recording(t, dy, z, fmax, d_K=3, **kw):
            seen.append(int(d_K))
            return real(t, dy, z, fmax, d_K=d_K, **kw)

        monkeypatch.setattr(lsmod, 'fap_baluev', recording)
        r = np.random.RandomState(3)
        t = np.sort(r.uniform(0, 100.0, 100))
        y = 0.06 * np.sin(2 * np.pi * t / 1.7) + 0.05 * r.randn(100)
        dy = 0.05 * np.ones(100)
        freqs = 0.002 * (50 + np.arange(1500))

        proc = LombScargleAsyncProcess()      # process default H = 1
        assert proc.nharmonics == 1
        proc.batched_run_const_nfreq([(t, y, dy)], freqs=freqs,
                                     nharmonics=2,
                                     only_return_best_freqs=True)
        assert seen == [5]
        # and the process default still gives d_K = 3 on the next call
        proc.batched_run_const_nfreq([(t, y, dy)], freqs=freqs,
                                     only_return_best_freqs=True)
        assert seen == [5, 3]
