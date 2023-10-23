from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import zip
from builtins import range
from builtins import object
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
lsrtol = 1E-2
lsatol = 1E-2
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
