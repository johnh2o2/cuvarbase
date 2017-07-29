import numpy as np
import pytest

from numpy.testing import assert_allclose
from astropy.stats.lombscargle import LombScargle

from ..lombscargle import LombScargleAsyncProcess
from pycuda.tools import mark_cuda_test
spp = 10
nfac = 10
lsrtol = 1E-2
lsatol = 1E-5
nfft_sigma = 2

@pytest.fixture
def data(seed=100, sigma=0.1, ndata=100, freq=3.):

    rand = np.random.RandomState(seed)

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

    #print(zip(diff, p0, p))
    assert(all(diff < lsrtol * 0.5 * (p + p0) + lsatol ))


@mark_cuda_test
def test_against_astropy_double():
    t, y, err = data()
    ls_proc = LombScargleAsyncProcess(use_double=True, sigma=nfft_sigma)

    results = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                                             samples_per_peak=spp)
    ls_proc.finish()

    fgpu, pgpu = zip(*(results[0]))

    power = LombScargle(t, y, err).power(fgpu)

    assert_similar(power, pgpu)

@mark_cuda_test
def test_against_astropy_single():
    t, y, err = data()
    ls_proc = LombScargleAsyncProcess(use_double=False, sigma=nfft_sigma)

    results = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                                             samples_per_peak=spp)
    ls_proc.finish()
    fgpu, pgpu = zip(*(results[0]))

    power = LombScargle(t, y, err).power(fgpu)

    assert_similar(power, pgpu)



@mark_cuda_test
def test_ls_kernel():
    t, y, err = data()
    ls_proc = LombScargleAsyncProcess(use_double=True, sigma=nfft_sigma)

    results = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                                             samples_per_peak=spp, use_cpu_nfft=True)
    ls_proc.finish()
    fgpu, pgpu = zip(*(results[0]))

    power = LombScargle(t, y, err, fit_mean=True, center_data=False).power(fgpu)

    assert_similar(power, pgpu)

@mark_cuda_test
def test_ls_kernel_direct_sums():
    t, y, err = data()
    ls_proc = LombScargleAsyncProcess(use_double=True, sigma=nfft_sigma)

    results = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                              samples_per_peak=spp, use_fft=False)
    ls_proc.finish()
    fgpu, pgpu = zip(*(results[0]))

    power = LombScargle(t, y, err, fit_mean=True, center_data=True).power(fgpu)

    assert_similar(power, pgpu)


@mark_cuda_test
def test_ls_kernel_direct_sums_is_consistent():
    t, y, err = data()
    ls_proc = LombScargleAsyncProcess(use_double=True, sigma=nfft_sigma)

    results_ds = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                                             samples_per_peak=spp, use_fft=False)
    ls_proc.finish()

    fgpu_ds, pgpu_ds = zip(*(results_ds[0]))

    results_reg = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                                             samples_per_peak=spp, use_cpu_nfft=True)
    ls_proc.finish()

    fgpu_reg, pgpu_reg = zip(*(results_reg[0]))

    assert_similar(pgpu_reg, pgpu_ds)


@mark_cuda_test
def test_ls_kernel_direct_sums_against_python():

    t, y, err = data()
    ls_proc = LombScargleAsyncProcess(use_double=True, sigma=nfft_sigma)

    result_ds = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                             samples_per_peak=spp, use_fft=False)
    ls_proc.finish()

    fgpu_ds, pgpu_ds = zip(*(result_ds[0]))

    result_reg = ls_proc.run([(t, y, err)], nyquist_factor=nfac,
                                               samples_per_peak=spp, use_fft=False,
                                               python_dir_sums=True)
    ls_proc.finish()
    fgpu_reg, pgpu_reg = zip(*(result_reg[0]))


    assert_similar(pgpu_reg, pgpu_ds)


@mark_cuda_test
def test_multiple_datasets():

    ndatas = 5
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
        fb, pb = zip(*rb)
        fnb, pnb = zip(*rnb)

        assert_allclose(pnb, pb, rtol=lsrtol, atol=lsatol)
        assert_allclose(fnb, fb, rtol=lsrtol, atol=lsatol)

@mark_cuda_test
def test_run_batch():

    ndatas = 25
    batch_size = 5

    datas = [data() for i in range(ndatas)]
    ls_proc = LombScargleAsyncProcess(sigma=nfft_sigma)

    batched_results = ls_proc.batched_run(datas, nyquist_factor=nfac,
                                  samples_per_peak=spp)
    ls_proc.finish()

    non_batched_results = []
    for d in datas:
        r = ls_proc.run([d], nyquist_factor=nfac, samples_per_peak=spp)
        ls_proc.finish()
        non_batched_results.extend(r)

    for rb, rnb in zip(batched_results, non_batched_results):
        fb, pb = zip(*rb)
        fnb, pnb = zip(*rnb)

        assert_allclose(pnb, pb, rtol=lsrtol, atol=lsatol)
        assert_allclose(fnb, fb, rtol=lsrtol, atol=lsatol)
