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
nfft_sigma = 4

@pytest.fixture
def data(seed=100, sigma=0.1, ndata=100, freq=3.):

    rand = np.random.RandomState(seed)

    t = np.sort(rand.rand(ndata))
    y = np.cos(2 * np.pi * freq * t)

    y += sigma * rand.randn(len(t))

    err = sigma * np.ones_like(y)

    return t, y, err

def assert_similar(pdg0, pdg, top=5):
    inds = np.argsort(pdg0)[::-1]

    pdiff = np.absolute(pdg[inds] - pdg0[inds]) / pdg0[inds]

    assert(all(pdiff[:top] < lsrtol))


@mark_cuda_test
def test_against_astropy_double():
    t, y, err = data()
    t[0] = 0
    ls_proc = LombScargleAsyncProcess(use_double=True, sigma=nfft_sigma)

    w = np.power(err, -2)
    w /= sum(w)

    freqs_gpu, power_gpu = ls_proc.run([(t, y, w)], nyquist_factor=nfac,
                                             samples_per_peak=spp)
    ls_proc.finish()

    fgpu = freqs_gpu[0]
    pgpu = power_gpu[0]

    power = LombScargle(t, y, err).power(fgpu)

    assert_similar(power, pgpu)

@mark_cuda_test
def test_against_astropy_single():
    t, y, err = data()
    t[0] = 0
    ls_proc = LombScargleAsyncProcess(use_double=False, sigma=nfft_sigma)

    w = np.power(err, -2)
    w /= sum(w)

    freqs_gpu, power_gpu = ls_proc.run([(t, y, w)], nyquist_factor=nfac,
                                             samples_per_peak=spp)
    ls_proc.finish()

    fgpu = freqs_gpu[0]
    pgpu = power_gpu[0]

    power = LombScargle(t, y, err).power(fgpu)

    assert_similar(power, pgpu)

@mark_cuda_test
def test_ls_kernel():
    t, y, err = data()
    t[0] = 0
    ls_proc = LombScargleAsyncProcess(use_double=True, sigma=nfft_sigma)

    w = np.power(err, -2)
    w /= sum(w)

    freqs_gpu, power_gpu = ls_proc.run([(t, y, w)], nyquist_factor=nfac,
                                             samples_per_peak=spp, use_cpu_nfft=True)
    ls_proc.finish()

    fgpu = freqs_gpu[0]
    pgpu = power_gpu[0]

    power = LombScargle(t, y, err, fit_mean=True, center_data=False).power(fgpu)

    assert_similar(power, pgpu)

@mark_cuda_test
def test_ls_kernel_direct_sums():
    t, y, err = data()
    t[0] = 0
    ls_proc = LombScargleAsyncProcess(use_double=True, sigma=nfft_sigma)

    w = np.power(err, -2)
    w /= sum(w)

    freqs_gpu, power_gpu = ls_proc.run([(t, y, w)], nyquist_factor=nfac,
                                             samples_per_peak=spp, use_fft=False)
    ls_proc.finish()

    fgpu = freqs_gpu[0]
    pgpu = power_gpu[0]

    power = LombScargle(t, y, err, fit_mean=True, center_data=True).power(fgpu)

    assert_similar(power, pgpu)


@mark_cuda_test
def test_ls_kernel_direct_sums_is_consistent():
    t, y, err = data()
    t[0] = 0
    ls_proc = LombScargleAsyncProcess(use_double=True, sigma=nfft_sigma)

    w = np.power(err, -2)
    w /= sum(w)

    freqs_gpu_ds, power_gpu_ds = ls_proc.run([(t, y, w)], nyquist_factor=nfac,
                                             samples_per_peak=spp, use_fft=False)
    ls_proc.finish()

    fgpu_ds = freqs_gpu_ds[0]
    pgpu_ds = power_gpu_ds[0]

    freqs_gpu_reg, power_gpu_reg = ls_proc.run([(t, y, w)], nyquist_factor=nfac,
                                             samples_per_peak=spp, use_cpu_nfft=True)
    ls_proc.finish()

    fgpu_reg    = freqs_gpu_reg[0]
    pgpu_reg    = power_gpu_reg[0]

    assert_similar(pgpu_reg, pgpu_ds)


@mark_cuda_test
def test_ls_kernel_direct_sums_against_python():

    t, y, err = data()
    t[0] = 0
    ls_proc = LombScargleAsyncProcess(use_double=True, sigma=nfft_sigma)

    w = np.power(err, -2)
    w /= sum(w)

    freqs_gpu_ds, power_gpu_ds = ls_proc.run([(t, y, w)], nyquist_factor=nfac,
                                             samples_per_peak=spp, use_fft=False)
    ls_proc.finish()

    fgpu_ds = freqs_gpu_ds[0]
    pgpu_ds = power_gpu_ds[0]

    freqs_gpu_reg, power_gpu_reg = ls_proc.run([(t, y, w)], nyquist_factor=nfac,
                                               samples_per_peak=spp, use_fft=False,
                                               python_dir_sums=True)
    ls_proc.finish()

    fgpu_reg    = freqs_gpu_reg[0]
    pgpu_reg    = power_gpu_reg[0]

    assert_similar(pgpu_reg, pgpu_ds)
