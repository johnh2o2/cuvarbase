import numpy as np
from numpy.testing import assert_allclose
import pytest
from ..utils import weights
from ..pdm import pdm2_cpu, binless_pdm_cpu, PDMAsyncProcess

pytest.nbins = 10
pytest.seed = 100
pytest.nfreqs = 100
pytest.ndata = 10
pytest.sigma = 0.1

@pytest.fixture(scope="function")
def pow_cpu(request):
    rand = np.random.RandomState(pytest.seed)

    t = np.sort(rand.rand(pytest.ndata))
    y = np.cos(2 * np.pi * (10./(max(t) - min(t))) * t)

    y += pytest.sigma * rand.randn(len(t))

    err = pytest.sigma * np.ones_like(y)

    w = weights(err)
    freqs = np.linspace(0, 100./(max(t) - min(t)), pytest.nfreqs)
    freqs += 0.5 * (freqs[1] - freqs[0])

    pow_cpu = pdm2_cpu(t, y, w, freqs,
                       linterp=(request.param == 'binned_linterp'),
                       nbins=pytest.nbins)

    return pow_cpu

@pytest.fixture(scope="function")
def binless_pow_cpu(request):
    rand = np.random.RandomState(pytest.seed)

    t = np.sort(rand.rand(pytest.ndata))
    y = np.cos(2 * np.pi * (10./(max(t) - min(t))) * t)

    y += pytest.sigma * rand.randn(len(t))

    err = pytest.sigma * np.ones_like(y)

    w = weights(err)
    freqs = np.linspace(0, 100./(max(t) - min(t)), pytest.nfreqs)
    freqs += 0.5 * (freqs[1] - freqs[0])

    pow_cpu = binless_pdm_cpu(t, y, w, freqs, tophat=(request.param == 'binless_tophat'))

    return pow_cpu

@pytest.fixture(scope="function")
def pow_gpu(request):
    rand = np.random.RandomState(pytest.seed)

    t = np.sort(rand.rand(pytest.ndata))
    y = np.cos(2 * np.pi * (10./(max(t) - min(t))) * t)

    y += pytest.sigma * rand.randn(len(t))

    err = pytest.sigma * np.ones_like(y)

    w = weights(err)
    freqs = np.linspace(0, 100./(max(t) - min(t)), pytest.nfreqs)
    freqs += 0.5 * (freqs[1] - freqs[0])

    pdm_proc = PDMAsyncProcess()
    # Test deprecated format
    with pytest.warns(DeprecationWarning):
        results = pdm_proc.run([(t, y, w, freqs)], kind=request.param, nbins=pytest.nbins)
    pdm_proc.finish()

    return results[0]

@pytest.mark.parametrize(["pow_cpu","pow_gpu"], [("binned_linterp","binned_linterp")], indirect=True)
def test_cuda_pdm_binned_linterp(pow_cpu,pow_gpu):
    assert_allclose(pow_cpu, pow_gpu, atol=1E-2, rtol=0)

@pytest.mark.parametrize(["pow_cpu","pow_gpu"], [("binned_step","binned_step")], indirect=True)
def test_cuda_pdm_binned_step(pow_cpu,pow_gpu):
    assert_allclose(pow_cpu, pow_gpu, atol=1E-2, rtol=0)


@pytest.mark.parametrize(["binless_pow_cpu","pow_gpu"], [("binless_gauss","binless_gauss")], indirect=True)
def test_cuda_pdm_binless_gauss(binless_pow_cpu,pow_gpu):
    assert_allclose(binless_pow_cpu, pow_gpu, atol=1E-2, rtol=0)


@pytest.mark.parametrize(["binless_pow_cpu","pow_gpu"], [("binless_tophat","binless_tophat")], indirect=True)
def test_cuda_pdm_binless_tophat(binless_pow_cpu,pow_gpu):
    assert_allclose(binless_pow_cpu, pow_gpu, atol=1E-2, rtol=0)


@pytest.mark.parametrize(["pow_cpu", "pow_gpu"], [("binned_linterp", "binned_linterp_fast")], indirect=True)
def test_cuda_pdm_binned_linterp_fast(pow_cpu, pow_gpu):
    assert_allclose(pow_cpu, pow_gpu, atol=1E-2, rtol=0)


@pytest.mark.parametrize(["pow_cpu", "pow_gpu"], [("binned_step", "binned_step_fast")], indirect=True)
def test_cuda_pdm_binned_step_fast(pow_cpu, pow_gpu):
    assert_allclose(pow_cpu, pow_gpu, atol=1E-2, rtol=0)


@pytest.mark.parametrize(["binless_pow_cpu", "pow_gpu"], [("binless_gauss", "binless_gauss_fast")], indirect=True)
def test_cuda_pdm_binless_gauss_fast(binless_pow_cpu ,pow_gpu):
    assert_allclose(binless_pow_cpu, pow_gpu, atol=1E-2, rtol=0)


@pytest.mark.parametrize(["binless_pow_cpu", "pow_gpu"], [("binless_tophat", "binless_tophat_fast")], indirect=True)
def test_cuda_pdm_binless_tophat_fast(binless_pow_cpu, pow_gpu):
    assert_allclose(binless_pow_cpu, pow_gpu, atol=1E-2, rtol=0)


def test_pdm_new_format():
    rand = np.random.RandomState(pytest.seed)

    t = np.sort(rand.rand(pytest.ndata))
    y = np.cos(2 * np.pi * (10./(max(t) - min(t))) * t)
    y += pytest.sigma * rand.randn(len(t))
    err = pytest.sigma * np.ones_like(y)

    freqs = np.linspace(0, 100./(max(t) - min(t)), pytest.nfreqs)
    freqs += 0.5 * (freqs[1] - freqs[0])

    pdm_proc = PDMAsyncProcess()

    # Test (t, y, err) with explicit freqs as array
    results = pdm_proc.run([(t, y, err)], freqs=freqs, kind='binned_linterp', nbins=pytest.nbins)
    assert_allclose(results[0][0], freqs)
    # Test (t, y, err) with explicit freqs as list
    results = pdm_proc.run([(t, y, err)], freqs=list(freqs), kind='binned_linterp', nbins=pytest.nbins)
    assert_allclose(results[0][0], freqs)

    # Test (t, y, err) with automatic freqs
    results_auto = pdm_proc.run([(t, y, err)], kind='binned_linterp', nbins=pytest.nbins)
    assert len(results_auto[0][0]) > 0
    assert len(results_auto[0][1]) == len(results_auto[0][0])

    pdm_proc.finish()
