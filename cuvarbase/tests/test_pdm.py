import numpy as np
from numpy.testing import assert_allclose
import pytest
from ..utils import weights
from ..pdm import pdm2_cpu, PDMAsyncProcess
from pycuda.tools import mark_cuda_test

@pytest.fixture
def data(seed=100, sigma=0.1, ndata=250):

    rand = np.random.RandomState(seed)

    t = np.sort(rand.rand(ndata))
    y = np.cos(2 * np.pi * (10./(max(t) - min(t))) * t)

    y += sigma * rand.randn(len(t))

    err = sigma * np.ones_like(y)

    return t, y, err

@mark_cuda_test
def test_cuda_pdm():

    kind = 'binned_linterp'
    nbins = 10
    seed = 100
    nfreqs = 1000
    ndata = 250

    t, y, err = data(seed=seed, ndata=ndata)


    w = weights(err)
    freqs = np.linspace(0, 100./(max(t) - min(t)), nfreqs)
    freqs += 0.5 * (freqs[1] - freqs[0])

    pow_cpu = pdm2_cpu(t, y, w, freqs, linterp=(kind == 'binned_linterp'), nbins=nbins)

    pdm_proc = PDMAsyncProcess()
    results = pdm_proc.run([(t, y, w, freqs)], kind=kind, nbins=nbins)
    pdm_proc.finish()

    pow_gpu = results[0]

    assert_allclose(pow_cpu, pow_gpu, atol=1E-2, rtol=0)