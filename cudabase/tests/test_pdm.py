import numpy as np
from numpy.testing import all_close
import pytest
from utils import weights
from pdm import pdm2_cpu, PDMAsyncProcess

@pytest.fixture
def data(seed=100, sigma=0.1, ndata=100):

    rand = np.random.RandomState(seed)

    t = np.sort(rand.rand(ndata))
    y = np.cos(2 * np.pi * (3./(max(t) - min(t))) * t)

    y += sigma * rand.randn(len(t))

    err = sigma * np.ones_like(y)

    return t, y, err


def test_pdm(data, kind='binned_linterp', nbins=30, seed=100, nfreqs=100):
    from numpy.testing import assert_allclose
    t, y, err = data
    w = weights(err)
    freqs = np.linspace(0, 1./(max(t) - min(t)), nfreqs)
    freqs += 0.5 * (freqs[1] - freqs[0])

    
    pow_cpu = pdm2_cpu(t, y, w, freqs, linterp=(kind == 'binned_linterp'), nbins=nbins)
    
    pdm_proc = PDMAsyncProcess()
    results = pdm_proc.run([(t, y, w, freqs)], kind=kind, nbins=nbins)
    pdm_proc.finish()

    pow_gpu = results[0]

    assert_allclose(pow_cpu, pow_gpu, atol=1E-5, rtol=1E-3)