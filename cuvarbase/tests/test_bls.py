import numpy as np
import pytest
from numpy.testing import assert_allclose
from pycuda.tools import mark_cuda_test
from pycuda import gpuarray
from ..bls import eebls_gpu

nfft_sigma = 2
nfft_m = 12
nfft_rtol = 1E-4
nfft_atol = 1E-5
ntests = 5


def transit_model(phi0, q, delta, q1=0.):
    def model(t, freq, q=q, phi0=phi0, delta=delta):

        dphi = (t * freq - phi0) % 1.0

        if not hasattr(t, '__iter__'):
            if dphi < 0:
                dphi += 1
            dphi -= 0.5
            return delta if np.absolute(dphi) < 0.5 * q else 0
        dphi[dphi < 0] += 1.0
        dphi -= 0.5
        y = np.zeros(len(t))
        y[np.absolute(dphi) < 0.5 * q] -= delta

        return y
    return model


@pytest.fixture
def data(seed=100, sigma=0.1, ybar=12., snr=10, ndata=500, freq=10.,
         q=0.01, phi0=None):

    rand = np.random.RandomState(seed)

    if phi0 is None:
        phi0 = rand.rand()

    delta = snr * sigma / np.sqrt(ndata * q * (1 - q))

    model = transit_model(phi0, q, delta)

    t = np.sort(rand.rand(ndata))
    y = model(t, freq) + sigma * rand.randn(len(t))
    y += ybar - np.mean(y)
    err = sigma * np.ones_like(y)

    return t, y, err


# TODO: Find out a way to split this into multiple tests using
#       pytest.parametrize. Doing things the usual way won't work
#       because mark_cuda_test expects no arguments
@mark_cuda_test
def test_bls_parameter_recovery(seed=100):
    rand = np.random.RandomState(seed)
    for test in range(ntests):
        freq = 10 * rand.rand() + 10
        q = 0.05 + 0.1 * rand.rand()
        phi0 = rand.rand()
        alpha = 1.1

        outstr = "TEST {test} / {ntests}: freq={freq}, q={q}, phi0={phi0}, alpha={alpha}"
        print(outstr.format(test=test, ntests=ntests, freq=freq,
                            q=q, phi0=phi0, alpha=alpha))
        t, y, err = data(snr=10, q=q, phi0=phi0, freq=freq)
        freqs = np.linspace(freq-0.5, freq+0.5, 1000)
        df = freqs[1] - freqs[0]

        power, sols = eebls_gpu(t, y, err, freqs,
                                qmin=0.001, qmax=0.3, alpha=alpha)

        q_sol, phi_sol = sols[np.argmax(power)]

        assert(abs(freqs[np.argmax(power)] - freq) / freq < 2E-2)
        assert(abs(q_sol - q) < alpha * q)

        dphi = phi_sol - phi0
        if dphi < 0:
            dphi += 1

        if dphi > 0.5:
            dphi = 1 - dphi

        assert(dphi < q)
