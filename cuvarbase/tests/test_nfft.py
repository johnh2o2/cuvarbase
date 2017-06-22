import numpy as np
import pytest
from nfft import nfft_adjoint as nfft_adjoint_cpu
from nfft.utils import nfft_matrix
from nfft.kernels import KERNELS
from numpy.testing import assert_allclose
from scipy import fftpack
import skcuda.fft as cufft
from ..cunfft import NFFTAsyncProcess
from pycuda.tools import mark_cuda_test
from pycuda import gpuarray

@pytest.fixture
def data(seed=100, sigma=0.1, ndata=100):

    rand = np.random.RandomState(seed)

    t = np.sort(rand.rand(ndata))
    y = np.cos(2 * np.pi * (3./(max(t) - min(t))) * t)

    y += sigma * rand.randn(len(t))

    err = sigma * np.ones_like(y)

    return t, y, err


def simple_gpu_nfft(t, y, N, sigma=2, m=8, block_size=128, **kwargs):
    proc = NFFTAsyncProcess()
    results = proc.run([(t, y, N)], sigma=sigma, m=m, block_size=block_size, **kwargs)
    proc.finish()
    return results[0]


def get_cpu_grid(t, y, N, sigma=2, m=8):
    kernel = KERNELS.get('gaussian', 'gaussian')
    mat = nfft_matrix(t, int(N * sigma), m, sigma, kernel, truncated=True)
    return mat.T.dot(y)

def test_nfft_adjoint_async(data, sigma=2, m=8, block_size=128):
    def test():
        t, y, N = data
        gpu_nfft = simple_gpu_nfft(t, y, N, sigma=sigma, m=m, block_size=block_size)

        cpu_nfft = nfft_adjoint_cpu(t, y, N, sigma=sigma, m=m, use_fft=True, truncated=True) 

        assert_allclose(gpu_nfft.real, cpu_nfft.real, atol=1E-4, rtol=1E-2)
        assert_allclose(gpu_nfft.imag, cpu_nfft.imag, atol=1E-4, rtol=1E-2)
    return mark_cuda_test(test)


def test_fast_gridding(data, sigma=2, m=8, block_size=160):
    def test():
        t, y, N = data

        gpu_grid = simple_gpu_nfft(t, y, N, sigma=sigma, m=m, block_size=block_size,
                            just_return_gridded_data=True, fast_grid=True)

        # get CPU grid
        cpu_grid = get_cpu_grid(t, y, N, sigma=sigma, m=m)

        assert_allclose(gpu_grid, cpu_grid, rtol=1E-3)
    return mark_cuda_test(test)

def test_slow_gridding(data, sigma=2, m=8, block_size=160):
    def test(): 
        t, y, N = data

        gpu_grid = simple_gpu_nfft(t, y, N, sigma=sigma, m=m, block_size=block_size,
                            just_return_gridded_data=True, fast_grid=False)

        # get CPU grid
        cpu_grid = get_cpu_grid(t, y, N, sigma=sigma, m=m)

        assert_allclose(gpu_grid, cpu_grid, rtol=1E-3)
        
    return mark_cuda_test(test)


def test_ffts(data):
    def test():
        t, y = data

        yhat = np.empty(len(y))

        yg = gpuarray.to_gpu(y.astype(np.complex64))
        yghat = gpuarray.to_gpu(yhat.astype(np.complex64))

        plan = Plan(len(y), np.complex64, np.complex64)
        ifft(yg, yghat, plan)

        yhat = fftpack.ifft(y) * len(y)

        assert_allclose(yhat, yghat.get())
    return mark_cuda_test(test)

def test_nfft_against_existing_impl(data, sigma=2, m=m):

    def test():
        t, y = data

        gpu_nfft = simple_gpu_nfft(t, y, N, sigma=sigma, m=m)

        cpu_nfft = nfft_adjoint(t, y, N, sigma=sigma, m=m)
        assert_allclose(np.real(cpu_nfft), np.real(gpu_nfft))
        assert_allclose(np.imag(cpu_nfft), np.imag(gpu_nfft))
    return mark_cuda_test(test)
