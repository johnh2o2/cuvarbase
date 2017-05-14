import numpy as np
import pytest
from nfft import nfft_adjoint as nfft_adjoint_cpu
from nfft.utils import nfft_matrix
from nfft.kernels import KERNELS
from numpy.testing import assert_allclose
from scipy import fftpack
import skcuda.fft as cufft
import ..cunfft

def get_cpu_grid(t, y, N, sigma=2, m=8):
    kernel = KERNELS.get('gaussian', 'gaussian')
    mat = nfft_matrix(t, int(N * sigma), m, sigma, kernel, truncated=True)
    return mat.T.dot(y)


def test_nfft_adjoint_async(t, y, N, sigma=2, m=8, block_size=160):
    
    nfft_adjoint_async(stream, data, gpu_data, result, functions, m=m, 
                        sigma=sigma, block_size=block_size, fast_grid=False)

    stream.synchronize()

    cpu_result = nfft_adjoint_cpu(t, y, N, sigma=sigma, m=m, use_fft=True, truncated=True) 
    pow_gpu = np.power(np.absolute(result), 2)
    pow_cpu = np.power(np.absolute(cpu_result), 2)

    assert_allclose(cpu_result.real, result.real, atol=1E-4, rtol=1E-2)
    assert_allclose(cpu_result.imag, result.imag, atol=1E-4, rtol=1E-2)


def test_post_gridding(t, y, N, sigma=2, m=8, block_size=160):
    stream = cuda.Stream()

    functions = compile_nfft_functions()
    data, gpu_data, result = allocate_data_for_nfft_adjoint_async(stream, t, y, N, sigma=sigma)

    # get CPU grid
    grid_cpu = get_cpu_grid(t, y, N, sigma=sigma, m=m)

    # test results using CPU grid
    nfft_adjoint_async(stream, data, gpu_data, result, functions, m=m, 
                        sigma=sigma, block_size=block_size, use_grid=grid_cpu.astype(np.float32)) 
    
    # get NFFT using CPU routine
    cpu_result = nfft_adjoint_cpu(t, y, N, sigma=sigma, m=m, use_fft=True, truncated=True) 

    assert_allclose(cpu_result.real, result.real, atol=1E-4, rtol=1E-3)
    assert_allclose(cpu_result.imag, result.imag, atol=1E-4, rtol=1E-3)


def test_fast_gridding(t, y, N, sigma=2, m=8, block_size=160):
    stream = cuda.Stream()
    functions = compile_nfft_functions()
    data, gpu_data, result = allocate_data_for_nfft_adjoint_async(stream, t, y, N, sigma=sigma)

    grid = nfft_adjoint_async(stream, data, gpu_data, result, functions, m=m, 
                        sigma=sigma, block_size=block_size, 
                        just_return_gridded_data=True, fast_grid=True)

    # get CPU grid
    grid_cpu = get_cpu_grid(t, y, N, sigma=sigma, m=m)

    assert_allclose(grid, grid_cpu, rtol=1E-3)

def test_slow_gridding(t, y, N, sigma=2, m=8, block_size=160):
    stream = cuda.Stream()
    functions = compile_nfft_functions()
    data, gpu_data, result = allocate_data_for_nfft_adjoint_async(stream, t, y, N, sigma=sigma)

    grid = nfft_adjoint_async(stream, data, gpu_data, result, functions, m=m, 
                        sigma=sigma, block_size=block_size, 
                        just_return_gridded_data=True, fast_grid=False)

    # get CPU grid
    grid_cpu = get_cpu_grid(t, y, N, sigma=sigma, m=m)

    assert_allclose(grid, grid_cpu, rtol=1E-3)

def test_ffts():
    ndata = 1000
    rand = np.random.RandomState(100)
    y = rand.randn(ndata)

    yhat = np.empty(len(y))

    yg = gpuarray.to_gpu(y.astype(np.complex64))
    yghat = gpuarray.to_gpu(yhat.astype(np.complex64))

    plan = Plan(len(y), np.complex64, np.complex64)
    ifft(yg, yghat, plan)

    yhat = fftpack.ifft(y) * len(y)

    print((yhat - yghat.get() )/ np.absolute(yhat))