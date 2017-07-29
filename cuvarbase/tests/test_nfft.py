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

nfft_sigma = 2
nfft_m = 12
nfft_rtol = 1E-4
nfft_atol = 1E-5


@pytest.fixture
def data(seed=100, sigma=0.1, ndata=100):

    rand = np.random.RandomState(seed)

    t = np.sort(rand.rand(ndata))
    y = np.cos(2 * np.pi * (3./(max(t) - min(t))) * t)

    y += sigma * rand.randn(len(t))

    err = sigma * np.ones_like(y)

    return t, y, err

def get_b(sigma, m):
    return (2. * sigma * m) / ((2 * sigma - 1) * np.pi)

def precomp_psi(t, b, n, m):
    xg = m + n * t - np.floor(n * t)

    q1 = np.exp(-xg ** 2 / b) / np.sqrt(np.pi * b)
    q2 = np.exp(2 * xg / b)
    q3 = np.exp(-np.arange(2 * m + 1) ** 2 / b)

    return q1, q2, q3

def gpu_grid_scalar(t, y, sigma, m, N):
    b = get_b(sigma, m)

    n = int(sigma * N)

    q1, q2, q3 = precomp_psi(t, b, n, m)

    u = (np.floor(n * (t + 0.5) - m)).astype(np.int)

    grid = np.zeros(n)

    inds = np.arange(2 * m + 1)
    for i, (U, Y) in enumerate(zip(u, y)):
        q2vals = np.array([pow(q2[i], j) for j in inds])
        grid[(U + inds) % len(grid)] += Y * q1[i] * q2vals * q3

    return grid

def simple_gpu_nfft(t, y, N, sigma=nfft_sigma, m=nfft_m, **kwargs):
    proc = NFFTAsyncProcess(sigma=sigma, m=m, use_double=True)
    results = proc.run([(t, y, N)], **kwargs)
    proc.finish()
    return results[0]


def get_cpu_grid(t, y, N, sigma=nfft_sigma, m=nfft_m):
    kernel = KERNELS.get('gaussian', 'gaussian')
    mat = nfft_matrix(t, int(N * sigma), m, sigma, kernel, truncated=True)
    return mat.T.dot(y)

@mark_cuda_test
def test_nfft_adjoint_async():
    t, y, err = data()

    N = int(nfft_sigma * len(t))
    gpu_nfft = simple_gpu_nfft(t, y, N, sigma=nfft_sigma, m=nfft_m)

    cpu_nfft = nfft_adjoint_cpu(t, y, N, sigma=nfft_sigma, m=nfft_m,
                                use_fft=True, truncated=True)

    assert_allclose(gpu_nfft.real, cpu_nfft.real, rtol=nfft_rtol, atol=nfft_atol)
    assert_allclose(gpu_nfft.imag, cpu_nfft.imag, rtol=nfft_rtol, atol=nfft_atol)

@mark_cuda_test
def test_fast_gridding_with_jvdp_nfft():
    t, y, err = data()

    N = int(nfft_sigma * len(t))
    gpu_grid = simple_gpu_nfft(t, y, N, sigma=nfft_sigma, m=nfft_m,
                        just_return_gridded_data=True, fast_grid=True)

    # get CPU grid
    cpu_grid = get_cpu_grid(t, y, N, sigma=nfft_sigma, m=nfft_m)

    assert_allclose(gpu_grid, cpu_grid, atol=1E-4, rtol=0)

@mark_cuda_test
def test_fast_gridding_against_scalar_version():
    t, y, err = data()

    N = int(nfft_sigma * len(t))
    gpu_grid = simple_gpu_nfft(t, y, N, sigma=nfft_sigma, m=nfft_m,
                        just_return_gridded_data=True, fast_grid=True)

    # get python version of gpu grid calculation
    cpu_grid = gpu_grid_scalar(t, y, nfft_sigma, nfft_m, N)

    assert_allclose(gpu_grid, cpu_grid, rtol=nfft_rtol, atol=nfft_atol)


@mark_cuda_test
def test_slow_gridding_against_scalar_fast_gridding():
    t, y, err = data()

    N = int(nfft_sigma * len(t))
    gpu_grid = simple_gpu_nfft(t, y, N, sigma=nfft_sigma, m=nfft_m,
                        just_return_gridded_data=True, fast_grid=False)

    # get python version of gpu grid calculation
    cpu_grid = gpu_grid_scalar(t, y, nfft_sigma, nfft_m, N)

    assert_allclose(gpu_grid, cpu_grid, rtol=nfft_rtol, atol=nfft_atol)

@mark_cuda_test
def test_slow_gridding_against_jvdp_nfft():
    t, y, err = data()

    N = int(nfft_sigma * len(t))
    gpu_grid = simple_gpu_nfft(t, y, N, sigma=nfft_sigma, m=nfft_m,
                        just_return_gridded_data=True, fast_grid=False)

    # get CPU grid
    cpu_grid = get_cpu_grid(t, y, N, sigma=nfft_sigma, m=nfft_m)

    diffs = np.absolute(gpu_grid - cpu_grid)
    inds = (np.argsort(diffs)[::-1])[:10]

    for i, gpug, cpug, d in zip(inds, gpu_grid[inds],
                                cpu_grid[inds],
                                diffs[inds]):
        print(i, gpug, cpug, d)

    assert_allclose(gpu_grid, cpu_grid, rtol=nfft_rtol, atol=nfft_atol)

@mark_cuda_test
def test_ffts():
    t, y, err = data()

    yhat = np.empty(len(y))

    yg = gpuarray.to_gpu(y.astype(np.complex128))
    yghat = gpuarray.to_gpu(yhat.astype(np.complex128))

    plan = cufft.Plan(len(y), np.complex128, np.complex128)
    cufft.ifft(yg, yghat, plan)

    yhat = fftpack.ifft(y) * len(y)

    assert_allclose(yhat, yghat.get(), rtol=nfft_rtol, atol=nfft_atol)

@mark_cuda_test
def test_nfft_against_existing_impl():
    t, y, err = data()

    N = int(nfft_sigma * len(t))
    gpu_nfft = simple_gpu_nfft(t, y, N, sigma=nfft_sigma, m=nfft_m)

    cpu_nfft = nfft_adjoint_cpu(t, y, N, sigma=nfft_sigma, m=nfft_m)
    assert_allclose(np.real(cpu_nfft), np.real(gpu_nfft), atol=nfft_atol, rtol=nfft_rtol)
    assert_allclose(np.imag(cpu_nfft), np.imag(gpu_nfft), atol=nfft_atol, rtol=nfft_rtol)
