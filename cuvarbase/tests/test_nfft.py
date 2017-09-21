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
nfft_sigma = 5
nfft_m = 8
nfft_rtol = 5E-3
nfft_atol = 5E-3
spp = 1


def direct_sums(t, y, freqs):
    C = [np.sum(y * np.cos(2 * np.pi * t * f)) for f in freqs]
    S = [np.sum(y * np.sin(2 * np.pi * t * f)) for f in freqs]

    return np.array([c + 1j * s for c, s in zip(C, S)])


def scale_time(t, samples_per_peak):
    return (t - min(t)) / (samples_per_peak * (max(t) - min(t))) - 0.5


@pytest.fixture
def data(seed=100, sigma=0.1, ndata=100, samples_per_peak=spp):

    rand = np.random.RandomState(seed)

    t = np.sort(rand.rand(ndata))
    y = np.cos(2 * np.pi * (3./(max(t) - min(t))) * t)

    tscl = scale_time(t, samples_per_peak=samples_per_peak)

    y += sigma * rand.randn(len(t))

    err = sigma * np.ones_like(y)

    return t, tscl, y, err


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


def simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, use_double=False,
                    m=nfft_m, samples_per_peak=spp, **kwargs):
    proc = NFFTAsyncProcess(sigma=sigma, m=m, autoset_m=False,
                            use_double=use_double)

    for stream in proc.streams:
        stream.synchronize()

    nfft_kwargs = dict(samples_per_peak=samples_per_peak)
    nfft_kwargs.update(kwargs)
    results = proc.run([(t, y, nf)], **nfft_kwargs)

    proc.finish()
    return results[0]


def get_cpu_grid(t, y, nf, sigma=nfft_sigma, m=nfft_m):
    kernel = KERNELS.get('gaussian', 'gaussian')
    mat = nfft_matrix(t, int(nf * sigma), m, sigma, kernel, truncated=True)
    return mat.T.dot(y)


@mark_cuda_test
def test_fast_gridding_with_jvdp_nfft():
    t, tsc, y, err = data()

    nf = int(nfft_sigma * len(t))
    gpu_grid = simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, m=nfft_m,
                               just_return_gridded_data=True, fast_grid=True,
                               minimum_frequency=-int(nf/2),
                               samples_per_peak=spp)

    # get CPU grid
    cpu_grid = get_cpu_grid(tsc, y, nf, sigma=nfft_sigma, m=nfft_m)

    assert_allclose(gpu_grid, cpu_grid, atol=1E-4, rtol=0)


@mark_cuda_test
def test_fast_gridding_against_scalar_version():
    t, tsc, y, err = data()

    nf = int(nfft_sigma * len(t))
    gpu_grid = simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, m=nfft_m,
                               just_return_gridded_data=True, fast_grid=True,
                               minimum_frequency=-int(nf/2),
                               samples_per_peak=spp)

    # get python version of gpu grid calculation
    cpu_grid = gpu_grid_scalar(tsc, y, nfft_sigma, nfft_m, nf)

    tols = dict(rtol=nfft_rtol, atol=nfft_atol)
    assert_allclose(gpu_grid, cpu_grid, **tols)


@mark_cuda_test
def test_slow_gridding_against_scalar_fast_gridding():
    t, tsc, y, err = data()

    nf = int(nfft_sigma * len(t))
    gpu_grid = simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, m=nfft_m,
                               just_return_gridded_data=True, fast_grid=False,
                               minimum_frequency=-int(nf/2),
                               samples_per_peak=spp)

    # get python version of gpu grid calculation
    cpu_grid = gpu_grid_scalar(tsc, y, nfft_sigma, nfft_m, nf)

    tols = dict(rtol=nfft_rtol, atol=nfft_atol)
    assert_allclose(gpu_grid, cpu_grid, **tols)


@mark_cuda_test
def test_slow_gridding_against_jvdp_nfft():
    t, tsc, y, err = data()

    nf = int(nfft_sigma * len(t))
    gpu_grid = simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, m=nfft_m,
                               just_return_gridded_data=True, fast_grid=False,
                               minimum_frequency=-int(nf/2),
                               samples_per_peak=spp)

    # get CPU grid
    cpu_grid = get_cpu_grid(tsc, y, nf, sigma=nfft_sigma, m=nfft_m)

    diffs = np.absolute(gpu_grid - cpu_grid)
    inds = (np.argsort(diffs)[::-1])[:10]

    for i, gpug, cpug, d in zip(inds, gpu_grid[inds],
                                cpu_grid[inds],
                                diffs[inds]):
        print(i, gpug, cpug, d)

    tols = dict(rtol=nfft_rtol, atol=nfft_atol)
    assert_allclose(gpu_grid, cpu_grid, **tols)


@mark_cuda_test
def test_ffts():
    t, tsc, y, err = data()

    yhat = np.empty(len(y))

    yg = gpuarray.to_gpu(y.astype(np.complex128))
    yghat = gpuarray.to_gpu(yhat.astype(np.complex128))

    plan = cufft.Plan(len(y), np.complex128, np.complex128)
    cufft.ifft(yg, yghat, plan)

    yhat = fftpack.ifft(y) * len(y)

    tols = dict(rtol=nfft_rtol, atol=nfft_atol)
    assert_allclose(yhat, yghat.get(), **tols)


@mark_cuda_test
def nfft_against_direct_sums(samples_per_peak=spp, f0=None, scaled=True):
    t, tsc, y, err = data(samples_per_peak=samples_per_peak)

    nf = int(nfft_sigma * len(t))

    df = 1./(samples_per_peak * (max(t) - min(t)))
    if f0 is None:
        f0 = -0.5 * nf * df
    k0 = int(f0 / df)

    f0 = k0 if scaled else k0 * df
    tg = tsc if scaled else t
    sppg = samples_per_peak

    gpu_nfft = simple_gpu_nfft(tg, y, nf, sigma=nfft_sigma, m=nfft_m,
                               minimum_frequency=f0,
                               samples_per_peak=sppg)

    freqs = (float(k0) + np.arange(nf))
    if not scaled:
        freqs *= df
    direct_dft = direct_sums(tg, y, freqs)

    tols = dict(rtol=nfft_rtol, atol=nfft_atol)

    def dsort(arr0, arr):
        d = np.absolute(arr0 - arr)
        return np.argsort(-d)

    inds = dsort(np.real(direct_dft), np.real(gpu_nfft))

    npr = 5
    q = zip(inds[:npr], direct_dft[inds[:npr]], gpu_nfft[inds[:npr]])
    for i, dft, gnfft in q:
        print(i, dft, gnfft)
    assert_allclose(np.real(direct_dft), np.real(gpu_nfft), **tols)
    assert_allclose(np.imag(direct_dft), np.imag(gpu_nfft), **tols)


@mark_cuda_test
def test_nfft_against_existing_impl_scaled_centered_spp1():
    nfft_against_direct_sums(samples_per_peak=1, scaled=True, f0=None)


@mark_cuda_test
def test_nfft_against_existing_impl_scaled_centered_spp5():
    nfft_against_direct_sums(samples_per_peak=5, scaled=True, f0=None)


@mark_cuda_test
def test_nfft_against_existing_impl_scaled_uncentered_spp1():
    nfft_against_direct_sums(samples_per_peak=1, scaled=True, f0=0.)


@mark_cuda_test
def test_nfft_against_existing_impl_unscaled_centered_spp1():
    nfft_against_direct_sums(samples_per_peak=1, scaled=False, f0=None)


@mark_cuda_test
def test_nfft_against_existing_impl_unscaled_uncentered_spp5():
    nfft_against_direct_sums(samples_per_peak=5, scaled=False, f0=0.)


@mark_cuda_test
def test_nfft_adjoint_async(f0=0., ndata=10, batch_size=3, use_double=False):
    datas = []
    for i in range(ndata):
        t, tsc, y, err = data()
        nf = int(nfft_sigma * len(t))

        datas.append((t, y, nf))

    kwargs = dict(minimum_frequency=f0, samples_per_peak=spp)

    proc = NFFTAsyncProcess(sigma=nfft_sigma, m=nfft_m, autoset_m=False,
                            use_double=use_double)

    single_nffts = [simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, m=nfft_m,
                                    use_double=use_double, **kwargs)
                    for t, y, nf in datas]

    multi_nffts = proc.run(datas, **kwargs)

    batch_nffts = proc.batched_run(datas, batch_size=batch_size, **kwargs)
    proc.finish()

    tols = dict(rtol=nfft_rtol, atol=nfft_atol)
    for ghat_m, ghat_s, ghat_b in zip(multi_nffts, single_nffts, batch_nffts):
        print("testing...")
        assert_allclose(ghat_s.real, ghat_m.real, **tols)
        assert_allclose(ghat_s.imag, ghat_m.imag, **tols)

        assert_allclose(ghat_s.real, ghat_b.real, **tols)
        assert_allclose(ghat_s.imag, ghat_b.imag, **tols)
