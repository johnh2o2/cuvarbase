import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy import fftpack

from pycuda.tools import mark_cuda_test
from pycuda import gpuarray

from .. import _cufft as cufft

pytest.importorskip(
    "nfft", reason="the optional 'nfft' package is the CPU reference "
                   "for these tests")
from nfft import nfft_adjoint as nfft_adjoint_cpu  # noqa: E402
from nfft.utils import nfft_matrix  # noqa: E402
from nfft.kernels import KERNELS  # noqa: E402

from ..cunfft import NFFTAsyncProcess

nfft_sigma = 5
nfft_m = 8
nfft_rtol = 5E-3
nfft_atol = 5E-3
spp = 1


def direct_sums(t, y, freqs):
    def sfunc(func):
        return [np.sum(y * func(2 * np.pi * t * f)) for f in freqs]
    return np.asarray(sfunc(np.cos)) + 1j * np.asarray(sfunc(np.sin))


def scale_time(t, samples_per_peak):
    return (t - min(t)) / (samples_per_peak * (max(t) - min(t))) - 0.5


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

    u = (np.floor(n * (t + 0.5) - m)).astype(int)

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


#@mark_cuda_test
class TestNFFT(object):

    def test_fast_gridding_with_jvdp_nfft(self):
        t, tsc, y, err = data()

        nf = int(nfft_sigma * len(t))
        gpu_grid = simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, m=nfft_m,
                                   just_return_gridded_data=True,
                                   fast_grid=True,
                                   minimum_frequency=-int(nf/2),
                                   samples_per_peak=spp)

        # get CPU grid
        cpu_grid = get_cpu_grid(tsc, y, nf, sigma=nfft_sigma, m=nfft_m)

        assert_allclose(gpu_grid, cpu_grid, atol=1E-4, rtol=0)

    def test_fast_gridding_against_scalar_version(self):
        t, tsc, y, err = data()

        nf = int(nfft_sigma * len(t))
        gpu_grid = simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, m=nfft_m,
                                   just_return_gridded_data=True,
                                   fast_grid=True,
                                   minimum_frequency=-int(nf/2),
                                   samples_per_peak=spp)

        # get python version of gpu grid calculation
        cpu_grid = gpu_grid_scalar(tsc, y, nfft_sigma, nfft_m, nf)

        tols = dict(rtol=nfft_rtol, atol=nfft_atol)
        assert_allclose(gpu_grid, cpu_grid, **tols)

    def test_slow_gridding_against_scalar_fast_gridding(self):
        t, tsc, y, err = data()

        nf = int(nfft_sigma * len(t))
        gpu_grid = simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, m=nfft_m,
                                   just_return_gridded_data=True,
                                   fast_grid=False,
                                   minimum_frequency=-int(nf/2),
                                   samples_per_peak=spp)

        # get python version of gpu grid calculation
        cpu_grid = gpu_grid_scalar(tsc, y, nfft_sigma, nfft_m, nf)

        tols = dict(rtol=nfft_rtol, atol=nfft_atol)
        assert_allclose(gpu_grid, cpu_grid, **tols)

    def test_slow_gridding_against_jvdp_nfft(self):
        t, tsc, y, err = data()

        nf = int(nfft_sigma * len(t))
        gpu_grid = simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, m=nfft_m,
                                   just_return_gridded_data=True,
                                   fast_grid=False,
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

    def test_ffts(self):
        t, tsc, y, err = data()

        yhat = np.empty(len(y))

        yg = gpuarray.to_gpu(y.astype(np.complex128))
        yghat = gpuarray.to_gpu(yhat.astype(np.complex128))

        plan = cufft.Plan(len(y), np.complex128, np.complex128)
        cufft.ifft(yg, yghat, plan)

        yhat = fftpack.ifft(y) * len(y)

        tols = dict(rtol=nfft_rtol, atol=nfft_atol)
        assert_allclose(yhat, yghat.get(), **tols)

    def nfft_against_direct_sums(self, samples_per_peak=spp,
                                 f0=None, scaled=True):
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
        q = list(zip(inds[:npr], direct_dft[inds[:npr]], gpu_nfft[inds[:npr]]))
        for i, dft, gnfft in q:
            print(i, dft, gnfft)
        assert_allclose(np.real(direct_dft), np.real(gpu_nfft), **tols)
        assert_allclose(np.imag(direct_dft), np.imag(gpu_nfft), **tols)

    def test_nfft_against_existing_impl_scaled_centered_spp1(self):
        self.nfft_against_direct_sums(samples_per_peak=1, scaled=True, f0=None)

    def test_nfft_against_existing_impl_scaled_centered_spp5(self):
        self.nfft_against_direct_sums(samples_per_peak=5, scaled=True, f0=None)

    def test_nfft_against_existing_impl_scaled_uncentered_spp1(self):
        self.nfft_against_direct_sums(samples_per_peak=1, scaled=True, f0=10.)

    def test_nfft_against_existing_impl_unscaled_centered_spp1(self):
        self.nfft_against_direct_sums(samples_per_peak=1, scaled=False,
                                      f0=None)

    def test_nfft_against_existing_impl_unscaled_uncentered_spp5(self):
        self.nfft_against_direct_sums(samples_per_peak=5, scaled=False, f0=0.)

    @pytest.mark.parametrize("use_double,tol", [(False, 1e-2),
                                                (True, 1e-2),
                                                (True, 1e-6)])
    def test_autoset_m_l1_bound_meets_tolerance(self, use_double, tol):
        # autoset_m sizes the filter radius m from the data-driven
        # L1-norm *truncation* bound (cunfft.estimate_m). We check both
        # that estimate_m returns the closed-form bound value and that
        # the realized GPU NFFT then meets the requested absolute
        # tolerance against the exact DFT.
        #
        # float32 is held at tol=1e-2: single precision has a genuine
        # ~1e-3 absolute error floor (float32 trig on large phases +
        # grid/FFT roundoff -- see estimate_m's docstring). float64 is
        # additionally checked at tol=1e-6, which the double path meets
        # since the float-PI phase-factor fix (A3, Jul 2026); before
        # that fix the realized error floored at ~1e-3 in both
        # precisions. Note ||y||_1 (~67) < nf (500) here, so the
        # chosen m is *smaller* than the old N-based heuristic -- this
        # validates the rigorous-but-tighter direction.
        t, tsc, y, err = data()
        nf = int(nfft_sigma * len(t))
        sigma = 2

        proc = NFFTAsyncProcess(sigma=sigma, autoset_m=True, tol=tol,
                                use_double=use_double)

        # estimate_m returns the smallest m with
        #   4 exp(-m pi (1 - 1/(2 sigma - 1))) ||y||_1 <= tol
        l1 = float(np.sum(np.abs(y)))
        D = np.pi * (1. - 1. / (2. * sigma - 1.))
        m_expected = max(1, int(np.ceil(-np.log(0.25 * tol / l1) / D)))
        assert proc.estimate_m(y=y) == m_expected

        results = proc.run([(tsc, y, nf)],
                           minimum_frequency=-int(nf / 2),
                           samples_per_peak=spp)
        proc.finish()
        gpu_nfft = results[0]

        freqs = -int(nf / 2) + np.arange(nf)
        direct_dft = direct_sums(tsc, y, freqs)

        # float32 gridding/FFT roundoff adds noise unrelated to the
        # truncation bound under test
        roundoff = 1e-10 if use_double else 5e-6
        err_max = np.max(np.absolute(direct_dft - gpu_nfft))
        assert err_max <= tol + roundoff * np.sum(np.abs(y))

    def test_double_precision_tracks_truncation_bound(self):
        # Regression test for the float-PI phase-factor bug (A3,
        # Jul 2026): cunfft.cu defined PI as a float32 literal, so the
        # nfft_shift/normalize phases carried a ~2.8e-8 relative error
        # that, multiplied by unreduced phase arguments up to
        # 2*pi*|k0|, produced an m-independent ~1e-3 absolute error
        # floor even at float64 (amplified with m by the Gaussian
        # deconvolution). With the fix, the realized float64 error
        # tracks the L1 truncation bound 4*exp(-m*D)*||y||_1; on the
        # A5000 the m=12 error is 1.2e-10 vs a 3.3e-9 bound. We assert
        # a 100x margin (buggy value was ~1e6 x the bound).
        t, tsc, y, err = data()
        nf = int(nfft_sigma * len(t))
        m, sigma = 12, 2

        gpu_nfft = simple_gpu_nfft(tsc, y, nf, sigma=sigma, m=m,
                                   use_double=True,
                                   minimum_frequency=-int(nf / 2),
                                   samples_per_peak=1)

        freqs = -int(nf / 2) + np.arange(nf)
        direct_dft = direct_sums(tsc, y, freqs)

        D = np.pi * (1. - 1. / (2. * sigma - 1.))
        bound = 4. * np.exp(-m * D) * np.sum(np.abs(y))
        err_max = np.max(np.absolute(direct_dft - gpu_nfft))
        assert err_max <= 100. * bound

    def test_fast_grid_double_precision_floor(self):
        # Regression test for floorf() on the double grid coordinate in
        # fast_gaussian_grid (Sep 2026): a point whose scaled position
        # ng*x - m lies within a float32 ulp below an integer K was
        # floored to K after the float32 rounding, so its window was
        # deposited one cell right of where precompute_psi (exact
        # fraction) centred it. ng*t1 - m = 16 - 2^-30 here: floor is
        # 15 in double, 16 after rounding to float32. t1 is exact in
        # binary (ng is a power of two), so the case is deterministic.
        nf, sigma, m = 32, 2, 4
        ng = sigma * nf
        K = 20
        t1 = (K - 2.0 ** -30) / ng
        t = np.array([0.0, t1, 1.0])
        y = np.array([0.0, 1.0, 0.0])
        b = get_b(sigma, m)

        ref = np.zeros(ng)
        u = int(np.floor(ng * t1 - m))
        for k in range(2 * m + 1):
            ref[(u + k) % ng] += np.exp(-((ng * t1 - (u + k)) ** 2) / b) \
                / np.sqrt(np.pi * b)
        assert u == K - m - 1

        grid = simple_gpu_nfft(t, y, nf, sigma=sigma, m=m,
                               use_double=True,
                               just_return_gridded_data=True,
                               fast_grid=True, minimum_frequency=0.,
                               samples_per_peak=1)
        grid = np.asarray(grid, dtype=np.float64)

        nonzero = np.flatnonzero(grid)
        assert nonzero.min() == u and nonzero.max() == u + 2 * m
        assert np.max(np.abs(grid - ref)) < 1e-12

    @pytest.mark.parametrize("use_double,mag_tol,phase_tol",
                             [(False, 5e-5, 1e-3), (True, 3e-7, 1e-5)])
    def test_absolute_times_bjd(self, use_double, mag_tol, phase_tol):
        # Regression test for defect 12 (nfft-absolute-time, Sep 2026):
        # NFFTMemory.fromdata cast absolute times to float32 as given,
        # so at BJD scale (~2.457e6 d, float32 spacing 0.25 d) the
        # transform's MAGNITUDES were wrong (rel. error 0.8 on this
        # data). fromdata now subtracts epoch = floor(min(t)) in
        # float64 first and records it as memory.epoch; the phases are
        # relative to that epoch (class docstring).
        rng = np.random.RandomState(5)
        n = 400
        t = np.sort(rng.rand(n)) * 30.0
        y = np.cos(2 * np.pi * 1.3 * t) + 0.1 * rng.randn(n)
        nf = 256
        T = t.max() - t.min()
        freqs = np.arange(nf) / T

        def exact(tt, epoch):
            return direct_sums(tt - epoch, y, freqs)

        proc = NFFTAsyncProcess(sigma=nfft_sigma, m=nfft_m,
                                autoset_m=False, use_double=use_double)

        mem0 = proc.allocate([(t, y, nf)])
        proc.run([(t, y, nf)], memory=mem0)
        proc.finish()
        g0 = np.array(mem0[0].ghat_c)
        assert mem0[0].epoch == 0.0
        scale = np.abs(exact(t, 0.0)).max()

        for offset in (1000.5, 2457000.5):
            tb = t + offset
            mem = proc.allocate([(tb, y, nf)])
            proc.run([(tb, y, nf)], memory=mem)
            proc.finish()
            g = np.array(mem[0].ghat_c)

            epoch = mem[0].epoch
            assert epoch == np.floor(tb.min())

            # magnitudes are shift-invariant and must match t ~ 0
            assert np.max(np.abs(np.abs(g) - np.abs(g0))) / scale < mag_tol
            # phases follow the documented convention: relative to epoch
            ref = exact(tb, epoch)
            assert np.max(np.abs(g - ref)) / scale < mag_tol
            phase_err = np.angle(g * np.conj(ref))
            assert np.sqrt(np.mean(phase_err ** 2)) < phase_tol

    @staticmethod
    def _high_k0_case(seed=9, N=500, T=365.0, k0=20000, nf=2000, spp=5.0):
        rng = np.random.RandomState(seed)
        t = np.sort(rng.rand(N)) * T
        y = rng.randn(N)
        df = 1.0 / (spp * (t.max() - t.min()))
        return t, y, df, k0, nf, spp

    def _run_band(self, proc, t, y, f0, k0, nf, spp):
        # the one-sided modes k0 .. k0 + nf - 1 must sit inside the
        # Gaussian window's alias-free band, so allocate k0 + nf modes
        # (grid sigma * (k0 + nf)) and read the first nf entries
        g = proc.run([(t, y, k0 + nf)], minimum_frequency=f0,
                     samples_per_peak=spp)[0]
        proc.finish()
        return np.array(g)[:nf]

    def test_minimum_frequency_rounds_to_an_integer_mode(self):
        # id 104 (Sep 2026): nfft_shift / normalize computed the first
        # mode k0 = f0 * spp * T as a FLT and used it as is; the periodic
        # grid only has integer modes, so a fractional value -- from a
        # user's f0 that is not a multiple of df, or from float32
        # rounding of the product (~k0 * 2e-7) -- produced a Dirichlet-
        # leakage mixture. The kernels now round k0 to the nearest
        # integer; f0 = (k0 + 0.3) df is therefore identical to k0 df.
        t, y, df, k0, nf, spp = self._high_k0_case()
        proc = NFFTAsyncProcess(sigma=4, m=8, autoset_m=False)
        g_int = self._run_band(proc, t, y, k0 * df, k0, nf, spp)
        g_frac = self._run_band(proc, t, y, (k0 + 0.3) * df, k0, nf, spp)
        scale = np.abs(g_int).max()
        assert np.max(np.abs(g_frac - g_int)) <= 1e-6 * scale

    @pytest.mark.parametrize("use_double,tol", [(False, 2e-3),
                                                (True, 5e-9)])
    def test_large_k0_band_matches_exact_dft(self, use_double, tol):
        # ids 98/160 (Sep 2026): the shift phase 2 pi (i mod ng) k0 / ng
        # and the normalize phase 2 pi f_k x0 were evaluated un-reduced
        # in float32 (arguments ~1e5 rad at k0 = 2e4 .. 5e5), giving
        # 0.1-0.4 rad phase errors at the top of high-frequency bands.
        # They are now reduced modulo one cycle (exact integer
        # arithmetic / double) before the trig. Measured on an A40
        # (m = 12): float32 4.4e-3 -> 1.0e-3 relative to max|exact|
        # (the rest is the float32 storage of t); double 3.5e-10.
        t, y, df, k0, nf, spp = self._high_k0_case()
        proc = NFFTAsyncProcess(sigma=4, m=12, autoset_m=False,
                                use_double=use_double)
        g = self._run_band(proc, t, y, k0 * df, k0, nf, spp)
        # phases are relative to epoch = floor(min t) (NFFTMemory notes)
        exact = direct_sums(t - np.floor(t.min()), y,
                            (k0 + np.arange(nf)) * df)
        err = np.max(np.abs(g - exact)) / np.abs(exact).max()
        assert err < tol, err

    def test_nfft_adjoint_async(self, f0=0., ndata=10,
                                batch_size=3, use_double=False):
        datas = []
        for i in range(ndata):
            t, tsc, y, err = data()
            nf = int(nfft_sigma * len(t))

            datas.append((t, y, nf))

        kwargs = dict(minimum_frequency=f0, samples_per_peak=spp)

        proc = NFFTAsyncProcess(sigma=nfft_sigma, m=nfft_m, autoset_m=False,
                                use_double=use_double)

        single_nffts = []
        for t, y, nf in datas:
            nfft = simple_gpu_nfft(t, y, nf, sigma=nfft_sigma, m=nfft_m,
                                   use_double=use_double, **kwargs)
            single_nffts.append(nfft)

        multi_nffts = proc.run(datas, **kwargs)

        batch_nffts = proc.batched_run(datas, batch_size=batch_size, **kwargs)
        proc.finish()

        tols = dict(rtol=nfft_rtol, atol=nfft_atol)
        for ghat_m, ghat_s, ghat_b in zip(multi_nffts, single_nffts,
                                          batch_nffts):
            assert_allclose(ghat_s.real, ghat_m.real, **tols)
            assert_allclose(ghat_s.imag, ghat_m.imag, **tols)

            assert_allclose(ghat_s.real, ghat_b.real, **tols)
            assert_allclose(ghat_s.imag, ghat_b.imag, **tols)
