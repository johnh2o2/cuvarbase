import numpy as np
from numpy.testing import assert_allclose
import pytest
from pycuda.tools import mark_cuda_test
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


class TestCpuFunctionsDoNotMutateInputs(object):
    """The CPU reference functions used to do `t -= mean(t)` in place,
    silently modifying the caller's arrays."""

    def _data(self):
        rand = np.random.RandomState(7)
        t = np.sort(10 * rand.rand(40))
        y = np.cos(2 * np.pi * 2.0 * t) + 0.1 * rand.randn(40)
        w = np.ones_like(y) / len(y)
        return t, y, w

    def test_binless_pdm_cpu(self):
        from ..pdm import binless_pdm_cpu
        t, y, w = self._data()
        t0, y0, w0 = t.copy(), y.copy(), w.copy()
        binless_pdm_cpu(t, y, w, np.array([1.0, 2.0]))
        assert np.array_equal(t, t0)
        assert np.array_equal(y, y0)
        assert np.array_equal(w, w0)

    def test_pdm2_cpu(self):
        from ..pdm import pdm2_cpu
        t, y, w = self._data()
        t0, y0, w0 = t.copy(), y.copy(), w.copy()
        pdm2_cpu(t, y, w, np.array([1.0, 2.0]))
        assert np.array_equal(t, t0)
        assert np.array_equal(y, y0)
        assert np.array_equal(w, w0)

    def test_pdm2_single_freq(self):
        from ..pdm import pdm2_single_freq
        t, y, w = self._data()
        t0, y0, w0 = t.copy(), y.copy(), w.copy()
        pdm2_single_freq(t, y, w, 2.0)
        assert np.array_equal(t, t0)
        assert np.array_equal(y, y0)
        assert np.array_equal(w, w0)


# ---------------------------------------------------------------------------
# Regression tests from the Sep-2026 algorithm audit (finding ids 110, 111, 113)
# ---------------------------------------------------------------------------

def _ref_binned_step_float32(t, y, w, freqs, nbins):
    """float64 ``1 - SS_within / SS_total`` for ``kind='binned_step'`` with
    the phase fold emulated in float32 exactly as ``pdm.cu`` does it
    (``PHASE(t, f) = t*f - floorf(t*f)``, ``bin = int(phase*nbins) % nbins``).

    Returns ``(power, n_occupied_bins)`` per frequency.
    """
    t = t - np.mean(t)
    y = y - np.mean(y)
    w = w / np.sum(w)
    ybar = np.dot(w, y)
    ss_tot = np.dot(w, (y - ybar) ** 2)
    t32 = t.astype(np.float32)
    f32 = np.asarray(freqs).astype(np.float32)
    power = np.empty(len(f32))
    n_occ = np.empty(len(f32), dtype=int)
    for i, f in enumerate(f32):
        tf = t32 * f
        phase = (tf - np.floor(tf)).astype(np.float64)
        b = (phase * nbins).astype(int) % nbins
        wtot = np.bincount(b, weights=w, minlength=nbins)
        wsum = np.bincount(b, weights=w * y, minlength=nbins)
        means = np.where(wtot > 0, wsum / np.where(wtot > 0, wtot, 1.0), 0.0)
        power[i] = 1 - np.dot(w, (y - means[b]) ** 2) / ss_tot
        n_occ[i] = np.count_nonzero(wtot)
    return power, n_occ


def _phase_exactly_one_lightcurve():
    """Times whose float64 mean is ~0 and which contain a point at t = -1e-9.

    In float32, ``t*f`` for that point lies in (-2**-25, 0) at the trial
    frequencies returned here, so ``PHASE(t, f) = t*f - floorf(t*f)`` rounds
    to exactly 1.0f and ``(int)(PHASE * NBINS)`` is ``NBINS`` -- one past the
    end of the per-thread bin arrays unless the kernel wraps it.
    """
    rand = np.random.RandomState(4)
    base = np.array([-3.0, 3.0, -2.5, 2.5, -1.7, 1.7, -0.9, 0.9, -1e-9, 1e-9])
    more = 3 * rand.rand(40)
    t = np.concatenate([base, more, -more])
    freqs = np.array([2.0, 3.0, 0.5, 4.0])
    # precondition: the -1e-9 point really folds to float32 phase 1.0
    t32 = (t - np.mean(t)).astype(np.float32)
    tf = t32[8] * freqs.astype(np.float32)
    assert np.all(tf - np.floor(tf) == np.float32(1.0))
    return t, freqs


@mark_cuda_test
def test_binned_step_phase_exactly_one_no_oob_read():
    """Audit id 113: ``var_step_function`` (kind='binned_step') indexed
    ``bin_means[NBINS]`` when a float32 phase rounds to exactly 1.0, while
    its own accumulation loop, the linterp kernel and the ``_fast`` kernels
    all wrap with ``bin % NBINS``.  Pre-fix (A40): |binned_step -
    binned_step_fast| = 0.012..0.024 at the affected frequencies; post-fix
    the kernels agree to float32 round-off (< 1e-7).
    """
    t, freqs = _phase_exactly_one_lightcurve()
    err = np.ones_like(t)
    rand = np.random.RandomState(113)
    proc = PDMAsyncProcess()

    def run(kind, t, y, err):
        res = proc.run([(t, y, err)], freqs=freqs, kind=kind, nbins=10)
        proc.finish()
        return np.copy(res[0][1])

    worst_vs_fast, worst_vs_ref = 0.0, 0.0
    for _ in range(10):
        y = 12 + rand.randn(len(t))
        step = run('binned_step', t, y, err)
        fast = run('binned_step_fast', t, y, err)
        ref, _ = _ref_binned_step_float32(t, y, weights(err), freqs, 10)
        assert np.all(np.isfinite(step))
        worst_vs_fast = max(worst_vs_fast, np.max(np.abs(step - fast)))
        worst_vs_ref = max(worst_vs_ref, np.max(np.abs(step - ref)))
    assert worst_vs_fast < 5e-6
    assert worst_vs_ref < 5e-6

    # the statistic must not depend on the order of the observations
    perm = rand.permutation(len(t))
    assert_allclose(run('binned_step', t[perm], y[perm], err[perm]), step,
                    atol=5e-6, rtol=0)


@mark_cuda_test
def test_deprecated_format_normalizes_weights():
    """Audit id 111: the deprecated ``(t, y, w, freqs)`` input format assumed
    ``sum(w) == 1``.  With raw inverse-variance weights (or all ones) the
    host-side weighted mean and variance were scaled by ``sum(w)`` and every
    kind returned a flat spectrum of exactly 1.0.  ``run()`` now normalizes
    ``w`` (the statistic is invariant to the scale of ``w``), so the legacy
    path must agree with the modern ``(t, y, err)`` path for any scaling.
    """
    rand = np.random.RandomState(111)
    n = 300
    t = np.sort(30 * rand.rand(n))
    y = 12 + np.sin(2 * np.pi * 1.7 * t) + 0.2 * rand.randn(n)
    err = 0.2 * (0.5 + rand.rand(n))
    freqs = np.linspace(0.05, 5.0, 400)
    proc = PDMAsyncProcess()

    def run(data, kind, **kw):
        res = proc.run(data, kind=kind, nbins=10, dphi=0.05, **kw)
        proc.finish()
        return res

    cases = [
        (err ** -2, err),           # raw inverse variance: sum(w) != 1
        (weights(err), err),        # already normalized
        (np.ones(n), np.ones(n)),   # uniform weights: sum(w) == n
    ]
    for kind in ['binned_linterp', 'binned_step_fast',
                 'binless_tophat', 'binless_gauss_fast']:
        for w, err_equiv in cases:
            modern = np.copy(run([(t, y, err_equiv)], kind, freqs=freqs)[0][1])
            with pytest.warns(DeprecationWarning):
                legacy = np.copy(run([(t, y, w, freqs)], kind)[0])
            assert np.ptp(modern) > 0.5          # a real periodogram
            assert_allclose(legacy, modern, atol=1e-6, rtol=0)
