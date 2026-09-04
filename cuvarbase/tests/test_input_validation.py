"""Input validation on every public entry point (Sep 2026 audit,
defect 23 ``input-validation``).

Before 1.0 nothing checked the light curve. A single NaN in ``t`` gave
a finite periodogram with the wrong argmax on the BLS and CE paths;
``dy = 0`` or a non-finite ``y`` gave an all-NaN spectrum (PDM), an
undocumented power of ``-1`` at every frequency (Lomb-Scargle) or a
chi2 off by a factor 1.3e3 (TLS fast); and a NaN per-frequency ``q``
bound, ``qmax >= 1``, or a Keplerian grid built from fewer points than
one transit needs crashed the kernel with

    cuMemcpyDtoH failed: an illegal memory access was encountered

which **kills the CUDA context for the rest of the process** -- every
later call in the same interpreter then fails with
``cuMemAlloc failed: an illegal memory access``.

Every entry point now calls :func:`cuvarbase.utils.check_lightcurve`
and :func:`cuvarbase.utils.check_freqs` before any device work
(compilation included), so almost all of these tests run without a
GPU: the ``ValueError`` is raised on the host. The one genuinely
device-bound test is
:func:`test_cuda_context_survives_rejected_calls`, which is the whole
point of the defect.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..utils import check_lightcurve, check_freqs
from ..bls import (sparse_bls_cpu, sparse_bls_gpu, eebls_gpu,
                   eebls_gpu_fast, eebls_gpu_fast_optimized,
                   eebls_gpu_fast_adaptive, eebls_gpu_custom,
                   eebls_gpu_batch, eebls_transit, eebls_transit_gpu,
                   single_bls, fmin_transit, transit_autofreq)
from .. import tls as TLS
from ..lombscargle import LombScargleAsyncProcess, lomb_scargle_simple
from ..ce import ConditionalEntropyAsyncProcess
from ..pdm import PDMAsyncProcess
from ..cunfft import NFFTAsyncProcess
from ..nufft_lrt import NUFFTLRTAsyncProcess


# ---------------------------------------------------------------- data

def make_lc(ndata=60, baseline=10., seed=42, freq=1.0, q=0.15,
            depth=0.05, sigma=0.005):
    """Deterministic box-transit light curve."""
    rand = np.random.RandomState(seed)
    t = baseline * np.sort(rand.rand(ndata))
    y = np.ones(ndata)
    phase = (t * freq) % 1.0
    y[phase < q] -= depth
    dy = sigma * np.ones(ndata)
    y = y + dy * rand.randn(ndata)
    return t, y, dy


BLS_FREQS = np.linspace(0.5, 2.0, 12)
LS_FREQS = 0.1 * (1 + np.arange(32))          # df * (k0 + arange(nf))
TLS_PERIODS = np.linspace(0.8, 1.4, 6)


# ------------------------------------------------------- entry points
#
# Each entry is (name, callable(t, y, dy), min_n, takes_dy, grid_kind).
# ``grid_kind`` names the validator the trial grid goes through:
# 'freqs' -> check_freqs, 'periods' -> the TLS/LRT period validator,
# None -> the entry point takes no grid.

def _ls_proc():
    return LombScargleAsyncProcess()


def _ce_proc():
    return ConditionalEntropyAsyncProcess()


def _pdm_proc():
    return PDMAsyncProcess()


ENTRY_POINTS = [
    # ---- BLS -------------------------------------------------------
    ('sparse_bls_cpu',
     lambda t, y, dy, f=None: sparse_bls_cpu(
         t, y, dy, BLS_FREQS if f is None else f), 2, True, 'freqs'),
    ('sparse_bls_gpu',
     lambda t, y, dy, f=None: sparse_bls_gpu(
         t, y, dy, BLS_FREQS if f is None else f), 2, True, 'freqs'),
    ('eebls_gpu',
     lambda t, y, dy, f=None: eebls_gpu(
         t, y, dy, BLS_FREQS if f is None else f), 2, True, 'freqs'),
    ('eebls_gpu_fast',
     lambda t, y, dy, f=None: eebls_gpu_fast(
         t, y, dy, BLS_FREQS if f is None else f), 2, True, 'freqs'),
    ('eebls_gpu_fast_optimized',
     lambda t, y, dy, f=None: eebls_gpu_fast_optimized(
         t, y, dy, BLS_FREQS if f is None else f), 2, True, 'freqs'),
    ('eebls_gpu_fast_adaptive',
     lambda t, y, dy, f=None: eebls_gpu_fast_adaptive(
         t, y, dy, BLS_FREQS if f is None else f), 2, True, 'freqs'),
    ('eebls_gpu_custom',
     lambda t, y, dy, f=None: eebls_gpu_custom(
         t, y, dy, BLS_FREQS if f is None else f,
         np.array([0.05, 0.1]), np.linspace(0, 1, 8, endpoint=False)),
     2, True, 'freqs'),
    ('eebls_gpu_batch',
     lambda t, y, dy, f=None: eebls_gpu_batch(
         [(t, y, dy)], BLS_FREQS if f is None else f), 2, True, 'freqs'),
    ('eebls_transit',
     lambda t, y, dy, f=None: eebls_transit(
         t, y, dy, freqs=BLS_FREQS if f is None else f,
         qvals=np.full(len(BLS_FREQS if f is None else f), 0.1)),
     2, True, 'freqs'),
    ('eebls_transit_gpu',
     lambda t, y, dy, f=None: eebls_transit_gpu(
         t, y, dy, freqs=BLS_FREQS if f is None else f,
         qvals=np.full(len(BLS_FREQS if f is None else f), 0.1)),
     2, True, 'freqs'),
    ('single_bls',
     lambda t, y, dy, f=None: single_bls(t, y, dy, 1.0, 0.1, 0.0),
     2, True, None),
    # ---- TLS -------------------------------------------------------
    ('tls_search',
     lambda t, y, dy, f=None: TLS.tls_search(
         t, y, dy, periods=TLS_PERIODS if f is None else f),
     2, True, 'periods'),
    ('tls_search_gpu',
     lambda t, y, dy, f=None: TLS.tls_search_gpu(
         t, y, dy, periods=TLS_PERIODS if f is None else f),
     2, True, 'periods'),
    ('tls_transit',
     lambda t, y, dy, f=None: TLS.tls_transit(
         t, y, dy, period_min=0.8, period_max=1.4),
     2, True, None),
    ('tls_search_batch',
     lambda t, y, dy, f=None: TLS.tls_search_batch(
         [(t, y, dy)], periods=TLS_PERIODS if f is None else f),
     2, True, 'periods'),
    # ---- Lomb-Scargle ----------------------------------------------
    ('lomb_scargle_simple',
     lambda t, y, dy, f=None: lomb_scargle_simple(
         t, y, dy, freqs=[LS_FREQS if f is None else f]),
     4, True, 'freqs'),
    ('LombScargleAsyncProcess.run',
     lambda t, y, dy, f=None: _ls_proc().run(
         [(t, y, dy)], freqs=[LS_FREQS if f is None else f]),
     4, True, 'freqs'),
    ('LombScargleAsyncProcess.batched_run_const_nfreq',
     lambda t, y, dy, f=None: _ls_proc().batched_run_const_nfreq(
         [(t, y, dy)], freqs=LS_FREQS if f is None else f),
     4, True, 'freqs'),
    # ---- conditional entropy ---------------------------------------
    ('ConditionalEntropyAsyncProcess.run',
     lambda t, y, dy, f=None: _ce_proc().run(
         [(t, y, dy)], freqs=[BLS_FREQS if f is None else f]),
     2, True, 'freqs'),
    ('ConditionalEntropyAsyncProcess.large_run',
     lambda t, y, dy, f=None: _ce_proc().large_run(
         [(t, y, dy)], freqs=[BLS_FREQS if f is None else f]),
     2, True, 'freqs'),
    ('ConditionalEntropyAsyncProcess.batched_run_const_nfreq',
     lambda t, y, dy, f=None: _ce_proc().batched_run_const_nfreq(
         [(t, y, dy)], freqs=BLS_FREQS if f is None else f),
     2, True, 'freqs'),
    # ---- PDM -------------------------------------------------------
    ('PDMAsyncProcess.run',
     lambda t, y, dy, f=None: _pdm_proc().run(
         [(t, y, dy)], freqs=BLS_FREQS if f is None else f),
     2, True, 'freqs'),
    ('PDMAsyncProcess.batched_run_const_nfreq',
     lambda t, y, dy, f=None: _pdm_proc().batched_run_const_nfreq(
         [(t, y, dy)], freqs=BLS_FREQS if f is None else f),
     2, True, 'freqs'),
    ('PDMAsyncProcess.large_run',
     lambda t, y, dy, f=None: _pdm_proc().large_run(
         [(t, y, dy)], freqs=BLS_FREQS if f is None else f),
     2, True, 'freqs'),
    # ---- NFFT / NUFFT-LRT ------------------------------------------
    ('NFFTAsyncProcess.run',
     lambda t, y, dy, f=None: NFFTAsyncProcess().run([(t, y, 64)]),
     2, False, None),
    ('NUFFTLRTAsyncProcess.run',
     lambda t, y, dy, f=None: NUFFTLRTAsyncProcess().run(
         t, y, TLS_PERIODS if f is None else f,
         durations=np.array([0.1])),
     3, False, 'periods'),
]

ALL_IDS = [e[0] for e in ENTRY_POINTS]
WITH_DY = [e for e in ENTRY_POINTS if e[3]]
WITH_FREQS = [e for e in ENTRY_POINTS if e[4] == 'freqs']
WITH_PERIODS = [e for e in ENTRY_POINTS if e[4] == 'periods']


def _ids(entries):
    return [e[0] for e in entries]


# ------------------------------------------------ the validators alone

class TestCheckLightcurve(object):
    """``utils.check_lightcurve`` itself: message content and the
    guarantee that it does not touch valid input."""

    def test_accepts_valid_input_unchanged(self):
        t, y, dy = make_lc(20)
        t2, y2, dy2 = check_lightcurve(t, y, dy, min_n=5, name='x')
        # returned as-is (no copy, no cast) so it cannot perturb results
        assert t2 is t and y2 is y and dy2 is dy

    def test_dy_none_is_allowed(self):
        t, y, _ = make_lc(20)
        t2, y2, dy2 = check_lightcurve(t, y, None)
        assert dy2 is None

    def test_integer_arrays_are_accepted(self):
        t = np.arange(10)
        y = np.arange(10) * 2
        check_lightcurve(t, y, np.ones(10, dtype=np.int64))

    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_dtypes(self, dtype):
        t, y, dy = make_lc(10)
        check_lightcurve(t.astype(dtype), y.astype(dtype),
                         dy.astype(dtype))

    def test_nan_in_t_names_t_and_the_index(self):
        t, y, dy = make_lc(20)
        t = t.copy()
        t[7] = np.nan
        with pytest.raises(ValueError) as exc:
            check_lightcurve(t, y, dy, name='thing')
        msg = str(exc.value)
        assert msg.startswith('thing: t ')
        assert '1 non-finite' in msg
        assert 'index/indices 7' in msg

    def test_counts_and_first_indices(self):
        t, y, dy = make_lc(20)
        y = y.copy()
        y[[2, 5, 9, 11, 13, 17]] = np.inf
        with pytest.raises(ValueError, match=r'y contains 6 non-finite'):
            check_lightcurve(t, y, dy)
        with pytest.raises(ValueError,
                           match=r'indices 2, 5, 9, 11, 13, \.\.\.'):
            check_lightcurve(t, y, dy)

    def test_dy_zero_and_negative(self):
        t, y, dy = make_lc(20)
        for bad in (0.0, -1e-3):
            d = dy.copy()
            d[3] = bad
            with pytest.raises(ValueError) as exc:
                check_lightcurve(t, y, d, name='thing')
            msg = str(exc.value)
            assert 'dy must be > 0' in msg
            assert '1 of 20' in msg
            assert 'indices 3' in msg

    def test_length_mismatch(self):
        t, y, dy = make_lc(20)
        with pytest.raises(ValueError, match='t and y must have the same'):
            check_lightcurve(t, y[:-1], dy)
        with pytest.raises(ValueError, match='t and dy must have the same'):
            check_lightcurve(t, y, dy[:-1])

    def test_min_n(self):
        t, y, dy = make_lc(3)
        with pytest.raises(ValueError, match='at least 5 observation'):
            check_lightcurve(t, y, dy, min_n=5)
        with pytest.raises(ValueError, match='at least 1 observation'):
            check_lightcurve(t[:0], y[:0], dy[:0])

    def test_shape_and_dtype_guards(self):
        with pytest.raises(ValueError, match='1-D'):
            check_lightcurve(np.zeros((2, 3)), np.zeros((2, 3)))
        with pytest.raises(ValueError, match='numeric'):
            check_lightcurve(np.array(['a', 'b']), np.zeros(2))


class TestCheckFreqs(object):

    def test_accepts_valid_grid_unchanged(self):
        f = check_freqs(BLS_FREQS)
        assert f is BLS_FREQS

    def test_non_finite(self):
        f = BLS_FREQS.copy()
        f[2] = np.nan
        with pytest.raises(ValueError, match='freqs contains 1 non-finite'):
            check_freqs(f, name='thing')

    def test_non_positive(self):
        f = np.array([-1.0, 0.0, 1.0])
        with pytest.raises(ValueError) as exc:
            check_freqs(f, name='thing')
        msg = str(exc.value)
        assert 'thing: freqs must be > 0' in msg
        assert '2 of 3' in msg
        assert 'indices 0, 1' in msg

    def test_empty(self):
        with pytest.raises(ValueError, match='non-empty'):
            check_freqs(np.array([]))


# ------------------------------------- every entry point, every poison

class TestEntryPointsRejectBadLightcurves(object):
    """Each public entry point must raise ``ValueError`` naming the
    offending array, on the host, before any GPU work."""

    @pytest.mark.parametrize('entry', ENTRY_POINTS, ids=ALL_IDS)
    def test_nan_in_t(self, entry):
        _, fn, min_n, _, _ = entry
        t, y, dy = make_lc(60)
        t = t.copy()
        t[17] = np.nan
        with pytest.raises(ValueError, match=r'\bt\b.*non-finite'):
            fn(t, y, dy)

    @pytest.mark.parametrize('entry', ENTRY_POINTS, ids=ALL_IDS)
    def test_inf_in_y(self, entry):
        _, fn, min_n, _, _ = entry
        t, y, dy = make_lc(60)
        y = y.copy()
        y[3] = np.inf
        with pytest.raises(ValueError, match=r'\by\b.*non-finite'):
            fn(t, y, dy)

    @pytest.mark.parametrize('entry', WITH_DY, ids=_ids(WITH_DY))
    def test_nan_in_dy(self, entry):
        _, fn, min_n, _, _ = entry
        t, y, dy = make_lc(60)
        dy = dy.copy()
        dy[41] = np.nan
        with pytest.raises(ValueError, match=r'\bdy\b.*non-finite'):
            fn(t, y, dy)

    @pytest.mark.parametrize('entry', WITH_DY, ids=_ids(WITH_DY))
    def test_zero_dy(self, entry):
        _, fn, min_n, _, _ = entry
        t, y, dy = make_lc(60)
        dy = dy.copy()
        dy[0] = 0.0
        with pytest.raises(ValueError, match='dy must be > 0'):
            fn(t, y, dy)

    @pytest.mark.parametrize('entry', WITH_DY, ids=_ids(WITH_DY))
    def test_negative_dy(self, entry):
        _, fn, min_n, _, _ = entry
        t, y, dy = make_lc(60)
        dy = dy.copy()
        dy[59] = -dy[59]
        with pytest.raises(ValueError, match='dy must be > 0'):
            fn(t, y, dy)

    @pytest.mark.parametrize('entry', ENTRY_POINTS, ids=ALL_IDS)
    def test_mismatched_lengths(self, entry):
        _, fn, min_n, takes_dy, _ = entry
        t, y, dy = make_lc(60)
        with pytest.raises(ValueError, match='same length'):
            fn(t, y[:-1], dy)
        if takes_dy:
            with pytest.raises(ValueError, match='same length'):
                fn(t, y, dy[:-1])

    @pytest.mark.parametrize('entry', ENTRY_POINTS, ids=ALL_IDS)
    def test_empty_arrays(self, entry):
        _, fn, min_n, _, _ = entry
        t, y, dy = make_lc(60)
        with pytest.raises(ValueError, match='at least'):
            fn(t[:0], y[:0], dy[:0])

    @pytest.mark.parametrize('entry', ENTRY_POINTS, ids=ALL_IDS)
    def test_below_method_minimum(self, entry):
        name, fn, min_n, _, _ = entry
        if min_n < 2:
            pytest.skip('%s accepts a single observation' % name)
        t, y, dy = make_lc(min_n - 1)
        with pytest.raises(ValueError,
                           match='at least %d observation' % min_n):
            fn(t, y, dy)


class TestEntryPointsRejectBadGrids(object):

    @pytest.mark.parametrize('entry', WITH_FREQS, ids=_ids(WITH_FREQS))
    def test_non_finite_freqs(self, entry):
        _, fn, min_n, _, _ = entry
        t, y, dy = make_lc(60)
        bad = np.array([0.5, np.nan, 1.5, 2.0])
        with pytest.raises(ValueError, match='freqs contains 1 non-finite'):
            fn(t, y, dy, bad)

    @pytest.mark.parametrize('entry', WITH_FREQS, ids=_ids(WITH_FREQS))
    def test_non_positive_freqs(self, entry):
        _, fn, min_n, _, _ = entry
        t, y, dy = make_lc(60)
        bad = np.array([0.0, 0.5, 1.0, 1.5])
        with pytest.raises(ValueError, match='freqs must be > 0'):
            fn(t, y, dy, bad)

    @pytest.mark.parametrize('entry', WITH_PERIODS, ids=_ids(WITH_PERIODS))
    def test_non_finite_periods(self, entry):
        _, fn, min_n, _, _ = entry
        t, y, dy = make_lc(60)
        with pytest.raises(ValueError, match='periods'):
            fn(t, y, dy, np.array([1.0, np.nan]))

    @pytest.mark.parametrize('entry', WITH_PERIODS, ids=_ids(WITH_PERIODS))
    def test_non_positive_periods(self, entry):
        _, fn, min_n, _, _ = entry
        t, y, dy = make_lc(60)
        with pytest.raises(ValueError, match='periods'):
            fn(t, y, dy, np.array([1.0, -1.0]))


# --------------------------------------------------- BLS q-bound rules

#: entry points whose kernels bin phase, so ``0 < qmin <= qmax <= 1``
BINNED_Q_ENTRIES = [
    ('eebls_gpu', lambda t, y, dy, **kw: eebls_gpu(
        t, y, dy, BLS_FREQS, **kw)),
    ('eebls_gpu_fast', lambda t, y, dy, **kw: eebls_gpu_fast(
        t, y, dy, BLS_FREQS, **kw)),
    ('eebls_gpu_fast_optimized',
     lambda t, y, dy, **kw: eebls_gpu_fast_optimized(
         t, y, dy, BLS_FREQS, **kw)),
    ('eebls_gpu_fast_adaptive',
     lambda t, y, dy, **kw: eebls_gpu_fast_adaptive(
         t, y, dy, BLS_FREQS, **kw)),
    ('eebls_gpu_batch', lambda t, y, dy, **kw: eebls_gpu_batch(
        [(t, y, dy)], BLS_FREQS, **kw)),
]
BINNED_Q_IDS = [e[0] for e in BINNED_Q_ENTRIES]

SPARSE_Q_ENTRIES = [
    ('sparse_bls_cpu', lambda t, y, dy, **kw: sparse_bls_cpu(
        t, y, dy, BLS_FREQS, **kw)),
    ('sparse_bls_gpu', lambda t, y, dy, **kw: sparse_bls_gpu(
        t, y, dy, BLS_FREQS, **kw)),
]
SPARSE_Q_IDS = [e[0] for e in SPARSE_Q_ENTRIES]


class TestQBoundValidation(object):
    """``BLSMemory.setdata`` casts ``1/qmin`` and ``1/qmax`` to uint32:
    ``(1 / [nan, 0.01, 5, inf]).astype(uint32)`` is ``[0, 100, 0, 0]``,
    and a zero bin count divides by zero in the kernel and atomicAdds
    outside the shared-memory histogram (illegal memory access, dead
    CUDA context). Every bound is checked before that cast."""

    @pytest.mark.parametrize('entry', BINNED_Q_ENTRIES + SPARSE_Q_ENTRIES,
                             ids=BINNED_Q_IDS + SPARSE_Q_IDS)
    def test_nan_scalar_q(self, entry):
        _, fn = entry
        t, y, dy = make_lc(60)
        with pytest.raises(ValueError, match='finite'):
            fn(t, y, dy, qmin=np.nan, qmax=0.2)
        with pytest.raises(ValueError, match='finite'):
            fn(t, y, dy, qmin=0.01, qmax=np.nan)

    @pytest.mark.parametrize('entry', BINNED_Q_ENTRIES + SPARSE_Q_ENTRIES,
                             ids=BINNED_Q_IDS + SPARSE_Q_IDS)
    def test_nan_per_frequency_q(self, entry):
        """The exact case that killed the CUDA context."""
        _, fn = entry
        t, y, dy = make_lc(60)
        qmin = np.full(len(BLS_FREQS), 0.01)
        qmax = np.full(len(BLS_FREQS), 0.2)
        qmin[4] = np.nan
        with pytest.raises(ValueError, match='finite'):
            fn(t, y, dy, qmin=qmin, qmax=qmax)
        qmin[4] = 0.01
        qmax[7] = np.nan
        with pytest.raises(ValueError, match='finite'):
            fn(t, y, dy, qmin=qmin, qmax=qmax)

    @pytest.mark.parametrize('entry', BINNED_Q_ENTRIES + SPARSE_Q_ENTRIES,
                             ids=BINNED_Q_IDS + SPARSE_Q_IDS)
    def test_inverted_q(self, entry):
        _, fn = entry
        t, y, dy = make_lc(60)
        with pytest.raises(ValueError, match='qmin > qmax'):
            fn(t, y, dy, qmin=0.3, qmax=0.1)

    @pytest.mark.parametrize('entry', BINNED_Q_ENTRIES, ids=BINNED_Q_IDS)
    def test_qmax_at_or_above_one(self, entry):
        """``qmax >= 1`` makes ``nbins0 = floor(1/qmax) = 0``: a device
        divide by zero that returned finite garbage."""
        _, fn = entry
        t, y, dy = make_lc(60)
        with pytest.raises(ValueError, match='qmax must be <= 1'):
            fn(t, y, dy, qmin=0.01, qmax=1.5)
        # inf is caught one step earlier, by the finiteness check
        with pytest.raises(ValueError, match='finite'):
            fn(t, y, dy, qmin=0.01, qmax=np.inf)

    @pytest.mark.parametrize('entry', BINNED_Q_ENTRIES, ids=BINNED_Q_IDS)
    def test_qmin_zero(self, entry):
        """``qmin = 0`` asks for infinitely many phase bins (and casts
        to a bin count of 0)."""
        _, fn = entry
        t, y, dy = make_lc(60)
        with pytest.raises(ValueError, match='qmin must be > 0'):
            fn(t, y, dy, qmin=0.0, qmax=0.2)

    def test_qmax_exactly_one_is_allowed(self):
        """The boundary must stay usable: nbins0 = 1 is a single box
        covering the whole period."""
        from ..bls import _validate_fast_q_bounds
        _validate_fast_q_bounds(4, 0.01, 1.0)


class TestKeplerianGridGuards(object):
    """``fmin_transit`` gives ``q = min_obs_per_transit / N > 1`` below
    ``min_obs_per_transit`` points, and ``freq_transit`` of ``q > 1`` is
    NaN: ``transit_autofreq`` used to return ``freqs = [nan]``,
    ``q = [nan]``, which reached the kernels as a zero uint32 bin count
    and killed the CUDA context on the ``use_fast=True`` path."""

    def test_fmin_transit_raises_below_min_obs(self):
        t = np.linspace(0, 10, 4)
        with pytest.raises(ValueError, match='min_obs_per_transit'):
            fmin_transit(t)
        # explicitly lowering the requirement still works
        assert np.isfinite(fmin_transit(t, min_obs_per_transit=2))

    def test_fmin_transit_rejects_non_finite_times(self):
        t = np.linspace(0, 10, 20)
        t[3] = np.nan
        with pytest.raises(ValueError, match='finite'):
            fmin_transit(t)
        with pytest.raises(ValueError, match='non-empty'):
            fmin_transit(t[:0])

    def test_transit_autofreq_rejects_non_finite_times(self):
        t = np.linspace(0, 10, 20)
        t[3] = np.nan
        with pytest.raises(ValueError, match='finite'):
            transit_autofreq(t)

    def test_transit_autofreq_grid_is_finite(self):
        t = np.linspace(0, 100, 500)
        freqs, qvals = transit_autofreq(t)
        assert np.all(np.isfinite(freqs)) and np.all(freqs > 0)
        assert np.all(np.isfinite(qvals)) and np.all(qvals > 0)

    @pytest.mark.parametrize('fn', [eebls_transit, eebls_transit_gpu],
                             ids=['eebls_transit', 'eebls_transit_gpu'])
    @pytest.mark.parametrize('use_fast', [False, True])
    def test_keplerian_entry_points_with_too_few_points(self, fn,
                                                        use_fast):
        t, y, dy = make_lc(4)
        with pytest.raises(ValueError):
            fn(t, y, dy, use_fast=use_fast)

    @pytest.mark.parametrize('fn', [eebls_transit, eebls_transit_gpu],
                             ids=['eebls_transit', 'eebls_transit_gpu'])
    @pytest.mark.parametrize('use_fast', [False, True])
    def test_keplerian_entry_points_with_nan_time(self, fn, use_fast):
        t, y, dy = make_lc(60)
        t = t.copy()
        t[11] = np.nan
        with pytest.raises(ValueError, match=r'\bt\b.*non-finite'):
            fn(t, y, dy, use_fast=use_fast)


class TestBatchEntryPoints(object):
    """The batch APIs validate every light curve and name the bad one."""

    def test_bls_batch_names_the_bad_lightcurve(self):
        good = make_lc(60, seed=1)
        bad = list(make_lc(60, seed=2))
        bad[2] = bad[2].copy()
        bad[2][5] = 0.0
        with pytest.raises(ValueError, match='lightcurve 1'):
            eebls_gpu_batch([good, tuple(bad)], BLS_FREQS)

    def test_tls_batch_names_the_bad_lightcurve(self):
        good = make_lc(60, seed=1)
        bad = list(make_lc(60, seed=2))
        bad[0] = bad[0].copy()
        bad[0][5] = np.nan
        with pytest.raises(ValueError, match='lightcurve 1'):
            TLS.tls_search_batch([good, tuple(bad)], periods=TLS_PERIODS)

    def test_ls_run_names_the_bad_lightcurve(self):
        good = make_lc(60, seed=1)
        bad = list(make_lc(60, seed=2))
        bad[1] = bad[1].copy()
        bad[1][5] = np.nan
        with pytest.raises(ValueError, match='lightcurve 1'):
            _ls_proc().run([good, tuple(bad)], freqs=[LS_FREQS] * 2)

    def test_ce_run_names_the_bad_lightcurve(self):
        good = make_lc(60, seed=1)
        bad = list(make_lc(60, seed=2))
        bad[1] = bad[1].copy()
        bad[1][5] = np.inf
        with pytest.raises(ValueError, match='lightcurve 1'):
            _ce_proc().run([good, tuple(bad)], freqs=BLS_FREQS)

    def test_pdm_run_names_the_bad_lightcurve(self):
        good = make_lc(60, seed=1)
        bad = list(make_lc(60, seed=2))
        bad[2] = bad[2].copy()
        bad[2][5] = -1.0
        with pytest.raises(ValueError, match='lightcurve 1'):
            _pdm_proc().run([good, tuple(bad)], freqs=BLS_FREQS)

    def test_pdm_deprecated_format_is_validated(self):
        from ..utils import weights
        t, y, dy = make_lc(60)
        w = weights(dy)
        bad_w = w.copy()
        bad_w[3] = 0.0
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError, match='w must be finite'):
                _pdm_proc().run([(t, y, bad_w, BLS_FREQS)])
        t_bad = t.copy()
        t_bad[3] = np.nan
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError, match=r'\bt\b.*non-finite'):
                _pdm_proc().run([(t_bad, y, w, BLS_FREQS)])


class TestSingleBlsScalarGuards(object):

    def test_non_finite_scalars(self):
        t, y, dy = make_lc(60)
        for kw in ({'freq': np.nan}, {'q': np.nan}, {'phi0': np.nan}):
            args = dict(freq=1.0, q=0.1, phi0=0.0)
            args.update(kw)
            with pytest.raises(ValueError, match='finite'):
                single_bls(t, y, dy, **args)

    def test_non_positive_frequency(self):
        t, y, dy = make_lc(60)
        with pytest.raises(ValueError, match='freq must be > 0'):
            single_bls(t, y, dy, 0.0, 0.1, 0.0)


class TestNFFTGuards(object):

    def test_nf_must_be_a_positive_integer(self):
        t, y, _ = make_lc(30)
        for bad in (0, -8, 12.5, np.nan):
            with pytest.raises(ValueError, match='nf'):
                NFFTAsyncProcess().run([(t, y, bad)])

    def test_adjoint_scalar_guards(self):
        """``nfft_adjoint_async`` gets its light curve through
        ``memory``; its own scalars are still checked."""
        from ..cunfft import nfft_adjoint_async
        for bad in (np.nan, np.inf, -1.0):
            with pytest.raises(ValueError, match='minimum_frequency'):
                nfft_adjoint_async(None, None, minimum_frequency=bad)
        for bad in (np.nan, 0.0, -2.0):
            with pytest.raises(ValueError, match='samples_per_peak'):
                nfft_adjoint_async(None, None, samples_per_peak=bad)


# -------------------------------------------------------- on a device
#
# These two need a real GPU. They are NOT decorated with
# ``pycuda.tools.mark_cuda_test`` -- like every other BLS test
# (``test_bls.py`` uses none) -- because that decorator runs each test
# in a freshly created, non-primary CUDA context while ``bls.py`` keeps
# a process-wide LRU cache of compiled kernels: a cache entry compiled
# under an earlier test's context raises ``cuFuncSetBlockShape failed:
# invalid resource handle`` when it is reused under a new one. Calling
# the entry points directly runs them in cuvarbase's own primary
# context; on a GPU-less machine the root ``conftest.py`` turns the
# resulting ``GPUStubError`` into a skip.

def test_cuda_context_survives_rejected_calls():
    """The payoff of defect 23.

    Each of these calls used to raise ``cuMemcpyDtoH failed: an illegal
    memory access was encountered`` from inside the kernel, which
    destroys the process's CUDA context: every subsequent GPU call in
    the same interpreter then failed with ``cuMemAlloc failed: an
    illegal memory access``, so a single bad light curve in a survey
    pipeline poisoned the whole worker. They must now be rejected on
    the host, leaving the context untouched.
    """
    freq = 1.0
    t, y, dy = make_lc(ndata=400, baseline=20., freq=freq, q=0.12,
                       depth=0.05, sigma=0.004, seed=5)
    freqs = np.linspace(0.6, 1.6, 400)

    # reference periodogram on a healthy context
    ref = eebls_gpu_fast(t, y, dy, freqs, qmin=0.03, qmax=0.3)
    assert np.all(np.isfinite(ref))
    assert abs(freqs[np.argmax(ref)] - freq) < 0.02

    t_nan = t.copy()
    t_nan[137] = np.nan
    qmin_nan = np.full(len(freqs), 0.03)
    qmin_nan[10] = np.nan
    qmax_nan = np.full(len(freqs), 0.3)
    qmax_nan[10] = np.nan

    bad_calls = [
        # per-frequency NaN q bounds on the fast kernel
        lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=qmin_nan,
                               qmax=qmax_nan),
        # qmax >= 1 -> nbins0 = 0 -> device divide by zero
        lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=0.03, qmax=np.inf),
        lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=0.03, qmax=5.0),
        # NaN timestamp on the fast kernel
        lambda: eebls_gpu_fast(t_nan, y, dy, freqs, qmin=0.03, qmax=0.3),
        # the Keplerian wrapper with too few points and with a NaN time
        lambda: eebls_transit_gpu(t[:4], y[:4], dy[:4], use_fast=True),
        lambda: eebls_transit_gpu(t_nan, y, dy, use_fast=True),
        # dy = 0 on the binned and batch kernels
        lambda: eebls_gpu(t, y, np.where(np.arange(len(t)) == 3, 0., dy),
                          freqs, qmin=0.03, qmax=0.3),
        lambda: eebls_gpu_batch([(t_nan, y, dy)], freqs, qmin=0.03,
                                qmax=0.3),
    ]
    for call in bad_calls:
        with pytest.raises(ValueError):
            call()

    # ... and the context is still alive and correct, in this module
    again = eebls_gpu_fast(t, y, dy, freqs, qmin=0.03, qmax=0.3)
    assert np.all(np.isfinite(again))
    assert np.argmax(again) == np.argmax(ref)
    assert_allclose(again, ref, rtol=1e-5, atol=1e-7)

    # ... and in another module that allocates its own device memory
    proc = LombScargleAsyncProcess()
    ls_freqs = (1. / 20.) * (1 + np.arange(512))
    power = np.copy(proc.run([(t, y, dy)], freqs=[ls_freqs])[0][1])
    proc.finish()
    assert np.all(np.isfinite(power))
    assert np.all(power >= 0)          # -1 sentinel must not appear
    assert power.max() > 0.1


def test_valid_input_is_unaffected_by_the_validators():
    """The acceptance criterion of defect 23: no change for valid
    input. The validators must not perturb, copy or re-cast the data
    they pass through, so repeated calls stay reproducible to the
    kernels' float32 atomic-accumulation noise and float32 inputs are
    still accepted (they used to reach the kernels untouched, and they
    still do)."""
    t, y, dy = make_lc(ndata=300, baseline=20., seed=11)
    freqs = np.linspace(0.6, 1.6, 256)
    a = eebls_gpu_fast(t, y, dy, freqs, qmin=0.03, qmax=0.3, noverlap=1)
    b = eebls_gpu_fast(t, y, dy, freqs, qmin=0.03, qmax=0.3, noverlap=1)
    assert_allclose(a, b, rtol=1e-6, atol=1e-8)

    # float32 inputs are accepted unchanged by the validator
    c = eebls_gpu_fast(t.astype(np.float32), y.astype(np.float32),
                       dy.astype(np.float32), freqs.astype(np.float32),
                       qmin=0.03, qmax=0.3, noverlap=1)
    assert_allclose(c, a, rtol=1e-3, atol=1e-5)
