import types

import pytest
from pycuda.tools import mark_cuda_test
import pycuda.gpuarray as gpuarray
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import ndtr
from .. import ce as ce_module
from ..ce import (ConditionalEntropyAsyncProcess, _needs_compile,
                  _CE_KERNELS, _is_single_freq_grid)
from ..memory import ConditionalEntropyMemory
from ..utils import normalize_light_curves
lsrtol = 1E-2
lsatol = 1E-5
seed = 100

rand = np.random.RandomState(seed)


def data(sigma=0.1, ndata=500, freq=3., snr=1000, t0=0.):

    t = np.sort(rand.rand(ndata)) + t0
    y = snr * sigma * np.cos(2 * np.pi * freq * t) / np.sqrt(len(t))

    y += sigma * rand.randn(len(t))

    err = sigma * np.ones_like(y)

    return t, y, err


def assert_similar(pdg0, pdg, top=5):
    inds = (np.argsort(pdg0)[::-1])[:top]

    p0 = np.asarray(pdg0)[inds]
    p = np.asarray(pdg)[inds]
    diff = np.absolute(p - p0)

    assert(all(diff < lsrtol * 0.5 * (p + p0) + lsatol))


# ---------------------------------------------------------------------------
# Independent CPU references (float64 sums, cuvarbase's bin conventions)
# ---------------------------------------------------------------------------

def _prep(t, y, dtype):
    """Emulate normalize_light_curves + ConditionalEntropyMemory.setdata."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    t = (t - t.mean()).astype(dtype)
    y = (y - y.mean()).astype(dtype)
    yscale = y.max() - y.min()
    y0 = y.min()
    return t, ((y - y0) / yscale).astype(dtype), yscale


def _phase_bins(t, f, nphase, dtype):
    ft = (t * dtype(f)).astype(dtype)
    ph = ft - np.floor(ft)
    return (np.floor(ph.astype(np.float64) * nphase).astype(int)) % nphase


def cpu_ce(t, y, freqs, nphase, nmag, phase_overlap=0, mag_overlap=0,
           dtype=np.float32):
    """Graham et al. (2013) conditional entropy with cuvarbase's bin
    definitions (uniform magnitude bins over [min, max], the brightest
    point in the top bin), overlap handling and its density offset
    ``log(dm)``; histogram counts are exact integers and the entropy sum
    runs in float64."""
    t, y01, _ = _prep(t, y, dtype)
    m0 = np.minimum(np.floor(y01 * dtype(nmag)).astype(int), nmag - 1)
    dm0 = (mag_overlap + 1.0) / nmag
    mm = np.arange(nmag)
    dm = np.where(mm + mag_overlap + 1 > nmag,
                  (nmag - mm) * dm0 / (1.0 + mag_overlap), dm0)
    out = np.empty(len(freqs))
    for k, f in enumerate(freqs):
        n0 = _phase_bins(t, f, nphase, dtype)
        H = np.zeros((nphase, nmag))
        for dn in range(phase_overlap + 1):
            for dmm in range(mag_overlap + 1):
                m = m0 - dmm
                ok = m >= 0
                np.add.at(H, ((n0[ok] - dn) % nphase, m[ok]), 1)
        Nphi = H.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(H > 0,
                            H * np.log(dm[None, :] * Nphi
                                       / np.where(H > 0, H, 1)), 0.0)
        out[k] = term.sum() / H.sum()
    return out


def exact_weighted_hist(t, y, dy, freqs, nphase, nmag, mag_overlap=0):
    """Weighted-CE histogram with the EXACT Gaussian probability mass of
    every point in every magnitude bin (no truncation).

    With ``mag_overlap > 0`` the weighted kernel widens every bin
    upwards without clipping, so bin ``m`` spans
    ``[m / nmag, (m + 1 + mag_overlap) / nmag]``.
    """
    t, Y, yscale = _prep(t, y, np.float32)
    Y = Y.astype(np.float64)
    DY = (np.asarray(dy, dtype=np.float32) / yscale).astype(np.float64)
    m = np.arange(nmag)
    P = (ndtr(((m + 1 + mag_overlap) / nmag - Y[:, None]) / DY[:, None])
         - ndtr((m / nmag - Y[:, None]) / DY[:, None]))
    H = np.zeros((len(freqs), nphase, nmag))
    for i, f in enumerate(freqs):
        n0 = _phase_bins(t, f, nphase, np.float32)
        np.add.at(H, (i, n0), P)
    return H


def weighted_ce_from_hist(H, nmag, mag_overlap=0):
    Nphi = H.sum(axis=2, keepdims=True)
    # ``weighted_ce`` uses the constant window width for every bin
    # (unlike the unweighted kernels, which truncate the top bins)
    dm = (mag_overlap + 1.0) / nmag
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where((H > 0) & (Nphi > 1e-10),
                        H * np.log(dm * Nphi / np.where(H > 0, H, 1)), 0)
    return term.sum(axis=(1, 2)) / H.sum(axis=(1, 2))


def run_ce(proc, t, y, dy, freqs, **kw):
    r = proc.run([(t, y, dy)], freqs=freqs, **kw)
    proc.finish()
    return np.copy(r[0][1])


def run_ce_with_memory(proc, t, y, dy, freqs, **kw):
    """Run and also return the memory object (to inspect ``bins_g``)."""
    mems = proc.allocate(normalize_light_curves([(t, y, dy)]),
                         freqs=[freqs], **kw)
    mems[0].transfer_freqs_to_gpu()
    r = proc.run([(t, y, dy)], memory=mems, freqs=[freqs], **kw)
    proc.finish()
    return np.copy(r[0][1]), mems[0]


def balance_magbins_cpu(mag_bins, y):
    """``ConditionalEntropyMemory.balance_magbins`` without a CUDA context.

    The method is pure numpy; only ``mag_bins``, ``real_type`` and the
    ``balanced_min_width`` class attribute are used, so it can be checked
    on a machine without a GPU (the constructor would retain the primary
    context).
    """
    stub = types.SimpleNamespace(
        mag_bins=mag_bins, real_type=np.float32,
        balanced_min_width=ConditionalEntropyMemory.balanced_min_width)
    return ConditionalEntropyMemory.balance_magbins(stub, y)


def lightcurve(ndata, seed, baseline=30., f0=1.3, noise=0.1, amp=0.3):
    r = np.random.RandomState(seed)
    t = np.sort(r.uniform(0, baseline, ndata))
    y = amp * np.sin(2 * np.pi * f0 * t) + noise * r.randn(ndata)
    return t, y, noise * np.ones(ndata)


class TestCE(object):
    plot = False

    @pytest.mark.parametrize('ndatas', [1, 5, 10])
    def test_multiple_datasets(self, ndatas, **kwargs):
        datas = [data() for i in range(ndatas)]
        proc = ConditionalEntropyAsyncProcess(**kwargs)

        df = 0.02
        max_freq = 1.1
        min_freq = 0.9
        nf = int((max_freq - min_freq) / df)
        freqs = min_freq + df * np.arange(nf)

        mult_results = proc.run(datas, freqs=freqs)
        proc.finish()

        sing_results = []

        for d in datas:
            sing_results.extend(proc.run([d], freqs=freqs))
            proc.finish()

        for rb, rnb in zip(mult_results, sing_results):
            fb, pb = rb
            fnb, pnb = rnb

            assert(not any(np.isnan(pb)))
            assert(not any(np.isnan(pnb)))

            assert_allclose(pnb, pb, rtol=lsrtol, atol=lsatol)
            assert_allclose(fnb, fb, rtol=lsrtol, atol=lsatol)

    @pytest.mark.parametrize('ndatas', [1, 7])
    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('use_double', [True, False])
    @pytest.mark.parametrize('use_fast,weighted,shmem_lc,freq_batch_size',
                             [(True, False, False, 1),
                              (True, False, True, None),
                              (False, True, False, None),
                              (False, False, False, None)])
    @pytest.mark.parametrize('phase_bins,phase_overlap',
                             [(10, 1)])
    @pytest.mark.parametrize('mag_bins,mag_overlap',
                             [(5, 0)])
    def test_batched_run(self, ndatas, batch_size, use_double,
                         mag_bins, phase_bins, mag_overlap,
                         phase_overlap, use_fast,
                         shmem_lc, weighted,
                         freq_batch_size):

        datas = [data(ndata=rand.randint(50, 100))
                 for i in range(ndatas)]
        kwargs = dict(use_double=use_double,
                      mag_bins=mag_bins,
                      phase_bins=phase_bins,
                      phase_overlap=phase_overlap,
                      mag_overlap=mag_overlap,
                      use_fast=use_fast,
                      weighted=weighted)
        proc = ConditionalEntropyAsyncProcess(**kwargs)
        df = 0.02
        max_freq = 1.1
        min_freq = 0.9
        nf = int((max_freq - min_freq) / df)
        freqs = min_freq + df * np.arange(nf)

        run_kw = dict(shmem_lc=shmem_lc, freqs=freqs,
                      freq_batch_size=freq_batch_size)
        batched_results = proc.batched_run(datas, **run_kw)
        proc.finish()

        non_batched_results = []
        for d in datas:
            r = proc.run([d], freqs=freqs)
            proc.finish()
            non_batched_results.extend(r)

        for rb, rnb in zip(batched_results, non_batched_results):
            fb, pb = rb
            fnb, pnb = rnb

            assert(not any(np.isnan(pb)))
            assert(not any(np.isnan(pnb)))

            assert_allclose(pnb, pb, rtol=lsrtol, atol=lsatol)
            assert_allclose(fnb, fb, rtol=lsrtol, atol=lsatol)

    @pytest.mark.parametrize('ndatas', [1, 7])
    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('use_double', [True, False])
    @pytest.mark.parametrize('use_fast,weighted,shmem_lc,freq_batch_size',
                             [(True, False, False, 1),
                              (True, False, True, None),
                              (False, True, False, None),
                              (False, False, False, None)])
    @pytest.mark.parametrize('phase_bins,phase_overlap',
                             [(10, 1)])
    @pytest.mark.parametrize('mag_bins,mag_overlap',
                             [(5, 0)])
    def test_batched_run_const_nfreq(self, ndatas, batch_size, use_double,
                                     mag_bins, phase_bins, mag_overlap,
                                     phase_overlap, use_fast, weighted,
                                     shmem_lc, freq_batch_size):
        frequencies = np.sort(10 + rand.rand(ndatas) * 100.)
        datas = [data(ndata=rand.randint(50, 100),
                      freq=freq)
                 for i, freq in enumerate(frequencies)]

        kwargs = dict(use_double=use_double,
                      mag_bins=mag_bins,
                      phase_bins=phase_bins,
                      phase_overlap=phase_overlap,
                      mag_overlap=mag_overlap,
                      use_fast=use_fast)
        proc = ConditionalEntropyAsyncProcess(**kwargs)

        df = 0.02
        max_freq = 1.1
        min_freq = 0.9
        nf = int((max_freq - min_freq) / df)
        freqs = min_freq + df * np.arange(nf)

        run_kw = dict(shmem_lc=shmem_lc, freqs=freqs,
                      freq_batch_size=freq_batch_size)
        batched_results = proc.batched_run_const_nfreq(datas, **run_kw)
        proc.finish()

        procnb = ConditionalEntropyAsyncProcess(**kwargs)

        non_batched_results = []
        for d, (frq, p) in zip(datas, batched_results):
            r = procnb.run([d], **run_kw)
            procnb.finish()
            non_batched_results.extend(r)

        for f0, (fb, pb), (fnb, pnb) in zip(frequencies, batched_results,
                                            non_batched_results):

            if self.plot:
                import matplotlib.pyplot as plt
                plt.plot(fnb, pnb, color='k', lw=3)
                plt.plot(fb, pb, color='r')
                plt.axvline(f0)
                plt.show()
            assert(not any(np.isnan(pb)))
            assert(not any(np.isnan(pnb)))

            assert_allclose(pnb, pb, rtol=lsrtol, atol=lsatol)
            assert_allclose(fnb, fb, rtol=lsrtol, atol=lsatol)

    # balanced_magbins is only implemented for the standard, unweighted
    # kernel (the other combinations raise ValueError); it used to be
    # parametrized independently, which silently ran the uniform kernel
    # because the constructor dropped the flag.
    @pytest.mark.parametrize('use_double', [True, False])
    @pytest.mark.parametrize(
        'use_fast,weighted,shmem_lc,freq_batch_size,balanced_magbins',
        [(True, False, False, 1, False),
         (True, False, True, None, False),
         (False, True, False, None, False),
         (False, False, False, None, False),
         (False, False, False, None, True)])
    @pytest.mark.parametrize('phase_bins,phase_overlap',
                             [(10, 1)])
    @pytest.mark.parametrize('mag_bins,mag_overlap',
                             [(5, 0)])
    @pytest.mark.parametrize('freq', [10.0])
    @pytest.mark.parametrize('t0', [0.0])
    def test_inject_and_recover(self, freq,
                                use_double, mag_bins, phase_bins, mag_overlap,
                                phase_overlap, use_fast, t0, balanced_magbins,
                                weighted, shmem_lc, freq_batch_size):

        kwargs = dict(use_double=use_double,
                      mag_bins=mag_bins,
                      phase_bins=phase_bins,
                      phase_overlap=phase_overlap,
                      mag_overlap=mag_overlap,
                      use_fast=use_fast,
                      balanced_magbins=balanced_magbins,
                      weighted=weighted)
        proc = ConditionalEntropyAsyncProcess(**kwargs)
        t, y, err = data(freq=freq, t0=t0)

        df = 1. / (max(t) - min(t)) / 10
        max_freq = 1.1 * freq
        min_freq = 0.9 * freq
        nf = int((max_freq - min_freq) / df)
        freqs = min_freq + df * np.arange(nf)

        run_kw = dict(shmem_lc=shmem_lc, freq_batch_size=freq_batch_size)
        results = proc.large_run([(t, y, err)],
                                 freqs=freqs, **run_kw)
        proc.finish()
        frq, p = results[0]
        best_freq = frq[np.argmin(p)]

        if self.plot:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots()
            ax.plot(frq, p)
            ax.axvline(freq, ls='-', color='k')
            ax.axvline(best_freq, ls=':', color='r')
            plt.show()

        # print best_freq, freq, abs(best_freq - freq) / freq
        assert(not any(np.isnan(p)))
        assert(abs(best_freq - freq) / freq < 3E-2)

    def test_large_run(self, make_plot=False, **kwargs):
        proc = ConditionalEntropyAsyncProcess(**kwargs)
        t, y, dy = data(sigma=0.01, ndata=100, freq=4.)
        df = 0.001
        max_freq = 100.
        min_freq = df
        nf = int((max_freq - min_freq) / df)
        freqs = min_freq + df * np.arange(nf)

        r0 = proc.run([(t, y, dy)], freqs=freqs)
        r1 = proc.large_run([(t, y, dy)], freqs=freqs, max_memory=1e7)

        f0, p0 = r0[0]
        f1, p1 = r1[0]

        rel_err = max(np.absolute(p0 - p1)) / np.median(np.absolute(p0))
        print(max(np.absolute(p0 - p1)), rel_err)
        assert_allclose(p0, p1, rtol=1e-4, atol=1e-2)

    @pytest.mark.parametrize('use_double', [True, False])
    @pytest.mark.parametrize(
        'use_fast,weighted,shmem_lc,freq_batch_size,balanced_magbins',
        [(True, False, False, 1, False),
         (False, False, False, None, True)])
    @pytest.mark.parametrize('phase_bins,phase_overlap',
                             [(10, 1)])
    @pytest.mark.parametrize('mag_bins,mag_overlap',
                             [(5, 0)])
    @pytest.mark.parametrize('freq', [10.0])
    def test_time_shift_invariance(self, freq,
                                   use_double, mag_bins, phase_bins,
                                   mag_overlap, phase_overlap, use_fast,
                                   balanced_magbins, weighted,
                                   shmem_lc, freq_batch_size):

        kwargs = dict(use_double=use_double,
                      mag_bins=mag_bins,
                      phase_bins=phase_bins,
                      phase_overlap=phase_overlap,
                      mag_overlap=mag_overlap,
                      use_fast=use_fast,
                      balanced_magbins=balanced_magbins,
                      weighted=weighted)
        proc = ConditionalEntropyAsyncProcess(**kwargs)

        run_kw = dict(shmem_lc=shmem_lc, freq_batch_size=freq_batch_size)
        for t0 in [-1e4, 1e4]:
            t, y, err = data(freq=freq)

            df = 1. / (max(t) - min(t)) / 10
            max_freq = 1.1 * freq
            min_freq = 0.9 * freq
            nf = int((max_freq - min_freq) / df)

            freqs = min_freq + df * np.arange(nf)

            results = proc.run([(t, y, err)], freqs=freqs, **run_kw)
            proc.finish()
            frq, p = results[0]

            results_shift = proc.run([(t + t0, y, err)], freqs=freqs, **run_kw)
            frq_shft, p_shft = results_shift[0]

            best_freq = frq[np.argmin(p)]
            best_freq_shft = frq_shft[np.argmin(p_shft)]

            if self.plot:
                import matplotlib.pyplot as plt
                f, ax = plt.subplots()
                ax.plot(frq, p)
                ax.plot(frq_shft, p_shft)
                ax.axvline(freq, ls='-', color='k')
                ax.axvline(best_freq, ls=':', color='r')
                plt.show()

            assert(not any(np.isnan(p)))
            assert(not any(np.isnan(p_shft)))

            baseline = max(t) - min(t)
            delta_f = abs(best_freq - best_freq_shft)
            top_freq_is_close = delta_f * baseline < 1

            diffs = np.absolute(p - p_shft)
            atol, rtol = 1e-1 * max(np.absolute(p)), 2e-1
            upper_limit = atol + rtol * np.absolute(p)

            pct_out_of_bounds = sum(diffs > upper_limit) / len(diffs)

            print(pct_out_of_bounds, delta_f * baseline)
            assert(top_freq_is_close and pct_out_of_bounds < 5e-2)

    # (phase_bins, mag_bins) combinations with (mag_bins + 1) * phase_bins
    # odd -- (5, 4), (7, 6), (3, 4) -- used to crash the double-precision
    # fast kernels with 'misaligned address' (defect 17).
    @pytest.mark.parametrize('use_double', [True, False])
    @pytest.mark.parametrize('shmem_lc', [True, False])
    @pytest.mark.parametrize('freq_batch_size', [1, None])
    @pytest.mark.parametrize('phase_bins,phase_overlap,mag_bins,mag_overlap',
                             [(10, 0, 5, 0), (10, 1, 5, 1), (5, 0, 4, 0),
                              (7, 0, 6, 0), (3, 0, 4, 0)])
    @pytest.mark.parametrize('freq', [12.0])
    @pytest.mark.parametrize('t0', [0.0])
    @pytest.mark.parametrize('balanced_magbins', [False])
    @pytest.mark.parametrize('weighted', [False])
    @pytest.mark.parametrize('force_nblocks', [1, None])
    @pytest.mark.parametrize('ndata', [300])
    def test_fast(self, freq, use_double, mag_bins, phase_bins, mag_overlap,
                  phase_overlap, t0, balanced_magbins, weighted,
                  shmem_lc, freq_batch_size, force_nblocks, ndata):

        kwargs = dict(use_double=use_double,
                      mag_bins=mag_bins,
                      phase_bins=phase_bins,
                      phase_overlap=phase_overlap,
                      mag_overlap=mag_overlap,
                      balanced_magbins=balanced_magbins,
                      weighted=weighted)
        proc_fast = ConditionalEntropyAsyncProcess(use_fast=True, **kwargs)
        proc_slow = ConditionalEntropyAsyncProcess(use_fast=False, **kwargs)
        t, y, err = data(freq=freq, t0=t0, ndata=ndata)

        df = 1. / (max(t) - min(t)) / 10
        max_freq = 1.1 * freq
        min_freq = 0.9 * freq
        nf = int((max_freq - min_freq) / df)
        freqs = min_freq + df * np.arange(nf)

        run_kw = dict(shmem_lc=shmem_lc,
                      freq_batch_size=freq_batch_size,
                      force_nblocks=force_nblocks)
        results_fast = proc_fast.run([(t + t0, y, err)], freqs=freqs,
                                     **run_kw)
        proc_fast.finish()
        frq_fast, p_fast = results_fast[0]

        results_slow = proc_slow.run([(t + t0, y, err)], freqs=freqs)
        proc_slow.finish()
        frq_slow, p_slow = results_slow[0]

        max_diff = 2e-2 * max(np.absolute(p_slow))
        if self.plot and \
                not all(np.absolute(p_slow - p_fast) < max_diff):
            import matplotlib.pyplot as plt

            f, ax = plt.subplots()
            ax.plot(frq_slow, p_slow, alpha=0.5)
            ax.plot(frq_fast, p_fast, alpha=0.5)
            ax.axvline(freq, ls='-', color='k')
            plt.show()

            f, ax = plt.subplots()
            ax.plot(frq_slow, (p_slow - p_fast) / max(np.absolute(p_slow)))
            ax.axvline(freq, ls='-', color='k')
            plt.show()
        # print best_freq, freq, abs(best_freq - freq) / freq
        assert(not any(np.isnan(p_slow)))
        assert(not any(np.isnan(p_fast)))
        # Both kernels histogram the same integer bins; the only
        # difference is float summation order (the old 2e-2 * max
        # tolerance hid the brightest-point mis-binning of defect 9).
        assert_allclose(p_slow, p_fast, rtol=0,
                        atol=(1e-10 if use_double else 1e-5))


# ---------------------------------------------------------------------------
# Regression tests for the Sep-2026 audit defects
# ---------------------------------------------------------------------------

class TestCEBrightestPoint(object):
    """Defect 9 (ce-brightest-bin): the brightest point (normalized
    magnitude exactly 1.0) got bin index ``mag_bins`` and spilled into the
    next phase bin / next frequency / past the end of ``bins_g``."""

    @pytest.mark.parametrize('phase_overlap,mag_overlap',
                             [(0, 0), (1, 0), (0, 1), (1, 1)])
    def test_histogram_totals_exact(self, phase_overlap, mag_overlap):
        N = 100
        t, y, dy = lightcurve(N, seed=3)
        freqs = np.linspace(0.3, 1.2, 50)
        proc = ConditionalEntropyAsyncProcess(phase_overlap=phase_overlap,
                                              mag_overlap=mag_overlap)
        _, mem = run_ce_with_memory(proc, t, y, dy, freqs)
        assert mem.y[:N].max() == proc.mag_bins - 1
        bins = mem.bins_g.get().reshape(len(freqs), proc.phase_bins,
                                        proc.mag_bins)
        totals = bins.sum(axis=(1, 2))
        # every point is counted (phase_overlap + 1) times in each of its
        # (mag_overlap + 1) magnitude bins, except that overlapping bins
        # below bin 0 do not exist; the total is the same at EVERY
        # frequency (it used to be N - 1 .. N + 1 from the spilled point)
        m0 = mem.y[:N].astype(int)
        expected = (phase_overlap + 1) * np.minimum(m0 + 1,
                                                    mag_overlap + 1).sum()
        if mag_overlap == 0:
            assert expected == N * (phase_overlap + 1)
        assert_array_equal(totals, np.full(len(freqs), expected))

    def test_no_write_past_bins(self):
        """The brightest point in the LAST phase bin of the LAST frequency
        used to be written one element past ``bins_g``."""
        N = 100
        t, y, dy = lightcurve(N, seed=3)
        imax = np.argmax(y)
        tt = np.float32(t - t.mean())

        def phase_bin(f):
            return _phase_bins(tt[imax:imax + 1], f, 10, np.float32)[0]

        cands = [f for f in np.linspace(0.3, 1.3, 4000) if phase_bin(f) == 9]
        freqs = np.concatenate([np.linspace(0.5, 0.9, 63), [cands[0]]])
        proc = ConditionalEntropyAsyncProcess()
        mems = proc.allocate([(t, y, dy)], freqs=[freqs])
        mem = mems[0]
        # ``allocate`` only creates a zero-filled ``freqs_g``; without this
        # upload every trial frequency would be f = 0, the brightest point
        # would never reach the last phase bin of the last frequency and
        # the guard below could not fire (defect 19 closes the same trap
        # inside ``run``, this makes the test independent of it)
        mem.transfer_freqs_to_gpu()
        nb = mem.nbins
        guard = np.uint32(0xDEAD)
        big = gpuarray.zeros(nb + 8, dtype=np.uint32)
        big.fill(guard)
        mem.bins_g = big[:nb]
        proc.run([(t, y, dy)], memory=mems, freqs=[freqs])
        proc.finish()
        assert mem.freqs_g.get().max() > 0
        full = big.get()
        assert_array_equal(full[nb:], np.full(8, guard))
        totals = full[:nb].reshape(len(freqs), -1).sum(axis=1)
        assert_array_equal(totals, np.full(len(freqs), N))
        # the count that used to be written one element past ``bins_g``:
        # brightest magnitude bin, last phase bin, last frequency
        bins = full[:nb].reshape(len(freqs), proc.phase_bins, proc.mag_bins)
        assert bins[-1, -1, -1] > 0

    @pytest.mark.parametrize('ndata', [5, 60])
    @pytest.mark.parametrize('use_double', [False, True])
    @pytest.mark.parametrize('use_fast', [False, True])
    def test_matches_cpu_reference(self, ndata, use_double, use_fast):
        t, y, dy = lightcurve(ndata, seed=1)
        freqs = np.linspace(0.05, 3.0, 200)
        proc = ConditionalEntropyAsyncProcess(use_double=use_double,
                                              use_fast=use_fast)
        p = run_ce(proc, t, y, dy, freqs)
        dtype = np.float64 if use_double else np.float32
        ref = cpu_ce(t, y, freqs, 10, 5, dtype=dtype)
        assert np.all(np.isfinite(p))
        atol = 1e-10 if use_double else 2e-6
        assert_allclose(p, ref, rtol=0, atol=atol)
        # (at N = 5 the CE takes few distinct values, so the argmin can
        # legitimately land on a tied minimum: compare the values)
        assert abs(ref[np.argmin(p)] - ref.min()) <= atol

    @pytest.mark.parametrize('phase_overlap,mag_overlap', [(1, 1), (2, 1)])
    def test_matches_cpu_reference_overlap(self, phase_overlap, mag_overlap):
        t, y, dy = lightcurve(60, seed=1)
        freqs = np.linspace(0.05, 3.0, 200)
        proc = ConditionalEntropyAsyncProcess(phase_overlap=phase_overlap,
                                              mag_overlap=mag_overlap,
                                              phase_bins=8, mag_bins=6)
        p = run_ce(proc, t, y, dy, freqs)
        ref = cpu_ce(t, y, freqs, 8, 6, phase_overlap, mag_overlap)
        assert_allclose(p, ref, rtol=0, atol=2e-6)

    @pytest.mark.parametrize('use_fast', [False, True])
    def test_frequency_grid_order_invariance(self, use_fast):
        """The standard kernel's output depended on the ORDER of the grid
        because the spilled count landed in the next frequency's bin."""
        t, y, dy = lightcurve(500, seed=1)
        freqs = np.linspace(0.05, 3.0, 200)
        proc = ConditionalEntropyAsyncProcess(use_fast=use_fast)
        fwd = run_ce(proc, t, y, dy, freqs)
        rev = run_ce(proc, t, y, dy, freqs[::-1].copy())[::-1]
        assert_array_equal(fwd, rev)

    def test_mag_bin_fracs_sum_to_one(self):
        t, y, dy = lightcurve(100, seed=3)
        mem = ConditionalEntropyMemory(phase_bins=10, mag_bins=5,
                                       compute_log_prob=True)
        mem.setdata(t - t.mean(), y - y.mean())
        assert mem.y.max() == 4
        assert_allclose(mem.mag_bin_fracs.sum(), 1.0, rtol=0, atol=1e-6)


class TestCEWeighted(object):
    """Defect 16 (ce-weighted-asym): the weighted histogram skipped a bin
    by the distance to its LOWER edge only, dropping the mass of bins
    below the datum, and the brightest point entirely."""

    def test_hand_placed_points_match_exact_masses(self):
        MB, PB, sig = 5, 1, 0.02
        Yc = np.array([0.0, 0.41, 0.5, 0.59, 1.0])
        proc = ConditionalEntropyAsyncProcess(phase_bins=PB, mag_bins=MB,
                                              weighted=True, max_phi=3.0)
        _, mem = run_ce_with_memory(proc, np.linspace(0, 1, 5), Yc,
                                    sig * np.ones(5), np.array([0.0]))
        bins = mem.bins_g.get().reshape(1, PB, MB)[0, 0]
        m = np.arange(MB)
        P = (ndtr(((m + 1) / MB - Yc[:, None]) / sig)
             - ndtr((m / MB - Yc[:, None]) / sig))
        # old kernel: [0.5, 0, 2.38, 0.31, 0] (bin 1 and the Y=1 point lost)
        assert_allclose(bins, P.sum(axis=0), rtol=0, atol=1e-4)
        assert bins[1] > 0.3 and bins[4] > 0.49

    @pytest.mark.parametrize('mag_bins', [5, 10])
    @pytest.mark.parametrize('noise', [0.05, 0.15])
    @pytest.mark.parametrize('mag_overlap', [0, 1, 2])
    def test_bins_and_ce_vs_ndtr_reference(self, mag_bins, noise,
                                           mag_overlap):
        # ``mag_overlap > 0`` is where the symmetric-truncation fix
        # matters most (the audit measured a 0.21 nat change in the CE
        # itself, 0.14 on the default lightcurve); the overlapping
        # window makes bin m span [m, m + 1 + mag_overlap] / mag_bins.
        r = np.random.RandomState(3)
        N = 300
        t = np.sort(r.rand(N)) * 20.0
        y = (12 + np.sin(2 * np.pi * 1.3 * t) + 0.3 * np.sin(4 * np.pi * 1.3 * t)
             + noise * r.randn(N))
        dy = noise * np.ones(N)
        freqs = np.linspace(0.1, 3.0, 40)
        He = exact_weighted_hist(t, y, dy, freqs, 10, mag_bins,
                                 mag_overlap=mag_overlap)
        ce_exact = weighted_ce_from_hist(He, mag_bins,
                                         mag_overlap=mag_overlap)

        # default max_phi=3: only bins wholly beyond 3 sigma are skipped
        proc = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=mag_bins,
                                              mag_overlap=mag_overlap,
                                              weighted=True, max_phi=3.0)
        ce, mem = run_ce_with_memory(proc, t, y, dy, freqs)
        bins = mem.bins_g.get().reshape(len(freqs), 10, mag_bins)
        assert np.all(np.isfinite(ce))
        # audit-measured post-fix levels: bins 6e-3 (mag_overlap 0) and
        # 4.2e-3 (mag_overlap 1-2), CE 1.1e-3 (old: 1.5-4.2 in the bins,
        # 2e-2 .. 5e-2 in the CE)
        assert_allclose(bins, He, rtol=0, atol=2e-2)
        assert_allclose(ce, ce_exact, rtol=0, atol=5e-3)
        # the per-frequency mass totals match the exact ones to the mass
        # of the skipped > 3-sigma bins (points near the range edges
        # legitimately lose the mass outside [0, 1]; old: -2 .. -12%)
        assert_allclose(bins.sum(axis=(1, 2)), He.sum(axis=(1, 2)),
                        rtol=3e-3, atol=0)

        # with a wide max_phi nothing is truncated: float32 normcdf level
        proc = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=mag_bins,
                                              mag_overlap=mag_overlap,
                                              weighted=True, max_phi=50.0)
        ce, mem = run_ce_with_memory(proc, t, y, dy, freqs)
        bins = mem.bins_g.get().reshape(len(freqs), 10, mag_bins)
        assert_allclose(bins, He, rtol=0, atol=2e-3)
        assert_allclose(bins.sum(axis=(1, 2)), He.sum(axis=(1, 2)),
                        rtol=1e-5, atol=0)
        assert_allclose(ce, ce_exact, rtol=0, atol=1e-4)

    def test_large_max_phi_is_finite(self):
        """Tiny bin masses used to make ``dm * p_phi / pmn`` overflow to
        inf (3 of 3000 frequencies for this lightcurve)."""
        r = np.random.RandomState(2)
        N = 200
        t = np.sort(r.rand(N)) * 20.0
        y = 12 + np.sin(2 * np.pi * 1.3 * t) + 0.3 * np.sin(4 * np.pi * 1.3 * t) + 0.05 * r.randn(N)
        dy = 0.05 * np.ones(N)
        freqs = np.linspace(0.1, 3.0, 3000)
        proc = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5,
                                              weighted=True, max_phi=1e6)
        ce = run_ce(proc, t, y, dy, freqs)
        assert np.all(np.isfinite(ce))
        proc3 = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5,
                                               weighted=True, max_phi=3.0)
        ce3 = run_ce(proc3, t, y, dy, freqs)
        assert np.all(np.isfinite(ce3))
        assert abs(freqs[np.argmin(ce3)] - 1.3) < 0.01
        assert abs(freqs[np.argmin(ce)] - 1.3) < 0.01


class TestCEDoubleFast(object):
    """Defect 17 (ce-double-fast-crash): shared-memory misalignment for
    ``use_double=True, use_fast=True`` when (mag_bins + 1) * phase_bins is
    odd, and a 4-byte shared-memory shortfall for odd ndata."""

    @pytest.mark.parametrize('ndata', [200, 201])
    @pytest.mark.parametrize('shmem_lc', [True, False])
    @pytest.mark.parametrize('phase_bins,mag_bins',
                             [(5, 4), (7, 6), (3, 4), (10, 5)])
    def test_double_fast_matches_double_standard(self, phase_bins, mag_bins,
                                                 shmem_lc, ndata):
        r = np.random.RandomState(0)
        t = np.sort(r.rand(ndata) * 20)
        y = 12 + 0.3 * np.cos(2 * np.pi * t * 1.7) + 0.05 * r.randn(ndata)
        dy = 0.05 * np.ones(ndata)
        freqs = np.linspace(0.1, 3.0, 256)
        ref = run_ce(ConditionalEntropyAsyncProcess(
            phase_bins=phase_bins, mag_bins=mag_bins, use_double=True),
            t, y, dy, freqs)
        proc = ConditionalEntropyAsyncProcess(phase_bins=phase_bins,
                                              mag_bins=mag_bins,
                                              use_double=True, use_fast=True)
        p = run_ce(proc, t, y, dy, freqs, shmem_lc=shmem_lc)
        assert np.all(np.isfinite(p))
        assert_allclose(p, ref, rtol=0, atol=1e-10)
        cpu = cpu_ce(t, y, freqs, phase_bins, mag_bins, dtype=np.float64)
        assert_allclose(p, cpu, rtol=0, atol=1e-10)


class TestCEBalanced(object):
    """Defect 18 (ce-balanced-ignored) and ids 105/106."""

    @staticmethod
    def _lc():
        r = np.random.RandomState(0)
        N = 400
        t = np.sort(30 * r.rand(N))
        y = 12 + 0.3 * np.cos(2 * np.pi * 3.1 * t) + 0.05 * r.randn(N)
        y[:3] += 5.0     # outliers: balanced bins differ strongly from uniform
        return t, y, 0.05 * np.ones(N)

    def test_constructor_flag_is_forwarded(self):
        t, y, dy = self._lc()
        freqs = np.linspace(2.5, 3.7, 1000)
        plain = run_ce(ConditionalEntropyAsyncProcess(), t, y, dy, freqs)

        def large(proc, **kw):
            r = proc.large_run([(t, y, dy)], freqs=freqs, **kw)
            proc.finish()
            return np.copy(r[0][1])

        def batched(proc, **kw):
            r = proc.batched_run_const_nfreq([(t, y, dy)], freqs=freqs, **kw)
            return np.copy(r[0][1])

        for fn in (run_ce, large, batched):
            if fn is run_ce:
                ctor = fn(ConditionalEntropyAsyncProcess(balanced_magbins=True),
                          t, y, dy, freqs)
                runkw = fn(ConditionalEntropyAsyncProcess(), t, y, dy, freqs,
                           balanced_magbins=True)
            else:
                ctor = fn(ConditionalEntropyAsyncProcess(balanced_magbins=True))
                runkw = fn(ConditionalEntropyAsyncProcess(),
                           balanced_magbins=True)
            assert_array_equal(ctor, runkw)
            assert np.max(np.abs(ctor - plain)) > 0.1

        proc = ConditionalEntropyAsyncProcess(balanced_magbins=True)
        assert proc.balanced_magbins
        mems = proc.allocate([(t, y, dy)], freqs=[freqs])
        assert mems[0].balanced_magbins
        proc.preallocate(len(t), freqs, nlcs=1)
        assert proc.memory[0].balanced_magbins

    def test_widen_mag_range_is_forwarded(self):
        t, y, dy = self._lc()
        freqs = np.linspace(2.5, 3.7, 500)
        plain = run_ce(ConditionalEntropyAsyncProcess(weighted=True),
                       t, y, dy, freqs)
        ctor = run_ce(ConditionalEntropyAsyncProcess(weighted=True,
                                                     widen_mag_range=True),
                      t, y, dy, freqs)
        runkw = run_ce(ConditionalEntropyAsyncProcess(weighted=True),
                       t, y, dy, freqs, widen_mag_range=True)
        assert_allclose(ctor, runkw, rtol=0, atol=1e-6)
        assert np.max(np.abs(ctor - plain)) > 1e-3
        proc = ConditionalEntropyAsyncProcess(weighted=True,
                                              widen_mag_range=True)
        proc.preallocate(len(t), freqs, nlcs=1)
        assert proc.memory[0].widen_mag_range

    def test_unsupported_combinations_raise_in_constructor(self):
        # CPU-runnable: the checks run before the GPU context is touched
        bad = [dict(weighted=True, use_fast=True),
               dict(weighted=True, balanced_magbins=True),
               dict(weighted=True, compute_log_prob=True),
               dict(use_fast=True, balanced_magbins=True),
               dict(balanced_magbins=True, compute_log_prob=True),
               dict(mag_overlap=1, balanced_magbins=True)]
        for kw in bad:
            with pytest.raises(ValueError):
                ConditionalEntropyAsyncProcess(**kw)

    @pytest.mark.parametrize('ctor', [dict(weighted=True), dict(use_fast=True),
                                      dict(compute_log_prob=True),
                                      dict(mag_overlap=1)])
    def test_unsupported_combinations_raise_for_run_kwargs(self, ctor):
        t, y, dy = self._lc()
        freqs = np.linspace(2.5, 3.7, 100)
        proc = ConditionalEntropyAsyncProcess(**ctor)
        with pytest.raises(ValueError):
            proc.run([(t, y, dy)], freqs=freqs, balanced_magbins=True)
        with pytest.raises(ValueError):
            proc.preallocate(len(t), freqs, balanced_magbins=True)

    def test_balanced_matches_reference(self):
        t, y, dy = self._lc()
        freqs = np.linspace(2.5, 3.7, 300)
        proc = ConditionalEntropyAsyncProcess(balanced_magbins=True)
        p, mem = run_ce_with_memory(proc, t, y, dy, freqs)
        ybins = mem.y[:mem.n0].astype(int)
        bwf = mem.mag_bwf.astype(np.float64)
        # each bin holds N / mag_bins points; widths tile [0, 1]
        assert_array_equal(np.bincount(ybins), np.full(5, 80))
        assert_allclose(bwf.sum(), 1.0, rtol=0, atol=1e-6)
        t32, _, _ = _prep(t, y, np.float32)
        H = np.zeros((len(freqs), 10, 5))
        for i, f in enumerate(freqs):
            np.add.at(H, (i, _phase_bins(t32, f, 10, np.float32), ybins), 1)
        Nphi = H.sum(axis=2, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(H > 0, H * np.log(bwf[None, None, :] * Nphi
                                              / np.where(H > 0, H, 1)), 0)
        ref = term.sum(axis=(1, 2)) / H.sum(axis=(1, 2))
        assert_allclose(p, ref, rtol=0, atol=2e-6)
        assert abs(freqs[np.argmin(p)] - 3.1) < 0.01

    def test_quantized_magnitudes_are_finite(self):
        """id 106: a bin of identical values had zero width -> CE = -inf."""
        r = np.random.RandomState(4)
        N = 400
        t = np.sort(r.rand(N) * 20)
        y = np.round(12 + np.sin(2 * np.pi * 1.3 * t) + 0.3 * r.randn(N))
        assert len(np.unique(y)) <= 6
        dy = np.ones(N)
        freqs = np.linspace(0.1, 3.0, 300)
        proc = ConditionalEntropyAsyncProcess(balanced_magbins=True)
        p, mem = run_ce_with_memory(proc, t, y, dy, freqs)
        assert np.all(np.isfinite(p))
        assert np.all(mem.mag_bwf > 0)
        assert_allclose(mem.mag_bwf.sum(), 1.0, rtol=0, atol=1e-5)
        assert abs(freqs[np.argmin(p)] - 1.3) < 0.02

    @pytest.mark.parametrize('mag_bins', [2, 3, 5, 7, 11, 20])
    def test_balanced_bin_bounds_cover_every_point(self, mag_bins):
        """Defect 18 (2nd round): the group boundaries were
        ``int(i * len(y) / mag_bins)``, and for 471 of the 37,810
        ``(mag_bins, N)`` combinations with ``mag_bins`` in 2..20 and
        ``N`` up to 2000 (e.g. ``(7, 61)``) the float product fell short
        of ``len(y)``, so the brightest point(s) were never assigned and
        kept ``ybins = 0`` -- the brightest star of the lightcurve was put
        in the FAINTEST magnitude bin.  CPU-only (pure numpy)."""
        r = np.random.RandomState(7)
        for n in range(mag_bins, 4 * mag_bins + 260):
            y = r.rand(n)
            ybins, bwf = balance_magbins_cpu(mag_bins, y)
            ybins = ybins.astype(int)
            counts = np.bincount(ybins, minlength=mag_bins)
            # every point is assigned, and to a group of the right size
            assert counts.sum() == n
            assert counts.min() == n // mag_bins
            assert counts.max() == -(-n // mag_bins)
            # bins increase monotonically with magnitude
            assert np.all(np.diff(ybins[np.argsort(y, kind='stable')]) >= 0)
            assert ybins[np.argmax(y)] == mag_bins - 1
            assert ybins[np.argmin(y)] == 0
            # widths still tile the magnitude range
            assert len(bwf) == mag_bins
            assert np.all(bwf > 0)
            assert abs(float(bwf.astype(np.float64).sum()) - 1.0) < 1e-4

    def test_balanced_brightest_point_on_gpu_ragged_n(self):
        """End-to-end version of the above: ``mag_bins=7``, ``N=61`` was
        one of the affected combinations (the brightest point landed in
        bin 0, giving ``bincount = [9 9 9 8 9 9 8]``)."""
        N, mag_bins = 61, 7
        t, y, dy = lightcurve(N, seed=11)
        freqs = np.linspace(0.5, 2.5, 200)
        proc = ConditionalEntropyAsyncProcess(mag_bins=mag_bins,
                                              balanced_magbins=True)
        p, mem = run_ce_with_memory(proc, t, y, dy, freqs)
        ybins = mem.y[:mem.n0].astype(int)
        counts = np.bincount(ybins, minlength=mag_bins)
        assert counts.sum() == N
        expected = np.full(mag_bins, N // mag_bins)
        expected[:N % mag_bins] += 1
        assert_array_equal(np.sort(counts), np.sort(expected))
        assert ybins[np.argmax(y)] == mag_bins - 1
        assert np.all(np.isfinite(p))
        assert_allclose(mem.mag_bwf.astype(np.float64).sum(), 1.0,
                        rtol=0, atol=1e-5)


class TestCEPreallocate(object):
    """Defect 19 (ce-preallocate): ``preallocate()`` never uploaded the
    frequency grid (every frequency evaluated at f = 0) and left
    ``memory.stream = None`` (results read before the copy landed)."""

    @staticmethod
    def _lc(N, seed):
        r = np.random.RandomState(seed)
        t = np.sort(r.uniform(0, 100, N))
        y = 0.3 * np.sin(2 * np.pi * t / 1.7) + 0.05 * r.randn(N)
        return t, y, 0.05 * np.ones(N)

    @pytest.mark.parametrize('use_fast', [False, True])
    def test_preallocate_then_run(self, use_fast):
        F = np.linspace(0.05, 5.0, 4000)
        B = self._lc(900, 2)
        C = self._lc(300, 5)
        proc = ConditionalEntropyAsyncProcess(use_fast=use_fast)
        fB = run_ce(proc, *B, F)
        fC = run_ce(proc, *C, F)
        assert fB.std() > 0 and fC.std() > 0

        proc.preallocate(max_nobs=900, freqs=F, nlcs=1)
        mem = proc.memory[0]
        assert mem.stream is proc.streams[0]
        assert_allclose(mem.freqs_g.get(), F.astype(np.float32),
                        rtol=0, atol=0)
        for k in range(3):
            for lc, ref in ((B, fB), (C, fC)):
                r = proc.run([lc], freqs=[F])
                proc.finish()
                assert_array_equal(np.copy(r[0][1]), ref)

    def test_preallocate_batch(self):
        F = np.linspace(0.05, 5.0, 2000)
        lcs = [self._lc(n, s) for n, s in ((900, 2), (300, 5), (600, 7))]
        proc = ConditionalEntropyAsyncProcess()
        refs = [run_ce(proc, *lc, F) for lc in lcs]
        proc.preallocate(max_nobs=900, freqs=F, nlcs=3)
        assert len(proc.memory) == 3
        assert len(set(id(m.stream) for m in proc.memory)) == 3
        r = proc.run(lcs, freqs=F)
        proc.finish()
        for (f, p), ref in zip(r, refs):
            assert_array_equal(np.copy(p), ref)
        with pytest.raises(ValueError):
            proc.run(lcs + [lcs[0]], freqs=F)

    def test_run_reuploads_changed_freqs(self):
        F1 = np.linspace(0.05, 5.0, 2000)
        F2 = np.linspace(0.5, 2.5, 2000)
        F3 = np.linspace(0.5, 2.5, 1000)
        lc = self._lc(500, 2)
        proc = ConditionalEntropyAsyncProcess()
        ref2 = run_ce(proc, *lc, F2)
        proc.preallocate(max_nobs=500, freqs=F1, nlcs=1)
        r = proc.run([lc], freqs=F2)
        proc.finish()
        assert_array_equal(np.copy(r[0][1]), ref2)
        assert_allclose(proc.memory[0].freqs_g.get(), F2.astype(np.float32),
                        rtol=0, atol=0)
        with pytest.raises(ValueError):
            proc.run([lc], freqs=F3)


    def test_preallocate_then_large_run(self):
        # large_run slices the grid into batches, so a preallocated
        # self.memory (nf = the full grid) can never serve them: the
        # combination raised "memory was allocated for N frequencies".
        # large_run now allocates per batch and passes it explicitly.
        F = np.linspace(0.05, 5.0, 3000)
        lc = self._lc(400, 11)
        proc = ConditionalEntropyAsyncProcess()
        ref = proc.large_run([lc], freqs=F, max_memory=1e5)
        proc.finish()
        ref = np.copy(ref[0][1])
        assert ref.std() > 0

        proc.preallocate(max_nobs=400, freqs=F, nlcs=1)
        r = proc.large_run([lc], freqs=F, max_memory=1e5)
        proc.finish()
        assert_array_equal(np.copy(r[0][1]), ref)
        # the preallocated memory is untouched and still usable
        assert_allclose(proc.memory[0].freqs_g.get(), F.astype(np.float32),
                        rtol=0, atol=0)
        r2 = proc.run([lc], freqs=F)
        proc.finish()
        assert np.all(np.isfinite(r2[0][1]))

    def test_preallocate_then_run_without_freqs(self):
        # run(freqs=None) used to build an autofrequency grid whose
        # length is never mem.nf, so it raised on preallocated memory.
        # The grid preallocate() uploaded is the one to use.
        F = np.linspace(0.05, 5.0, 2000)
        lc = self._lc(500, 13)
        proc = ConditionalEntropyAsyncProcess()
        ref = run_ce(proc, *lc, F)

        proc.preallocate(max_nobs=500, freqs=F, nlcs=1)
        r = proc.run([lc])
        proc.finish()
        assert_array_equal(np.asarray(r[0][0]), F.astype(np.float32))
        assert_array_equal(np.copy(r[0][1]), ref)

        # explicit memory from allocate() behaves the same way
        proc2 = ConditionalEntropyAsyncProcess()
        mems = proc2.allocate(normalize_light_curves([lc]), freqs=[F])
        r = proc2.run([lc], memory=mems)
        proc2.finish()
        assert_array_equal(np.asarray(r[0][0]), F.astype(np.float32))
        assert_allclose(np.copy(r[0][1]), ref, rtol=0, atol=1e-6)

    def test_run_without_freqs_and_without_memory_uses_autofrequency(self):
        lc = self._lc(200, 17)
        proc = ConditionalEntropyAsyncProcess()
        r = proc.run([lc])
        proc.finish()
        assert len(r[0][0]) == len(proc.autofrequency(lc[0]))


class TestCEFastSharedMemoryLimit(object):
    """audit section 4 row 122: a phase_bins x mag_bins histogram that
    does not fit in shared memory died with an opaque pycuda
    ``LogicError: cuLaunchKernel failed: invalid argument``."""

    def test_oversized_histogram_raises_value_error(self):
        t, y, dy = lightcurve(200, seed=1)
        freqs = np.linspace(0.1, 3.0, 64)
        proc = ConditionalEntropyAsyncProcess(use_fast=True,
                                              phase_bins=200, mag_bins=50)
        with pytest.raises(ValueError,
                           match=r"shared memory.*200 x 50"):
            proc.run([(t, y, dy)], freqs=freqs)

    def test_small_histogram_still_runs(self):
        t, y, dy = lightcurve(200, seed=1)
        freqs = np.linspace(0.1, 3.0, 64)
        proc = ConditionalEntropyAsyncProcess(use_fast=True,
                                              phase_bins=10, mag_bins=5)
        ce = run_ce(proc, t, y, dy, freqs)
        assert np.all(np.isfinite(ce))


class TestCEReuse(object):
    """id 112 (``set_data=False`` accumulated histograms across calls) and
    CE-1 (the module was recompiled with nvcc on every call)."""

    @pytest.mark.parametrize('compute_log_prob', [False, True])
    def test_set_data_false_repeat_is_idempotent(self, compute_log_prob):
        r = np.random.RandomState(0)
        N = 60
        t = np.sort(r.rand(N) * 20)
        d = [(t, r.randn(N), np.ones(N))]
        freqs = np.linspace(0.1, 3.0, 50)
        proc = ConditionalEntropyAsyncProcess(compute_log_prob=compute_log_prob)
        mems = proc.allocate(normalize_light_curves(d), freqs=[freqs])
        mems[0].transfer_freqs_to_gpu()
        first = None
        for k in range(3):
            res = proc.run(d, memory=mems, freqs=[freqs], set_data=(k == 0))
            proc.finish()
            p = np.copy(res[0][1])
            assert mems[0].bins_g.get().sum() == N * len(freqs)
            if first is None:
                first = p
            else:
                assert_array_equal(p, first)

    def test_compile_gate_logic(self):
        # CPU-runnable
        assert _needs_compile({})
        assert _needs_compile(None)
        assert _needs_compile({'ce_wt': object()})   # the old sentinel
        assert not _needs_compile({k: object() for k in _CE_KERNELS})
        assert _needs_compile({k: object() for k in _CE_KERNELS[:-1]})

    def test_compiles_once_per_process(self, monkeypatch):
        calls = []
        real = ce_module.SourceModule

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(ce_module, 'SourceModule', counting)
        t, y, dy = lightcurve(100, seed=3)
        freqs = np.linspace(0.3, 1.2, 50)
        proc = ConditionalEntropyAsyncProcess()
        run_ce(proc, t, y, dy, freqs)
        run_ce(proc, t, y, dy, freqs)
        proc.large_run([(t, y, dy)], freqs=freqs, max_memory=1e5)
        assert len(calls) == 1


class TestCEFrequencyInput(object):
    """ids 108/163: float32 (or any non-Python-float) frequency arrays
    were rejected with a misleading 'number of frequency grids' error."""

    def test_single_grid_detection(self):
        # CPU-runnable
        assert _is_single_freq_grid(np.linspace(0, 1, 5).astype(np.float32))
        assert _is_single_freq_grid(np.linspace(0, 1, 5))
        assert _is_single_freq_grid([0.1, 0.2, 0.3])
        assert _is_single_freq_grid(np.arange(5))
        assert not _is_single_freq_grid([np.linspace(0, 1, 5)])
        assert not _is_single_freq_grid([[0.1, 0.2], [0.3, 0.4, 0.5]])
        assert not _is_single_freq_grid(np.ones((2, 5)))

    @pytest.mark.parametrize('ctor', [dict(), dict(use_fast=True),
                                      dict(weighted=True)])
    def test_float32_freqs_accepted(self, ctor):
        t, y, dy = lightcurve(60, seed=0)
        freqs = np.linspace(0.1, 3.0, 50)
        proc = ConditionalEntropyAsyncProcess(**ctor)
        ref = run_ce(proc, t, y, dy, freqs)

        def same(p):
            # (the weighted kernel's float32 atomicAdd order varies
            # between runs at the 1e-7 level)
            assert_allclose(p, ref, rtol=0, atol=1e-6)

        same(run_ce(proc, t, y, dy, freqs.astype(np.float32)))
        same(run_ce(proc, t, y, dy, list(freqs)))
        r = proc.large_run([(t, y, dy)], freqs=freqs.astype(np.float32))
        proc.finish()
        same(np.copy(r[0][1]))
        # a list of per-lightcurve grids still works
        r = proc.run([(t, y, dy), (t, y, dy)],
                     freqs=[freqs.astype(np.float32), freqs])
        proc.finish()
        same(np.copy(r[0][1]))
        same(np.copy(r[1][1]))
        mems = proc.allocate([(t, y, dy)], freqs=freqs.astype(np.float32))
        assert mems[0].nf == len(freqs)
