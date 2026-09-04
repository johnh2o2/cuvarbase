import pytest
from pycuda.tools import mark_cuda_test
import pycuda.gpuarray as gpuarray
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import ndtr
from ..ce import ConditionalEntropyAsyncProcess
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


def exact_weighted_hist(t, y, dy, freqs, nphase, nmag):
    """Weighted-CE histogram with the EXACT Gaussian probability mass of
    every point in every magnitude bin (no truncation)."""
    t, Y, yscale = _prep(t, y, np.float32)
    Y = Y.astype(np.float64)
    DY = (np.asarray(dy, dtype=np.float32) / yscale).astype(np.float64)
    m = np.arange(nmag)
    P = (ndtr(((m + 1) / nmag - Y[:, None]) / DY[:, None])
         - ndtr((m / nmag - Y[:, None]) / DY[:, None]))
    H = np.zeros((len(freqs), nphase, nmag))
    for i, f in enumerate(freqs):
        n0 = _phase_bins(t, f, nphase, np.float32)
        np.add.at(H, (i, n0), P)
    return H


def weighted_ce_from_hist(H, nmag):
    Nphi = H.sum(axis=2, keepdims=True)
    dm = 1.0 / nmag
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
    @pytest.mark.parametrize('freq', [10.0])
    @pytest.mark.parametrize('t0', [0.0])
    @pytest.mark.parametrize('balanced_magbins', [True, False])
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
    @pytest.mark.parametrize('use_fast,weighted,shmem_lc,freq_batch_size',
                             [(True, False, False, 1)])
    @pytest.mark.parametrize('phase_bins,phase_overlap',
                             [(10, 1)])
    @pytest.mark.parametrize('mag_bins,mag_overlap',
                             [(5, 0)])
    @pytest.mark.parametrize('freq', [10.0])
    @pytest.mark.parametrize('balanced_magbins', [True, False])
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

    @pytest.mark.parametrize('use_double', [True, False])
    @pytest.mark.parametrize('shmem_lc', [True, False])
    @pytest.mark.parametrize('freq_batch_size', [1, None])
    @pytest.mark.parametrize('phase_bins,phase_overlap,mag_bins,mag_overlap',
                             [(10, 0, 5, 0), (10, 1, 5, 1)])
    @pytest.mark.parametrize('freq', [12.0])
    @pytest.mark.parametrize('t0', [0.0])
    #@pytest.mark.parametrize('balanced_magbins', [True, False])
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
        nb = mem.nbins
        guard = np.uint32(0xDEAD)
        big = gpuarray.zeros(nb + 8, dtype=np.uint32)
        big.fill(guard)
        mem.bins_g = big[:nb]
        proc.run([(t, y, dy)], memory=mems, freqs=[freqs])
        proc.finish()
        full = big.get()
        assert_array_equal(full[nb:], np.full(8, guard))
        totals = full[:nb].reshape(len(freqs), -1).sum(axis=1)
        assert_array_equal(totals, np.full(len(freqs), N))

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
    def test_bins_and_ce_vs_ndtr_reference(self, mag_bins, noise):
        r = np.random.RandomState(3)
        N = 300
        t = np.sort(r.rand(N)) * 20.0
        y = (12 + np.sin(2 * np.pi * 1.3 * t) + 0.3 * np.sin(4 * np.pi * 1.3 * t)
             + noise * r.randn(N))
        dy = noise * np.ones(N)
        freqs = np.linspace(0.1, 3.0, 40)
        He = exact_weighted_hist(t, y, dy, freqs, 10, mag_bins)
        ce_exact = weighted_ce_from_hist(He, mag_bins)

        # default max_phi=3: only bins wholly beyond 3 sigma are skipped
        proc = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=mag_bins,
                                              weighted=True, max_phi=3.0)
        ce, mem = run_ce_with_memory(proc, t, y, dy, freqs)
        bins = mem.bins_g.get().reshape(len(freqs), 10, mag_bins)
        assert np.all(np.isfinite(ce))
        # audit-measured post-fix levels: bins 6e-3, CE 1.1e-3 (old: 1.5-4.2
        # in the bins, 2e-2 .. 5e-2 in the CE)
        assert_allclose(bins, He, rtol=0, atol=2e-2)
        assert_allclose(ce, ce_exact, rtol=0, atol=5e-3)
        # the per-frequency mass totals match the exact ones to the mass
        # of the skipped > 3-sigma bins (points near the range edges
        # legitimately lose the mass outside [0, 1]; old: -2 .. -12%)
        assert_allclose(bins.sum(axis=(1, 2)), He.sum(axis=(1, 2)),
                        rtol=3e-3, atol=0)

        # with a wide max_phi nothing is truncated: float32 normcdf level
        proc = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=mag_bins,
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
