from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import zip
from builtins import range
from builtins import object
import pytest
from pycuda.tools import mark_cuda_test
import numpy as np
from numpy.testing import assert_allclose
from ..ce import ConditionalEntropyAsyncProcess
lsrtol = 1E-2
lsatol = 1E-5
seed = 100

rand = np.random.RandomState(seed)


@pytest.fixture
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
        assert_allclose(p_slow, p_fast, atol=2e-2 * max(np.absolute(p_slow)))
