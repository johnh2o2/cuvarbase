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
def data(seed=100, sigma=0.1, ndata=100, freq=3., snr=10, t0=0.):

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

    def test_multiple_datasets(self, **kwargs):

        ndatas = 5
        datas = [data() for i in range(ndatas)]
        proc = ConditionalEntropyAsyncProcess(**kwargs)

        mult_results = proc.run(datas)
        proc.finish()

        sing_results = []

        for d in datas:
            sing_results.extend(proc.run([d]))
            proc.finish()

        for rb, rnb in zip(mult_results, sing_results):
            fb, pb = rb
            fnb, pnb = rnb

            assert(not any(np.isnan(pb)))
            assert(not any(np.isnan(pnb)))

            assert_allclose(pnb, pb, rtol=lsrtol, atol=lsatol)
            assert_allclose(fnb, fb, rtol=lsrtol, atol=lsatol)

    def test_batched_run(self, ndatas=7, batch_size=3, **kwargs):

        datas = [data(ndata=rand.randint(50, 100))
                 for i in range(ndatas)]
        proc = ConditionalEntropyAsyncProcess(**kwargs)

        batched_results = proc.batched_run(datas)
        proc.finish()

        non_batched_results = []
        for d in datas:
            r = proc.run([d])
            proc.finish()
            non_batched_results.extend(r)

        for rb, rnb in zip(batched_results, non_batched_results):
            fb, pb = rb
            fnb, pnb = rnb

            assert(not any(np.isnan(pb)))
            assert(not any(np.isnan(pnb)))

            assert_allclose(pnb, pb, rtol=lsrtol, atol=lsatol)
            assert_allclose(fnb, fb, rtol=lsrtol, atol=lsatol)

    def test_batched_run_const_nfreq(self, make_plot=False, ndatas=7,
                                     batch_size=3, seed=100,
                                     **kwargs):
        frequencies = np.sort(10 + rand.rand(ndatas) * 100.)
        datas = [data(ndata=rand.randint(50, 100),
                      freq=freq)
                 for i, freq in enumerate(frequencies)]
        proc = ConditionalEntropyAsyncProcess(**kwargs)

        batched_results = proc.batched_run_const_nfreq(datas, **kwargs)
        proc.finish()

        procnb = ConditionalEntropyAsyncProcess(**kwargs)

        non_batched_results = []
        for d, (frq, p) in zip(datas, batched_results):
            r = procnb.run([d], freqs=frq, **kwargs)
            procnb.finish()
            non_batched_results.extend(r)

        # for f0, (fb, pb), (fnb, pnb) in zip(frequencies, batched_results,
        #                                    non_batched_results):
        #    print f0, fb[np.argmax(pb)], fnb[np.argmax(pnb)]

        for f0, (fb, pb), (fnb, pnb) in zip(frequencies, batched_results,
                                            non_batched_results):

            if make_plot:
                import matplotlib.pyplot as plt
                plt.plot(fnb, pnb, color='k', lw=3)
                plt.plot(fb, pb, color='r')
                plt.axvline(f0)
                plt.show()
            assert(not any(np.isnan(pb)))
            assert(not any(np.isnan(pnb)))

            assert_allclose(pnb, pb, rtol=lsrtol, atol=lsatol)
            assert_allclose(fnb, fb, rtol=lsrtol, atol=lsatol)

    def test_inject_and_recover(self, make_plot=False, **kwargs):

        proc = ConditionalEntropyAsyncProcess(**kwargs)
        for freq in [5.0, 10.0, 50.0]:
            t0 = kwargs.get('t0', 0.)
            t, y, err = data(sigma=0.01, snr=100, ndata=200,
                             freq=freq, t0=t0)

            df = 0.001
            max_freq = 100.
            min_freq = df
            nf = int((max_freq - min_freq) / df)
            freqs = min_freq + df * np.arange(nf)
            results = proc.large_run([(t, y, err)],
                                     freqs=freqs,
                                     max_memory=1e8)
            proc.finish()
            frq, p = results[0]
            best_freq = frq[np.argmin(p)]

            if make_plot:
                import matplotlib.pyplot as plt
                f, ax = plt.subplots()
                ax.plot(frq, p)
                ax.axvline(freq, ls='-', color='k')
                ax.axvline(best_freq, ls=':', color='r')
                plt.show()

            # print best_freq, freq, abs(best_freq - freq) / freq
            assert(not any(np.isnan(p)))
            assert(abs(best_freq - freq) / freq < 1E-2)

    def test_balanced_magbins(self, make_plot=False, **kwargs):
        self.test_inject_and_recover(make_plot=make_plot,
                                     balanced_magbins=True,
                                     **kwargs)

    def test_inject_and_recover_weighted(self, make_plot=False, **kwargs):
        kwargs.update({'weighted': True})
        self.test_inject_and_recover(make_plot=make_plot, **kwargs)

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

        assert_allclose(p0, p1)

    def test_double(self, make_plot=False, **kwargs):

        proc1 = ConditionalEntropyAsyncProcess(**kwargs)
        proc2 = ConditionalEntropyAsyncProcess(use_double=True, **kwargs)
        freq = 10.

        t, y, err = data(sigma=0.1, snr=20, ndata=100, freq=freq)

        # y[5] = np.median(y) + 10.

        df = 0.001
        max_freq = 100.
        min_freq = df
        nf = int((max_freq - min_freq) / df)
        freqs = min_freq + df * np.arange(nf)

        results = proc1.large_run([(t, y, err)], freqs=freqs, max_memory=1e7)
        proc1.finish()
        frq, p = results[0]

        results1 = proc2.large_run([(t, y, err)], freqs=freqs, max_memory=1e7)
        proc2.finish()
        frq1, p1 = results1[0]

        best_freq = frq[np.argmin(p)]

        if make_plot:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots()
            ax.plot(frq, p)
            ax.plot(frq1, p1)
            ax.axvline(freq, ls=':', color='k', alpha=0.5)
            # ax.axvline(best_freq, ls=':', color='r')
            plt.show()

        # print best_freq, freq, abs(best_freq - freq) / freq
        assert(not any(np.isnan(p)))
        assert(not any(np.isnan(p1)))

        spows = sorted(zip(p, p1), key=lambda x: -abs(x[1] - x[0]))

        diffs = np.absolute(p - p1)
        tolerance = 1e-2 * 0.5 * (np.absolute(p) + np.absolute(p1)) + 1e-3
        bad_frac = sum(diffs > tolerance) / float(len(p))

        for P, P1 in spows:
            if abs(P - P1) > 1e-3:
                print(P, P1, abs(P - P1))

        assert bad_frac < 1e-2
        # assert_allclose(p, p1, rtol=1e-2, atol=1e-3)

    def test_double_weighted(self, make_plot=False, **kwargs):
        self.test_double(weighted=True, make_plot=make_plot, **kwargs)

    def test_time_shift_invariance(self, make_plot=False,
                                   use_double=True, **kwargs):
        proc = ConditionalEntropyAsyncProcess(use_double=use_double, **kwargs)
        freq = 10.
        for t0 in [-1e4, 1e4]:
            t, y, err = data(seed=100, sigma=0.01, ndata=200, freq=freq)

            df = 0.001
            max_freq = 100.
            min_freq = df
            nf = int((max_freq - min_freq) / df)
            freqs = min_freq + df * np.arange(nf)
            results = proc.run([(t, y, err)], freqs=freqs)
            proc.finish()
            frq, p = results[0]

            results1 = proc.run([(t + t0, y, err)], freqs=freqs)
            frq1, p1 = results1[0]

            best_freq = frq[np.argmin(p)]

            if make_plot:
                import matplotlib.pyplot as plt
                f, ax = plt.subplots()
                ax.plot(frq, p)
                ax.plot(frq1, p1)
                ax.axvline(freq, ls='-', color='k')
                ax.axvline(best_freq, ls=':', color='r')
                plt.show()

            # print best_freq, freq, abs(best_freq - freq) / freq
            assert(not any(np.isnan(p)))
            assert(not any(np.isnan(p1)))
            assert_allclose(p, p1, rtol=1e-3)

    def test_fast(self, make_plot=True,
                  use_double=True, **kwargs):
        proc = ConditionalEntropyAsyncProcess(use_double=use_double,
                                              use_fast=True, **kwargs)
        freq = 10.
        for t0 in [-1e4, 1e4]:
            t, y, err = data(seed=100, sigma=0.01, ndata=200, freq=freq)

            df = 0.001
            max_freq = 100.
            min_freq = df
            nf = int((max_freq - min_freq) / df)
            freqs = min_freq + df * np.arange(nf)
            results = proc.run([(t, y, err)], freqs=freqs)
            proc.finish()
            frq, p = results[0]

            results1 = proc.run([(t + t0, y, err)], freqs=freqs)
            frq1, p1 = results1[0]

            best_freq = frq[np.argmin(p)]

            if make_plot:
                import matplotlib.pyplot as plt
                f, ax = plt.subplots()
                ax.plot(frq, p)
                ax.plot(frq1, p1)
                ax.axvline(freq, ls='-', color='k')
                ax.axvline(best_freq, ls=':', color='r')
                plt.show()

            # print best_freq, freq, abs(best_freq - freq) / freq
            assert(not any(np.isnan(p)))
            assert(not any(np.isnan(p1)))
            assert_allclose(p, p1, rtol=1e-3)
