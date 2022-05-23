from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import zip
from builtins import range
from builtins import object
import pytest
import numpy as np
from numpy.testing import assert_allclose
from pycuda.tools import mark_cuda_test
from ..bls import eebls_gpu, eebls_transit_gpu, \
                  q_transit, compile_bls, hone_solution,\
                  single_bls, eebls_gpu_custom, eebls_gpu_fast


def transit_model(phi0, q, delta, q1=0.):
    def model(t, freq, q=q, phi0=phi0, delta=delta):

        phi = t * freq - phi0
        phi -= np.floor(phi)

        if not hasattr(t, '__iter__'):
            return -delta if np.absolute(phi) < q else 0
        y = np.zeros(len(t))
        y[np.absolute(phi) < q] -= delta

        return y
    return model


def plot_bls_sol(t, y, dy, freq, q, phi0):

    w = np.power(dy, -2)
    w /= sum(w)

    phi_plot = np.linspace(0, 1, 50./q)

    phi = (t * freq)
    phi -= np.floor(phi)

    dphi = phi - phi0 - np.floor(phi - phi0)
    mask = dphi < q

    ybt = np.dot(w[mask], y[mask]) / sum(w[mask])
    yb0 = np.dot(w[~mask], y[~mask]) / sum(w[~mask])

    delta = yb0 - ybt

    model = transit_model(phi0, q, delta)

    ym = model(phi_plot, 1.) + yb0

    import matplotlib.pyplot as plt

    f, ax = plt.subplots()

    ax.scatter(phi[~mask], y[~mask], c='k', s=1, alpha=0.1)
    ax.scatter(phi[mask], y[mask], c='g', s=1, alpha=0.8)
    ax.plot(phi_plot, ym, color='r')
    ax.axvline(phi0, color='k', ls=':')
    ax.axvline(phi0 + q, color='k', ls=':')

    plt.show()


def data(seed=100, sigma=0.1, ybar=12., snr=10, ndata=200, freq=10.,
         q=0.01, phi0=None, baseline=1.):

    rand = np.random.RandomState(seed)

    if phi0 is None:
        phi0 = rand.rand()

    delta = snr * sigma / np.sqrt(ndata * q * (1 - q))

    model = transit_model(phi0, q, delta)

    t = baseline * np.sort(rand.rand(ndata))
    y = model(t, freq) + sigma * rand.randn(len(t))
    y += ybar - np.mean(y)
    err = sigma * np.ones_like(y)

    return t, y, err


def get_total_nbins(nbins0, nbinsf, dlogq):
    nbins_tot = 0
    while (int(x * nbins0) <= nbinsf):
        nb = int(x * nbins0)
        x *= 1 + dlogq

        nbins_tot += nb

    return nbins_tot


def mod1(x):
    return x - np.floor(x)


def manual_binning(t, y, dy, freqs, nbins0, nbinsf, dlogq,
                   phi_min, phi_max, noverlap):
    """
    for possible tests of the binning procedure. this
    method has *not* been tested!
    """

    w = np.power(dy, -2)
    w /= sum(w)

    yw = np.multiply(y, w)

    nbins_tot = get_total_nbins(nbins0, nbinsf, dlogq)

    yw_bins = np.zeros(nbins_tot * len(freqs) * noverlap)
    w_bins = np.zeros(nbins_tot * len(freqs) * noverlap)

    dphi = 1. / noverlap
    for i, freq in enumerate(freqs):
        nb = nbins0
        nbtot = 0
        x = 1.
        while (int(x * nbins0) <= nbinsf):
            nb = int(x * nbins0)
            x *= 1 + dlogq

            q = 1./nb

            for s in range(noverlap):
                phi = t * freq
                bf = np.floor(nb * mod1(phi - s * q * dphi))

                bf += i * nbins_tot * noverlap + s * nb + noverlap * nbtot
                for b, YW, W in zip(bf[mask], yw[mask], w[mask]):
                    yw_bins[b] += YW
                    w_bins[b] += W

            nbtot += nb
    return yw_bins, w_bins


class TestBLS(object):
    seed = 100
    rand = np.random.RandomState(seed)
    plot = False
    rtol = 1e-3
    atol = 1e-5

    @pytest.mark.parametrize("freq", [0.3])
    @pytest.mark.parametrize("phi0", [0.0, 0.5])
    @pytest.mark.parametrize("dlogq", [0.2, -1])
    @pytest.mark.parametrize("nstreams", [1, 3])
    @pytest.mark.parametrize("freq_batch_size", [1, 3, None])
    def test_transit_parameter_consistency(self, freq, phi0, dlogq, nstreams,
                                           freq_batch_size):
        q = q_transit(freq)

        t, y, dy = data(snr=30, q=q, phi0=phi0, freq=freq, baseline=365.)

        freqs, power, sols = eebls_transit_gpu(t, y, dy,
                                               samples_per_peak=2,
                                               freq_batch_size=freq_batch_size,
                                               nstreams=nstreams,
                                               dlogq=dlogq,
                                               fmin=freq * 0.99,
                                               fmax=freq * 1.01)
        pcpu = [single_bls(t, y, dy, x[0], *x[1])
                for x in zip(freqs, sols)]
        pcpu = np.asarray(pcpu)

        if self.plot:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots()
            ax.plot(freqs, pcpu)
            ax.plot(freqs, power)
            plt.show()

        sorted_results = sorted(zip(pcpu, power, freqs, sols),
                                key=lambda x: -abs(x[1] - x[0]))

        for i, (pcs, pgs, freq, (qs, phs)) in enumerate(sorted_results):
            if i > 10:
                break
            print(pcs, pgs, (qs, phs))
            if self.plot:
                plot_bls_sol(t, y, dy, freq, qs, phs)

        pows, diffs = list(zip(*sorted(zip(pcpu,
                                           np.absolute(power - pcpu)),
                                       key=lambda x: -x[1])))

        upper_bound = self.rtol * np.array(pows) + self.atol
        mostly_ok = sum(np.array(diffs) > upper_bound) / len(pows) < 1e-2
        not_too_bad = max(diffs) < 1e-1

        print(max(diffs))
        assert mostly_ok and not_too_bad

    @pytest.mark.parametrize("freq", [1.0])
    @pytest.mark.parametrize("phi_index", [0, 10, -1])
    @pytest.mark.parametrize("q_index", [0, 5, -1])
    @pytest.mark.parametrize("nstreams", [1, 3])
    @pytest.mark.parametrize("freq_batch_size", [1, 3, None])
    def test_custom(self, freq, q_index, phi_index, freq_batch_size, nstreams):
        q_values = np.logspace(-1.1, -0.8, num=10)
        phi_values = np.linspace(0, 1, int(np.ceil(2./min(q_values))))

        q = q_values[q_index]
        phi = phi_values[phi_index]

        t, y, dy = data(snr=10, q=q, phi0=phi, freq=freq,
                        baseline=365.)

        df = min(q_values) / (10 * (max(t) - min(t)))
        freqs = np.linspace(freq - 10 * df, freq + 10 * df, 20)

        power, gsols = eebls_gpu_custom(t, y, dy, freqs,
                                        q_values, phi_values,
                                        freq_batch_size=freq_batch_size,
                                        nstreams=nstreams)

        # Now get CPU values
        qgrid, phigrid = np.meshgrid(q_values, phi_values)
        bls = np.vectorize(single_bls, excluded=set([0, 1, 2, 3]))

        blses = []

        max_bls = []
        cpu_bls = []
        sols = []
        for freq, (qg, phg) in zip(freqs, gsols):
            p = bls(t, y, dy, freq, qgrid, phigrid)
            pc = single_bls(t, y, dy, freq, qg, phg)
            mind = np.unravel_index(p.argmax(), p.shape)
            qsol = qgrid[mind]
            phisol = phigrid[mind]

            max_bls.append(p[mind])
            cpu_bls.append(pc)
            sols.append((qsol, phisol))
            blses.append(p)
        qsg, phsg = list(zip(*gsols))
        qs, phs = list(zip(*sols))
        if self.plot:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots()

            ax.plot(freqs, max_bls)
            ax.plot(freqs, power)

            plt.show()

            f, ax = plt.subplots()
            ax.plot(freqs, qs)
            ax.plot(freqs, qsg)

            plt.show()

            f, ax = plt.subplots()
            ax.plot(freqs, phs)
            ax.plot(freqs, phsg)

            plt.show()

        for bls_arr in [max_bls, cpu_bls]:
            max_diffs = 0.5 * self.rtol * (bls_arr + power) + self.atol
            diffs = np.absolute(bls_arr - power)
            nbad = sum(diffs > max_diffs)

            mostly_ok = nbad / float(len(power)) < 1e-2 or nbad == 1
            not_too_bad = max(np.absolute(bls_arr - power) / power) < 1e-1

            print(max(diffs) / max_diffs[np.argmax(max_diffs)])
            if not (mostly_ok and not_too_bad):
                print(nbad / float(len(power)), not_too_bad)
                import matplotlib.pyplot as plt
                plt.plot(freqs, power)
                plt.plot(freqs, bls_arr)
                plt.show()
            assert(mostly_ok and not_too_bad)

    @pytest.mark.parametrize("freq", [1.0])
    @pytest.mark.parametrize("phi_index", [0, 10, -1])
    @pytest.mark.parametrize("q_index", [0, 5, -1])
    @pytest.mark.parametrize("nstreams", [1, 3])
    @pytest.mark.parametrize("freq_batch_size", [1, 3, None])
    def test_standard(self, freq, q_index, phi_index, nstreams, freq_batch_size):

        q_values = np.logspace(-1.5, np.log10(0.1), num=100)
        phi_values = np.linspace(0, 1, int(np.ceil(2./min(q_values))))

        q = q_values[q_index]
        phi = phi_values[phi_index]

        t, y, dy = data(snr=10, q=q, phi0=phi, freq=freq,
                        baseline=365.)

        df = min(q_values) / (10 * (max(t) - min(t)))

        delta_f = 5 * df / freq
        freqs = np.linspace(freq * (1 - delta_f),
                            (1 + delta_f) * freq,
                            int(5. * 2 * delta_f * freq / df))
        power, gsols = eebls_gpu(t, y, dy, freqs,
                                 qmin=0.1 * q, qmax=2.0 * q,
                                 nstreams=nstreams, noverlap=2, dlogq=0.5,
                                 freq_batch_size=freq_batch_size)

        bls_c = [single_bls(t, y, dy, x[0], *x[1]) for x in zip(freqs, gsols)]
        if self.plot:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots()

            ax.plot(freqs, bls_c)
            ax.plot(freqs, power)

            plt.show()

            inds = sorted(np.arange(len(power)),
                          key=lambda i: -abs(power[i] - bls_c[i]))

            all_qs, all_phis = zip(*gsols)

            for i in inds[:100]:
                qs, phis = gsols[i]
                print(power[i], bls_c[i], abs(power[i] - bls_c[i]),
                      qs, phis)
                #plot_bls_sol(t, y, dy, freqs[i], qs, phis)

        pows, diffs = list(zip(*sorted(zip(bls_c, np.absolute(power - bls_c)),
                               key=lambda x: -x[1])))

        upper_bound = self.rtol * np.array(pows) + self.atol
        mostly_ok = sum(np.array(diffs) > upper_bound) / len(pows) < 1e-2
        not_too_bad = max(diffs) < 1e-1

        print(diffs[0], pows[0])
        assert mostly_ok and not_too_bad
        # assert_allclose(bls_c, power, rtol=1e-3, atol=1e-5)

    @pytest.mark.parametrize("freq", [1.0])
    @pytest.mark.parametrize("dlogq", [0.5, -1.0])
    @pytest.mark.parametrize("freq_batch_size", [1, 10, None])
    @pytest.mark.parametrize("phi0", [0.0])
    @pytest.mark.parametrize("use_fast", [True, False])
    @pytest.mark.parametrize("nstreams", [1, 4])
    def test_transit(self, freq, use_fast, freq_batch_size, nstreams, phi0, dlogq):
        q = q_transit(freq)
        samples_per_peak = 2
        noverlap = 2

        t, y, err = data(snr=10, q=q, phi0=phi0, freq=freq,
                         baseline=365.)

        kw = dict(samples_per_peak=samples_per_peak,
                  freq_batch_size=freq_batch_size, dlogq=dlogq,
                  nstreams=nstreams, noverlap=noverlap,
                  fmin=0.9 * freq, fmax=1.1 * freq,
                  use_fast=use_fast)

        if use_fast:
            freqs, power = eebls_transit_gpu(t, y, err, **kw)

            kw['use_fast'] = False
            freqs, power_slow, sols = eebls_transit_gpu(t, y, err, **kw)

            dfsol = freqs[np.argmax(power)] - freqs[np.argmax(power_slow)]
            close_enough = abs(dfsol) * (max(t) - min(t)) / q < 3
            if not close_enough:
                import matplotlib.pyplot as plt
                plt.plot(freqs, power, alpha=0.5)
                plt.plot(freqs, power_slow, alpha=0.5)
                plt.show()

            assert(close_enough)
            return

        freqs, power, sols = eebls_transit_gpu(t, y, err, **kw)
        power_cpu = np.array([single_bls(t, y, err, x[0], *x[1]) for x in zip(freqs, sols)])

        if self.plot:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots()

            ax.plot(freqs, power_cpu)
            ax.plot(freqs, power)

            pows, diffs = list(zip(*sorted(zip(power_cpu, power - power_cpu),
                               key=lambda x: -abs(x[1]))))
            print(list(zip(pows[:10], diffs[:10])))
            plt.show()

        diffs = np.absolute(power - power_cpu)
        upper_bound = 1e-3 * np.array(power_cpu) + 1e-5
        mostly_ok = sum(np.array(diffs) > upper_bound) / len(diffs) < 1e-2
        not_too_bad = max(diffs) < 1e-1

        print(max(diffs))
        assert mostly_ok and not_too_bad

    @pytest.mark.parametrize("freq", [1.0])
    @pytest.mark.parametrize("q", [0.1])
    @pytest.mark.parametrize("phi0", [0.0])
    @pytest.mark.parametrize("dphi", [0.0, 1.0])
    @pytest.mark.parametrize("freq_batch_size", [None, 100])
    @pytest.mark.parametrize("dlogq", [0.5, -1.0])
    def test_fast_eebls(self, freq, q, phi0, freq_batch_size, dlogq, dphi,
                        **kwargs):
        t, y, err = data(snr=50, q=q, phi0=phi0, freq=freq,
                         baseline=365.)

        df = 0.25 * q / (max(t) - min(t))
        fmin = 0.9 * freq
        fmax = 1.1 * freq
        nf = int(np.ceil((fmax - fmin) / df))
        freqs = fmin + df * np.arange(nf)

        kw = dict(qmin=1e-2, qmax=0.5, dphi=dphi,
                  freq_batch_size=freq_batch_size, dlogq=dlogq)

        kw.update(kwargs)

        power = eebls_gpu_fast(t, y, err, freqs, **kw)

        power0, sols = eebls_gpu(t, y, err, freqs, **kw)
        if self.plot:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots()
            ax.plot(freqs, power, alpha=0.5)
            ax.axvline(freq, ls=':', color='k')
            ax.plot(freqs, power0, alpha=0.5)
            ax.set_yscale('log')
            plt.show()

        # this is janky. Need better test
        # to ensure we're getting the best results,
        # but no apples-to-apples comparison is
        # possible for eebls_gpu and eebls_gpu_fast
        fmax_fast = freqs[np.argmax(power)]
        fmax_regular = freqs[np.argmax(power0)]
        assert(abs(fmax_fast - fmax_regular) * (max(t) - min(t)) / q < 3)
