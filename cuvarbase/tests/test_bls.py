import numpy as np
import pytest
from numpy.testing import assert_allclose
from pycuda.tools import mark_cuda_test
from pycuda import gpuarray
import sys
from time import time
from ..bls import eebls_gpu, eebls_transit_gpu, \
                  q_transit, compile_bls, hone_solution,\
                  single_bls, eebls_gpu_custom

ntests = 3


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


@pytest.fixture
def data(seed=100, sigma=0.1, ybar=12., snr=10, ndata=500, freq=10.,
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


@mark_cuda_test
def test_transit_parameter_consistency(seed=100, plot=False):
    rand = np.random.RandomState(seed)
    for test in range(ntests):
        freq = 0.1 + 0.1 * rand.rand()
        q = q_transit(freq)
        phi0 = rand.rand()
        dlogq = 0.1

        outstr = "TEST {test} / {ntests}: freq={freq}, q={q}, phi0={phi0}, "\
                 "dlogq={dlogq}"
        print(outstr.format(test=test, ntests=ntests, freq=freq,
                            q=q, phi0=phi0, dlogq=dlogq))
        t, y, dy = data(snr=30, q=q, phi0=phi0, freq=freq, baseline=365.)

        freqs, power, sols = eebls_transit_gpu(t, y, dy, samples_per_peak=2,
                                               batch_size=1, nstreams=1,
                                               dlogq=dlogq, fmin=freq * 0.99,
                                               fmax=freq * 1.01)

        pcpu = map(lambda (f, (q, phi0)): single_bls(t, y, dy,
                                                     f, q, phi0),
                   zip(freqs, sols))
        pcpu = np.asarray(pcpu)

        sorted_results = sorted(zip(pcpu, power, freqs, sols),
                                key=lambda x: -abs(x[1] - x[0]))

        for i, (pcs, pgs, freq, (qs, phs)) in enumerate(sorted_results):
            if i > 10:
                break

            print pcs, pgs
            if plot:
                plot_bls_sol(t, y, dy, freq, qs, phs)

        pows, diffs = zip(*sorted(zip(pcpu, np.absolute(power - pcpu)),
                                  key=lambda x: -x[1]))

        upper_bound = 1e-3 * np.array(pows) + 1e-5
        mostly_ok = sum(np.array(diffs) > upper_bound) / len(pows) < 1e-2
        not_too_bad = max(diffs) < 1e-1

        print max(diffs)
        assert mostly_ok and not_too_bad
        # assert_allclose(pcpu, power, atol=1e-5, rtol=1e-3)


@mark_cuda_test
def test_custom(seed=100, plot=False):
    rand = np.random.RandomState(seed)
    for ntest in range(ntests):
        freq = 0.1 + np.random.rand()

        q_values = np.logspace(-1.1, -0.8, num=10)
        phi_values = np.linspace(0, 1, int(np.ceil(2./min(q_values))))

        q = q_values[rand.randint(len(q_values))]
        phi = phi_values[rand.randint(len(phi_values))]

        t, y, dy = data(snr=10, q=q, phi0=phi, freq=freq, ndata=300,
                        baseline=365.)

        df = min(q_values) / (10 * (max(t) - min(t)))

        freqs = np.linspace(freq - 10 * df, freq + 10 * df, 20)
        power, gsols = eebls_gpu_custom(t, y, dy, freqs, q_values, phi_values,
                                        batch_size=100, nstreams=5)

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
        qsg, phsg = zip(*gsols)
        qs, phs = zip(*sols)
        if plot:
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

        assert_allclose(max_bls, power, rtol=1e-3, atol=1e-5)
        assert_allclose(cpu_bls, power, rtol=1e-3, atol=1e-5)

    # assert_allclose(qs, qsg, rtol=1e-1, atol=1e-5)
    # assert_allclose(phs, phsg, rtol=1e-1, atol=1e-5)


@mark_cuda_test
def test_standard(seed=100, plot=False):
    rand = np.random.RandomState(seed)
    for ntest in range(ntests):
        freq = 0.1 + rand.rand()

        q_values = np.logspace(-1.5, -0.8, num=5)
        phi_values = np.linspace(0, 1, int(np.ceil(2./min(q_values))))

        q = q_values[rand.randint(len(q_values))]
        phi = phi_values[rand.randint(len(phi_values))]

        t, y, dy = data(snr=10, q=q, phi0=phi, freq=freq, ndata=300,
                        baseline=365.)

        df = min(q_values) / (10 * (max(t) - min(t)))

        delta_f = 0.02
        freqs = np.linspace(freq * (1 - delta_f),
                            (1 + delta_f) * freq,
                            int(5. * 2 * delta_f * freq / df))
        power, gsols = eebls_gpu(t, y, dy, freqs, qmin=0.3 * q, qmax=3 * q,
                                 nstreams=5, noverlap=3, dlogq=0.5,
                                 batch_size=100)

        bls_c = map(lambda (f, (qs, ps)): single_bls(t, y, dy, f, qs, ps),
                    zip(freqs, gsols))
        if plot:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots()

            ax.plot(freqs, bls_c)
            ax.plot(freqs, power)

            plt.show()

            inds = sorted(np.arange(len(power)),
                          key=lambda i: -abs(power[i] - bls_c[i]))

            for i in inds[:10]:
                qs, phis = gsols[i]
                print power[i], bls_c[i]
                plot_bls_sol(t, y, dy, freqs[i], qs, phis)

        pows, diffs = zip(*sorted(zip(bls_c, np.absolute(power - bls_c)),
                                  key=lambda x: -x[1]))

        upper_bound = 1e-3 * np.array(pows) + 1e-5
        mostly_ok = sum(np.array(diffs) > upper_bound) / len(pows) < 1e-2
        not_too_bad = max(diffs) < 1e-1

        print diffs[0], pows[0]
        assert mostly_ok and not_too_bad
    # assert_allclose(bls_c, power, rtol=1e-3, atol=1e-5)


@mark_cuda_test
def test_transit(seed=100, plot=False):
    rand = np.random.RandomState(seed)
    freq = 1.0 + 0.1 * rand.rand()
    q = q_transit(freq)
    phi0 = rand.rand()
    dlogq = 0.3
    samples_per_peak = 2
    noverlap = 2

    outstr = "freq={freq}, q={q}, phi0={phi0}, "\
             "dlogq={dlogq}"
    print(outstr.format(freq=freq,
                        q=q, phi0=phi0, dlogq=dlogq))
    t, y, err = data(snr=10, q=q, phi0=phi0, freq=freq, ndata=300,
                     baseline=365.)

    freqs, power, sols = eebls_transit_gpu(t, y, err,
                                           samples_per_peak=samples_per_peak,
                                           batch_size=50, dlogq=dlogq,
                                           nstreams=5, noverlap=noverlap,
                                           fmin=0.75 * freq, fmax=1.25 * freq)

    power_cpu = np.array(map(lambda (f, (qs, phis)):
                         single_bls(t, y, err, f, qs, phis),
                         zip(freqs, sols)))

    if plot:
        import matplotlib.pyplot as plt
        f, ax = plt.subplots()

        ax.plot(freqs, power_cpu)
        ax.plot(freqs, power)

        pows, diffs = zip(*sorted(zip(power_cpu, power - power_cpu),
                          key=lambda x: -abs(x[1])))
        print(zip(pows[:10], diffs[:10]))
        plt.show()

    diffs = np.absolute(power - power_cpu)
    upper_bound = 1e-3 * np.array(power_cpu) + 1e-5
    mostly_ok = sum(np.array(diffs) > upper_bound) / len(diffs) < 1e-2
    not_too_bad = max(diffs) < 1e-1

    print max(diffs)
    assert mostly_ok and not_too_bad
