from itertools import product 
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..bls import eebls_gpu, eebls_transit_gpu, \
                  q_transit, compile_bls, hone_solution,\
                  single_bls, eebls_gpu_custom, eebls_gpu_fast, \
                  eebls_gpu_fast_optimized, \
                  sparse_bls_cpu, sparse_bls_gpu, eebls_transit


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
         q=0.01, phi0=None, baseline=1., negative_delta=False):

    rand = np.random.RandomState(seed)

    if phi0 is None:
        phi0 = rand.rand()

    delta = snr * sigma / np.sqrt(ndata * q * (1 - q))

    if negative_delta:
        delta *= -1

    model = transit_model(phi0, q, delta)

    t = baseline * np.sort(rand.rand(ndata))
    y = model(t, freq) + sigma * rand.randn(len(t))
    y += ybar - np.mean(y)
    err = sigma * np.ones_like(y)

    return t, y, err


def get_total_nbins(nbins0, nbinsf, dlogq):
    nbins_tot = 0
    x = 1.
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
                for b, YW, W in zip(bf.astype(int), yw, w):
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

    # TODO: tests that have specific bls values; test single_bls function returns
    #       what you expect it to for several example problems
    class SolutionParams(object):
        def __init__(self, freq, phi0, q, baseline, ybar, snr, negative_delta):
            self.freq = freq
            self.phi0 = phi0
            self.q = q
            self.baseline = baseline
            self.ybar = ybar
            self.snr = snr
            self.negative_delta = negative_delta

    @pytest.mark.parametrize("args", [(
            SolutionParams(freq=0.3, phi0=0.5, q=0.2, baseline=365., ybar=0., snr=50.,
                           negative_delta=True),
            {'bls0': 0.8902446483898836, 'bls_ignore': 0}
        )
    ])
    def test_ignore_positive_sols(self, args):
        solution, bls_values = args
        t, y_neg, dy = data(snr=solution.snr,
                            q=solution.q,
                            phi0=solution.phi0,
                            freq=solution.freq,
                            baseline=solution.baseline,
                            ybar=solution.ybar,
                            negative_delta=solution.negative_delta)
        
        freq, q, phi0 = solution.freq, solution.q, solution.phi0

        # single_bls folds epoch-subtracted times (phases relative to
        # floor(min(t))); shift the injected absolute-time phase to match
        phi0 = (phi0 - np.floor(np.min(t)) * freq) % 1.0

        bls_default = single_bls(t, y_neg, dy, freq, q, phi0)
        bls0 = single_bls(t, y_neg, dy, freq, q, phi0, ignore_negative_delta_sols=False)
        bls_ignore = single_bls(t, y_neg, dy, freq, q, phi0, 
                                ignore_negative_delta_sols=True)
        assert np.allclose(bls_values['bls0'] , bls0)
        assert bls_values['bls_ignore'] == bls_ignore
        assert (bls0 == bls_default)

    @pytest.mark.parametrize("freq", [0.3])
    @pytest.mark.parametrize("phi0", [0.0, 0.5])
    @pytest.mark.parametrize("dlogq", [0.2, -1])
    @pytest.mark.parametrize("nstreams", [1, 3])
    @pytest.mark.parametrize("freq_batch_size", [1, 3, None])
    @pytest.mark.parametrize("ignore_negative_delta_sols", [True, False])
    def test_transit_parameter_consistency(self, freq, phi0, dlogq, nstreams,
                                           freq_batch_size, ignore_negative_delta_sols):
        q = q_transit(freq)

        t, y, dy = data(snr=30, q=q, phi0=phi0, freq=freq, baseline=365.)

        freqs, power, sols = eebls_transit_gpu(t, y, dy,
                                               samples_per_peak=2,
                                               freq_batch_size=freq_batch_size,
                                               nstreams=nstreams,
                                               dlogq=dlogq,
                                               ignore_negative_delta_sols=ignore_negative_delta_sols,
                                               fmin=freq * 0.99,
                                               fmax=freq * 1.01)
        pcpu = [single_bls(t, y, dy, x[0], *x[1], ignore_negative_delta_sols=ignore_negative_delta_sols)
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

        qsols = np.array([s[0] for s in sols])
        pows, diffs, qq = list(zip(*sorted(zip(pcpu,
                                               np.absolute(power - pcpu),
                                               qsols),
                                           key=lambda x: -x[1])))

        # The binned (GPU) and exact (single_bls) powers can disagree
        # by ~power/n_in_transit when a single point's float32 phase
        # lands on the opposite side of a bin edge in the kernel's
        # fast-math fold vs numpy's. For tiny-q solutions (n ~ ndata*q
        # points in transit) that single-point jitter is O(0.1), so
        # both criteria are scale-aware: tight where boxes hold >= ~8
        # points, loose (one-point jitter) below.
        ndata = len(t)
        n_in_transit = ndata * np.array(qq)
        well_populated = n_in_transit >= 8

        upper_bound = self.rtol * np.array(pows) + self.atol
        viol = (np.array(diffs) > upper_bound) & well_populated
        mostly_ok = viol.sum() / len(pows) < 1e-2

        cap = np.where(well_populated, 1e-1, 2.5e-1)
        not_too_bad = np.all(np.array(diffs) < cap)

        print(max(diffs))
        assert mostly_ok and not_too_bad

    @pytest.mark.parametrize("freq", [1.0])
    @pytest.mark.parametrize("phi_index", [0, 10])
    @pytest.mark.parametrize("q_index", [0, 5])
    @pytest.mark.parametrize("nstreams", [1, 3])
    @pytest.mark.parametrize("freq_batch_size", [1, 3, None])
    @pytest.mark.parametrize("ignore_negative_delta_sols", [True, False])
    def test_custom(self, freq, q_index, phi_index, freq_batch_size, nstreams,
                    ignore_negative_delta_sols):
        q_values = np.logspace(-1.1, -0.8, num=10)
        phi_values = np.linspace(0, 1, int(np.ceil(2./min(q_values))))

        q = q_values[q_index]
        phi = phi_values[phi_index]

        t, y, dy = data(snr=10, q=q, phi0=phi, freq=freq,
                        baseline=365., ndata=500)

        df = min(q_values) / (10 * (max(t) - min(t)))
        freqs = np.linspace(freq - 10 * df, freq + 10 * df, 20)

        power, gsols = eebls_gpu_custom(t, y, dy, freqs,
                                        q_values, phi_values,
                                        ignore_negative_delta_sols=ignore_negative_delta_sols,
                                        freq_batch_size=freq_batch_size,
                                        nstreams=nstreams)

        for freq, (qg, phg), gpower in zip(freqs, gsols, power):
            q_and_phis = product(q_values, phi_values)
            
            best_q, best_phi, best_p = None, None, None
            for Q, PHI in q_and_phis:
                p = single_bls(t, y, dy, freq, Q, PHI,
                               ignore_negative_delta_sols=ignore_negative_delta_sols)
                if best_p is None or p > best_p:
                    best_p = p
                    best_q = Q
                    best_phi = PHI
            
            assert np.abs(best_p - gpower) < 1e-5

    @pytest.mark.parametrize("freq", [1.0])
    @pytest.mark.parametrize("phi_index", [0, 10, -1])
    @pytest.mark.parametrize("q_index", [0, 5, -1])
    @pytest.mark.parametrize("nstreams", [1, 3])
    @pytest.mark.parametrize("freq_batch_size", [1, 3, None])
    @pytest.mark.parametrize("ignore_negative_delta_sols", [True, False])
    def test_standard(self, freq, q_index, phi_index, nstreams, freq_batch_size,
                      ignore_negative_delta_sols):

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
                                 freq_batch_size=freq_batch_size,
                                 ignore_negative_delta_sols=ignore_negative_delta_sols)

        bls_c = [single_bls(t, y, dy, x[0], *x[1],
                            ignore_negative_delta_sols=ignore_negative_delta_sols)
                 for x in zip(freqs, gsols)]
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
    @pytest.mark.parametrize("ignore_negative_delta_sols", [True, False])
    def test_transit(self, freq, use_fast, freq_batch_size, nstreams, phi0, dlogq,
                     ignore_negative_delta_sols):
        q = q_transit(freq)
        samples_per_peak = 2
        noverlap = 2

        t, y, err = data(snr=10, q=q, phi0=phi0, freq=freq,
                         baseline=365.)

        kw = dict(samples_per_peak=samples_per_peak,
                  freq_batch_size=freq_batch_size, dlogq=dlogq,
                  ignore_negative_delta_sols=ignore_negative_delta_sols,
                  nstreams=nstreams, noverlap=noverlap,
                  fmin=0.9 * freq, fmax=1.1 * freq,
                  use_fast=use_fast)

        if use_fast:
            freqs, power = eebls_transit_gpu(t, y, err, **kw)

            kw['use_fast'] = False
            freqs, power_slow, sols = eebls_transit_gpu(t, y, err, **kw)
            kw['use_fast'] = True
            dfsol = freqs[np.argmax(power)] - freqs[np.argmax(power_slow)]
            close_enough = abs(dfsol) * (max(t) - min(t)) / q < 3
            if not close_enough and self.plot:
                import matplotlib.pyplot as plt
                plt.plot(freqs, power, alpha=0.5)
                plt.plot(freqs, power_slow, alpha=0.5)
                plt.show()

            assert(close_enough)
            return

        freqs, power, sols = eebls_transit_gpu(t, y, err, **kw)
        power_cpu = np.array([single_bls(t, y, err, x[0], *x[1],
                                         ignore_negative_delta_sols=ignore_negative_delta_sols)
                              for x in zip(freqs, sols)])

        if self.plot:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots()

            ax.plot(freqs, power_cpu)
            ax.plot(freqs, power)

            pows, diffs = list(zip(*sorted(zip(power_cpu, power - power_cpu),
                               key=lambda x: -abs(x[1]))))
            print(list(zip(pows[:10], diffs[:10])))
            plt.show()

        # Same scale-aware criteria as test_transit_parameter_consistency:
        # at freq=1 the Keplerian q is ~0.017, so every box holds < 8
        # points and binned-vs-exact powers jitter by ~power/n when a
        # single point's float32 phase crosses a bin edge.
        diffs = np.absolute(power - power_cpu)
        qsols = np.array([s[0] for s in sols])
        well_populated = len(t) * qsols >= 8

        upper_bound = 1e-3 * np.array(power_cpu) + 1e-5
        viol = (diffs > upper_bound) & well_populated
        mostly_ok = viol.sum() / len(diffs) < 1e-2
        not_too_bad = np.all(diffs < np.where(well_populated, 1e-1, 2.5e-1))

        print(max(diffs))
        assert mostly_ok and not_too_bad

    @pytest.mark.parametrize("freq", [1.0])
    @pytest.mark.parametrize("q", [0.1])
    @pytest.mark.parametrize("phi0", [0.0])
    @pytest.mark.parametrize("dphi", [0.0, 1.0])
    @pytest.mark.parametrize("freq_batch_size", [None, 100])
    @pytest.mark.parametrize("dlogq", [0.5, -1.0])
    @pytest.mark.parametrize("ignore_negative_delta_sols", [True, False])
    def test_fast_eebls(self, freq, q, phi0, freq_batch_size, dlogq, dphi,
                        ignore_negative_delta_sols, **kwargs):
        t, y, err = data(snr=50, q=q, phi0=phi0, freq=freq,
                         baseline=365.)

        df = 0.25 * q / (max(t) - min(t))
        fmin = 0.9 * freq
        fmax = 1.1 * freq
        nf = int(np.ceil((fmax - fmin) / df))
        freqs = fmin + df * np.arange(nf)

        kw = dict(qmin=1e-2, qmax=0.5, dphi=dphi,
                  ignore_negative_delta_sols=ignore_negative_delta_sols,
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

    # ---- Sparse BLS tests: ground-truth correctness ----

    @staticmethod
    def _brute_force_bls(t, y, dy, freq, ignore_negative_delta_sols=False,
                         qmin=0.0, qmax=0.5):
        """Exhaustive BLS over all observation-pair transit boundaries."""
        t = np.asarray(t, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        dy = np.asarray(dy, dtype=np.float32)

        ndata = len(t)
        w = np.power(dy, -2, dtype=np.float32)
        w /= np.sum(w)

        phi = (t * freq) % 1.0
        idx = np.argsort(phi)
        phi_s, y_s, w_s = phi[idx], y[idx], w[idx]

        ybar = np.dot(w, y)
        YY = np.dot(w, (y - ybar) ** 2)

        max_bls, best_q, best_phi = 0.0, 0.0, 0.0

        # Non-wrapped pairs
        for i in range(ndata):
            W_acc, YW_acc = 0.0, 0.0
            for j in range(i + 1, ndata + 1):
                W_acc += w_s[j - 1]
                YW_acc += w_s[j - 1] * y_s[j - 1]
                if j < ndata:
                    q = 0.5 * (phi_s[j] + phi_s[j - 1]) - phi_s[i]
                else:
                    q = phi_s[ndata - 1] - phi_s[i] + 1e-7
                if q <= 0 or q < qmin or q > qmax:
                    continue
                W = W_acc
                YW = YW_acc - ybar * W
                if W < 1e-9 or W > 1.0 - 1e-9:
                    continue
                if YW > 0 and ignore_negative_delta_sols:
                    continue
                bls = (YW ** 2) / (W * (1 - W)) / YY
                if bls > max_bls:
                    max_bls, best_q, best_phi = bls, q, phi_s[i]

        # Wrapped pairs
        for i in range(ndata):
            W_tail = float(np.sum(w_s[i:]))
            YW_tail = float(np.dot(w_s[i:], y_s[i:]))
            W_head, YW_head = 0.0, 0.0
            for k in range(i):
                if k > 0:
                    W_head += w_s[k - 1]
                    YW_head += w_s[k - 1] * y_s[k - 1]
                phi0 = phi_s[i]
                if k > 0:
                    q = (1.0 - phi0) + 0.5 * (phi_s[k - 1] + phi_s[k])
                else:
                    q = 1.0 - phi0 + 1e-7
                if q <= 0 or q < qmin or q > qmax:
                    continue
                W = W_tail + W_head
                YW = (YW_tail + YW_head) - ybar * W
                if W < 1e-9 or W > 1.0 - 1e-9:
                    continue
                if YW > 0 and ignore_negative_delta_sols:
                    continue
                bls = (YW ** 2) / (W * (1 - W)) / YY
                if bls > max_bls:
                    max_bls, best_q, best_phi = bls, q, phi0

        return max_bls, best_q, best_phi

    @pytest.mark.parametrize("ndata", [10, 15, 20])
    @pytest.mark.parametrize("freq", [1.0, 2.5])
    @pytest.mark.parametrize("seed", [42, 123])
    @pytest.mark.parametrize("ignore_negative_delta_sols", [True, False])
    def test_sparse_bls_vs_exhaustive(self, ndata, freq, seed,
                                      ignore_negative_delta_sols):
        """Verify sparse_bls_cpu matches exhaustive brute-force search."""
        rand = np.random.RandomState(seed)
        sigma = 0.1
        q_true, phi0_true = 0.1, 0.3
        delta = 5.0 * sigma / np.sqrt(ndata * q_true)

        t = np.sort(rand.rand(ndata))
        y = np.zeros(ndata)
        phi = (t * freq - phi0_true) % 1.0
        y[phi < q_true] -= delta
        y += sigma * rand.randn(ndata)
        dy = sigma * np.ones(ndata)

        freqs = np.array([freq], dtype=np.float32)
        power, sols = sparse_bls_cpu(
            t, y, dy, freqs,
            ignore_negative_delta_sols=ignore_negative_delta_sols)
        bf_power, _, _ = self._brute_force_bls(
            t, y, dy, freq,
            ignore_negative_delta_sols=ignore_negative_delta_sols)

        assert np.abs(power[0] - bf_power) < 1e-5, \
            f"sparse={power[0]:.8f}, brute={bf_power:.8f}"

    @pytest.mark.parametrize("freq", [1.0, 2.0])
    @pytest.mark.parametrize("q", [0.05, 0.1])
    @pytest.mark.parametrize("phi0", [0.0, 0.3, 0.5])
    @pytest.mark.parametrize("ndata", [100, 200])
    def test_sparse_bls_ground_truth(self, freq, q, phi0, ndata):
        """Verify sparse_bls_cpu recovers a known injected transit."""
        t, y, dy = data(snr=50, q=q, phi0=phi0, freq=freq,
                        baseline=365., ndata=ndata)

        df = q / (10 * (max(t) - min(t)))
        freqs = np.linspace(freq - 5 * df, freq + 5 * df, 21)

        power, sols = sparse_bls_cpu(t, y, dy, freqs)

        # Best frequency should be within the searched range
        best_idx = np.argmax(power)
        best_freq = freqs[best_idx]
        T = max(t) - min(t)
        assert np.abs(best_freq - freq) < q / T, \
            f"Expected freq~{freq}, got {best_freq}"

        # Verify solution is consistent with single_bls
        q_found, phi_found = sols[best_idx]
        p_single = single_bls(t, y, dy, best_freq, q_found, phi_found)
        assert np.abs(power[best_idx] - p_single) < 1e-4, \
            f"sparse={power[best_idx]}, single_bls={p_single}"

    @pytest.mark.parametrize("freq", [1.0])
    @pytest.mark.parametrize("phi0", [0.95, 0.98])
    @pytest.mark.parametrize("q", [0.08, 0.1])
    @pytest.mark.parametrize("ndata", [80, 120])
    def test_sparse_bls_phase_wrapping(self, freq, phi0, q, ndata):
        """Verify sparse_bls_cpu correctly finds transits that wrap phase 0/1."""
        t, y, dy = data(snr=50, q=q, phi0=phi0, freq=freq,
                        baseline=365., ndata=ndata)

        df = q / (10 * (max(t) - min(t)))
        freqs = np.linspace(freq - 5 * df, freq + 5 * df, 21)

        power, sols = sparse_bls_cpu(t, y, dy, freqs)

        best_idx = np.argmax(power)
        best_freq = freqs[best_idx]

        # Should find transit near the true frequency
        T = max(t) - min(t)
        assert np.abs(best_freq - freq) < q / T, \
            f"Expected freq~{freq}, got {best_freq}"

        # Power should be significant (SNR=50 should give high power)
        assert power[best_idx] > 0.5, \
            f"Power too low: {power[best_idx]}"

        # Verify against brute-force at the best frequency
        bf_power, _, _ = self._brute_force_bls(t, y, dy, best_freq)
        assert np.abs(power[best_idx] - bf_power) < 1e-5, \
            f"sparse={power[best_idx]:.8f}, brute={bf_power:.8f}"

    @pytest.mark.parametrize("freq", [1.0, 2.0])
    @pytest.mark.parametrize("ndata", [50, 100])
    def test_sparse_bls_optimality(self, freq, ndata):
        """Verify sparse_bls_cpu finds the global max (no pairs missed)."""
        t, y, dy = data(snr=30, q=0.08, phi0=0.5, freq=freq,
                        baseline=365., ndata=ndata)

        freqs = np.array([freq], dtype=np.float32)
        power, sols = sparse_bls_cpu(t, y, dy, freqs)
        bf_power, _, _ = self._brute_force_bls(t, y, dy, freq)

        assert np.abs(power[0] - bf_power) < 1e-5, \
            f"sparse={power[0]:.8f} != brute={bf_power:.8f}"

    # ---- Sparse BLS q-bound (qmin/qmax) tests ----

    @pytest.mark.parametrize("freq", [1.0, 2.0])
    @pytest.mark.parametrize("qbounds", [(0.02, 0.08), (0.05, 0.15)])
    @pytest.mark.parametrize("ndata", [50, 100])
    def test_sparse_bls_cpu_q_bounds_vs_brute(self, freq, qbounds, ndata):
        """sparse_bls_cpu with q bounds matches the bounded brute force."""
        qmin, qmax = qbounds
        t, y, dy = data(snr=30, q=0.1, phi0=0.4, freq=freq,
                        baseline=365., ndata=ndata)

        freqs = np.array([freq], dtype=np.float32)
        power, sols = sparse_bls_cpu(t, y, dy, freqs, qmin=qmin, qmax=qmax)
        bf_power, _, _ = self._brute_force_bls(t, y, dy, freq,
                                               qmin=qmin, qmax=qmax)

        assert np.abs(power[0] - bf_power) < 1e-5, \
            f"sparse={power[0]:.8f}, brute={bf_power:.8f}"
        q_found, _ = sols[0]
        if power[0] > 0:
            assert qmin <= q_found <= qmax

    def test_sparse_bls_cpu_q_bounds_change_solution(self):
        """A qmax below the injected duration must exclude the
        unbounded optimum (bounds demonstrably constrain the search)."""
        t, y, dy = data(snr=50, q=0.2, phi0=0.3, freq=1.0,
                        baseline=365., ndata=100)
        freqs = np.array([1.0])

        power_free, sols_free = sparse_bls_cpu(t, y, dy, freqs)
        power_bound, sols_bound = sparse_bls_cpu(t, y, dy, freqs,
                                                 qmin=0.01, qmax=0.05)

        assert sols_free[0][0] > 0.05  # unbounded finds the q~0.2 dip
        assert power_bound[0] < power_free[0]
        if power_bound[0] > 0:
            assert 0.01 <= sols_bound[0][0] <= 0.05

    def test_sparse_bls_cpu_q_bounds_per_frequency(self):
        """Per-frequency qmin/qmax arrays bound each frequency
        independently."""
        t, y, dy = data(snr=30, q=0.1, phi0=0.4, freq=1.0,
                        baseline=365., ndata=80)
        freqs = np.array([0.8, 1.0, 1.25])
        qmins = np.array([0.01, 0.05, 0.02])
        qmaxes = np.array([0.05, 0.15, 0.3])

        power, sols = sparse_bls_cpu(t, y, dy, freqs,
                                     qmin=qmins, qmax=qmaxes)

        for i in range(len(freqs)):
            bf_power, _, _ = self._brute_force_bls(
                t, y, dy, freqs[i], qmin=qmins[i], qmax=qmaxes[i])
            assert np.abs(power[i] - bf_power) < 1e-5, \
                f"freq={freqs[i]}: sparse={power[i]:.8f}, " \
                f"brute={bf_power:.8f}"
            if power[i] > 0:
                assert qmins[i] <= sols[i][0] <= qmaxes[i]

    def test_sparse_bls_cpu_q_bounds_bad_length_raises(self):
        t, y, dy = data(ndata=50)
        freqs = np.array([0.9, 1.0, 1.1])
        with pytest.raises(ValueError, match="qmin"):
            sparse_bls_cpu(t, y, dy, freqs, qmin=np.array([0.01, 0.02]))
        with pytest.raises(ValueError, match="qmax"):
            sparse_bls_cpu(t, y, dy, freqs, qmax=np.array([0.1] * 5))

    def test_sparse_bls_q_bounds_keyword_only(self):
        """qmin/qmax were inserted mid-signature in v1.0: a pre-v1.0
        positional call like sparse_bls_cpu(t, y, dy, freqs, True)
        (ignore_negative_delta_sols) would silently become qmin=True
        -> qmin=1.0 > qmax and an all-zero periodogram. The bounds are
        keyword-only so legacy positional calls fail loudly instead."""
        t, y, dy = data(ndata=50)
        freqs = np.array([0.9, 1.0, 1.1])
        with pytest.raises(TypeError):
            sparse_bls_cpu(t, y, dy, freqs, True)
        with pytest.raises(TypeError):
            sparse_bls_gpu(t, y, dy, freqs, False, 128)

    def test_sparse_bls_inverted_q_bounds_raise(self):
        """qmin > qmax used to silently return an all-zero periodogram
        (every candidate rejected) — a pipeline reads that as 'no
        transit'. It must raise. Validation runs before any GPU work,
        so the GPU variant is CPU-testable too."""
        t, y, dy = data(ndata=50)
        freqs = np.array([0.9, 1.0, 1.1])
        for fn in (sparse_bls_cpu, sparse_bls_gpu):
            with pytest.raises(ValueError, match="qmin > qmax"):
                fn(t, y, dy, freqs, qmin=0.2, qmax=0.1)
            with pytest.raises(ValueError, match="finite"):
                fn(t, y, dy, freqs, qmin=np.nan)
            with pytest.raises(ValueError, match="qmax"):
                fn(t, y, dy, freqs, qmax=0.0)

    @pytest.mark.parametrize("use_simple", [False, True])
    def test_sparse_bls_gpu_q_bounds(self, use_simple):
        """GPU sparse BLS honors per-frequency q bounds (matches CPU)."""
        t, y, dy = data(snr=30, q=0.1, phi0=0.3, freq=1.0,
                        baseline=365., ndata=80)
        freqs = np.linspace(0.95, 1.05, 11)
        qmins = np.full(len(freqs), 0.03)
        qmaxes = np.full(len(freqs), 0.2)

        power_cpu, _ = sparse_bls_cpu(t, y, dy, freqs,
                                      qmin=qmins, qmax=qmaxes)
        power_gpu, sols_gpu = sparse_bls_gpu(t, y, dy, freqs,
                                             qmin=qmins, qmax=qmaxes,
                                             use_simple=use_simple)

        assert_allclose(power_cpu, power_gpu, rtol=1e-3, atol=1e-5)
        for (q_g, _), p in zip(sols_gpu, power_gpu):
            if p > 0:
                assert qmins[0] - 1e-6 <= q_g <= qmaxes[0] + 1e-6

    @pytest.mark.parametrize("freq", [1.0, 2.0])
    @pytest.mark.parametrize("q", [0.02, 0.1])
    @pytest.mark.parametrize("phi0", [0.0, 0.5])
    @pytest.mark.parametrize("ndata", [50, 100])
    def test_sparse_bls_gpu(self, freq, q, phi0, ndata):
        """Test GPU sparse BLS matches CPU and both match ground truth."""
        t, y, dy = data(snr=30, q=q, phi0=phi0, freq=freq,
                        baseline=365., ndata=ndata)

        df = q / (10 * (max(t) - min(t)))
        freqs = np.linspace(freq - 5 * df, freq + 5 * df, 11)

        power_cpu, sols_cpu = sparse_bls_cpu(t, y, dy, freqs)
        power_gpu, sols_gpu = sparse_bls_gpu(t, y, dy, freqs)

        # Powers should match closely across all frequencies
        assert_allclose(power_cpu, power_gpu, rtol=1e-3, atol=1e-5,
                       err_msg=f"Power mismatch for freq={freq}, q={q}, phi0={phi0}")

        # Best powers should be close (argmax may differ due to float precision)
        assert np.abs(np.max(power_cpu) - np.max(power_gpu)) < 1e-4, \
            f"Best power mismatch: cpu={np.max(power_cpu)}, gpu={np.max(power_gpu)}"

    @pytest.mark.parametrize("freq", [1.0])
    @pytest.mark.parametrize("phi0", [0.95])
    @pytest.mark.parametrize("q", [0.08])
    @pytest.mark.parametrize("ndata", [80])
    def test_sparse_bls_gpu_phase_wrapping(self, freq, phi0, q, ndata):
        """Test GPU sparse BLS with wrapped transits matches CPU."""
        t, y, dy = data(snr=50, q=q, phi0=phi0, freq=freq,
                        baseline=365., ndata=ndata)

        df = q / (10 * (max(t) - min(t)))
        freqs = np.linspace(freq - 5 * df, freq + 5 * df, 11)

        power_cpu, _ = sparse_bls_cpu(t, y, dy, freqs)
        power_gpu, _ = sparse_bls_gpu(t, y, dy, freqs)

        assert_allclose(power_cpu, power_gpu, rtol=1e-4, atol=1e-6)

        # Both should find significant power
        assert np.max(power_gpu) > 0.1

    @pytest.mark.parametrize("ndata", [50, 100])
    @pytest.mark.parametrize("use_sparse_override", [None, True])
    def test_eebls_transit_auto_select(self, ndata, use_sparse_override):
        """Test eebls_transit automatic selection with sparse BLS."""
        freq_true = 1.0
        q = 0.05
        phi0 = 0.3

        t, y, dy = data(snr=30, q=q, phi0=phi0, freq=freq_true,
                        baseline=365., ndata=ndata)

        freqs, powers, sols = eebls_transit(
            t, y, dy,
            fmin=freq_true * 0.99,
            fmax=freq_true * 1.01,
            use_sparse=use_sparse_override,
            sparse_threshold=150
        )

        assert len(freqs) > 0
        assert len(powers) == len(freqs)
        assert sols is not None
        assert len(sols) == len(freqs)

        best_freq = freqs[np.argmax(powers)]
        T = max(t) - min(t)
        # the peak-frequency uncertainty is ~q/T (one phase-smear
        # width); with only 50 points the peak can statistically land
        # a couple of widths off, so allow 2 units
        assert np.abs(best_freq - freq_true) < 2 * q / T

    @pytest.mark.parametrize("ndata", [50, 100])
    def test_eebls_transit_standard_returns_3(self, ndata):
        """Test eebls_transit always returns 3 values, even with use_fast."""
        freq_true = 1.0
        q = 0.05
        phi0 = 0.3

        t, y, dy = data(snr=30, q=q, phi0=phi0, freq=freq_true,
                        baseline=365., ndata=ndata)

        # use_fast=True should still return 3 values (sols=None)
        result = eebls_transit(
            t, y, dy,
            fmin=freq_true * 0.99,
            fmax=freq_true * 1.01,
            use_sparse=False,
            use_fast=True
        )
        assert len(result) == 3
        freqs, powers, sols = result
        assert sols is None


class TestEeblsTransitSparseKwargs(object):
    """Regression tests: eebls_transit's sparse path must tolerate the
    documented pass-through kwargs (rho, samples_per_peak, dlogq, ...)
    instead of crashing with TypeError (sparse_bls_gpu has a closed
    signature), and must honor the Keplerian q constraints."""

    def _data(self, ndata=100):
        t, y, dy = data(snr=20, q=0.05, phi0=0.3, freq=1.0,
                        baseline=365., ndata=ndata)
        return t, y, dy

    def test_sparse_gpu_path_accepts_documented_kwargs(self):
        # Before the fix: TypeError('sparse_bls_gpu() got an unexpected
        # keyword argument "rho"') raised at call time, before any GPU
        # work. GPU-runtime errors (e.g. on CPU-only test machines) are
        # acceptable here -- we are only asserting the kwarg plumbing.
        t, y, dy = self._data()
        try:
            eebls_transit(t, y, dy, rho=1.5, samples_per_peak=2,
                          fmin=0.95, fmax=1.05, use_gpu=True)
        except TypeError as e:
            pytest.fail("sparse path crashed on documented kwarg: %s" % e)
        except Exception:
            pass  # GPU unavailable (stubbed) -- plumbing already verified

    def test_sparse_cpu_path_accepts_documented_kwargs(self):
        t, y, dy = self._data()
        freqs, powers, sols = eebls_transit(t, y, dy, rho=1.0,
                                            fmin=0.95, fmax=1.05,
                                            use_gpu=False)
        assert len(freqs) == len(powers)
        assert np.all(np.isfinite(powers))

    def test_sparse_path_honors_q_constraints(self):
        # The sparse path applies the same per-frequency Keplerian
        # qmin_fac/qmax_fac bounds as the standard path (no more
        # discontinuity warning across the sparse_threshold boundary).
        import warnings as _warnings
        t, y, dy = self._data()
        qmin_fac, qmax_fac = 0.3, 1.5
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            freqs, powers, sols = eebls_transit(
                t, y, dy, qmin_fac=qmin_fac, qmax_fac=qmax_fac,
                fmin=0.95, fmax=1.05, use_gpu=False)

        qvals = q_transit(freqs)
        for (q_found, _), p, qv in zip(sols, powers, qvals):
            if p > 0:
                assert qmin_fac * qv - 1e-6 <= q_found
                assert q_found <= qmax_fac * qv + 1e-6

    def test_standard_path_unaffected(self):
        # No warning and no kwargs filtering on the standard path
        import warnings as _warnings
        t, y, dy = self._data(ndata=100)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            try:
                eebls_transit(t, y, dy, fmin=0.95, fmax=1.05,
                              use_sparse=False)
            except UserWarning:
                pytest.fail("standard path should not warn")
            except Exception:
                pass  # GPU unavailable (stubbed)


class TestEeblsGpuFastNoverlap(object):
    """eebls_gpu_fast(noverlap=k) must equal the elementwise max over
    k dphi-shifted single passes (the manual re-run procedure the
    docstring used to recommend; previously noverlap was silently
    ignored on the fast path)."""

    def _data(self):
        return data(snr=30, q=0.05, phi0=0.317, freq=1.0,
                    baseline=365., ndata=300)

    def test_noverlap_validation(self):
        # Runs CPU-side: validation precedes any GPU work.
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 20)
        for bad in (0, -1, 1.5, "2"):
            with pytest.raises(ValueError, match="noverlap"):
                eebls_gpu_fast(t, y, dy, freqs, noverlap=bad)
        with pytest.raises(ValueError, match="noverlap"):
            eebls_gpu_fast_optimized(t, y, dy, freqs, noverlap=0)

    @pytest.mark.parametrize("use_optimized", [False, True])
    def test_noverlap_matches_manual_dphi_runs(self, use_optimized):
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 200)
        fn = eebls_gpu_fast_optimized if use_optimized else eebls_gpu_fast
        k = 3
        kw = dict(qmin=0.01, qmax=0.1, dlogq=0.2)

        power_k = fn(t, y, dy, freqs, noverlap=k, **kw)
        manual = np.max([fn(t, y, dy, freqs, noverlap=1,
                            dphi=float(i) / k, **kw)
                         for i in range(k)], axis=0)

        assert_allclose(power_k, manual, rtol=1e-4, atol=1e-6)

    def test_noverlap_never_decreases_power(self):
        # Pass 0 of the noverlap=3 run is exactly the noverlap=1 run,
        # so the elementwise max can only gain power.
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 200)
        kw = dict(qmin=0.01, qmax=0.1)

        p1 = eebls_gpu_fast(t, y, dy, freqs, noverlap=1, **kw)
        p3 = eebls_gpu_fast(t, y, dy, freqs, noverlap=3, **kw)

        assert np.all(p3 >= p1 - 1e-6)


class TestPowerConventions(object):
    """convert_bls_power + the convention= kwarg (#17): conversions
    validated against astropy.timeseries.BoxLeastSquares definitions
    on shared (period, duration, phase) solutions."""

    def _data(self, ndata=120, freq=1.0, q=0.06, phi0=0.42, seed=7):
        rand = np.random.RandomState(seed)
        t = np.sort(365.0 * rand.rand(ndata))
        t -= np.floor(t.min())
        sigma = 0.01
        y = np.zeros(ndata)
        phi = (t * freq) % 1.0
        y[(phi > phi0) & (phi < phi0 + q)] -= 12 * sigma / np.sqrt(
            ndata * q)
        y += sigma * rand.randn(ndata)
        dy = sigma * np.ones(ndata)
        return t, y, dy

    @staticmethod
    def _chi2_0(y, dy):
        w = np.power(np.asarray(dy, dtype=np.float64), -2)
        ybar = np.dot(w, y) / np.sum(w)
        return float(np.dot(w, (np.asarray(y) - ybar) ** 2))

    def test_invalid_convention_raises(self):
        from ..bls import convert_bls_power
        t, y, dy = self._data()
        with pytest.raises(ValueError, match="convention"):
            convert_bls_power(0.5, y, dy, convention='banana')
        with pytest.raises(ValueError, match="convention"):
            sparse_bls_cpu(t, y, dy, np.array([1.0]),
                           convention='banana')
        with pytest.raises(ValueError, match="convention"):
            eebls_gpu_fast(t, y, dy, np.array([1.0]),
                           convention='banana')
        # eebls_gpu_custom used to validate only at the return
        # statement, i.e. AFTER the full GPU grid search. The
        # ValueError (not a GPU error) must come before any GPU work.
        with pytest.raises(ValueError, match="convention"):
            eebls_gpu_custom(t, y, dy, np.array([1.0]),
                             q_values=np.array([0.05, 0.1]),
                             phi_values=np.linspace(0, 1, 10),
                             convention='banana')

    def test_chi2ratio_is_identity(self):
        from ..bls import convert_bls_power
        t, y, dy = self._data()
        p = np.array([0.0, 0.1, 0.5])
        assert convert_bls_power(p, y, dy) is p

    def test_conversion_definitions(self):
        from ..bls import convert_bls_power
        t, y, dy = self._data()
        chi2_0 = self._chi2_0(y, dy)
        p = np.array([0.0, 0.05, 0.3])
        assert_allclose(convert_bls_power(p, y, dy, 'snr'),
                        np.sqrt(chi2_0 * p))
        assert_allclose(convert_bls_power(p, y, dy, 'loglik'),
                        0.5 * chi2_0 * p)

    def _astropy_results(self, t, y, dy, objective):
        astropy_ts = pytest.importorskip('astropy.timeseries')
        model = astropy_ts.BoxLeastSquares(t, y, dy=dy)
        periods = np.linspace(0.95, 1.05, 9)
        durations = np.array([0.04, 0.06, 0.08])
        return model.power(periods, durations, method='slow',
                           oversample=10, objective=objective)

    def _our_power_at(self, t, y, dy, period, duration, transit_time):
        # Evaluate the native power at astropy's exact solution.
        # astropy's transit_time is mid-transit; single_bls phases are
        # relative to floor(min(t)) and phi0 is the transit start.
        freq = 1.0 / period
        q = duration / period
        epoch = np.floor(t.min())
        phi0 = ((transit_time - 0.5 * duration - epoch) * freq) % 1.0
        return single_bls(t, y, dy, freq, q, phi0), q

    def test_snr_matches_astropy(self):
        from ..bls import convert_bls_power
        t, y, dy = self._data()
        res = self._astropy_results(t, y, dy, 'snr')
        for i in range(len(res.period)):
            p_native, _ = self._our_power_at(
                t, y, dy, res.period[i], res.duration[i],
                res.transit_time[i])
            snr = convert_bls_power(p_native, y, dy, 'snr')
            assert np.abs(snr - res.power[i]) <= 2e-3 * abs(res.power[i]), \
                f"period={res.period[i]}: ours={snr}, astropy={res.power[i]}"

    def test_loglik_matches_astropy_up_to_reference(self):
        # astropy's likelihood objective uses the out-of-transit level
        # as the null reference, so its power equals our 'loglik'
        # (constant-weighted-mean reference) divided by (1 - r), with
        # r the in-transit fraction of total statistical weight.
        from ..bls import convert_bls_power
        t, y, dy = self._data()
        res = self._astropy_results(t, y, dy, 'likelihood')
        w = np.power(dy, -2.0)
        for i in range(len(res.period)):
            p_native, q = self._our_power_at(
                t, y, dy, res.period[i], res.duration[i],
                res.transit_time[i])
            loglik = convert_bls_power(p_native, y, dy, 'loglik')

            period, dur = res.period[i], res.duration[i]
            hp = 0.5 * period
            t0 = (res.transit_time[i] - t.min()) % period
            m_in = np.abs((t - t.min() - t0 + hp) % period - hp) \
                < 0.5 * dur
            r = np.sum(w[m_in]) / np.sum(w)

            expected = res.power[i] * (1.0 - r)
            assert np.abs(loglik - expected) <= 2e-3 * abs(expected), \
                f"period={period}: ours={loglik}, astropy(1-r)={expected}"

    def test_sparse_cpu_convention_consistency(self):
        from ..bls import convert_bls_power
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 11)
        p_native, sols = sparse_bls_cpu(t, y, dy, freqs)
        p_snr, sols_snr = sparse_bls_cpu(t, y, dy, freqs,
                                         convention='snr')
        assert_allclose(p_snr, convert_bls_power(p_native, y, dy, 'snr'),
                        rtol=1e-6)
        # solutions are convention-independent
        assert sols == sols_snr

    def test_eebls_transit_sparse_path_convention(self):
        t, y, dy = self._data()
        freqs, p_native, _ = eebls_transit(t, y, dy, fmin=0.95,
                                           fmax=1.05, use_gpu=False)
        freqs2, p_loglik, _ = eebls_transit(t, y, dy, fmin=0.95,
                                            fmax=1.05, use_gpu=False,
                                            convention='loglik')
        chi2_0 = self._chi2_0(y, dy)
        assert_allclose(p_loglik, 0.5 * chi2_0 * p_native, rtol=1e-6)

    def test_gpu_entry_points_convention(self):
        # GPU smoke test (pod): the kwarg flows through the standard
        # and fast call chains and converts the returned host array.
        from ..bls import convert_bls_power
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 50)

        p0, sols = eebls_gpu(t, y, dy, freqs, qmin=0.01, qmax=0.2)
        p_snr, _ = eebls_gpu(t, y, dy, freqs, qmin=0.01, qmax=0.2,
                             convention='snr')
        assert_allclose(p_snr, convert_bls_power(p0, y, dy, 'snr'),
                        rtol=1e-4, atol=1e-6)

        f0 = eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.2)
        f_log = eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.2,
                               convention='loglik')
        assert_allclose(f_log, convert_bls_power(f0, y, dy, 'loglik'),
                        rtol=1e-4, atol=1e-6)


class TestCompileBlsValidation(object):
    """compile_bls should fail loudly on bad block sizes and on filter
    results that would otherwise surface as confusing KeyErrors."""

    def test_bad_block_size_raises(self):
        from ..bls import _validate_block_size
        for bad in (0, 16, 31, 48, 100, -64, 2.5, "256"):
            with pytest.raises(ValueError):
                _validate_block_size(bad)
        for good in (32, 64, 128, 256, 512, 1024):
            _validate_block_size(good)  # should not raise

    def test_compile_bls_rejects_bad_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            compile_bls(block_size=48)

    def test_compile_bls_empty_filter_raises_value_error(self):
        # full_bls_no_sol only exists in the standard kernel; requesting
        # it alone with use_optimized=True used to produce an empty
        # function dict and downstream KeyErrors.
        with pytest.raises(ValueError, match="no loadable functions"):
            compile_bls(function_names=['full_bls_no_sol'],
                        use_optimized=True)
        with pytest.raises(ValueError, match="no loadable functions"):
            compile_bls(function_names=['full_bls_no_sol_optimized'],
                        use_optimized=False)


class TestEpochHandling(object):
    """Times must be epoch-subtracted before any float32 cast.

    With raw BJD-scale timestamps (~2.45e6 days), float32 phase folding
    loses essentially all phase information: float32 carries ~7
    significant digits, so the fractional part of ``t * freq`` is
    dominated by rounding error. All BLS paths subtract ``min(t)`` (in
    float64) before casting, and phases are reported relative to it.
    """

    # Integer offset: epoch = floor(min(t)) makes the shifted and
    # unshifted time arrays exactly identical, so powers must match to
    # float rounding. (A fractional offset would rotate all phases by
    # frac * freq mod 1 -- powers are invariant in exact math but bin
    # alignments shift.)
    bjd_offset = 2455197.0

    def _signal(self, ndata=120, baseline=365., freq=0.3, q=0.05,
                phi0=0.3, snr=50., sigma=0.01, seed=42):
        rand = np.random.RandomState(seed)
        t = baseline * np.sort(rand.rand(ndata))
        t -= t.min()  # absolute and epoch-relative phases coincide
        delta = snr * sigma / np.sqrt(ndata * q * (1 - q))
        phi = (t * freq) % 1.0
        y = -delta * ((phi > phi0) & (phi < phi0 + q)).astype(float)
        y += sigma * rand.randn(ndata)
        dy = sigma * np.ones(ndata)
        return t, y, dy, freq, q, phi0

    def test_single_bls_bjd_invariance(self):
        t, y, dy, freq, q, phi0 = self._signal()
        p_rel = single_bls(t, y, dy, freq, q, phi0)
        p_raw = single_bls(t + self.bjd_offset, y, dy, freq, q, phi0)
        assert p_rel > 0.5  # signal actually detected
        assert abs(p_raw - p_rel) < 1e-3 * p_rel

    def test_sparse_bls_cpu_bjd_invariance(self):
        t, y, dy, freq, q, phi0 = self._signal(ndata=60)
        freqs = np.array([0.9 * freq, freq, 1.1 * freq])
        p_rel, _ = sparse_bls_cpu(t, y, dy, freqs)
        p_raw, _ = sparse_bls_cpu(t + self.bjd_offset, y, dy, freqs)
        assert p_rel[1] > 0.5
        assert_allclose(p_raw, p_rel, rtol=1e-3, atol=1e-4)

    def test_bls_memory_epoch_subtraction(self):
        # Runs on GPU only (BLSMemory allocates pinned arrays); the
        # conftest stub converts it to a skip on CPU-only machines.
        from ..bls import BLSMemory
        t, y, dy, freq, q, phi0 = self._signal()
        freqs = np.linspace(0.2, 0.4, 10)
        mem = BLSMemory.fromdata(t + self.bjd_offset, y, dy,
                                 qmin=1e-2, qmax=0.5, freqs=freqs,
                                 transfer=False)
        assert_allclose(mem.t[:len(t)], t.astype(np.float32), atol=1e-3)
        assert mem.epoch == pytest.approx(
            np.floor(self.bjd_offset + t.min()))

    def test_bls_batch_memory_epoch_subtraction(self):
        # Runs on GPU only (pinned host arrays); skipped on CPU.
        from ..memory.bls_memory import BLSBatchMemory
        t, y, dy, freq, q, phi0 = self._signal()
        mem = BLSBatchMemory(len(t), 1, 8)
        mem.set_lightcurve(0, t + self.bjd_offset, y, dy)
        assert_allclose(mem.t[:len(t)], t.astype(np.float32), atol=1e-3)
        assert mem.epochs[0] == pytest.approx(
            np.floor(self.bjd_offset + t.min()))

    def test_eebls_gpu_bjd_invariance(self):
        # Full GPU path; skipped on CPU-only machines.
        t, y, dy, freq, q, phi0 = self._signal()
        freqs = np.linspace(0.95 * freq, 1.05 * freq, 50)
        p_rel, _ = eebls_gpu(t, y, dy, freqs, qmin=0.01, qmax=0.1)
        p_raw, _ = eebls_gpu(t + self.bjd_offset, y, dy, freqs,
                             qmin=0.01, qmax=0.1)
        # the binned estimator peaks well below the exact box power
        # (~0.48 vs ~0.95 here); 0.3 still clears the ~0.15 noise floor
        assert max(p_rel) > 0.3
        assert_allclose(p_raw, p_rel, rtol=1e-3, atol=1e-3)


class TestReductionMaxValidation(object):
    """_reduction_max used to 'validate' block_size with an assert that
    is always true under Python 3 division; a mismatched block_size
    silently corrupts the tree reduction on the GPU."""

    class _FakePtr(object):
        ptr = 0

    class _FakeKernel(object):
        def __init__(self):
            self.calls = []

        def prepared_async_call(self, *args):
            self.calls.append(args)

    def _call(self, block_size):
        from ..bls import _reduction_max
        kern = self._FakeKernel()
        _reduction_max(kern, self._FakePtr(), self._FakePtr(),
                       4, 64, None, self._FakePtr(), self._FakePtr(),
                       0, block_size)
        return kern

    def test_non_power_of_two_block_size_raises(self):
        for bad in (48, 100, 0, -64, 2.5, "256"):
            with pytest.raises(ValueError):
                self._call(bad)

    def test_valid_block_size_launches(self):
        kern = self._call(64)
        assert len(kern.calls) >= 1


class TestSparseBlsCpuVectorized:
    """sparse_bls_cpu used to be a pure-Python O(N^3) loop (each pair
    recomputed its slice sum) — minutes per frequency at the
    ndata=500 sparse threshold. The vectorized scan must stay fast."""

    def test_moderate_ndata_runs_in_seconds(self):
        import time
        t, y, dy = data(snr=20, q=0.05, phi0=0.4, freq=1.0,
                        baseline=365., ndata=250)
        start = time.time()
        power, _ = sparse_bls_cpu(t, y, dy,
                                  np.array([0.9, 1.0, 1.1]))
        elapsed = time.time() - start
        assert elapsed < 10.0  # pre-vectorization: minutes
        assert int(np.argmax(power)) == 1


class TestPinnedBufferStreamParity(object):
    """With page-locked host result buffers, device->host copies on a
    user stream are genuinely asynchronous. BLSMemory.transfer_data_to_cpu
    used to normalize (bls /= yy) right after enqueueing get_async, racing
    the DMA — the returned periodogram could be unnormalized or torn.
    Results on a user stream must match the default-stream results."""

    @pytest.mark.parametrize("use_optimized", [False, True])
    def test_fast_path_stream_matches_default(self, use_optimized):
        import pycuda.driver as cuda
        from ..core import ensure_context

        t, y, dy = data(snr=30, q=0.05, phi0=0.317, freq=1.0,
                        baseline=365., ndata=300)
        freqs = np.linspace(0.95, 1.05, 200)
        fn = eebls_gpu_fast_optimized if use_optimized else eebls_gpu_fast

        p_default = fn(t, y, dy, freqs)
        ensure_context()
        p_stream = fn(t, y, dy, freqs, stream=cuda.Stream())

        # rtol only needs to catch the failure modes (unnormalized:
        # off by the factor 1/yy; torn: garbage), not atomic-order
        # jitter between runs.
        assert_allclose(p_stream, p_default, rtol=1e-3)
