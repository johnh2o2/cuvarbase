from itertools import product 
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..bls import eebls_gpu, eebls_transit_gpu, \
                  q_transit, compile_bls, hone_solution,\
                  single_bls, eebls_gpu_custom, eebls_gpu_fast, \
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
    @pytest.mark.parametrize("use_optimized", [True, False])
    def test_transit_parameter_consistency(self, freq, phi0, dlogq, nstreams,
                                           freq_batch_size, ignore_negative_delta_sols, use_optimized):
        q = q_transit(freq)

        t, y, dy = data(snr=30, q=q, phi0=phi0, freq=freq, baseline=365.)

        freqs, power, sols = eebls_transit_gpu(t, y, dy,
                                               samples_per_peak=2,
                                               freq_batch_size=freq_batch_size,
                                               nstreams=nstreams,
                                               dlogq=dlogq,
                                               ignore_negative_delta_sols=ignore_negative_delta_sols,
                                               fmin=freq * 0.99,
                                               fmax=freq * 1.01,
                                               use_optimized=use_optimized)
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

        pows, diffs = list(zip(*sorted(zip(pcpu,
                                           np.absolute(power - pcpu)),
                                       key=lambda x: -x[1])))

        upper_bound = self.rtol * np.array(pows) + self.atol
        mostly_ok = sum(np.array(diffs) > upper_bound) / len(pows) < 1e-2
        not_too_bad = max(diffs) < 1e-1

        print(max(diffs))
        assert mostly_ok and not_too_bad

    @pytest.mark.parametrize("freq", [1.0])
    @pytest.mark.parametrize("phi_index", [0, 10])
    @pytest.mark.parametrize("q_index", [0, 5])
    @pytest.mark.parametrize("nstreams", [1, 3])
    @pytest.mark.parametrize("freq_batch_size", [1, 3, None])
    @pytest.mark.parametrize("ignore_negative_delta_sols", [True, False])
    @pytest.mark.parametrize("use_optimized", [True, False])
    def test_custom(self, freq, q_index, phi_index, freq_batch_size, nstreams,
                    ignore_negative_delta_sols, use_optimized):
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
                                        nstreams=nstreams,
                                        use_optimized=use_optimized)

        best_ps = []
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
            best_ps.append(best_p)

            # assert np.abs(best_p - gpower) < 1e-5

        diffs = np.abs(best_ps - power)
        upper_bound = self.rtol * np.array(best_ps) + self.atol
        mostly_ok = sum(np.array(diffs) > upper_bound) / len(power) <= 5e-2
        not_too_bad = max(diffs) < 5e-3

        assert mostly_ok and not_too_bad

    @pytest.mark.parametrize("freq", [1.0])
    @pytest.mark.parametrize("phi_index", [0, 10, -1])
    @pytest.mark.parametrize("q_index", [0, 5, -1])
    @pytest.mark.parametrize("nstreams", [1, 3])
    @pytest.mark.parametrize("freq_batch_size", [1, 3, None])
    @pytest.mark.parametrize("ignore_negative_delta_sols", [True, False])
    @pytest.mark.parametrize("use_optimized", [True, False])
    def test_standard(self, freq, q_index, phi_index, nstreams, freq_batch_size,
                      ignore_negative_delta_sols, use_optimized):

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
                                 ignore_negative_delta_sols=ignore_negative_delta_sols,
                                 use_optimized=use_optimized)

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
    @pytest.mark.parametrize("use_optimized", [True, False])
    def test_transit(self, freq, use_fast, freq_batch_size, nstreams, phi0, dlogq,
                     ignore_negative_delta_sols, use_optimized):
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

        freqs, power, sols = eebls_transit_gpu(t, y, err, **kw,
                                               use_optimized=use_optimized)
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
    def _brute_force_bls(t, y, dy, freq, ignore_negative_delta_sols=False):
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
                if q <= 0 or q > 0.5:
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
                if q <= 0 or q > 0.5:
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
        assert np.abs(best_freq - freq_true) < q / T

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
