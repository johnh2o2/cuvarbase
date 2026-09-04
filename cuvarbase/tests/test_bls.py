from itertools import product
import warnings

import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..bls import eebls_gpu, eebls_transit_gpu, \
                  q_transit, compile_bls, hone_solution,\
                  single_bls, eebls_gpu_custom, eebls_gpu_fast, \
                  eebls_gpu_fast_optimized, \
                  sparse_bls_cpu, sparse_bls_gpu, eebls_transit, \
                  count_tot_nbins, _bls_batch_table, _max_nbins_tot, \
                  _cap_freq_batch_size, _q_bounds_to_nbins, \
                  _MAX_FOLD_THREADS
from ..bls_frequencies import keplerian_freq_grid


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
         q=0.01, phi0=None, baseline=1., negative_delta=False,
         t0=4.5):

    rand = np.random.RandomState(seed)

    if phi0 is None:
        phi0 = rand.rand()

    delta = snr * sigma / np.sqrt(ndata * q * (1 - q))

    if negative_delta:
        delta *= -1

    model = transit_model(phi0, q, delta)

    # Non-zero T0 so every test exercises a non-trivial epoch
    # (floor(min(t)) > 0): phases reported by the BLS functions are in
    # the original input timescale, and the injected transit is at
    # original-timescale phase phi0 (the shift is applied BEFORE the
    # model is evaluated).
    t = baseline * np.sort(rand.rand(ndata)) + t0
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
            # Deterministic single_bls value at the injected solution
            # (pure-CPU float32 arithmetic; changes only if data() or
            # single_bls numerics change -- e.g. this was
            # 0.8902446483898836 before data() gained the t0=4.5 shift,
            # which rotates the fold and re-draws which points host the
            # injected dip).
            {'bls0': 0.9223771210115413, 'bls_ignore': 0}
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

        # single_bls now takes phi0 in the ORIGINAL input timescale (it
        # epoch-subtracts internally), so the injected phase is passed
        # through unchanged.
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

    # use_optimized=True swaps in the bls_optimized.cu module, whose
    # binning/store kernels are byte-shared with bls.cu via
    # bls_common.cuh -- only reduction_max differs (warp-shuffle finish
    # vs full tree). One focused equivalence test per entry point
    # exercises that reduction + store path; cross-multiplying
    # use_optimized into every test_standard/test_custom parametrization
    # would double the suite while varying nothing else in the kernel.
    def test_standard_use_optimized_matches(self):
        q = 0.05
        t, y, dy = data(snr=10, q=q, phi0=0.317, freq=1.0, baseline=365.)
        freqs = np.linspace(0.95, 1.05, 300)

        kw = dict(qmin=0.1 * q, qmax=2.0 * q, nstreams=1,
                  noverlap=2, dlogq=0.5)
        p_std, sols_std = eebls_gpu(t, y, dy, freqs, **kw)
        p_opt, sols_opt = eebls_gpu(t, y, dy, freqs, use_optimized=True,
                                    **kw)

        # identical binning kernels: powers agree to float32
        # atomic-ordering noise
        assert_allclose(p_opt, p_std, rtol=1e-4, atol=1e-6)

        # solutions may legitimately differ where two boxes tie in
        # power (the two reductions break ties differently), so compare
        # the powers of the solutions rather than the solutions
        for f, s_std, s_opt in zip(freqs, sols_std, sols_opt):
            if s_std != s_opt:
                b_std = single_bls(t, y, dy, f, *s_std)
                b_opt = single_bls(t, y, dy, f, *s_opt)
                # ties: same binned power; exact powers can differ by
                # one point's membership at most (~power / n_in_box)
                assert abs(b_std - b_opt) < 0.15 * max(b_std, b_opt) + 1e-5

    def test_custom_use_optimized_matches(self):
        q_values = np.logspace(-1.1, -0.8, num=10)
        phi_values = np.linspace(0, 1, int(np.ceil(2. / min(q_values))))
        t, y, dy = data(snr=10, q=q_values[5], phi0=phi_values[10],
                        freq=1.0, baseline=365., ndata=500)
        freqs = np.linspace(0.9999, 1.0001, 20)

        p_std, sols_std = eebls_gpu_custom(t, y, dy, freqs,
                                           q_values, phi_values)
        p_opt, sols_opt = eebls_gpu_custom(t, y, dy, freqs,
                                           q_values, phi_values,
                                           use_optimized=True)
        assert_allclose(p_opt, p_std, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("freq", [1.0])
    @pytest.mark.parametrize("dlogq", [0.5, -1.0])
    @pytest.mark.parametrize("freq_batch_size", [1, 10, None])
    @pytest.mark.parametrize("phi0", [0.0])
    # one axis for the three kernel paths: a use_fast x use_optimized
    # cross-product would add combinations (fast+optimized) that just
    # re-run the fast branch
    @pytest.mark.parametrize("mode", ["standard", "fast", "optimized"])
    @pytest.mark.parametrize("nstreams", [1, 4])
    @pytest.mark.parametrize("ignore_negative_delta_sols", [True, False])
    def test_transit(self, freq, mode, freq_batch_size, nstreams, phi0, dlogq,
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
                  use_fast=(mode == "fast"),
                  use_optimized=(mode == "optimized"))

        if mode in ("fast", "optimized"):
            freqs, power, no_sols = eebls_transit_gpu(t, y, err, **kw)
            # fast/optimized kernels do not track solutions but the
            # return is a uniform 3-tuple
            assert no_sols is None

            kw['use_fast'] = False
            kw['use_optimized'] = False
            freqs, power_slow, sols = eebls_transit_gpu(t, y, err, **kw)
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


class TestHoneSolution(object):
    """hone_solution refines an initial (f, q, phi) via successive
    eebls_gpu_custom grids. This is the regression coverage for the
    original-timescale phi convention through the whole custom chain:
    trial phi values are passed in the original input timescale and the
    kernel re-references them to the subtracted epoch. With the fixture
    epoch (floor(min(t)) = 5) and freq = 0.7 the phase rotation
    (epoch * freq) % 1 = 0.5 is maximal -- a convention slip anywhere
    in the chain puts every trial box half a cycle off the transit."""

    def test_hone_refines_and_matches_single_bls(self):
        freq, q, phi0 = 0.7, 0.05, 0.3
        t, y, dy = data(snr=50, q=q, phi0=phi0, freq=freq,
                        baseline=365.)

        q0 = 1.3 * q
        phi_start = phi0 + 0.03
        p_start = single_bls(t, y, dy, freq, q0, phi_start)

        f, pn, niter, (qs, phs) = hone_solution(
            t, y, dy, freq, 1e-6, q0, 0.3, phi_start,
            stop=1e-4, max_iter=10)

        # refinement must improve on the deliberately misaligned start
        assert pn > p_start

        # the reported (f, q, phi) must reproduce the reported power
        # through single_bls: custom-kernel boxes are exact (unbinned)
        # box memberships, so agreement is at the float32-accumulation
        # level. If phs were epoch-relative instead of original-scale,
        # single_bls would evaluate a box 0.5 cycles from the transit
        # and disagree at the 0.1-1 level.
        p_check = single_bls(t, y, dy, f, qs, phs)
        assert abs(pn - p_check) < 1e-3 * pn + 1e-4

        # the refined box overlaps the injected transit in the ORIGINAL
        # timescale (circular distance between box centers below q)
        c_found = (phs + 0.5 * qs) % 1.0
        c_true = (phi0 + 0.5 * q) % 1.0
        dist = abs(c_found - c_true)
        dist = min(dist, 1.0 - dist)
        assert dist < q

        # frequency recovered to within a few phase-smear widths
        assert abs(f - freq) * (np.max(t) - np.min(t)) / q < 3


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


class TestFusedNoverlapKernel(object):
    """The fused-noverlap kernel (full_bls_no_sol_fused /
    full_bls_batch_fused) replaces the dphi-shifted multi-pass host
    loop for power-of-two noverlap with dphi == 0: it histograms once
    at noverlap-times finer phase resolution and derives every pass's
    box sums from runs of fine bins. Bin assignment is bit-identical
    to the multi-pass launches on this path; box sums differ only at
    float32 accumulation-order level (which the multi-pass path
    already doesn't pin down, shared atomics being order-free)."""

    def _data(self, **kw):
        kw.setdefault('snr', 30)
        kw.setdefault('q', 0.05)
        kw.setdefault('phi0', 0.317)
        kw.setdefault('freq', 1.0)
        kw.setdefault('baseline', 365.)
        kw.setdefault('ndata', 300)
        return data(**kw)

    @pytest.mark.parametrize("use_optimized,k",
                             list(product([False, True], [2, 4])))
    def test_fused_matches_manual_dphi_runs(self, use_optimized, k):
        # For power-of-two k the fused kernel must reproduce the manual
        # k-pass elementwise max (same gate as the multi-pass loop).
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 200)
        fn = eebls_gpu_fast_optimized if use_optimized else eebls_gpu_fast
        kw = dict(qmin=0.01, qmax=0.1, dlogq=0.2)

        power_k = fn(t, y, dy, freqs, noverlap=k, **kw)
        manual = np.max([fn(t, y, dy, freqs, noverlap=1,
                            dphi=float(i) / k, **kw)
                         for i in range(k)], axis=0)

        assert_allclose(power_k, manual, rtol=1e-4, atol=1e-6)

    def test_fused_nonzero_dphi_falls_back(self):
        # dphi != 0 keeps the multi-pass path: noverlap=2 with base
        # dphi=0.25 must equal the manual dphi = 0.25, 0.75 passes.
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 200)
        kw = dict(qmin=0.01, qmax=0.1)

        p = eebls_gpu_fast(t, y, dy, freqs, noverlap=2, dphi=0.25, **kw)
        manual = np.max([eebls_gpu_fast(t, y, dy, freqs, noverlap=1,
                                        dphi=0.25 + 0.5 * i, **kw)
                         for i in range(2)], axis=0)
        assert_allclose(p, manual, rtol=1e-4, atol=1e-6)

    def test_fused_bjd_scale(self):
        # BJD-scale timestamps (epoch ~2.455e6): the fused kernel must
        # (i) keep the recovered peak at the same frequency as the
        # epoch-subtracted input and (ii) match the manual dphi-shifted
        # passes bit-tightly ON the BJD input. (Full periodogram
        # correlation between BJD and non-BJD inputs is NOT gated at
        # 0.999 here: the phase origin moves by epoch*f mod 1, so
        # bin-edge quantization decorrelates off-peak power on the
        # multi-pass path too -- measured corr 0.95 for the pre-fusion
        # noverlap=3 loop on this exact dataset.)
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 500)
        kw = dict(qmin=0.01, qmax=0.1)
        t_bjd = t + 2455197.5

        p0 = eebls_gpu_fast(t, y, dy, freqs, noverlap=2, **kw)
        p1 = eebls_gpu_fast(t_bjd, y, dy, freqs, noverlap=2, **kw)
        assert int(np.argmax(p0)) == int(np.argmax(p1))

        manual = np.max([eebls_gpu_fast(t_bjd, y, dy, freqs, noverlap=1,
                                        dphi=0.5 * i, **kw)
                         for i in range(2)], axis=0)
        assert_allclose(p1, manual, rtol=1e-4, atol=1e-6)

    def test_batch_fused_matches_manual_passes(self):
        from ..bls import eebls_gpu_batch

        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 200)
        kw = dict(qmin=0.01, qmax=0.1)

        p2 = eebls_gpu_batch([(t, y, dy)], freqs, noverlap=2, **kw)[0]
        manual = np.max([eebls_gpu_batch([(t, y, dy)], freqs,
                                         noverlap=1,
                                         dphi=0.5 * i, **kw)[0]
                         for i in range(2)], axis=0)
        assert_allclose(p2, manual, rtol=1e-4, atol=1e-6)


class TestBatchMemoryReuse(object):
    """eebls_gpu_batch(memory=...) reuses one BLSBatchMemory across
    calls and chunks (per-call pinned/device allocation costs several
    ms at survey nfreq); results must match the allocate-per-call
    path, including across chunked processing and back-to-back calls
    with different data."""

    @staticmethod
    def _lcs(seeds, ndatas, baseline=365.0):
        out = []
        for seed, nd in zip(seeds, ndatas):
            rand = np.random.RandomState(seed)
            t = np.sort(baseline * rand.rand(nd)) + 4.5
            phase = (t * 0.5) % 1.0
            y = 12.0 - 0.05 * (phase < 0.04)
            y += 0.01 * rand.randn(nd)
            dy = 0.01 * np.ones(nd)
            out.append((t, y, dy))
        return out

    def test_memory_reuse_matches_fresh(self):
        from ..bls import eebls_gpu_batch
        from ..memory.bls_memory import BLSBatchMemory
        import pycuda.driver as cuda

        freqs = np.linspace(0.1, 1.0, 500)
        mem = BLSBatchMemory(400, 2, len(freqs), stream=cuda.Stream())

        for seeds in ((1, 2), (3, 4)):
            lcs = self._lcs(seeds, (200, 400))
            expect = eebls_gpu_batch(lcs, freqs)
            got = eebls_gpu_batch(lcs, freqs, memory=mem)
            for a, b in zip(expect, got):
                assert_allclose(a, b, rtol=1e-4, atol=1e-6)

    def test_chunked_matches_single_chunk(self):
        from ..bls import eebls_gpu_batch

        freqs = np.linspace(0.1, 1.0, 300)
        lcs = self._lcs((5, 6, 7, 8, 9), (150, 220, 300, 80, 260))

        p_one = eebls_gpu_batch(lcs, freqs)
        p_chunks = eebls_gpu_batch(lcs, freqs, max_batch_lcs=2)
        for a, b in zip(p_one, p_chunks):
            assert_allclose(a, b, rtol=1e-4, atol=1e-6)

    def test_too_small_memory_raises(self):
        from ..bls import eebls_gpu_batch
        from ..memory.bls_memory import BLSBatchMemory

        freqs = np.linspace(0.1, 1.0, 100)
        lcs = self._lcs((1,), (200,))
        mem = BLSBatchMemory(100, 1, len(freqs))  # max_ndata too small
        with pytest.raises(ValueError, match="too small"):
            eebls_gpu_batch(lcs, freqs, memory=mem)

    def test_freq_chunked_batch_matches(self):
        # freq-chunked launches (occupancy-aware path) must reproduce
        # the single-launch result; odd chunk size to catch
        # offset/stride mistakes.
        from ..bls import eebls_gpu_batch

        freqs = np.linspace(0.1, 1.0, 500)
        lcs = self._lcs((1, 2), (200, 400))
        p_full = eebls_gpu_batch(lcs, freqs)
        p_chunk = eebls_gpu_batch(lcs, freqs, freq_batch_size=97)
        for a, b in zip(p_full, p_chunk):
            assert_allclose(a, b, rtol=1e-4, atol=1e-6)

    def test_oversized_memory_reuse_matches(self):
        # memory allocated for MORE freqs/LCs/ndata than the call uses:
        # output row pitch is the allocation, results must still match.
        from ..bls import eebls_gpu_batch
        from ..memory.bls_memory import BLSBatchMemory
        import pycuda.driver as cuda

        freqs = np.linspace(0.1, 1.0, 400)
        lcs = self._lcs((3, 4), (150, 250))
        mem = BLSBatchMemory(600, 4, 900, stream=cuda.Stream())
        expect = eebls_gpu_batch(lcs, freqs)
        got = eebls_gpu_batch(lcs, freqs, memory=mem)
        for a, b in zip(expect, got):
            assert len(a) == len(b) == len(freqs)
            assert_allclose(a, b, rtol=1e-4, atol=1e-6)


class TestAllWeightBoxStability(object):
    """Regression tests for the nondeterministic bogus-peak bug behind
    PR #65's fabs(ybar) guard (attila's HATPI reproducer): bls_value's
    upper w bound `1.f - 1e-10f` is a float32 no-op (compiles to
    `w < 1.f`), so a trial box capturing ALL the statistical weight --
    routine for single-site data at ~1 cycle/day aliases with q up to
    0.5 -- divided atomic roundoff by atomic roundoff, producing
    run-to-run-varying spurious power. The bound is now a meaningful
    1e-4 complement across bls_common.cuh / bls_batch.cu /
    sparse_bls.cu / single_bls / sparse_bls_cpu."""

    @staticmethod
    def _single_site_data(n_nights=60, per_night=50, seed=21):
        rand = np.random.RandomState(seed)
        nights = np.arange(n_nights)
        t = np.concatenate([n + 0.25 * np.sort(rand.rand(per_night))
                            for n in nights])
        y = 12.0 + 0.01 * rand.randn(len(t))
        dy = 0.01 * np.ones_like(y)
        return t, y, dy

    def test_single_bls_all_weight_box_is_zero(self):
        # deterministic CPU check: at f = 1/day the whole lightcurve
        # sits at phases < 0.25, so a q=0.5 box holds all the weight --
        # power must be exactly 0, not roundoff/roundoff
        t, y, dy = self._single_site_data()
        assert single_bls(t, y, dy, 1.0, 0.5, 0.0) == 0
        # a normal box is unaffected by the new bound
        assert np.isfinite(single_bls(t, y, dy, 0.31, 0.05, 0.1))

    def test_fast_path_repeatable_on_single_site_data(self):
        # the GPU symptom: identical calls returned different
        # periodograms (deviations > 1e-2, transient bogus peaks near
        # 1 cycle/day). With the fixed bound the all-weight boxes score
        # exactly 0 in every pass, so repeats must agree to float32
        # atomic-reordering noise and no order-0.01+ power appears in
        # pure noise.
        t, y, dy = self._single_site_data()
        freqs = np.linspace(0.95, 1.05, 500)
        kw = dict(qmin=0.01, qmax=0.5, noverlap=1)
        p0 = eebls_gpu_fast(t, y, dy, freqs, **kw)
        for _ in range(5):
            p = eebls_gpu_fast(t, y, dy, freqs, **kw)
            assert np.max(np.abs(p - p0)) < 1e-4
        assert np.max(p0) < 0.05

    def test_shallow_transit_survives_w_bound(self):
        # guard against "fixing" the instability with an absolute
        # amplitude threshold instead (the PR #65 approach): a 500 ppm
        # q=0.01 transit in normalized flux (kernel-internal
        # s ~ 5e-6) must still be recovered.
        rand = np.random.RandomState(42)
        ndata, freq_inj, q_inj = 3000, 0.4, 0.01
        t = np.sort(370.0 * rand.rand(ndata))
        y = np.ones(ndata) - 5e-4 * ((t * freq_inj) % 1.0 < q_inj)
        y += 1e-4 * rand.randn(ndata)
        dy = 1e-4 * np.ones(ndata)
        freqs = np.linspace(0.38, 0.42, 4001)
        power = eebls_gpu_fast(t, y, dy, freqs, qmin=0.005, qmax=0.05)
        fbest = freqs[int(np.argmax(power))]
        assert abs(fbest - freq_inj) < 5 * (freqs[1] - freqs[0])


class TestBatchFastParity(object):
    """E1 regression: eebls_gpu_batch must match eebls_gpu_fast on the
    same inputs. The batch kernel's noverlap argument is a no-op (like
    the fast kernels', the A2 finding), so the batch launch is wrapped
    in the same host-side dphi-shifted multi-pass; before that the
    batch path was effectively noverlap=1 while the fast/adaptive
    reference multi-passed, and the periodograms diverged at small
    ndata (corr 0.77, peak match 5/10 at ndata=200 in the Jun 2026
    GPU benchmark)."""

    @pytest.mark.parametrize('ndata', [200, 2000])
    def test_batch_matches_fast(self, ndata):
        from ..bls import eebls_gpu_batch, eebls_gpu_fast

        rand = np.random.RandomState(3)
        baseline = 365.0
        freq_inj, q_inj, delta = 0.5, 0.03, 0.05
        t = np.sort(baseline * rand.rand(ndata))
        phase = (t * freq_inj) % 1.0
        y = 12.0 - delta * (phase < q_inj)
        sigma = 0.01
        y += sigma * rand.randn(ndata)
        dy = sigma * np.ones(ndata)

        freqs = np.linspace(0.1, 1.0, 5000)
        p_fast = eebls_gpu_fast(t, y, dy, freqs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_batch = eebls_gpu_batch([(t, y, dy)], freqs)[0]

        corr = float(np.corrcoef(p_fast, p_batch)[0, 1])
        assert corr > 0.999, corr
        assert int(np.argmax(p_fast)) == int(np.argmax(p_batch))

    def test_batch_noverlap_1_single_pass(self):
        # noverlap=1 must reproduce the old single-pass behavior:
        # everywhere <= the multi-pass result (elementwise max).
        from ..bls import eebls_gpu_batch

        rand = np.random.RandomState(4)
        ndata = 300
        t = np.sort(365.0 * rand.rand(ndata))
        y = 1.0 + 0.01 * rand.randn(ndata)
        dy = 0.01 * np.ones(ndata)
        freqs = np.linspace(0.1, 1.0, 2000)

        p1 = eebls_gpu_batch([(t, y, dy)], freqs, noverlap=1)[0]
        p3 = eebls_gpu_batch([(t, y, dy)], freqs, noverlap=3)[0]
        assert np.all(p3 >= p1 - 1e-7)
        assert np.max(np.abs(p3 - p1)) > 0


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

    def test_snr_uses_loaded_data_on_memory_reuse(self):
        # A5 audit follow-up: on the memory-reuse path (memory= given,
        # transfer_to_device=False) the y/dy ARGUMENTS may not be the
        # data that produced the periodogram; the 'snr'/'loglik'
        # scaling must come from the chi2_0 of the data actually
        # loaded into the memory (recorded at setdata time), not from
        # the arguments.
        from ..bls import BLSMemory, eebls_gpu_fast
        t, y, dy = self._data()
        freqs = np.linspace(0.5, 1.5, 200)

        p_ref = eebls_gpu_fast(t, y, dy, freqs, convention='snr')

        mem = BLSMemory.fromdata(t, y, dy, qmin=1e-2, qmax=0.5,
                                 freqs=freqs, transfer=True)
        # deliberately junk arguments: must not affect the scaling
        y_junk = 100.0 + 5.0 * y
        dy_junk = 25.0 * dy
        p_reuse = eebls_gpu_fast(t, y_junk, dy_junk, freqs,
                                 memory=mem,
                                 transfer_to_device=False,
                                 convention='snr')
        assert_allclose(p_reuse, p_ref, rtol=1e-6)

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
        # phi0 is now in the ORIGINAL input timescale, so a time-shifted
        # run must use the covariantly shifted phase
        # (phi0 + offset * freq) mod 1 to refer to the same transit.
        t, y, dy, freq, q, phi0 = self._signal()
        p_rel = single_bls(t, y, dy, freq, q, phi0)
        phi0_raw = (phi0 + self.bjd_offset * freq) % 1.0
        p_raw = single_bls(t + self.bjd_offset, y, dy, freq, q, phi0_raw)
        assert p_rel > 0.5  # signal actually detected
        assert abs(p_raw - p_rel) < 1e-3 * p_rel

    def test_single_bls_phase_is_original_timescale(self):
        # The convention itself: evaluating at the UNshifted phi0 on
        # shifted times must MISS the transit (if it matched, phases
        # would still be epoch-relative and the covariance test above
        # would be vacuous). An integer-day offset o at freq=0.3 rotates
        # the transit by (o * freq) mod 1 = 0.5 in phase, so the
        # unshifted phi0 lands in pure out-of-transit noise.
        t, y, dy, freq, q, phi0 = self._signal()
        p_rel = single_bls(t, y, dy, freq, q, phi0)
        p_wrong = single_bls(t + 4325.0, y, dy, freq, q, phi0)
        assert p_rel > 0.5
        assert p_wrong < 0.25 * p_rel

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
        # staging buffers hold the samples in conflict-scattered order
        # (utils.conflict_scatter_perm); compare as sets via sort
        assert_allclose(np.sort(mem.t[:len(t)]),
                        np.sort(t.astype(np.float32)), atol=1e-3)
        assert mem.epoch == pytest.approx(
            np.floor(self.bjd_offset + t.min()))

    def test_bls_batch_memory_epoch_subtraction(self):
        # Runs on GPU only (pinned host arrays); skipped on CPU.
        from ..memory.bls_memory import BLSBatchMemory
        t, y, dy, freq, q, phi0 = self._signal()
        mem = BLSBatchMemory(len(t), 1, 8)
        mem.set_lightcurve(0, t + self.bjd_offset, y, dy)
        # staging buffers hold the samples in conflict-scattered order
        # (utils.conflict_scatter_perm); compare as sets via sort
        assert_allclose(np.sort(mem.t[:len(t)]),
                        np.sort(t.astype(np.float32)), atol=1e-3)
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


class TestBlsBatchSizing(object):
    """Defect 1 of the Sep 2026 audit (``bls-overflow-oob``), the
    default ``eebls_transit`` path for ndata >= 500 (``eebls_gpu``):

    (a) the fold kernels indexed their ``ndata * nfreq`` threads in 32
        bits while the host launched the exact product with an
        uncapped auto batch, so a TESS 2-min year (262,800 points x an
        18,551-frequency batch = 4.9e9 > 2^32) silently returned
        65,372 zero powers, a power of 1.678 (> 1) and the wrong peak;
    (b) the device bin buffers were sized from
        ``count_tot_nbins(grid-wide min nbins0, grid-wide max nbinsf)``,
        which is NOT an upper bound over batches (``count_tot_nbins``
        is non-monotone in ``nbins0``), so a Keplerian-q batched grid
        could overrun its buffers: ``eebls_transit(t, y, dy, fmin=0.02,
        fmax=0.5)`` on 70,000 points died with ``illegal memory
        access``.

    The kernels now index in 64 bits, the host caps ``freq_batch_size``
    at ``len(freqs)`` and ``(2^31 - 1) // ndata``, and the batch table
    is built before allocating so the buffers are sized from the
    actual maximum over batches.
    """

    # ---- pure-CPU checks of the sizing helpers ----

    def test_count_tot_nbins_is_not_monotone_in_nbins0(self):
        # the property that broke the old sizing (audit's numbers)
        assert [count_tot_nbins(nb0, 359, 0.2) for nb0 in (28, 29, 30)] \
            == [1875, 1939, 1704]

    def test_batch_table_sizes_from_the_actual_batches(self):
        # batch 0 starts at nbins0 = 29 (1939 cells per frequency)
        # although the grid-wide minimum nbins0 is 28 (1875 cells): the
        # old gs = freq_batch_size * 1875 * noverlap under-allocated
        # batch 0 and the fold kernel's atomics ran off the buffer
        nbins0 = np.array([29] * 5 + [28] * 5 + [30] * 5)
        nbinsf = np.full(15, 359)
        noverlap = 3
        table = _bls_batch_table(nbins0, nbinsf, 5, 0.2)
        assert [(b[0], b[1]) for b in table] == [(0, 5), (5, 10), (10, 15)]
        assert [(b[2], b[3]) for b in table] == [(29, 359), (28, 359),
                                                 (30, 359)]
        assert [b[4] for b in table] == [1939, 1875, 1704]

        old_gs = 5 * count_tot_nbins(int(nbins0.min()), int(nbinsf.max()),
                                     0.2) * noverlap
        new_gs = max((b[1] - b[0]) * b[4] for b in table) * noverlap
        batch0_bins = 5 * table[0][4] * noverlap
        assert batch0_bins > old_gs      # the overrun
        assert batch0_bins <= new_gs     # the fix

        # the memory-budget estimate is an upper bound over batches
        assert _max_nbins_tot(nbins0, nbinsf, 0.2) >= max(b[4]
                                                          for b in table)

    def test_batch_table_last_batch_and_uneven_grids(self):
        nbins0 = np.array([4, 4, 2, 2, 2, 8, 8])
        nbinsf = np.array([50, 40, 60, 60, 20, 100, 100])
        table = _bls_batch_table(nbins0, nbinsf, 3, 0.3)
        assert [(b[0], b[1]) for b in table] == [(0, 3), (3, 6), (6, 7)]
        assert table[0][2:4] == (2, 60)   # collapsed min nb0 / max nbf
        assert table[1][2:4] == (2, 100)
        assert table[2][2:4] == (8, 100)
        for b in table:
            assert b[4] == count_tot_nbins(b[2], b[3], 0.3)
        with pytest.raises(ValueError):
            _bls_batch_table(nbins0, nbinsf, 0, 0.3)

    def test_max_nbins_tot_bounds_every_batching_of_a_keplerian_grid(self):
        # HAT-like Keplerian grid with 0.5 q .. 2 q bounds, as
        # eebls_transit builds it: every batch of every batch size
        # needs at most the estimated number of cells
        freqs, qvals = keplerian_freq_grid(0.5, 100., 3650.,
                                           oversampling=2,
                                           return_qvals=True)
        qvals = qvals.astype(np.float64)[:5000]
        nbins0, nbinsf = _q_bounds_to_nbins(0.5 * qvals, 2.0 * qvals)
        for dlogq in (0.2, 0.3, -1.0):
            bound = _max_nbins_tot(nbins0, nbinsf, dlogq)
            for fbs in (1, 7, 100, 1234, len(qvals)):
                table = _bls_batch_table(nbins0, nbinsf, fbs, dlogq)
                assert max(b[4] for b in table) <= bound

    def test_cap_freq_batch_size(self):
        # (2^31 - 1) // ndata: the audit's 66,000-point case
        assert _cap_freq_batch_size(10 ** 9, 66000, 10 ** 9) \
            == _MAX_FOLD_THREADS // 66000 == 32537
        assert 66000 * 32537 <= 2 ** 31 - 1 < 66000 * 32538
        # never more than the grid
        assert _cap_freq_batch_size(500, 100, 300) == 300
        # never less than one frequency
        assert _cap_freq_batch_size(0, 100, 300) == 1
        # a sane request is left alone
        assert _cap_freq_batch_size(5, 100, 300) == 5

    def test_q_bounds_to_nbins(self):
        nb0, nbf = _q_bounds_to_nbins([0.01, 0.02], [0.5, 0.25])
        assert list(nb0) == [2, 4] and list(nbf) == [100, 50]
        with pytest.raises(ValueError, match="qmin must be > 0"):
            _q_bounds_to_nbins([0.0], [0.5])
        with pytest.raises(ValueError, match="qmax must be <= 1"):
            _q_bounds_to_nbins([0.1], [1.5])

    def test_eebls_gpu_rejects_bad_bounds_before_any_gpu_work(self):
        # used to be a ZeroDivisionError (qmin > qmax) or a device
        # divide-by-zero (qmax > 1); validation now precedes the compile,
        # so this runs on CPU-only machines too
        t, y, dy = data(ndata=50)
        freqs = np.array([0.9, 1.0, 1.1])
        with pytest.raises(ValueError, match="qmin > qmax"):
            eebls_gpu(t, y, dy, freqs, qmin=0.2, qmax=0.1)
        with pytest.raises(ValueError, match="qmax must be <= 1"):
            eebls_gpu(t, y, dy, freqs, qmin=0.1, qmax=2.0)
        with pytest.raises(ValueError, match="qmin must be > 0"):
            eebls_gpu(t, y, dy, freqs, qmin=0.0, qmax=0.5)
        with pytest.raises(ValueError, match="qmin"):
            eebls_gpu(t, y, dy, freqs, qmin=np.array([0.01, 0.02]))

    # ---- GPU ----

    @staticmethod
    def _big_lc(ndata=131072, seed=1):
        rng = np.random.RandomState(seed)
        t = np.sort(rng.uniform(0, 30., ndata))
        y = 1 - 0.01 * (((t * 0.5) % 1) < 0.3) + 0.002 * rng.randn(ndata)
        dy = np.full(ndata, 0.002)
        return t, y, dy

    def test_fold_kernel_index_is_64_bit(self):
        # Direct launch of bin_and_phase_fold_bst_multifreq with
        # ndata * nfreq = 131072 * 32769 = 4.295e9 > 2^32 (the host
        # entry points now cap the batch, so only a direct launch
        # reaches this). With the old 32-bit bound `i < ndata * nfreq`
        # the product wrapped to 65536: only half of frequency 0's
        # points were binned and every other frequency stayed empty.
        # One q level of 1024 bins keeps the atomics cheap (~1 s).
        import pycuda.gpuarray as gpuarray
        from ..bls import _function_signatures, _default_block_size
        ndata, nf, nb = 131072, 32769, 1024
        assert ndata * nf > 2 ** 32
        t, y, dy = self._big_lc(ndata)
        t32 = (t - np.floor(t.min())).astype(np.float32)
        rng = np.random.RandomState(5)
        yw = (1e-4 * rng.randn(ndata)).astype(np.float32)
        w = np.full(ndata, 1. / ndata, dtype=np.float32)
        freqs = np.linspace(0.3, 0.7, nf).astype(np.float32)

        funcs = compile_bls(
            function_names=['bin_and_phase_fold_bst_multifreq'])
        func = funcs['bin_and_phase_fold_bst_multifreq']
        t_g, yw_g, w_g, f_g = (gpuarray.to_gpu(a)
                               for a in (t32, yw, w, freqs))
        yw_bin = gpuarray.zeros(nf * nb, np.float32)
        w_bin = gpuarray.zeros(nf * nb, np.float32)
        bs = _default_block_size
        grid = (int(np.ceil(float(ndata) * nf / bs)), 1)
        args = (t_g.ptr, yw_g.ptr, w_g.ptr, yw_bin.ptr, w_bin.ptr, f_g.ptr)
        func.prepared_call(grid, (bs, 1, 1), *args, np.uint32(ndata),
                           np.uint32(nf), np.uint32(nb), np.uint32(nb),
                           np.uint32(0), np.uint32(1), np.float32(0.2),
                           np.uint32(nb))
        wb = w_bin.get()
        ywb = yw_bin.get()

        # float32 fold replica (bit-identical to the kernel's
        # mod1(t * f) / floorf(nb * phi) for dphi = 0); check the first,
        # a middle and the LAST frequency -- the last one's threads all
        # lie beyond the 2^32 boundary
        for k in (0, nf // 2, nf - 1):
            phi = np.float32(t32 * freqs[k])
            phi = phi - np.floor(phi)
            b = np.floor(np.float32(nb) * phi).astype(np.int64) % nb
            ref_w = np.bincount(b, weights=w.astype(np.float64),
                                minlength=nb)
            ref_yw = np.bincount(b, weights=yw.astype(np.float64),
                                 minlength=nb)
            assert_allclose(wb[k * nb:(k + 1) * nb], ref_w,
                            rtol=1e-5, atol=1e-9)
            assert_allclose(ywb[k * nb:(k + 1) * nb], ref_yw,
                            rtol=1e-3, atol=1e-8)
        # nothing was binned outside the requested cells, and every
        # frequency saw all the weight
        assert_allclose(wb.reshape(nf, nb).sum(axis=1), 1.0, rtol=1e-4)

    def test_eebls_gpu_above_2_32_threads_matches_safe_batching(self):
        # eebls_gpu with a user-supplied freq_batch_size whose
        # ndata * batch exceeds 2^32 (before the fix: zeros / powers > 1;
        # the audit's 66,000 x 66,000 case had corr -0.003 with the
        # correct periodogram). One q level of 1024 bins keeps the two
        # full-grid runs to well under a second each.
        ndata, nf = 131072, 32769
        t, y, dy = self._big_lc(ndata)
        freqs = np.linspace(0.3, 0.7, nf)
        q = 1. / 1024
        kw = dict(qmin=q, qmax=q, noverlap=1)
        p_big, sols_big = eebls_gpu(t, y, dy, freqs, freq_batch_size=nf,
                                    **kw)
        p_safe, sols_safe = eebls_gpu(t, y, dy, freqs,
                                      freq_batch_size=4096, **kw)
        assert not np.any(p_big == 0)
        assert np.all(p_big <= 1.0)
        assert_allclose(p_big, p_safe, rtol=1e-4, atol=1e-6)
        assert np.argmax(p_big) == np.argmax(p_safe)

    def test_eebls_gpu_keplerian_batches_do_not_overrun(self):
        # The audit's reproducer for (b): HAT-like Keplerian grid
        # (keplerian_freq_grid(0.5, 100, 3650), first 20,000
        # frequencies, qmin = 0.5 q, qmax = 2 q), 600 points,
        # freq_batch_size = 2435 (what a 1.5 GB budget gave the old
        # sizing). Batch 0 starts at nbins0 = 131 and needs 2706 cells
        # per frequency while the old buffers held 2565 (the grid-wide
        # (81, 570) count): `illegal memory access` before the fix.
        freqs, qvals = keplerian_freq_grid(0.5, 100., 3650.,
                                           oversampling=2,
                                           return_qvals=True)
        freqs = freqs.astype(np.float64)[:20000]
        qvals = qvals.astype(np.float64)[:20000]
        qmins, qmaxes = 0.5 * qvals, 2.0 * qvals
        nbins0, nbinsf = _q_bounds_to_nbins(qmins, qmaxes)
        fbs, dlogq, noverlap = 2435, 0.2, 3
        table = _bls_batch_table(nbins0, nbinsf, fbs, dlogq)
        old_cells = count_tot_nbins(int(nbins0.min()), int(nbinsf.max()),
                                    dlogq)
        # the configuration really is one the old sizing overran
        assert table[0][4] > old_cells

        rng = np.random.RandomState(0)
        ndata = 600
        t = np.sort(rng.uniform(0, 3650., ndata))
        y = 1 + 0.002 * rng.randn(ndata)
        dy = np.full(ndata, 0.002)
        p, sols = eebls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxes,
                            freq_batch_size=fbs, dlogq=dlogq,
                            noverlap=noverlap)
        assert np.all(np.isfinite(p))
        assert np.all((p >= 0) & (p <= 1))
        assert len(sols) == len(freqs)

    def test_eebls_gpu_small_grid_allocates_only_what_it_needs(self):
        # finding 135 / plan item BLS-2: a 300-frequency grid used to
        # allocate scratch for the ~100K-frequency batch the free
        # memory allowed (4 arrays x 5 streams x ~0.9 x free). The
        # batch is now capped at len(freqs), so the scratch buffers
        # hold exactly nfreq * cells * noverlap floats, and one scratch
        # set per batch (not per stream) is allocated. The periodogram
        # is unchanged (scalar q: batch boundaries never change it).
        import cuvarbase.bls as B
        t, y, dy = data(snr=10, q=0.05, phi0=0.3, freq=1.0, baseline=365.)
        freqs = np.linspace(0.95, 1.05, 300)
        qmin, qmax, noverlap, dlogq = 0.01, 0.1, 3, 0.2
        need = len(freqs) * count_tot_nbins(10, 100, dlogq) * noverlap

        sizes = []
        real = B.gpuarray

        class Recorder(object):
            to_gpu = staticmethod(real.to_gpu)
            maximum = staticmethod(real.maximum)

            @staticmethod
            def zeros(n, dtype=np.float32):
                sizes.append(int(n))
                return real.zeros(n, dtype=dtype)

        B.gpuarray = Recorder
        try:
            p, sols = eebls_gpu(t, y, dy, freqs, qmin=qmin, qmax=qmax,
                                noverlap=noverlap, dlogq=dlogq)
        finally:
            B.gpuarray = real
        assert max(sizes) == need
        # single batch -> one scratch set of 4 arrays (+ the 4
        # per-frequency result arrays)
        assert sizes.count(need) == 4

        p2, sols2 = eebls_gpu(t, y, dy, freqs, qmin=qmin, qmax=qmax,
                              noverlap=noverlap, dlogq=dlogq,
                              freq_batch_size=50)
        assert_allclose(p, p2, rtol=1e-4, atol=1e-6)
