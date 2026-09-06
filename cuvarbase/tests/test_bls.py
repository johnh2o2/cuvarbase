from itertools import product
import os
import warnings

import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..bls import eebls_gpu, eebls_transit_gpu, \
                  q_transit, compile_bls, hone_solution,\
                  single_bls, eebls_gpu_custom, eebls_gpu_fast, \
                  eebls_gpu_fast_optimized, \
                  sparse_bls_cpu, sparse_bls_gpu, eebls_transit, \
                  transit_autofreq, \
                  count_tot_nbins, _bls_batch_table, _max_nbins_tot, \
                  _per_freq_nbins_tot, _cap_freq_batch_size, \
                  _q_bounds_to_nbins, _MAX_FOLD_THREADS, \
                  _fast_path_nbins, _fast_bls_box_scan, \
                  _fast_bls_solutions
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
            
            best_p = None
            for Q, PHI in q_and_phis:
                p = single_bls(t, y, dy, freq, Q, PHI,
                               ignore_negative_delta_sols=ignore_negative_delta_sols)
                if best_p is None or p > best_p:
                    best_p = p
            
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
        """Exhaustive BLS over all observation-pair transit boundaries
        (float32 fold like the kernels; flux centred in float64 and
        sums in float64 -- the sparse paths centre in float64 since
        defect 8 of the Sep 2026 audit)."""
        t = np.asarray(t, dtype=np.float32)
        y64 = np.asarray(y, dtype=np.float64)
        w64 = np.power(np.asarray(dy, dtype=np.float64), -2)
        w64 /= w64.sum()
        y = (y64 - np.dot(w64, y64)).astype(np.float32)
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

    def test_use_simple_kernel_was_removed(self):
        """The bubble-sort sparse kernel (sparse_bls_simple.cu) shipped
        with the pre-PR#65 MAX_W_COMPLEMENT 1E-9 bound (powers up to
        4.6 in pure noise); it is gone and the old switch must fail
        loudly on every entry point that used to accept it."""
        from ..bls import compile_sparse_bls
        from ..utils import find_kernel
        t, y, dy = data(ndata=50)
        freqs = np.array([0.9, 1.0, 1.1])
        with pytest.raises(TypeError, match="use_simple"):
            sparse_bls_gpu(t, y, dy, freqs, use_simple=True)
        with pytest.raises(TypeError, match="use_simple"):
            compile_sparse_bls(use_simple=False)
        with pytest.raises(TypeError, match="use_simple"):
            eebls_transit(t, y, dy, fmin=0.9, fmax=1.1, use_simple=True)
        import os
        assert not os.path.exists(find_kernel('sparse_bls_simple'))

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

    def test_sparse_bls_gpu_q_bounds(self):
        """GPU sparse BLS honors per-frequency q bounds (matches CPU)."""
        t, y, dy = data(snr=30, q=0.1, phi0=0.3, freq=1.0,
                        baseline=365., ndata=80)
        freqs = np.linspace(0.95, 1.05, 11)
        qmins = np.full(len(freqs), 0.03)
        qmaxes = np.full(len(freqs), 0.2)

        power_cpu, _ = sparse_bls_cpu(t, y, dy, freqs,
                                      qmin=qmins, qmax=qmaxes)
        power_gpu, sols_gpu = sparse_bls_gpu(t, y, dy, freqs,
                                             qmin=qmins, qmax=qmaxes)

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

        # The sparse statistic is piecewise constant in frequency (the
        # power only changes when a point crosses a box edge): with 50
        # points the maximum is a plateau of ~20 grid frequencies
        # spanning +-7 q/T around the injected frequency, and which of
        # them argmax returns is a tie-break. Before the float64
        # centring (defect 8) float32 noise broke the tie by luck within
        # 2 q/T. Require the found peak to lie on the float64
        # reference's maximum plateau, and that plateau to cover the
        # injected frequency to within ~q/T (one phase-smear width).
        qv = q_transit(freqs)
        ref = _sparse_reference(t, y, dy, freqs, 0.5 * qv, 2.0 * qv)
        plateau = ref >= ref.max() * (1 - 1e-5)
        assert plateau[int(np.argmax(powers))]
        T = max(t) - min(t)
        assert np.min(np.abs(freqs[plateau] - freq_true)) < 2 * q / T

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
        # work. Runs on a device; on CPU-only hosts the conftest turns
        # the GPUStubError into a skip (it used to swallow every
        # exception and so passed while asserting nothing).
        t, y, dy = self._data()
        freqs, powers, sols = eebls_transit(t, y, dy, rho=1.5,
                                            samples_per_peak=2,
                                            fmin=0.95, fmax=1.05,
                                            use_gpu=True)
        assert len(freqs) == len(powers) == len(sols)
        assert len(freqs) > 0
        assert np.all(np.isfinite(powers))
        # the injected transit (freq = 1.0) is the peak
        assert abs(freqs[np.argmax(powers)] - 1.0) < 0.01

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
        # No warning and no kwargs filtering on the standard path.
        # Runs on a device (the conftest skips it on CPU-only hosts);
        # a UserWarning is an error here, so "should not warn" is
        # asserted rather than swallowed.
        import warnings as _warnings
        t, y, dy = self._data(ndata=100)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            freqs, powers, sols = eebls_transit(t, y, dy, fmin=0.95,
                                                fmax=1.05, use_sparse=False)
        assert len(freqs) == len(powers) == len(sols)
        assert len(freqs) > 0
        assert np.all(np.isfinite(powers))
        assert abs(freqs[np.argmax(powers)] - 1.0) < 0.01


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
        assert_allclose(convert_bls_power(p, y, dy, convention='snr'),
                        np.sqrt(chi2_0 * p))
        assert_allclose(convert_bls_power(p, y, dy, convention='loglik'),
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
        # astropy's transit_time is mid-transit; single_bls takes phi0
        # (the transit START phase) in the ORIGINAL input timescale and
        # re-references it to the subtracted epoch internally.
        freq = 1.0 / period
        q = duration / period
        phi0 = ((transit_time - 0.5 * duration) * freq) % 1.0
        return single_bls(t, y, dy, freq, q, phi0), q

    def test_snr_matches_astropy(self):
        from ..bls import convert_bls_power
        t, y, dy = self._data()
        res = self._astropy_results(t, y, dy, 'snr')
        for i in range(len(res.period)):
            p_native, _ = self._our_power_at(
                t, y, dy, res.period[i], res.duration[i],
                res.transit_time[i])
            snr = convert_bls_power(p_native, y, dy, convention='snr')
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
            loglik = convert_bls_power(p_native, y, dy, convention='loglik')

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
        assert_allclose(p_snr, convert_bls_power(p_native, y, dy, convention='snr'),
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
        assert_allclose(p_snr, convert_bls_power(p0, y, dy, convention='snr'),
                        rtol=1e-4, atol=1e-6)

        f0 = eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.2)
        f_log = eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.2,
                               convention='loglik')
        assert_allclose(f_log, convert_bls_power(f0, y, dy, convention='loglik'),
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
    float64) before casting; the phases they report are re-referenced
    to the ORIGINAL input timescale (see
    ``test_single_bls_phase_is_original_timescale``).
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
        assert list(_per_freq_nbins_tot(nbins0, nbinsf, 0.2)) \
            == [1939] * 5 + [1875] * 5 + [1704] * 5
        table = _bls_batch_table(nbins0, nbinsf, 5, 0.2)
        assert [(b[0], b[1]) for b in table] == [(0, 5), (5, 10), (10, 15)]
        assert [b[2] for b in table] == [1939, 1875, 1704]

        old_gs = 5 * count_tot_nbins(int(nbins0.min()), int(nbinsf.max()),
                                     0.2) * noverlap
        new_gs = max((b[1] - b[0]) * b[2] for b in table) * noverlap
        batch0_bins = 5 * table[0][2] * noverlap
        assert batch0_bins > old_gs      # the overrun
        assert batch0_bins <= new_gs     # the fix

        # a mixed batch: the stride is the per-frequency maximum, NOT
        # the count of the batch-wide (min nb0, max nbf) collapse
        # (1875 here, less than the 1939 cells its nb0 = 29 members
        # need)
        table = _bls_batch_table(nbins0, nbinsf, 7, 0.2)
        assert [(b[0], b[1]) for b in table] == [(0, 7), (7, 14), (14, 15)]
        assert [b[2] for b in table] == [1939, 1875, 1704]

        # the memory-budget estimate is an upper bound over batches
        assert _max_nbins_tot(nbins0, nbinsf, 0.2) >= max(b[2]
                                                          for b in table)

    def test_batch_table_last_batch_and_uneven_grids(self):
        nbins0 = np.array([4, 4, 2, 2, 2, 8, 8])
        nbinsf = np.array([50, 40, 60, 60, 20, 100, 100])
        per_f = [count_tot_nbins(a, b, 0.3) for a, b in zip(nbins0, nbinsf)]
        assert list(_per_freq_nbins_tot(nbins0, nbinsf, 0.3)) == per_f
        table = _bls_batch_table(nbins0, nbinsf, 3, 0.3)
        assert [(b[0], b[1]) for b in table] == [(0, 3), (3, 6), (6, 7)]
        assert [b[2] for b in table] == [max(per_f[0:3]), max(per_f[3:6]),
                                         per_f[6]]
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
                assert max(b[2] for b in table) <= bound

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
        from ..bls import _default_block_size
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
        nb_g = gpuarray.to_gpu(np.full(nf, nb, dtype=np.uint32))
        yw_bin = gpuarray.zeros(nf * nb, np.float32)
        w_bin = gpuarray.zeros(nf * nb, np.float32)
        bs = _default_block_size
        grid = (int(np.ceil(float(ndata) * nf / bs)), 1)
        args = (t_g.ptr, yw_g.ptr, w_g.ptr, yw_bin.ptr, w_bin.ptr, f_g.ptr,
                nb_g.ptr, nb_g.ptr)
        func.prepared_call(grid, (bs, 1, 1), *args, np.uint32(ndata),
                           np.uint32(nf), np.uint32(0), np.uint32(1),
                           np.float32(0.2), np.uint32(nb))
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
        # The two batchings sum the same float32 shared-memory atomics
        # in a different order, so near-zero powers differ by more than
        # a 1e-4 relative tolerance (observed: 1.472e-4 vs 1.438e-4 on
        # one of 32,769 frequencies). atol is set an order of magnitude
        # above that floor; the peak, its location and the overflow
        # invariants above are what this test is really guarding.
        assert_allclose(p_big, p_safe, rtol=1e-4, atol=1e-5)
        assert np.argmax(p_big) == np.argmax(p_safe)
        assert np.corrcoef(p_big, p_safe)[0, 1] > 0.9999

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
        # the configuration really is one the old sizing overran: its
        # batch-0 collapse (131, 570) needs more cells than the
        # grid-wide (81, 570) collapse the buffers were sized from
        nb0_b0 = int(nbins0[:fbs].min())
        nbf_b0 = int(nbinsf[:fbs].max())
        assert (nb0_b0, nbf_b0) == (131, 570)
        assert count_tot_nbins(nb0_b0, nbf_b0, dlogq) > old_cells
        # the per-frequency stride is what is allocated now
        assert table[0][2] == int(np.max(_per_freq_nbins_tot(
            nbins0[:fbs], nbinsf[:fbs], dlogq)))

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
        qs = np.array([s[0] for s in sols])
        assert np.all(qs >= 1. / nbinsf - 1e-6)
        assert np.all(qs <= 1. / nbins0 + 1e-6)

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


class TestPerFrequencyQBounds(object):
    """Defect 7 of the Sep 2026 audit (``bls-q-collapse``): ``eebls_gpu``
    reduced per-frequency ``qmin``/``qmax`` arrays to one scalar pair
    per batch (the batch-wide min/max) because the binned kernels took
    scalar bin counts per launch, so every frequency in a batch was
    searched over ``floor(1/max qmax) .. ceil(1/min qmin)`` bins:
    2670/3049 Keplerian-grid solutions fell outside their own window
    and the result changed with ``freq_batch_size`` / free memory.
    The kernels now read per-frequency bin-count arrays, and the
    ``eebls_transit`` default (ndata >= sparse_threshold) runs the fast
    kernel (which always honoured the bounds) with a top-K solution
    pass.
    """

    @staticmethod
    def _lc(ndata=1200, baseline=200., freq=0.2, q=0.03, phi0=0.6,
            snr=12., seed=11, sigma=0.01):
        rng = np.random.RandomState(seed)
        t = np.sort(rng.uniform(0, baseline, ndata)) + 100.3
        delta = snr * sigma / np.sqrt(ndata * q * (1 - q))
        y = 12. - delta * (((t * freq) - phi0) % 1.0 < q)
        y += sigma * rng.randn(ndata)
        dy = np.full(ndata, sigma)
        return t, y, dy

    # ---- CPU: the fast-kernel box scan used for the solution pass ----

    def test_fast_box_scan_matches_brute_force_over_the_kernel_grid(self):
        # every (q, phi) the fast kernel searches at one frequency is
        # q = m / nbf, phi0 = (n + s / noverlap) / nbf; the scan must
        # return the box single_bls scores highest
        from ..utils import subtract_epoch
        t, y, dy = self._lc(ndata=80, baseline=30., freq=1.0, q=0.1,
                            phi0=0.3, snr=20., seed=3)
        freq = 1.0
        qmin, qmax, dlogq, noverlap = 0.05, 0.25, 0.3, 2
        t64, epoch = subtract_epoch(t)
        w = dy ** -2
        w /= w.sum()
        ybar = np.dot(w, y)
        YY = np.dot(w, (y - ybar) ** 2)
        t32 = t64.astype(np.float32)
        nb0, nbf = _fast_path_nbins(np.float32([freq]), qmin, qmax)
        nb0, nbf = int(nb0[0]), int(nbf[0])
        assert (nb0, nbf) == (4, 20)

        val, q, phi = _fast_bls_box_scan(
            t32, ((y - ybar) * w).astype(np.float32),
            w.astype(np.float32), np.float32(freq), nb0, nbf, dlogq,
            noverlap)
        p_scan = val / YY

        # brute force over the same grid, in the original timescale.
        # The ladder runs up to and including nbf // nb0 = 5 (q = 0.25
        # = qmax); before the id-64 fix it stopped at 4 (q = 0.2).
        ms, m = [], 1
        while m <= nbf // nb0:
            ms.append(m)
            m += m * 3 // 10 if m * 3 // 10 > 0 else 1
        assert ms[-1] == nbf // nb0 and ms[-1] / nbf == qmax
        best = 0.
        for s_pass in range(noverlap):
            for m in ms:
                for n in range(nbf):
                    phi0 = ((n + s_pass / noverlap) / nbf
                            + epoch * freq) % 1.0
                    best = max(best, single_bls(t, y, dy, freq, m / nbf,
                                                phi0))
        assert abs(p_scan - best) < 1e-5 * max(best, 1e-3)
        # and the returned (q, phi) reproduces that power
        p_sol = single_bls(t, y, dy, freq, q,
                           (phi + epoch * freq) % 1.0)
        assert abs(p_sol - p_scan) < 1e-5 * max(best, 1e-3)
        assert q in [mm / nbf for mm in ms]

    def test_fast_solutions_selects_the_top_k_and_marks_the_rest(self):
        t, y, dy = self._lc(ndata=150, baseline=30., freq=1.0, q=0.1,
                            phi0=0.3, snr=20., seed=4)
        freqs = np.linspace(0.9, 1.1, 41)
        powers = np.exp(-0.5 * ((freqs - 1.0) / 0.01) ** 2)
        sols = _fast_bls_solutions(t, y, dy, freqs, powers, 0.05, 0.25, 5,
                                   dlogq=0.3, noverlap=2)
        assert len(sols) == len(freqs)
        filled = [i for i, s_ in enumerate(sols) if s_ is not None]
        assert set(filled) == set(np.argsort(-powers)[:5])
        assert 20 in filled
        for i in filled:
            q, phi = sols[i]
            assert 0.05 <= q <= 0.25 and 0. <= phi < 1.
        # zero-power frequencies get no solution; K = 0 -> all None
        assert all(s_ is None for s_ in
                   _fast_bls_solutions(t, y, dy, freqs, np.zeros(41),
                                       0.05, 0.25, 5))
        assert all(s_ is None for s_ in
                   _fast_bls_solutions(t, y, dy, freqs, powers,
                                       0.05, 0.25, 0))

    # ---- GPU: eebls_gpu with per-frequency bounds ----

    def test_eebls_gpu_array_bounds_are_honoured_per_frequency(self):
        # two frequencies with disjoint windows in ONE batch; the
        # injected transit at f = 0.05 has q = 0.15, outside that
        # frequency's [0.01, 0.02] window. The old batch-wide collapse
        # ([0.01, 0.2]) found q ~ 0.15 there.
        t, y, dy = self._lc(ndata=1200, baseline=200., freq=0.05, q=0.15,
                            phi0=0.2, snr=40., seed=5)
        freqs = np.array([0.05, 2.5])
        qmins = np.array([0.01, 0.10])
        qmaxes = np.array([0.02, 0.20])
        nb0, nbf = _q_bounds_to_nbins(qmins, qmaxes)

        p, sols = eebls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxes)
        p1, sols1 = eebls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxes,
                              freq_batch_size=1)
        for i in range(2):
            assert 1. / nbf[i] - 1e-6 <= sols[i][0] <= 1. / nb0[i] + 1e-6
        # one frequency per batch always had per-frequency semantics:
        # the default batching must now agree with it
        assert_allclose(p, p1, rtol=1e-5, atol=1e-7)
        assert [s_[0] for s_ in sols] == [s_[0] for s_ in sols1]
        # the wide (unconstrained) box the collapse used to return
        p_wide, sols_wide = eebls_gpu(t, y, dy, freqs[:1], qmin=0.01,
                                      qmax=0.2)
        assert sols_wide[0][0] > 0.1 and p_wide[0] > p[0]

    def test_eebls_gpu_keplerian_grid_independent_of_batching(self):
        t, y, dy = self._lc()
        freqs, q0 = transit_autofreq(t, qmin_fac=0.5, fmin=0.02, fmax=3.0)
        freqs = freqs[::max(1, len(freqs) // 600)]
        q0 = q_transit(freqs)
        qmins, qmaxes = 0.5 * q0, 2.0 * q0
        nb0, nbf = _q_bounds_to_nbins(qmins, qmaxes)

        runs = {}
        for fbs in (None, 200, 20, 1):
            runs[fbs] = eebls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxes,
                                  freq_batch_size=fbs)
        p_ref, sols_ref = runs[None]
        qs = np.array([s_[0] for s_ in sols_ref])
        # every solution inside its own window (bin-count rounding)
        assert np.all(qs >= 1. / nbf - 1e-6)
        assert np.all(qs <= 1. / nb0 + 1e-6)
        for fbs in (200, 20, 1):
            p, sols = runs[fbs]
            # float32 atomic-order noise only (the audit measured
            # 1.7e-2 differences between batchings before the fix)
            assert_allclose(p, p_ref, rtol=1e-4, atol=1e-6)
            qb = np.array([s_[0] for s_ in sols])
            # solutions may differ only where powers tie
            diff = qb != qs
            assert np.mean(diff) < 0.02

    # ---- GPU: the eebls_transit default path ----

    def test_eebls_transit_default_is_fast_kernel_plus_top_k_solutions(self):
        t, y, dy = self._lc(ndata=2000, seed=12)
        fr, p, sols = eebls_transit(t, y, dy, fmin=0.05, fmax=1.0)
        q0 = q_transit(fr)
        nb0, nbf = _fast_path_nbins(fr.astype(np.float32), 0.5 * q0,
                                    2.0 * q0)

        # the periodogram is the fast kernel's (per-frequency bounds)
        p_fast = eebls_gpu_fast(t, y, dy, fr, qmin=0.5 * q0, qmax=2.0 * q0)
        assert_allclose(p, p_fast, rtol=1e-4, atol=1e-6)
        assert abs(fr[np.argmax(p)] - 0.2) < 3 * 0.03 / 200.

        # top-10 solutions, None elsewhere, argmax included
        assert len(sols) == len(fr)
        filled = [i for i, s_ in enumerate(sols) if s_ is not None]
        assert set(filled) == set(np.argsort(-p, kind='stable')[:10])
        assert sols[int(np.argmax(p))] is not None

        for i in filled:
            q, phi = sols[i]
            # inside this frequency's own window: the ladder ends at
            # floor(nbf / nb0) fine bins (_fast_box_widths), so no
            # reported box may be wider than that
            assert q >= 1. / nbf[i] - 1e-6
            assert q <= (int(nbf[i]) // int(nb0[i])) / float(nbf[i]) + 1e-6
            assert q <= 2.0 * q0[i] * (1 + 1. / nb0[i]) + 1e-6
            # and it is the box that produced the power: single_bls
            # re-evaluates it exactly (float32 accumulation and, at a
            # bin edge, one point's membership may differ)
            p_single = single_bls(t, y, dy, fr[i], q, phi)
            n_box = len(t) * q
            assert abs(p_single - p[i]) < 1e-3 * p[i] + 1e-5 + 2. * p[i] / n_box

    def test_eebls_transit_default_independent_of_batching_and_memory(self):
        # the pre-1.0 default path changed 28,628/36,585 frequencies by
        # up to 2.3e-2 between a 24 GB and a 7 GB card (the auto batch
        # size set the collapsed window)
        t, y, dy = self._lc(ndata=1200)
        kw = dict(fmin=0.05, fmax=1.0)
        fr, p, sols = eebls_transit(t, y, dy, **kw)
        fr2, p2, sols2 = eebls_transit(t, y, dy, freq_batch_size=97, **kw)
        with pytest.warns(UserWarning, match="eebls_transit ignores"):
            fr3, p3, sols3 = eebls_transit(t, y, dy, max_memory=int(2e9),
                                           nstreams=2, **kw)
        assert_allclose(p2, p, rtol=1e-4, atol=1e-6)
        assert_allclose(p3, p, rtol=1e-4, atol=1e-6)
        for a, b in ((sols2, sols), (sols3, sols)):
            assert [i for i, s_ in enumerate(a) if s_ is not None] \
                == [i for i, s_ in enumerate(b) if s_ is not None]

    def test_eebls_transit_warns_about_ignored_eebls_gpu_kwargs(self):
        # the default path runs eebls_gpu_fast, so nstreams / max_memory
        # (a resource bound the caller may be relying on) do not apply;
        # dropping them silently was the complaint.
        t, y, dy = self._lc(ndata=600)
        kw = dict(fmin=0.1, fmax=0.5)
        for key, value in (('nstreams', 2), ('max_memory', int(2e9))):
            with pytest.warns(UserWarning) as rec:
                eebls_transit(t, y, dy, **dict(kw, **{key: value}))
            msgs = [str(w.message) for w in rec
                    if issubclass(w.category, UserWarning)]
            assert any(key in m and 'eebls_transit ignores' in m
                       for m in msgs), msgs
            assert any('eebls_gpu' in m for m in msgs), msgs
        # ... and no warning when they are not passed
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            eebls_transit(t, y, dy, **kw)
        assert not [w for w in rec
                    if 'eebls_transit ignores' in str(w.message)]

    def test_eebls_transit_solution_keywords(self):
        t, y, dy = self._lc(ndata=800)
        kw = dict(fmin=0.1, fmax=0.5)
        fr, p, sols = eebls_transit(t, y, dy, n_solutions=3, **kw)
        assert sum(s_ is not None for s_ in sols) == 3
        fr, p0, sols0 = eebls_transit(t, y, dy, n_solutions=0, **kw)
        assert len(sols0) == len(fr) and all(s_ is None for s_ in sols0)
        assert_allclose(p0, p, rtol=1e-4, atol=1e-6)
        # use_fast: same periodogram, no solution pass
        fr, pf, none = eebls_transit(t, y, dy, use_fast=True, **kw)
        assert none is None
        assert_allclose(pf, p, rtol=1e-4, atol=1e-6)
        # the binned search with a solution everywhere is still there
        fr, pg, sg = eebls_transit_gpu(t, y, dy, **kw)
        assert len(sg) == len(fr) and all(s_ is not None for s_ in sg)


def _sparse_reference(t, y, dy, freqs, qmin=0.0, qmax=0.5):
    """Exact float64 sparse-BLS reference (Panahi & Zucker 2021: every
    cyclic run of phase-sorted points), with the kernels' float32 fold
    and box definition (phi0 = first in-transit phase, q to the egress
    midpoint) and weight guards -- the reference of the Sep 2026 audit
    (repro/local/sparse-batch/sparse_exp.py). Returns 'chi2ratio'
    powers."""
    from ..utils import subtract_epoch
    t64, epoch = subtract_epoch(np.asarray(t, dtype=np.float64))
    y64 = np.asarray(y, dtype=np.float64)
    w = np.asarray(dy, dtype=np.float64) ** -2
    w /= w.sum()
    x = y64 - np.dot(w, y64)
    YY = np.dot(w, x ** 2)
    N = len(t64)
    out = np.zeros(len(freqs))
    qmin = np.broadcast_to(np.asarray(qmin, float), (len(freqs),))
    qmax = np.broadcast_to(np.asarray(qmax, float), (len(freqs),))
    i = np.arange(N)[:, None]
    L = np.arange(1, N)[None, :]
    j = i + L
    for k, f in enumerate(freqs):
        phi = (np.float32(t64) * np.float32(f)) % np.float32(1.0)
        phi = phi.astype(np.float64)
        o = np.argsort(phi, kind='stable')
        ps, ws, xs = phi[o], w[o], x[o]
        cw = np.concatenate([[0.0], np.cumsum(np.concatenate([ws, ws]))])
        cxw = np.concatenate([[0.0], np.cumsum(np.concatenate(
            [ws * xs, ws * xs]))])
        W = cw[j] - cw[i]
        S = cxw[j] - cxw[i]
        ps2 = np.concatenate([ps, ps + 1.0])
        last = ps2[j - 1]
        nxt = ps2[np.minimum(j, 2 * N - 1)]
        q = 0.5 * (last + nxt) - ps[:, None]
        valid = (q > 0) & (q >= qmin[k]) & (q <= qmax[k]) \
            & (W > 1e-9) & (W < 1.0 - 1e-4)
        with np.errstate(divide='ignore', invalid='ignore'):
            P = np.where(valid, S * S / (W * (1 - W)) / YY, 0.0)
        out[k] = P.max()
    return out


def _untied_frequencies(t, freqs):
    """Mask of the frequencies at which the kernels' float32 fold gives
    no two observations the same phase. At a tie the candidate runs
    depend on the sort order (bitonic vs argsort vs the stable sort of
    the reference) and the reported egress midpoint collapses onto the
    tied point (audit ids 65/74), so exact comparisons are only
    meaningful away from ties (~20-25 % of a 365-day mag-12 grid at
    f ~ 1.4 has one)."""
    from ..utils import subtract_epoch
    t64, _ = subtract_epoch(np.asarray(t, dtype=np.float64))
    t32 = t64.astype(np.float32)
    mask = np.ones(len(freqs), dtype=bool)
    for k, f in enumerate(freqs):
        phi = (t32 * np.float32(f)) % np.float32(1.0)
        mask[k] = len(np.unique(phi)) == len(phi)
    return mask


def _same_peak(p, ref, rtol=1e-4):
    """The reference power at the tested periodogram's argmax is the
    reference maximum (plateaus of equal power, e.g. 4 adjacent grid
    frequencies with the same in-transit set, break argmax ties by
    float32 rounding order)."""
    return ref[int(np.argmax(p))] >= ref.max() * (1 - rtol)


class TestSparseCentering(object):
    """Defect 8 of the Sep 2026 audit (``bls-sparse-uncentered``): the
    sparse kernels (and ``sparse_bls_cpu`` / ``single_bls``) accumulated
    float32 sums of raw ``w * y`` and subtracted ``ybar * W`` afterwards,
    so on mag-12 fluxes (the ``eebls_transit`` default for ndata < 500)
    the power was off by up to 1e-2 relative (argmax moved in 8/20
    seeds) and one point ~1e3x more precise than the rest gave powers
    up to 52 (> 1) at every frequency. The wrappers now centre in
    float64 before the float32 cast; the audit measured ~1e-6 after
    the fix."""

    @staticmethod
    def _mag12(N=200, seed=0, base=365.0, ybar=12.0, depth=5e-3,
               sig=5e-3, f=1.37, q=0.02):
        r = np.random.RandomState(seed)
        t = np.sort(base * r.rand(N))
        ph = (t * f) % 1
        y = ybar - depth * (ph < q) + sig * r.randn(N)
        dy = sig * (0.7 + 0.6 * r.rand(N))
        return t, y, dy

    @staticmethod
    def _grid():
        return 1.37 + (0.02 / 365 / 4) * np.arange(-100, 101)

    # ---- CPU ----

    def test_sparse_bls_cpu_mag12_matches_float64_reference(self):
        # before the fix: max rel 1.1e-2 (audit), 67-78 % of the grid
        # off by > 1e-3
        freqs = self._grid()
        qv = q_transit(freqs)
        for seed in (0, 1):
            t, y, dy = self._mag12(seed=seed)
            ref = _sparse_reference(t, y, dy, freqs, 0.5 * qv, 2.0 * qv)
            p, _ = sparse_bls_cpu(t, y, dy, freqs, qmin=0.5 * qv,
                                  qmax=2.0 * qv)
            ok = _untied_frequencies(t, freqs)
            assert ok.mean() > 0.7
            assert_allclose(p[ok], ref[ok], rtol=1e-4, atol=1e-7)
            assert _same_peak(p, ref)

    def test_sparse_bls_cpu_offset_invariance(self):
        freqs = self._grid()[::4]
        t, y, dy = self._mag12(seed=2)
        p0, s0 = sparse_bls_cpu(t, y, dy, freqs)
        p20, s20 = sparse_bls_cpu(t, y + 20., dy, freqs)
        assert_allclose(p20, p0, rtol=1e-5, atol=1e-8)
        assert [a[0] for a in s20] == [a[0] for a in s0]

    def test_single_bls_offset_invariance_and_reference(self):
        t, y, dy = self._mag12(seed=3)
        freqs = self._grid()[::8]
        ref = _sparse_reference(t, y, dy, freqs)
        _, sols = sparse_bls_cpu(t, y, dy, freqs)
        ok = _untied_frequencies(t, freqs)
        assert ok.sum() >= 15
        for k, f in enumerate(freqs):
            if not ok[k]:
                continue
            q, phi = sols[k]
            p = single_bls(t, y, dy, f, q, phi)
            p20 = single_bls(t, y + 20., dy, f, q, phi)
            assert abs(p20 - p) < 1e-5 * max(p, 1e-3)
            # the solution reproduces the reference power
            assert abs(p - ref[k]) < 1e-4 * max(ref[k], 1e-3)

    def test_cpu_one_precise_point_powers_stay_below_one(self):
        r = np.random.RandomState(3)
        N = 200
        t = np.sort(365 * r.rand(N))
        y = 12.0 + 0.01 * r.randn(N)
        dy0 = 0.01 * np.ones(N)
        freqs = np.linspace(0.5, 1.5, 51)
        ok = _untied_frequencies(t, freqs)
        assert ok.mean() > 0.7
        for R, tol in ((1e4, 1e-3), (1e6, 3e-2)):
            dy = dy0.copy()
            dy[17] = 0.01 / np.sqrt(R)
            ref = _sparse_reference(t, y, dy, freqs)
            p, _ = sparse_bls_cpu(t, y, dy, freqs)
            assert np.all(p <= 1.0)
            assert abs(p[ok].max() - ref[ok].max()) < tol * ref[ok].max()
            assert _same_peak(p[ok], ref[ok])

    # ---- GPU ----

    def test_sparse_bls_gpu_mag12_matches_float64_reference(self):
        freqs = self._grid()
        qv = q_transit(freqs)
        for seed in (0, 1, 2):
            t, y, dy = self._mag12(seed=seed)
            ref = _sparse_reference(t, y, dy, freqs, 0.5 * qv, 2.0 * qv)
            p, _ = sparse_bls_gpu(t, y, dy, freqs, qmin=0.5 * qv,
                                  qmax=2.0 * qv)
            ok = _untied_frequencies(t, freqs)
            assert ok.mean() > 0.7
            assert_allclose(p[ok], ref[ok], rtol=1e-4, atol=1e-7)
            assert _same_peak(p, ref)

    def test_eebls_transit_default_sparse_path_matches_reference(self):
        # the public default path (ndata < sparse_threshold) on mag-12
        # data: peak rel err was up to 7.2e-3 before the fix
        freqs = self._grid()
        t, y, dy = self._mag12(seed=4)
        fr, p, sols = eebls_transit(t, y, dy, freqs=freqs)
        qv = q_transit(fr)
        ref = _sparse_reference(t, y, dy, fr, 0.5 * qv, 2.0 * qv)
        ok = _untied_frequencies(t, fr)
        assert ok.mean() > 0.7
        assert_allclose(p[ok], ref[ok], rtol=1e-4, atol=1e-7)
        assert _same_peak(p, ref)

    def test_sparse_bls_gpu_offset_invariance(self):
        freqs = self._grid()[::2]
        t, y, dy = self._mag12(seed=5)
        p0, s0 = sparse_bls_gpu(t, y, dy, freqs)
        p20, s20 = sparse_bls_gpu(t, y + 20., dy, freqs)
        assert_allclose(p20, p0, rtol=1e-5, atol=1e-8)
        assert [a[0] for a in s20] == [a[0] for a in s0]

    def test_gpu_one_precise_point_powers_stay_below_one(self):
        # R = 1e6: 401/401 powers > 1 (max 52) before the fix
        r = np.random.RandomState(3)
        N = 200
        t = np.sort(365 * r.rand(N))
        y = 12.0 + 0.01 * r.randn(N)
        dy0 = 0.01 * np.ones(N)
        freqs = np.linspace(0.5, 1.5, 201)
        ok = _untied_frequencies(t, freqs)
        assert ok.mean() > 0.7
        for R, tol in ((1e4, 1e-3), (1e6, 3e-2)):
            dy = dy0.copy()
            dy[17] = 0.01 / np.sqrt(R)
            ref = _sparse_reference(t, y, dy, freqs)
            p, _ = sparse_bls_gpu(t, y, dy, freqs)
            assert np.all(p <= 1.0)
            assert abs(p[ok].max() - ref[ok].max()) < tol * ref[ok].max()
            assert _same_peak(p[ok], ref[ok])


class TestBlsPrecisionDocs(object):
    """Sep 2026 audit section 3.3: the float32 fold limit, the
    time-origin sensitivity of binned power and the run-to-run
    float32-atomic tolerance had to be stated somewhere a user reads.
    CPU-only."""

    @staticmethod
    def _bls_rst():
        here = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(here, 'docs', 'source', 'bls.rst')
        if not os.path.exists(path):
            pytest.skip("docs/source/bls.rst not found (running outside "
                        "the source tree)")
        with open(path, encoding='utf-8') as f:
            return f.read()

    def test_bls_rst_has_precision_section(self):
        rst = self._bls_rst()
        assert 'Precision and reproducibility' in rst
        # the section header must be underlined (valid rst)
        i = rst.index('Precision and reproducibility')
        underline = rst[i:].split('\n')[1]
        assert set(underline) == {'-'}
        assert len(underline) >= len('Precision and reproducibility')
        for phrase in (r'\mathrm{ulp}', r'q_\mathrm{min}',
                       r'n_\mathrm{overlap}', 'float32 atomics',
                       'fractional', 'bitwise'):
            assert phrase in rst, phrase

    def test_eebls_gpu_fast_docstring_mirrors_it(self):
        doc = ' '.join(eebls_gpu_fast.__doc__.split())
        assert 'ulp(T * max(freqs))' in doc
        assert 'qmin / noverlap' in doc
        assert '1e-8 to 1e-7' in doc


class TestFastPathQmaxBox(object):
    """Sep 2026 audit, id 64: the fast (shared-memory) kernels built
    their box ladder as ``max_bin_width = divrndup(nbinsf, nbins0)``
    and looped ``m < max_bin_width``. That is the same set of widths
    whenever ``nbins0`` does not divide ``nbinsf``, but one level short
    when it does, so ``qmax`` itself was never tested: with
    ``qmin=0.025, qmax=0.1`` (nbinsf=40, nbins0=10) the widest box
    searched was ``q = 0.075``, and an on-grid ``q = qmax`` transit was
    recovered at ~74-86 % of its exact power. The bound is now
    ``max_bin_width = nbinsf // nbins0`` with ``m <= max_bin_width``:
    the widest box with ``q = m/nbinsf <= 1/nbins0`` is included and no
    box wider than the discretized ``qmax`` is ever evaluated.
    """

    # ---- CPU: the ladder itself ----

    def test_ladder_includes_the_qmax_box(self):
        from ..bls import _fast_box_widths
        # qmin = 0.025, qmax = 0.1 -> nbinsf = 40, nbins0 = 10
        nb0, nbf = _fast_path_nbins(np.float32([1.0]), 0.025, 0.1)
        assert (int(nb0[0]), int(nbf[0])) == (10, 40)
        widths = _fast_box_widths(int(nbf[0]), int(nb0[0]), 0.3)
        assert widths == [1, 2, 3, 4]
        assert widths[-1] / int(nbf[0]) == 0.1        # == qmax
        # the old ladder stopped at 3 (q = 0.075)
        assert 4 in widths

    def test_helper_docstrings_state_the_floor_bound(self):
        # the bound is floor(nbinsf / nbins0), inclusive -- not the
        # pre-fix ceil(...); _fast_path_nbins' docstring said "ceil"
        # long after the kernels changed.
        from ..bls import _fast_box_widths
        for doc in (_fast_path_nbins.__doc__, _fast_box_widths.__doc__):
            assert 'nbinsf / nbins0' in doc or 'nbinsf // nbins0' in doc
        assert 'ceil(nbinsf /' not in _fast_path_nbins.__doc__.replace(
            '\n', ' ').replace('  ', ' ')
        assert 'floor(nbinsf / nbins0)' in ' '.join(
            _fast_path_nbins.__doc__.split())

    def test_ladder_never_exceeds_the_discretized_qmax(self):
        from ..bls import _fast_box_widths, dnbins
        for dlogq in (0.2, 0.3, 0.5, -1.0):
            for nb0 in range(1, 25):
                for nbf in range(nb0, 220, 7):
                    widths = _fast_box_widths(nbf, nb0, dlogq)
                    assert widths[0] == 1
                    # every searched q is within the discretized qmax
                    assert widths[-1] <= nbf // nb0
                    assert widths[-1] / nbf <= 1.0 / nb0 + 1e-12
                    # ... and it is the LAST rung that fits: the next
                    # step would overshoot
                    assert (widths[-1] + dnbins(widths[-1], dlogq)
                            > nbf // nb0)

    def test_default_bounds_are_unchanged_by_the_fix(self):
        # qmin=0.01, qmax=0.5 -> nbinsf=100, nbins0=2, max width 50,
        # but the geometric step jumps 48 -> 62, so the default fast
        # path searches exactly what it did before.
        from ..bls import _fast_box_widths
        assert _fast_box_widths(100, 2, 0.3)[-1] == 48
        assert _fast_box_widths(100, 2, 0.2)[-1] == 44

    # ---- CPU: the box scan used for the eebls_transit solution pass ----

    @staticmethod
    def _on_grid_box(nbf=40, m=4, n0=10, ndays=10, depth=0.02,
                     sigma=1e-3, seed=17):
        """Light curve whose flux dips in exactly bins ``n0 ..
        n0+m-1`` of an ``nbf``-bin phase grid at f = 1 c/d, i.e. a box
        of q = m/nbf starting at phi0 = n0/nbf."""
        rand = np.random.RandomState(seed)
        phase = (np.arange(nbf) + 0.5) / nbf
        t = np.concatenate([d + phase for d in range(ndays)])
        y = np.ones(len(t))
        b = np.tile(np.arange(nbf), ndays)
        y[(b >= n0) & (b < n0 + m)] -= depth
        y += sigma * rand.randn(len(t))
        dy = sigma * np.ones(len(t))
        return t, y, dy

    @staticmethod
    def _scan_power(t, y, dy, freq, qmin, qmax, dlogq=0.3, noverlap=2):
        """(power, q, phi0) of the fast-kernel box grid at one
        frequency, in the caller's timescale."""
        from ..utils import subtract_epoch
        t64, epoch = subtract_epoch(t)
        w = np.asarray(dy, dtype=np.float64) ** -2
        w /= w.sum()
        ybar = float(np.dot(w, y))
        YY = float(np.dot(w, (np.asarray(y) - ybar) ** 2))
        nb0, nbf = _fast_path_nbins(np.float32([freq]), qmin, qmax)
        val, q, phi = _fast_bls_box_scan(
            t64.astype(np.float32), ((y - ybar) * w).astype(np.float32),
            w.astype(np.float32), np.float32(freq),
            int(nb0[0]), int(nbf[0]), dlogq, noverlap)
        return val / YY, q, (phi + epoch * freq) % 1.0

    def test_box_scan_finds_the_qmax_wide_box(self):
        t, y, dy = self._on_grid_box()
        p, q, phi0 = self._scan_power(t, y, dy, 1.0, 0.025, 0.1)
        # the injected box is exactly qmax wide and on the bin grid
        assert q == pytest.approx(0.1, abs=1e-7)
        assert phi0 == pytest.approx(0.25, abs=1e-6)
        # the widest box the OLD ladder could reach (q = 0.075) leaves
        # a quarter of the transit out and scores clearly lower
        p3 = single_bls(t, y, dy, 1.0, 3. / 40., 0.25)
        p3 = max(p3, single_bls(t, y, dy, 1.0, 3. / 40., 0.275))
        assert p > 1.2 * p3
        # the reported solution reproduces the power exactly
        assert single_bls(t, y, dy, 1.0, q, phi0) == pytest.approx(
            p, rel=1e-5)

    # ---- GPU ----

    def test_fast_kernel_evaluates_the_qmax_box(self):
        # the kernel must agree with the CPU replica of its own grid
        # (which now includes m = nbinsf // nbins0) and must recover
        # the on-grid q = qmax transit at nearly its exact power
        t, y, dy = self._on_grid_box()
        freqs = np.array([1.0], dtype=np.float64)
        p_gpu = eebls_gpu_fast(t, y, dy, freqs, qmin=0.025, qmax=0.1,
                               dlogq=0.3, noverlap=2)
        p_ref, q_ref, phi_ref = self._scan_power(t, y, dy, 1.0,
                                                 0.025, 0.1)
        assert_allclose(p_gpu[0], p_ref, rtol=2e-4, atol=1e-6)
        assert q_ref == pytest.approx(0.1, abs=1e-7)
        exact = single_bls(t, y, dy, 1.0, 0.1, 0.25)
        assert p_gpu[0] > 0.95 * exact

    def test_optimized_kernel_matches_the_standard_one(self):
        t, y, dy = self._on_grid_box(seed=18)
        freqs = np.linspace(0.9, 1.1, 201)
        kw = dict(qmin=0.025, qmax=0.1, dlogq=0.3, noverlap=2)
        p_std = eebls_gpu_fast(t, y, dy, freqs, **kw)
        p_opt = eebls_gpu_fast_optimized(t, y, dy, freqs, **kw)
        assert_allclose(p_opt, p_std, rtol=1e-4, atol=1e-6)

    def test_batch_kernel_uses_the_same_ladder(self):
        # bls_batch.cu carries its own copy of the box loop; it must
        # keep the same widths as the single-LC kernels or the batch
        # periodogram silently differs at the widest box
        from ..bls import eebls_gpu_batch
        t, y, dy = self._on_grid_box(seed=19)
        freqs = np.linspace(0.9, 1.1, 201)
        kw = dict(qmin=0.025, qmax=0.1, dlogq=0.3, noverlap=2)
        p_fast = eebls_gpu_fast(t, y, dy, freqs, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_batch = eebls_gpu_batch([(t, y, dy)], freqs, **kw)[0]
        assert_allclose(p_batch, p_fast, rtol=1e-3, atol=1e-5)


class TestBatchNoverlapValidation(object):
    """Sep 2026 audit, id 75: ``eebls_gpu_batch(noverlap=0)`` computed
    ``n_passes = 1 if fused else noverlap`` and therefore launched
    nothing, returning the untouched device buffer -- all zeros, or a
    stale periodogram when a ``memory=`` was reused -- while
    ``eebls_gpu_fast(noverlap=0)`` raised. A non-integer ``noverlap``
    hit ``range(3.0)`` with a TypeError. The batch entry point now runs
    the same ``_validate_noverlap`` guard as the fast paths."""

    @staticmethod
    def _data():
        rand = np.random.RandomState(23)
        t = np.sort(365. * rand.rand(300))
        y = 1. + 0.01 * rand.randn(300)
        dy = 0.01 * np.ones(300)
        return t, y, dy

    def test_bad_noverlap_raises(self):
        # CPU-runnable: validation precedes any GPU work
        from ..bls import eebls_gpu_batch
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 20)
        for bad in (0, -1, 1.5, 3.0, "2", None):
            with pytest.raises(ValueError, match="noverlap"):
                eebls_gpu_batch([(t, y, dy)], freqs, noverlap=bad)

    def test_valid_noverlap_still_runs(self):
        from ..bls import eebls_gpu_batch
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = eebls_gpu_batch([(t, y, dy)], freqs, noverlap=1)[0]
        assert np.all(np.isfinite(p)) and np.max(p) > 0.


class TestSparseSharedMemoryLimit(object):
    """Sep 2026 audit, ids 77/126: ``sparse_bls_gpu`` sized its dynamic
    shared memory from ``ndata`` and never compared it with the
    device's per-block limit, so anything above ~2,000 points died with
    a bare ``cuLaunchKernel failed: invalid argument``. The size is now
    checked before the launch and reported with the point limit."""

    def test_shared_memory_formula(self):
        from ..bls import _sparse_shared_mem_bytes
        # matches the audit's measurements on a 48 KB device
        assert _sparse_shared_mem_bytes(2000, 64) == 41344
        assert _sparse_shared_mem_bytes(2500, 64) == 69920

    def test_max_ndata_is_the_largest_that_fits(self):
        from ..bls import _sparse_shared_mem_bytes, _sparse_max_ndata
        for lim in (16384, 49152, 65536, 101376):
            for block_size in (32, 64, 256):
                n = _sparse_max_ndata(lim, block_size)
                assert n > 0
                assert _sparse_shared_mem_bytes(n, block_size) <= lim
                assert _sparse_shared_mem_bytes(n + 1, block_size) > lim

    def test_too_many_points_raises_a_clear_error(self):
        rand = np.random.RandomState(29)
        ndata = 6000
        t = np.sort(365. * rand.rand(ndata))
        y = 1. + 0.01 * rand.randn(ndata)
        dy = 0.01 * np.ones(ndata)
        freqs = np.linspace(0.95, 1.05, 5)
        with pytest.raises(ValueError, match="shared memory"):
            sparse_bls_gpu(t, y, dy, freqs)

    def test_small_light_curve_still_runs(self):
        rand = np.random.RandomState(31)
        ndata = 200
        t = np.sort(365. * rand.rand(ndata))
        y = 1. + 0.01 * rand.randn(ndata)
        dy = 0.01 * np.ones(ndata)
        freqs = np.linspace(0.95, 1.05, 25)
        p, sols = sparse_bls_gpu(t, y, dy, freqs)
        assert np.all(np.isfinite(p)) and len(sols) == len(freqs)


class TestBLSMemoryKeywords(object):
    """Sep 2026 audit, id 67: ``BLSMemory.fromdata`` read
    ``max_ndata``/``max_nfreqs`` with ``kwargs.get`` and then forwarded
    the same ``kwargs`` to ``__init__``, so passing either raised
    ``TypeError: got multiple values for argument``. Reusing a memory
    with a different number of frequencies used to fail deep inside
    pycuda with ``ary and self must be the same size``."""

    @staticmethod
    def _data(ndata=200):
        rand = np.random.RandomState(37)
        t = np.sort(365. * rand.rand(ndata))
        y = 1. + 0.01 * rand.randn(ndata)
        dy = 0.01 * np.ones(ndata)
        return t, y, dy

    def test_fromdata_accepts_max_ndata_and_max_nfreqs(self):
        from ..bls import BLSMemory
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 50)
        mem = BLSMemory.fromdata(t, y, dy, qmin=1e-2, qmax=0.5,
                                 freqs=freqs, transfer=True,
                                 max_ndata=len(t) + 100,
                                 max_nfreqs=1000)
        assert mem.max_ndata == len(t) + 100
        assert mem.max_nfreqs == 1000
        assert len(mem.t) == len(t) + 100

    def test_reuse_with_a_different_nfreqs_raises_clearly(self):
        from ..bls import BLSMemory
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 50)
        mem = BLSMemory.fromdata(t, y, dy, qmin=1e-2, qmax=0.5,
                                 freqs=freqs, transfer=True)
        # same length: fine
        mem.setdata(t, y, dy, qmin=1e-2, qmax=0.5,
                    freqs=freqs + 0.01, transfer=True)
        with pytest.raises(ValueError, match="frequencies"):
            mem.setdata(t, y, dy, qmin=1e-2, qmax=0.5,
                        freqs=np.linspace(0.95, 1.05, 120),
                        transfer=True)


class TestKernelCompileCaching(object):
    """Sep 2026 audit, ids 3/7/43/60/126 (plan item BLS-1).

    ``eebls_gpu``, ``eebls_gpu_custom``, ``hone_solution`` and
    ``sparse_bls_gpu`` used to call ``compile_bls`` /
    ``compile_sparse_bls`` directly whenever the caller did not supply
    kernels, bypassing the LRU cache the fast/batch paths use. pycuda
    runs an ``nvcc --preprocess`` subprocess on every ``SourceModule``
    even when its own disk cache holds the cubin, so that cost ~0.4-0.5 s
    (standard) and ~0.4-1.6 s (sparse) *per call*.

    These tests assert compile *counts*, never wall times.
    """

    @staticmethod
    def _data(ndata=100):
        rand = np.random.RandomState(11)
        t = np.sort(100. * rand.rand(ndata))
        y = 1. + 0.01 * rand.randn(ndata)
        dy = 0.01 * np.ones(ndata)
        return t, y, dy

    @staticmethod
    def _counting(monkeypatch, name):
        """Swap in a fresh kernel cache and count real compiles."""
        from collections import OrderedDict
        from .. import bls as B
        monkeypatch.setattr(B, '_kernel_cache', OrderedDict())
        calls = []
        orig = getattr(B, name)

        def counted(*args, **kwargs):
            calls.append((args, tuple(sorted(kwargs.items()))))
            return orig(*args, **kwargs)

        monkeypatch.setattr(B, name, counted)
        return calls

    def test_sparse_bls_gpu_compiles_once_per_block_size(self, monkeypatch):
        calls = self._counting(monkeypatch, 'compile_sparse_bls')
        t, y, dy = self._data()
        freqs = np.linspace(0.9, 1.1, 30)

        p1, _ = sparse_bls_gpu(t, y, dy, freqs)
        assert len(calls) == 1
        p2, _ = sparse_bls_gpu(t, y, dy, freqs)
        assert len(calls) == 1, "second call recompiled the sparse kernel"
        # identical kernel, identical numbers
        assert np.array_equal(p1, p2)

        # a different block_size is a different kernel: compile again
        sparse_bls_gpu(t, y, dy, freqs, block_size=32)
        assert len(calls) == 2
        sparse_bls_gpu(t, y, dy, freqs, block_size=32)
        assert len(calls) == 2

    def test_eebls_transit_sparse_path_shares_the_cached_kernel(
            self, monkeypatch):
        calls = self._counting(monkeypatch, 'compile_sparse_bls')
        t, y, dy = self._data()
        freqs = np.linspace(0.9, 1.1, 30)
        qvals = q_transit(freqs)
        for _ in range(3):
            eebls_transit(t, y, dy, freqs=freqs, qvals=qvals,
                          use_sparse=True)
        assert len(calls) == 1

    def test_eebls_gpu_compiles_once(self, monkeypatch):
        calls = self._counting(monkeypatch, 'compile_bls')
        t, y, dy = self._data()
        freqs = np.linspace(0.9, 1.1, 20)

        p1, _ = eebls_gpu(t, y, dy, freqs)
        assert len(calls) == 1
        p2, _ = eebls_gpu(t, y, dy, freqs)
        assert len(calls) == 1, "second call recompiled the BLS kernels"
        # same kernels, same numbers (eebls_gpu's multi-stream global
        # atomics are not bit-reproducible run to run, hence allclose)
        assert_allclose(p1, p2, rtol=1e-5, atol=1e-7)

        # eebls_gpu_custom asks for the same (block_size, use_optimized,
        # function_names) key: still one compile
        eebls_gpu_custom(t, y, dy, freqs, np.array([0.05, 0.1]),
                         np.array([0.0, 0.5]))
        assert len(calls) == 1

        # a different block_size must recompile
        eebls_gpu(t, y, dy, freqs, block_size=128)
        assert len(calls) == 2
        eebls_gpu(t, y, dy, freqs, block_size=128)
        assert len(calls) == 2

    def test_prepare_false_bypasses_the_cache(self, monkeypatch):
        # prepare=False returns unprepared functions, which the cache
        # key does not model: it must fall through to a direct compile
        # every time rather than hand back prepared kernels.
        from .. import bls as B
        calls = self._counting(monkeypatch, 'compile_bls')
        fns1 = B._cached_compile_bls(prepare=False)
        assert len(calls) == 1
        fns2 = B._cached_compile_bls(prepare=False)
        assert len(calls) == 2
        assert fns1 is not fns2
        # ... while the default (prepare=True) is cached and shared
        c1 = B._cached_compile_bls()
        c2 = B._cached_compile_bls()
        assert c1 is c2
        assert len(calls) == 3


class TestAdaptiveUsesFusedKernel(object):
    """Sep 2026 audit, ids 40/63 (plan item BLS-5).

    ``eebls_gpu_fast_adaptive`` and ``eebls_transit(use_optimized=True)``
    loaded a function dict without ``full_bls_no_sol_fused``, so the
    shared implementation could only take the ``noverlap``-pass loop:
    two launches and 1.7-2.3x the GPU time of ``eebls_gpu_fast`` on
    identical inputs. They must now take the fused kernel wherever it is
    valid (power-of-two ``noverlap``, ``dphi == 0``, shared memory
    permitting) and keep the multi-pass fallback otherwise.
    """

    @staticmethod
    def _data():
        return data(snr=30, q=0.05, phi0=0.317, freq=1.0,
                    baseline=365., ndata=300)

    @staticmethod
    def _launch_counter(monkeypatch):
        """Count prepared launches by kernel name."""
        from .. import bls as B
        counts = {}

        class Spy(object):
            def __init__(self, name, func):
                self._name, self._func = name, func

            def prepared_call(self, *a, **k):
                counts[self._name] = counts.get(self._name, 0) + 1
                return self._func.prepared_call(*a, **k)

            def prepared_async_call(self, *a, **k):
                counts[self._name] = counts.get(self._name, 0) + 1
                return self._func.prepared_async_call(*a, **k)

            def __getattr__(self, k):
                return getattr(self._func, k)

        orig = B._get_cached_kernels

        def spied(*a, **k):
            return {name: Spy(name, f) for name, f in orig(*a, **k).items()}

        monkeypatch.setattr(B, '_get_cached_kernels', spied)
        return counts

    def test_adaptive_launches_the_fused_kernel_once(self, monkeypatch):
        from ..bls import eebls_gpu_fast_adaptive
        counts = self._launch_counter(monkeypatch)
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 200)

        eebls_gpu_fast_adaptive(t, y, dy, freqs, qmin=0.01, qmax=0.1,
                                noverlap=2)
        assert counts == {'full_bls_no_sol_fused': 1}, counts

    def test_transit_use_optimized_launches_the_fused_kernel_once(
            self, monkeypatch):
        counts = self._launch_counter(monkeypatch)
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 200)

        eebls_transit(t, y, dy, freqs=freqs, qvals=q_transit(freqs),
                      use_optimized=True, use_sparse=False, noverlap=2)
        assert counts == {'full_bls_no_sol_fused': 1}, counts

    @pytest.mark.parametrize("kw", [dict(noverlap=3), dict(dphi=0.25)])
    def test_adaptive_falls_back_when_fused_is_invalid(self, kw,
                                                       monkeypatch):
        # non-power-of-two noverlap / a non-zero base phase offset are
        # outside what the fused kernel implements
        from ..bls import eebls_gpu_fast_adaptive
        counts = self._launch_counter(monkeypatch)
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 200)

        eebls_gpu_fast_adaptive(t, y, dy, freqs, qmin=0.01, qmax=0.1,
                                **kw)
        assert 'full_bls_no_sol_fused' not in counts, counts
        assert counts.get('full_bls_no_sol_optimized', 0) >= 2, counts

    def test_fused_and_multipass_agree(self):
        # Parity of the two paths at the default noverlap: hand the
        # adaptive entry point a function dict WITHOUT the fused kernel
        # to force the multi-pass loop.
        from .. import bls as B
        from ..bls import eebls_gpu_fast_adaptive, eebls_gpu_fast
        t, y, dy = self._data()
        freqs = np.linspace(0.95, 1.05, 500)
        bs = B._choose_block_size(len(t))
        multi = B._get_cached_kernels(bs, True,
                                      ['full_bls_no_sol_optimized'])
        assert 'full_bls_no_sol_fused' not in multi

        kw = dict(qmin=0.01, qmax=0.1, block_size=bs)
        p_fused = eebls_gpu_fast_adaptive(t, y, dy, freqs, **kw)
        p_multi = eebls_gpu_fast_adaptive(t, y, dy, freqs,
                                          functions=multi, **kw)
        assert int(np.argmax(p_fused)) == int(np.argmax(p_multi))
        assert_allclose(p_fused, p_multi, rtol=1e-4, atol=1e-5)

        # and the adaptive path now returns what eebls_gpu_fast (fused
        # since 1.0) returns, to float32 atomic-ordering noise
        p_fast = eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.1)
        assert_allclose(p_fused, p_fast, rtol=1e-4, atol=1e-6)


class TestBLSMemoryHostStaging(object):
    """Sep 2026 audit, id 41 (plan item BLS-6).

    ``BLSMemory.allocate_host_arrays`` page-locked all six host buffers,
    three of which are never the source or destination of an async copy
    (``nbins0``/``nbinsf`` are replaced by fresh pageable arrays in
    ``setdata``; ``bls`` is only an async destination when a stream is
    attached).  And the ``memory=None`` fast path allocated a whole
    ``BLSMemory`` -- six host buffers plus four device buffers -- per
    call, so back-to-back calls of the same shape re-paid it every time.
    """

    @staticmethod
    def _data(ndata=200, seed=17):
        rand = np.random.RandomState(seed)
        t = np.sort(365. * rand.rand(ndata))
        y = 1. + 0.01 * rand.randn(ndata)
        dy = 0.01 * np.ones(ndata)
        return t, y, dy

    def test_only_transfer_buffers_are_page_locked(self):
        import pycuda.driver as cuda
        from ..bls import BLSMemory
        mem = BLSMemory(64, 128)
        pinned = cuda.pagelocked_empty(1, np.float32).base.__class__
        for attr in ('t', 'yw', 'w'):
            assert isinstance(getattr(mem, attr).base, pinned), attr
        for attr in ('bls', 'nbins0', 'nbinsf'):
            assert not isinstance(getattr(mem, attr).base, pinned), attr

    def test_result_buffer_is_page_locked_with_a_stream(self):
        # get_async into a pageable buffer is not asynchronous, and the
        # normalization after it would race the DMA (see
        # TestPinnedBufferStreamParity)
        import pycuda.driver as cuda
        from ..core import ensure_context
        from ..bls import BLSMemory
        ensure_context()
        mem = BLSMemory(64, 128, stream=cuda.Stream())
        pinned = cuda.pagelocked_empty(1, np.float32).base.__class__
        assert isinstance(mem.bls.base, pinned)

    def test_pooled_memory_stages_identical_bytes(self):
        # The pool must never hand back another light curve's data: the
        # staged host buffers, the uploaded device buffers and the
        # normalization scalars have to equal what a freshly-allocated
        # memory produces, bit for bit.
        from .. import bls as B
        from ..bls import BLSMemory
        freqs = np.linspace(0.95, 1.05, 64)
        B._memory_pool_tls.pool = None
        for seed in (1, 2, 3):
            t, y, dy = self._data(seed=seed)
            pooled = B._pooled_bls_memory(t, y, dy, 1e-2, 0.5, freqs, {})
            fresh = BLSMemory.fromdata(t, y, dy, qmin=1e-2, qmax=0.5,
                                       freqs=freqs, transfer=True)
            for attr in ('t', 'yw', 'w', 'freqs', 'nbins0', 'nbinsf'):
                assert np.array_equal(np.asarray(getattr(pooled, attr)),
                                      np.asarray(getattr(fresh, attr))), attr
            for attr in ('t_g', 'yw_g', 'w_g', 'freqs_g', 'nbins0_g',
                         'nbinsf_g'):
                assert np.array_equal(getattr(pooled, attr).get(),
                                      getattr(fresh, attr).get()), attr
            for attr in ('yy', 'chi2_0', 'ybar', 'epoch'):
                assert getattr(pooled, attr) == getattr(fresh, attr), attr
        B._memory_pool_tls.pool = None

    def test_pool_reuses_one_memory_per_shape(self):
        from .. import bls as B
        freqs = np.linspace(0.95, 1.05, 64)
        B._memory_pool_tls.pool = None
        t, y, dy = self._data()
        m1 = B._pooled_bls_memory(t, y, dy, 1e-2, 0.5, freqs, {})
        m2 = B._pooled_bls_memory(t, y, dy, 1e-2, 0.5, freqs, {})
        assert m1 is m2
        # a different ndata is a different entry
        t2, y2, dy2 = self._data(ndata=100)
        m3 = B._pooled_bls_memory(t2, y2, dy2, 1e-2, 0.5, freqs, {})
        assert m3 is not m1
        # ... and a different number of frequencies too (the device
        # frequency arrays keep their first size)
        f2 = np.linspace(0.95, 1.05, 128)
        m4 = B._pooled_bls_memory(t, y, dy, 1e-2, 0.5, f2, {})
        assert m4 is not m1
        assert len(m4.freqs_g) == 128
        B._memory_pool_tls.pool = None

    def test_pooled_and_unpooled_results_agree(self):
        # Interleave three different light curves through the pool and
        # compare against the allocate-per-call path; also check that
        # holding an earlier result across later calls is safe (the
        # returned array must not alias a pooled buffer).
        from .. import bls as B
        freqs = np.linspace(0.95, 1.05, 300)
        lcs = [data(snr=30, q=0.05, phi0=0.317, freq=1.0, baseline=365.,
                    ndata=300, seed=s) for s in (11, 12, 13)]

        old = B._MEMORY_POOL_MAX_SIZE
        try:
            B._MEMORY_POOL_MAX_SIZE = 0
            B._memory_pool_tls.pool = None
            ref = [eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.1)
                   for (t, y, dy) in lcs]
            B._MEMORY_POOL_MAX_SIZE = 2
            B._memory_pool_tls.pool = None
            got = [eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.1)
                   for (t, y, dy) in lcs]
        finally:
            B._MEMORY_POOL_MAX_SIZE = old
            B._memory_pool_tls.pool = None

        for a, b in zip(got, ref):
            assert int(np.argmax(a)) == int(np.argmax(b))
            assert_allclose(a, b, rtol=1e-4, atol=1e-6)
        # distinct light curves must give distinct periodograms (a pool
        # bug that reused stale data would make these equal)
        assert not np.allclose(got[0], got[1], rtol=1e-3)

    def test_pool_is_skipped_when_it_would_be_visible(self):
        # A stream-attached call hands back the pinned bls buffer, and
        # transfer_to_host=False hands back the raw buffer: neither may
        # come from the pool.
        import pycuda.driver as cuda
        from ..core import ensure_context
        from .. import bls as B
        ensure_context()
        t, y, dy = self._data(ndata=300)
        freqs = np.linspace(0.95, 1.05, 100)
        B._memory_pool_tls.pool = None
        eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.1,
                       stream=cuda.Stream())
        assert not getattr(B._memory_pool_tls, 'pool', None)
        eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.1,
                       transfer_to_host=False)
        assert not getattr(B._memory_pool_tls, 'pool', None)
        # the ordinary call does use it
        eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.1)
        assert len(B._memory_pool_tls.pool) == 1
        B._memory_pool_tls.pool = None


class TestNoBlasThreadpoolInPrologues(object):
    """Sep 2026 audit, id 45 (plan item BLS-8).

    ``np.dot`` on a long float vector goes to BLAS, which spawns a full
    threadpool; on CPU-quota-limited containers (RunPod, Kubernetes) the
    burst trips CFS throttling and stalls the process. The July 2026 work
    moved ``BLSMemory.setdata`` and ``_chi2_null`` to ``np.einsum``; the
    per-light-curve prologues of ``eebls_gpu``, ``eebls_gpu_custom``,
    ``single_bls`` and ``sparse_bls_cpu`` were still on ``np.dot``.
    Measured on the pod at ndata = 20000: the prologue's median went
    0.60 -> 0.17 ms with a 98 ms tail and 12 CFS throttle events per 50
    calls going to none, and ``single_bls`` 93.9 -> 0.7 ms.

    Source-level guard (there is no timing assertion anywhere here) plus
    a check that the change is a summation-order change only.
    """

    @pytest.mark.parametrize("name", ['eebls_gpu', 'eebls_gpu_custom',
                                      'single_bls', 'sparse_bls_cpu'])
    def test_prologue_does_not_call_np_dot(self, name):
        import inspect
        from .. import bls as B
        src = inspect.getsource(getattr(B, name))
        # comments mention np.dot on purpose; look at the code only
        code = '\n'.join(line.split('#')[0] for line in src.splitlines())
        assert 'np.dot' not in code, (
            "%s reintroduced np.dot: use np.einsum('i,i->', ...) so the "
            "per-light-curve prologue stays off the BLAS threadpool"
            % name)
        assert "np.einsum('i,i->'" in code

    def test_einsum_and_dot_agree_to_rounding(self):
        # The replacement is the same mathematical reduction in a
        # different summation order: a few float64 ulps.
        rand = np.random.RandomState(3)
        for ndata in (150, 2000, 20000):
            y = 1. + 0.01 * rand.randn(ndata)
            dy = 0.01 * np.ones(ndata)
            w = np.power(dy, -2.)
            w /= np.sum(w)
            ybar_dot = np.dot(w, y)
            ybar_ein = float(np.einsum('i,i->', w, y))
            assert abs(ybar_dot - ybar_ein) <= 64 * np.spacing(abs(ybar_ein))
            yy_dot = np.dot(w, np.power(y - ybar_dot, 2))
            yy_ein = float(np.einsum('i,i->', w, np.power(y - ybar_ein, 2)))
            assert abs(yy_dot - yy_ein) <= 64 * np.spacing(abs(yy_ein))

    def test_sparse_bls_cpu_still_matches_the_gpu_kernel(self):
        # sparse_bls_cpu is the CPU reference for sparse_bls_gpu; the
        # reordered sums must not move it away from the kernel.
        t, y, dy = data(snr=30, q=0.05, phi0=0.317, freq=1.0,
                        baseline=365., ndata=120)
        freqs = np.linspace(0.95, 1.05, 60)
        p_cpu, s_cpu = sparse_bls_cpu(t, y, dy, freqs)
        p_gpu, s_gpu = sparse_bls_gpu(t, y, dy, freqs)
        assert int(np.argmax(p_cpu)) == int(np.argmax(p_gpu))
        assert_allclose(p_cpu, p_gpu, rtol=1e-4, atol=1e-6)


class TestPerFrequencyHostWork(object):
    """Sep 2026 audit, ids 136/137 (plan item BLS-9).

    Three per-call host costs that were pure overhead:
    the per-frequency solution re-phasing comprehension (39 ms at 60,121
    frequencies, 74 ms at 117,403 -- more than the GPU work it followed),
    ``_chi2_null``'s second full pass over the light curve in
    ``BLSMemory.setdata`` when ``chi2_0`` follows from ``yy``, and
    ``conflict_scatter_perm`` rebuilt on every ``setdata`` although it is
    a pure function of ``ndata``.
    """

    @staticmethod
    def _lc(ndata=300, seed=5):
        rand = np.random.RandomState(seed)
        t = np.sort(365. * rand.rand(ndata)) + 2455197.5
        y = 1. + 0.01 * rand.randn(ndata)
        dy = 0.01 * np.ones(ndata)
        return t, y, dy

    @pytest.mark.parametrize("freqs_kind",
                             ['float64', 'float32', 'list', 'np_scalars'])
    def test_rephasing_matches_the_per_frequency_loop(self, freqs_kind):
        from ..bls import _rephase_solutions
        rand = np.random.RandomState(4)
        n = 500
        q = rand.rand(n).astype(np.float32)
        phi = rand.rand(n).astype(np.float32)
        base = np.linspace(0.01, 2.0, n)
        freqs = {'float64': base,
                 'float32': base.astype(np.float32),
                 'list': [float(x) for x in base],
                 'np_scalars': list(base)}[freqs_kind]
        # epoch is np.float64 everywhere in the package (subtract_epoch
        # returns np.floor(np.min(t))), which is what keeps the
        # expression in float64 for a float32 grid.
        epoch = np.float64(2455197.0)

        old = [(a, (b + (epoch * f)) % 1.0)
               for (a, b), f in zip(list(zip(q, phi)), freqs)]
        new = _rephase_solutions(q, phi, epoch, freqs)
        assert len(new) == len(old)
        assert np.array_equal(np.asarray(old, dtype=np.float64),
                              np.asarray(new, dtype=np.float64))
        # and it is the exact float64 answer
        exact = (phi.astype(np.float64)
                 + epoch * np.asarray(freqs, dtype=np.float64)) % 1.0
        assert np.array_equal(np.array([x[1] for x in new]), exact)

    def test_setdata_chi2_0_matches_the_two_pass_form(self):
        from ..bls import BLSMemory, _chi2_null
        freqs = np.linspace(0.95, 1.05, 64)
        for ndata in (150, 2000):
            t, y, dy = self._lc(ndata=ndata)
            mem = BLSMemory.fromdata(t, y, dy, qmin=1e-2, qmax=0.5,
                                     freqs=freqs, transfer=True)
            # chi2_0 = yy * sum(dy**-2): the same weighted sum of
            # squares with un-normalized weights
            assert_allclose(mem.chi2_0, _chi2_null(y, dy), rtol=1e-12)
            assert_allclose(mem.chi2_0,
                            mem.yy * np.sum(np.power(dy, -2.)), rtol=1e-14)

    def test_setdata_chi2_0_with_float32_inputs(self):
        # float32 y/dy make yy (and hence chi2_0) a float32-accumulated
        # sum where _chi2_null forced float64; the difference is ~1
        # float32 ulp, well inside the data's own precision.
        from ..bls import BLSMemory, _chi2_null
        freqs = np.linspace(0.95, 1.05, 64)
        t, y, dy = self._lc(ndata=2000)
        y = y.astype(np.float32)
        dy = dy.astype(np.float32)
        mem = BLSMemory.fromdata(t, y, dy, qmin=1e-2, qmax=0.5,
                                 freqs=freqs, transfer=True)
        assert_allclose(mem.chi2_0, _chi2_null(y, dy), rtol=1e-5)

    def test_scatter_perm_cache(self):
        from ..bls import _cached_conflict_scatter_perm
        from ..utils import conflict_scatter_perm
        for n in (63, 64, 150, 2000):
            cached = _cached_conflict_scatter_perm(n)
            direct = conflict_scatter_perm(n)
            if direct is None:
                assert cached is None
                continue
            assert np.array_equal(cached, direct)
            # same object on the second call, and read-only so a caller
            # cannot corrupt the shared permutation
            assert _cached_conflict_scatter_perm(n) is cached
            assert not cached.flags.writeable

    def test_conventions_still_consistent(self):
        # chi2_0 feeds the 'snr'/'loglik' conversions
        t, y, dy = self._lc(ndata=400)
        freqs = np.linspace(0.95, 1.05, 120)
        p = eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.1)
        p_snr = eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.1,
                               convention='snr')
        p_ll = eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.1,
                              convention='loglik')
        w = np.power(dy, -2.)
        ybar = float(np.einsum('i,i->', w, y)) / np.sum(w)
        chi2_0 = float(np.einsum('i,i->', w, (np.asarray(y) - ybar) ** 2))
        assert_allclose(p_snr, np.sqrt(chi2_0 * p), rtol=1e-5, atol=1e-6)
        assert_allclose(p_ll, 0.5 * chi2_0 * p, rtol=1e-5, atol=1e-6)


class TestAdaptiveBlockSize(object):
    """``eebls_gpu_fast_adaptive`` picks the CUDA block size from
    ``ndata`` (ported from ``scripts/test_adaptive_correctness.py``).
    The heuristic is CPU-checkable; the parity of the adaptive wrapper
    with ``eebls_gpu_fast_optimized`` at the same block size runs on a
    device (the conftest skips it on CPU-only hosts)."""

    EXPECTED = [(2, 32), (10, 32), (32, 32), (33, 64), (50, 64), (64, 64),
                (65, 128), (100, 128), (128, 128), (129, 256), (500, 256),
                (65536, 256)]

    @pytest.mark.parametrize("ndata,expected", EXPECTED)
    def test_choose_block_size(self, ndata, expected):
        from ..bls import _choose_block_size
        bs = _choose_block_size(ndata)
        assert bs == expected
        assert bs in (32, 64, 128, 256)

    def test_choose_block_size_is_monotonic(self):
        from ..bls import _choose_block_size
        sizes = [_choose_block_size(n) for n in range(2, 600)]
        assert all(a <= b for a, b in zip(sizes, sizes[1:]))
        assert set(sizes) == {32, 64, 128, 256}

    @staticmethod
    def _lc(ndata, seed=42):
        rand = np.random.RandomState(seed)
        t = np.sort(rand.uniform(0, 100, ndata))
        period, depth = 5.0, 0.01
        phase = (t % period) / period
        y = np.ones(ndata) - depth * ((phase > 0.4) & (phase < 0.5))
        y += rand.normal(0, 0.01, ndata)
        dy = 0.01 * np.ones(ndata)
        return t, y, dy

    @pytest.mark.parametrize("ndata", [10, 50, 100, 500])
    def test_adaptive_matches_optimized_at_the_chosen_block_size(self,
                                                                 ndata):
        # GPU only. The adaptive wrapper is eebls_gpu_fast_optimized with
        # block_size=_choose_block_size(ndata) and the same cached
        # kernel set; results must agree to float32 rounding (the
        # fold/bin arithmetic is identical, only the launch shape
        # differs) and the peak must be the same grid point.
        from ..bls import (eebls_gpu_fast_adaptive, eebls_gpu_fast_optimized,
                           _choose_block_size)
        t, y, dy = self._lc(ndata)
        freqs = np.linspace(0.05, 0.5, 100)
        bs = _choose_block_size(ndata)
        p_adaptive = eebls_gpu_fast_adaptive(t, y, dy, freqs)
        p_fixed = eebls_gpu_fast_optimized(t, y, dy, freqs, block_size=bs)
        assert p_adaptive.shape == freqs.shape
        assert np.all(np.isfinite(p_adaptive))
        assert_allclose(p_adaptive, p_fixed, rtol=1e-5, atol=1e-6)
        assert np.argmax(p_adaptive) == np.argmax(p_fixed)


class TestHostLadderMirrorsDevice(object):
    """The host q ladder (``dnbins`` and everything built on it:
    ``count_tot_nbins`` sizing ``eebls_gpu``'s bin rows, and
    ``_fast_box_widths`` replicating the fast kernels' box grid) must
    agree with the device ladder rung for rung. The kernels form
    ``floorf(dlogq * nbins)`` in float32 (``dlogq`` is a ``float``
    kernel argument), and the old float64 host arithmetic disagreed for
    e.g. ``dlogq = 0.65, nbins = 180`` (117.0 vs floorf(116.99999) =
    116), so ``eebls_gpu(dlogq=0.65)`` could under-size a frequency's
    row (host 180 cells, device 476) and the fold kernel's atomics ran
    into the next row (Sep 2026 fresh-eyes review, finding 26)."""

    DLOGQS = [round(0.05 * k, 2) for k in range(2, 21)]   # 0.1 .. 1.0

    @staticmethod
    def _device_dnbins(nbins, dlogq):
        # bls_common.cuh: `unsigned int n = (unsigned int) floorf(dlogq
        # * nbins); return (n == 0) ? 1 : n;` with float dlogq and
        # unsigned int nbins (exact in float32 below 2^24)
        if dlogq < 0:
            return 1
        n = int(np.floor(np.float32(dlogq) * np.float32(nbins)))
        return n if n > 0 else 1

    @pytest.mark.parametrize("dlogq", DLOGQS)
    def test_dnbins_matches_the_float32_device_arithmetic(self, dlogq):
        from ..bls import dnbins
        nb = np.arange(1, 200001)
        f32 = np.floor(np.float32(dlogq) * nb.astype(np.float32))
        f64 = np.floor(dlogq * nb.astype(np.float64))
        # every nbins where float32 and float64 disagree, plus a sample
        # of those where they agree (the whole range would be 200000
        # scalar calls per dlogq)
        differ = nb[f32 != f64]
        same = nb[f32 == f64][::997]
        for n in np.concatenate([differ, same]):
            assert dnbins(int(n), dlogq) == self._device_dnbins(int(n),
                                                                dlogq)
        if dlogq in (0.2, 0.3):
            # the defaults of eebls_gpu / the fast paths: the fix must
            # not move a single rung there
            assert len(differ) == 0
        elif dlogq in (0.35, 0.65, 0.7):
            # the values where the review found the divergence
            assert len(differ) > 0

    def test_count_tot_nbins_matches_the_device_count(self):
        # the review's cases: (nbins0, nbinsf, dlogq) -> device count
        from ..bls import count_tot_nbins

        def device_count(nb0, nbf, dlogq):
            tot, nb = 0, nb0
            while nb <= nbf:
                tot += nb
                nb += self._device_dnbins(nb, dlogq)
            return tot

        for nb0, nbf, dlogq, expect in [(180, 296, 0.65, 476),
                                        (180, 243, 0.35, 423),
                                        (90, 153, 0.7, 243)]:
            assert device_count(nb0, nbf, dlogq) == expect
            assert count_tot_nbins(nb0, nbf, dlogq) == expect
        # and the defaults are what they always were
        assert count_tot_nbins(2, 100, 0.2) == 2 + 3 + 4 + 5 + 6 + 7 + \
            8 + 9 + 10 + 12 + 14 + 16 + 19 + 22 + 26 + 31 + 37 + 44 + \
            52 + 62 + 74 + 88

    def test_fast_box_widths_matches_the_device_ladder(self):
        from ..bls import _fast_box_widths
        # (1, 6000, 0.53) is a pair where the old float64 host ladder
        # and the device's float32 ladder differ, so this test fails on
        # the pre-fix code (the smaller pairs happen to agree there)
        for dlogq in (0.35, 0.65, 0.7, 0.3, 0.53):
            for nb0, nbf in [(1, 180), (1, 340), (2, 360), (1, 90),
                             (1, 6000)]:
                widths = _fast_box_widths(nbf, nb0, dlogq)
                m, expect = 1, []
                while m <= nbf // nb0:
                    expect.append(m)
                    m += self._device_dnbins(m, dlogq)
                assert widths == expect


class TestFastSolutionLadderMatchesKernel(object):
    """``eebls_transit``'s top-K ``(q, phi)`` re-scan must walk the SAME
    bin ladder the kernel searched. ``BLSMemory.setdata`` computes the
    kernel's ``nbins0`` / ``nbinsf`` with ``_fast_path_nbins`` on the
    bounds as passed (their own dtype); ``_fast_bls_solutions`` used to
    promote them to float64 first, so for float32 ``qvals`` (the
    documented override, e.g. ``keplerian_freq_grid(return_qvals=True)``
    output) the two ladders were one bin apart at some frequencies and
    the reported box was one the kernel never evaluated (Sep 2026
    fresh-eyes review, finding 18)."""

    ROUND_Q32 = np.float32([0.025, 0.05, 1. / 7., 0.1, 0.2, 1. / 9.,
                            0.03, 0.07])

    @staticmethod
    def _kernel_ladder(freqs, qmin, qmax):
        # exactly BLSMemory.setdata: `self.freqs = np.asarray(freqs)
        # .astype(self.rtype)`; `_fast_path_nbins(self.freqs, qmin, qmax)`
        return _fast_path_nbins(np.asarray(freqs).astype(np.float32),
                                qmin, qmax)

    @staticmethod
    def _record_solution_ladder(monkeypatch, *args, **kwargs):
        """Run _fast_bls_solutions and return the (nbins0, nbinsf) it
        derived, captured from its _fast_path_nbins call."""
        import cuvarbase.bls as bls_mod
        seen = []
        real = bls_mod._fast_path_nbins

        def recorder(freqs32, qmin, qmax):
            out = real(freqs32, qmin, qmax)
            seen.append(out)
            return out

        monkeypatch.setattr(bls_mod, '_fast_path_nbins', recorder)
        sols = bls_mod._fast_bls_solutions(*args, **kwargs)
        assert len(seen) == 1
        return sols, seen[0]

    @staticmethod
    def _lc(n=300, seed=4):
        rand = np.random.RandomState(seed)
        t = np.sort(30. * rand.rand(n))
        y = 1. + 1e-3 * rand.randn(n)
        dy = 1e-3 * np.ones(n)
        return t, y, dy

    def test_float32_bounds_use_the_uploaded_ladder(self, monkeypatch):
        from ..bls import _broadcast_q_bound
        t, y, dy = self._lc()
        q32 = self.ROUND_Q32
        freqs = np.linspace(0.5, 1.5, len(q32))
        qmins, qmaxes = q32 * 0.5, q32 * 2.0      # as eebls_transit forms them
        assert qmins.dtype == np.float32 and qmaxes.dtype == np.float32

        nb0_k, nbf_k = self._kernel_ladder(freqs, qmins, qmaxes)
        _, (nb0_s, nbf_s) = self._record_solution_ladder(
            monkeypatch, t, y, dy, freqs, np.ones(len(freqs)),
            qmins, qmaxes, len(freqs))
        assert np.array_equal(nb0_s, nb0_k)
        assert np.array_equal(nbf_s, nbf_k)

        # ... and the test bites: the float64-promoted ladder the old
        # code walked differs at some of these 'round' float32 values
        nb0_p, nbf_p = _fast_path_nbins(
            freqs.astype(np.float32),
            _broadcast_q_bound(qmins, len(freqs), 1e-2, 'qmin'),
            _broadcast_q_bound(qmaxes, len(freqs), 0.5, 'qmax'))
        assert np.any(nbf_p != nbf_k) and np.any(nb0_p != nb0_k)

    def test_float64_default_path_is_unchanged(self, monkeypatch):
        # the default eebls_transit path (float64 qvals from
        # transit_autofreq): promoting to float64 was the identity, so
        # the ladder is bit-identical before and after the fix, and
        # identical to the kernel's
        from ..bls import _broadcast_q_bound
        t, y, dy = self._lc()
        freqs, q0 = transit_autofreq(t, fmin=0.2, fmax=2.0)
        freqs, q0 = freqs[::50], q0[::50]
        qmins, qmaxes = q0 * 0.5, q0 * 2.0
        assert qmins.dtype == np.float64
        nb0_k, nbf_k = self._kernel_ladder(freqs, qmins, qmaxes)
        nb0_old, nbf_old = _fast_path_nbins(
            freqs.astype(np.float32),
            _broadcast_q_bound(qmins, len(freqs), 1e-2, 'qmin'),
            _broadcast_q_bound(qmaxes, len(freqs), 0.5, 'qmax'))
        _, (nb0_s, nbf_s) = self._record_solution_ladder(
            monkeypatch, t, y, dy, freqs, np.ones(len(freqs)),
            qmins, qmaxes, len(freqs))
        for a in (nb0_old, nb0_s):
            assert np.array_equal(a, nb0_k)
        for a in (nbf_old, nbf_s):
            assert np.array_equal(a, nbf_k)

    def test_scalar_and_none_bounds_match_the_fast_path_defaults(
            self, monkeypatch):
        t, y, dy = self._lc()
        freqs = np.linspace(0.5, 1.5, 5)
        _, (nb0_s, nbf_s) = self._record_solution_ladder(
            monkeypatch, t, y, dy, freqs, np.ones(5), None, None, 5)
        nb0_k, nbf_k = self._kernel_ladder(freqs, 1e-2, 0.5)
        assert np.array_equal(nb0_s, nb0_k) and np.array_equal(nbf_s,
                                                                 nbf_k)
        _, (nb0_s, nbf_s) = self._record_solution_ladder(
            monkeypatch, t, y, dy, freqs, np.ones(5), 0.05, 0.25, 5)
        nb0_k, nbf_k = self._kernel_ladder(freqs, 0.05, 0.25)
        assert np.array_equal(nb0_s, nb0_k) and np.array_equal(nbf_s,
                                                                 nbf_k)

    def test_reported_box_is_on_the_kernel_grid_for_float32_bounds(self):
        # an on-grid q = 4/40 box at phi0 = 0.25; float32 bounds
        # qmin = 0.025, qmax = 0.2 give the kernel nbinsf = 40 and
        # nbins0 = 5, while the float64-promoted ladder is 39 / 4, on
        # which no q = m/39 box is the kernel's
        t, y, dy = TestFastPathQmaxBox._on_grid_box(nbf=40, m=4, n0=10)
        qmin, qmax = np.float32([0.025]), np.float32([0.2])
        nb0_k, nbf_k = self._kernel_ladder([1.0], qmin, qmax)
        assert (int(nb0_k[0]), int(nbf_k[0])) == (5, 40)
        sols = _fast_bls_solutions(t, y, dy, np.array([1.0]),
                                   np.array([1.0]), qmin, qmax, 1)
        q, phi0 = sols[0]
        assert q == pytest.approx(4. / 40., abs=1e-9)
        assert phi0 == pytest.approx(0.25, abs=1e-6)
        # a q = m/39 (the old ladder) is never within 1e-9 of m/40
        assert np.min(np.abs(q - np.arange(1, 40) / 39.)) > 1e-4


class TestSingleBlsQDomain(object):
    """``single_bls`` input domain: ``freq > 0``, ``q`` in ``[0, 1]``,
    ``phi0`` any finite phase. A negative or > 1 ``q`` used to return
    a silent power of 0 (Sep 2026 fresh-eyes review, finding 38)."""

    @staticmethod
    def _lc(n=200, seed=9):
        rand = np.random.RandomState(seed)
        t = np.sort(20. * rand.rand(n))
        y = 1. - 0.01 * (((t * 0.7) % 1.) < 0.1) + 1e-3 * rand.randn(n)
        dy = 1e-3 * np.ones(n)
        return t, y, dy

    @pytest.mark.parametrize("q", [-0.1, -1e-9, 1.0000001, 1.5, 7.])
    def test_q_outside_unit_interval_raises(self, q):
        t, y, dy = self._lc()
        with pytest.raises(ValueError, match=r"q must be in \[0, 1\]"):
            single_bls(t, y, dy, 0.7, q, 0.1)

    @pytest.mark.parametrize("freq", [0., -0.7])
    def test_non_positive_freq_raises(self, freq):
        t, y, dy = self._lc()
        with pytest.raises(ValueError, match="freq must be > 0"):
            single_bls(t, y, dy, freq, 0.1, 0.1)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_parameters_raise(self, bad):
        t, y, dy = self._lc()
        for args in [(bad, 0.1, 0.1), (0.7, bad, 0.1), (0.7, 0.1, bad)]:
            with pytest.raises(ValueError, match="must be finite"):
                single_bls(t, y, dy, *args)

    def test_q_endpoints_evaluate_to_zero_power(self):
        # q = 0 (the sparse paths' no-solution sentinel) is an empty
        # box; q = 1 is an all-weight box: both are power 0, not errors
        t, y, dy = self._lc()
        assert single_bls(t, y, dy, 0.7, 0.0, 0.1) == 0
        assert single_bls(t, y, dy, 0.7, 1.0, 0.1) == 0

    def test_phi0_is_any_finite_phase(self):
        # phi0 = 0 and negative phases are valid and wrap mod 1
        t, y, dy = self._lc()
        p0 = single_bls(t, y, dy, 0.7, 0.1, 0.0)
        assert np.isfinite(p0) and p0 > 0.5
        assert single_bls(t, y, dy, 0.7, 0.1, -0.3) == \
            single_bls(t, y, dy, 0.7, 0.1, 0.7)
        assert single_bls(t, y, dy, 0.7, 0.1, -1.0) == p0
