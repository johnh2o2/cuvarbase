"""Sparse BLS audit: exact float64 set-based reference (Panahi & Zucker 2021)
vs sparse_bls_cpu / sparse_bls_gpu (full + simple kernels).

Reference enumerates every cyclic-contiguous run of phase-sorted points and
maximizes s^2/(r(1-r))/YY (= SR^2/YY, chi2ratio convention). The phase fold
is done in float32 exactly like the kernels (fl32(t)*fl32(f), wrap) so that
only the arithmetic differs; a second, all-float64 fold quantifies the fold
error separately.
"""
import sys, time
import numpy as np
from cuvarbase.bls import (sparse_bls_cpu, sparse_bls_gpu, compile_sparse_bls,
                           single_bls)
from cuvarbase.utils import subtract_epoch


def ref_sparse(t, y, dy, freqs, qmin=0.0, qmax=0.5, qdef='cuv',
               fold='f32', ignore_neg=False, wmax_c=1e-4):
    """qdef: 'cuv' (phi0 = phi_i, egress midpoint), 'paper' (both midpoints),
    'none' (no q filter except set size < N)."""
    t64, epoch = subtract_epoch(t)
    y64 = np.asarray(y, dtype=np.float64)
    dy64 = np.asarray(dy, dtype=np.float64)
    w = dy64 ** -2
    w /= w.sum()
    ybar = np.dot(w, y64)
    x = y64 - ybar
    YY = np.dot(w, x ** 2)
    N = len(t)
    out = np.zeros(len(freqs))
    qout = np.zeros(len(freqs))
    phout = np.zeros(len(freqs))
    npts = np.zeros(len(freqs), dtype=int)
    qmin = np.broadcast_to(np.asarray(qmin, float), (len(freqs),))
    qmax = np.broadcast_to(np.asarray(qmax, float), (len(freqs),))
    for k, f in enumerate(freqs):
        if fold == 'f32':
            phi = (np.float32(t64) * np.float32(f)) % np.float32(1.0)
            phi = phi.astype(np.float64)
        else:
            phi = (t64 * float(f)) % 1.0
        o = np.argsort(phi, kind='stable')
        ps, ws, xs = phi[o], w[o], x[o]
        # doubled arrays for cyclic runs: start i in [0,N), length L in [1,N-1]
        ws2 = np.concatenate([ws, ws])
        xw2 = np.concatenate([ws * xs, ws * xs])
        cw = np.concatenate([[0.0], np.cumsum(ws2)])
        cxw = np.concatenate([[0.0], np.cumsum(xw2)])
        i = np.arange(N)[:, None]
        L = np.arange(1, N)[None, :]
        j = i + L  # exclusive end (in doubled index)
        W = cw[j] - cw[i]
        S = cxw[j] - cxw[i]
        # representative durations
        ps2 = np.concatenate([ps, ps + 1.0])
        last = ps2[j - 1]
        nxt = ps2[np.minimum(j, 2 * N - 1)]
        egress = 0.5 * (last + nxt)
        prev = np.concatenate([[ps[-1] - 1.0], ps[:-1]])[:, None]
        ingress_paper = 0.5 * (ps[:, None] + prev)
        if qdef == 'cuv':
            q = egress - ps[:, None]
        elif qdef == 'paper':
            q = egress - ingress_paper
        else:
            q = np.full_like(W, 0.25)
        valid = (q > 0) & (q >= qmin[k]) & (q <= qmax[k]) \
            & (W > 1e-9) & (W < 1.0 - wmax_c)
        if ignore_neg:
            valid &= (S <= 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            P = np.where(valid, S * S / (W * (1 - W)) / YY, 0.0)
        im = np.argmax(P)
        ii, ll = divmod(im, N - 1)
        out[k] = P.flat[im]
        qout[k] = q.flat[im]
        phout[k] = ps[ii]
        npts[k] = ll + 1
    return out, qout, phout, npts


def make_lc(N, kind, seed, baseline=365.0, q=0.05, phi0=0.3, f=1.3,
            depth_sig=8.0, sigma=0.01, offset=0.0, bjd=0.0):
    r = np.random.RandomState(seed)
    if kind == 'uniform':
        t = np.sort(baseline * r.rand(N))
    else:  # clustered: nightly blocks of 0.3 d
        nn = max(1, N // 8)
        nights = np.sort(r.choice(int(baseline), nn, replace=False))
        t = np.sort(np.concatenate([n + 0.3 * r.rand(8) for n in nights])[:N])
        if len(t) < N:
            t = np.sort(np.concatenate([t, baseline * r.rand(N - len(t))]))
    N = len(t)
    delta = depth_sig * sigma / np.sqrt(max(1, N * q))
    ph = (t * f - phi0) % 1.0
    y = offset - delta * (ph < q) + sigma * r.randn(N)
    dy = sigma * (0.7 + 0.6 * r.rand(N))
    return t + bjd, y, dy


def cmp(name, p, pref, argfreq=None):
    d = np.abs(p - pref)
    scale = max(pref.max(), 1e-12)
    print("  %-34s max|d|=%.3e (rel %.2e) argmax %s  P*=%.5f vs %.5f" % (
        name, d.max(), d.max() / scale,
        'SAME' if np.argmax(p) == np.argmax(pref) else 'DIFF(%d vs %d)' % (np.argmax(p), np.argmax(pref)),
        p.max(), pref.max()))
    return d.max() / scale


def main():
    kern = compile_sparse_bls(block_size=64)
    kern_s = compile_sparse_bls(block_size=64, use_simple=True)
    worst = {}
    for N in (20, 50, 100, 200, 300):
        for kind in ('uniform', 'clustered'):
            for offset, bjd, tag in ((0.0, 0.0, 'centered'),
                                     (1.0, 0.0, 'normflux'),
                                     (1.0, 2457000.5, 'normflux+BJD')):
                t, y, dy = make_lc(N, kind, seed=N + (kind == 'clustered'),
                                   offset=offset, bjd=bjd)
                f0 = 1.3
                freqs = np.concatenate([np.linspace(f0 - 0.01, f0 + 0.01, 33),
                                        np.random.RandomState(1).uniform(0.05, 5, 31)])
                pref, qref, phref, npref = ref_sparse(t, y, dy, freqs)
                pref64, _, _, _ = ref_sparse(t, y, dy, freqs, fold='f64')
                pc, sc = sparse_bls_cpu(t, y, dy, freqs)
                pg, sg = sparse_bls_gpu(t, y, dy, freqs, kernel=kern)
                pgs, sgs = sparse_bls_gpu(t, y, dy, freqs, kernel=kern_s,
                                          use_simple=True)
                print("N=%d %s %s  (ref P*=%.5f at f=%.5f, %d pts in box; f64-fold ref P*=%.5f)"
                      % (N, kind, tag, pref.max(), freqs[np.argmax(pref)],
                         npref[np.argmax(pref)], pref64.max()))
                worst[('cpu', tag)] = max(worst.get(('cpu', tag), 0), cmp('cpu vs ref(f32 fold)', pc, pref))
                worst[('gpu', tag)] = max(worst.get(('gpu', tag), 0), cmp('gpu-full vs ref(f32 fold)', pg, pref))
                worst[('gpus', tag)] = max(worst.get(('gpus', tag), 0), cmp('gpu-simple vs ref(f32 fold)', pgs, pref))
                cmp('ref(f32 fold) vs ref(f64 fold)', pref, pref64)
                # solution check at the peak: single_bls reproduces the power?
                k = int(np.argmax(pg))
                qg, phg = sg[k]
                ps = single_bls(t, y, dy, freqs[k], qg, phg)
                print("  peak sol gpu: q=%.5f phi=%.5f single_bls=%.5f (gpu %.5f, cpu q=%.5f phi=%.5f)"
                      % (qg, phg, ps, pg[k], sc[k][0], sc[k][1]))
    print("\nWORST relative max|diff| per implementation/data type:")
    for k, v in sorted(worst.items()):
        print("  %-22s %.3e" % (k, v))


if __name__ == '__main__':
    main()
