"""Independent float64 numpy references for the cuvarbase BLS kernels.

ref_fast   : replica of full_bls_no_sol / _fused (eebls_gpu_fast*) --
             histogram at nbf bins per frequency, box widths m in the
             dnbins sequence, noverlap phase-shifted passes, max.
ref_slow   : replica of eebls_gpu (bin_and_phase_fold_bst_multifreq +
             store_best_sols) -- one-bin boxes at every nb level.
exact_bls  : closed-form 1 - chi2/chi2_0 at (f, q, phi0) in float64.
"""
import numpy as np


def make_data(ndata=2000, baseline=30., freq=1.0, q=0.05, phi0=0.3,
              depth=None, sigma=0.01, snr=20., seed=1, t0=0.0,
              cadence=None):
    rng = np.random.RandomState(seed)
    if cadence is None:
        t = np.sort(baseline * rng.rand(ndata))
    else:
        t = np.arange(0, baseline, cadence)[:ndata]
        ndata = len(t)
    if depth is None:
        depth = snr * sigma / np.sqrt(ndata * q)
    y = np.zeros(ndata)
    phi = (t * freq) % 1.0
    intr = ((phi - phi0) % 1.0) < q
    y[intr] -= depth
    y += sigma * rng.randn(ndata)
    dy = sigma * np.ones(ndata)
    return t + t0, y, dy


def dnbins(nb, dlogq, f32=False):
    if dlogq < 0:
        return 1
    if f32:
        n = int(np.floor(np.float32(np.float32(dlogq) * np.float32(nb))))
    else:
        n = int(np.floor(dlogq * nb))
    return n if n > 0 else 1


def nb_levels(nb0, nbf, dlogq, f32=False):
    out = []
    nb = nb0
    while nb <= nbf:
        out.append(nb)
        nb += dnbins(nb, dlogq, f32)
    return out


def m_sequence(nbf, nb0, dlogq, f32=False):
    mbw = -(-nbf // nb0)  # divrndup
    ms = []
    m = 1
    while m < mbw:
        ms.append(m)
        m += dnbins(m, dlogq, f32)
    return ms


def _prep(t, y, dy):
    t = np.asarray(t, dtype=np.float64)
    epoch = np.floor(t.min())
    t = t - epoch
    w = np.power(np.asarray(dy, float), -2.)
    w /= w.sum()
    ybar = np.dot(w, y)
    yy = np.dot(w, (y - ybar) ** 2)
    yw = (y - ybar) * w
    return t, epoch, w, yw, yy


def bls_value(yw, w, ign):
    ok = (w > 1e-10) & (w < 1. - 1e-4)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(ok, yw * yw / (w * (1. - w)), 0.)
    if ign:
        p = np.where(yw > 0, 0., p)
    return p


def fold(t, f, f32fold=False):
    if f32fold:
        ph = np.float32(t.astype(np.float32) * np.float32(f))
        ph = ph - np.floor(ph)
        return ph.astype(np.float64)
    ph = t * f
    return ph - np.floor(ph)


def ref_fast(t, y, dy, freqs, qmin=1e-2, qmax=0.5, dlogq=0.3,
             noverlap=2, dphi=0.0, ignore_negative_delta_sols=False,
             f32fold=False, return_sols=False):
    t, epoch, w, yw, yy = _prep(t, y, dy)
    freqs = np.asarray(freqs, dtype=np.float64)
    nf = len(freqs)
    qmin = np.broadcast_to(np.asarray(qmin, float), (nf,))
    qmax = np.broadcast_to(np.asarray(qmax, float), (nf,))
    # BLSMemory.setdata: (ones_like(freqs32)/q).astype(uint32) (truncation)
    nbinsf = (np.ones(nf, dtype=np.float32) / qmin).astype(np.uint32)
    nbins0 = (np.ones(nf, dtype=np.float32) / qmax).astype(np.uint32)
    out = np.zeros(nf)
    sols = []
    for i, f in enumerate(freqs):
        nbf, nb0 = int(nbinsf[i]), int(nbins0[i])
        ff = np.float32(f) if f32fold else f
        ph = fold(t, ff, f32fold)
        best = 0.
        bsol = (0., 0.)
        ms = m_sequence(nbf, nb0, dlogq, f32=True)
        for s in range(noverlap):
            dp = dphi + float(s) / noverlap
            b = np.floor(nbf * ph - dp).astype(np.int64) % nbf
            hyw = np.bincount(b, weights=yw, minlength=nbf)
            hw = np.bincount(b, weights=w, minlength=nbf)
            cyw = np.concatenate(([0.], np.cumsum(np.concatenate((hyw, hyw)))))
            cw = np.concatenate(([0.], np.cumsum(np.concatenate((hw, hw)))))
            n = np.arange(nbf)
            for m in ms:
                byw = cyw[n + m] - cyw[n]
                bw = cw[n + m] - cw[n]
                p = bls_value(byw, bw, ignore_negative_delta_sols)
                j = int(np.argmax(p))
                if p[j] > best:
                    best = p[j]
                    bsol = (m / nbf, ((j + dp) / nbf) % 1.0)
        out[i] = best / yy
        sols.append(bsol)
    if return_sols:
        return out, sols
    return out


def ref_slow(t, y, dy, freqs, qmin=1e-2, qmax=0.5, dlogq=0.2, noverlap=3,
             ignore_negative_delta_sols=False, f32fold=False,
             per_freq_bounds=False):
    """Replica of eebls_gpu. By default mimics the batch-wide collapse
    (nbins0 = floor(1/max(qmax)), nbinsf = ceil(1/min(qmin))) assuming a
    single batch; per_freq_bounds=True applies the bounds per frequency."""
    t, epoch, w, yw, yy = _prep(t, y, dy)
    freqs = np.asarray(freqs, dtype=np.float64)
    nf = len(freqs)
    qmin = np.broadcast_to(np.asarray(qmin, float), (nf,))
    qmax = np.broadcast_to(np.asarray(qmax, float), (nf,))
    out = np.zeros(nf)
    sols = []
    for i, f in enumerate(freqs):
        if per_freq_bounds:
            nb0 = int(np.floor(1. / qmax[i]))
            nbf = int(np.ceil(1. / qmin[i]))
        else:
            nb0 = int(np.floor(1. / qmax.max()))
            nbf = int(np.ceil(1. / qmin.min()))
        ff = np.float32(f) if f32fold else f
        ph = fold(t, ff, f32fold)
        best = 0.
        bsol = (0., 0.)
        for nb in nb_levels(nb0, nbf, dlogq, f32=True):
            for s in range(noverlap):
                dp = float(s) / noverlap
                b = np.floor(nb * ph - dp).astype(np.int64) % nb
                hyw = np.bincount(b, weights=yw, minlength=nb)
                hw = np.bincount(b, weights=w, minlength=nb)
                p = bls_value(hyw, hw, ignore_negative_delta_sols)
                j = int(np.argmax(p))
                if p[j] > best:
                    best = p[j]
                    bsol = (1. / nb, (((j + dp) / nb) + epoch * f) % 1.0)
        out[i] = best / yy
        sols.append(bsol)
    return out, sols


def exact_bls(t, y, dy, f, q, phi0):
    """1 - chi2/chi2_0 for the box starting at phase phi0 (original
    timescale) with fractional width q, float64."""
    t = np.asarray(t, dtype=np.float64)
    w = np.power(np.asarray(dy, float), -2.)
    w /= w.sum()
    ybar = np.dot(w, y)
    yy = np.dot(w, (y - ybar) ** 2)
    ph = ((t * f) - phi0) % 1.0
    m = ph < q
    W = w[m].sum()
    YW = np.dot(w[m], y[m] - ybar)
    if W <= 1e-10 or W >= 1 - 1e-4:
        return 0.
    return YW ** 2 / (W * (1 - W)) / yy


def exact_best_at_freq(t, y, dy, f, q, nphi=4000):
    """max over a fine phase grid (and the data-phase edges) of exact_bls"""
    t = np.asarray(t, float)
    ph = (t * f) % 1.0
    cands = np.concatenate((np.linspace(0, 1, nphi, endpoint=False),
                            ph, (ph - q) % 1.0))
    best = 0.
    bp = 0.
    for p0 in cands:
        v = exact_bls(t, y, dy, f, q, p0)
        if v > best:
            best, bp = v, p0
    return best, bp
