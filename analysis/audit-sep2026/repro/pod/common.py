import numpy as np

def make_lc(N=300, T=365.0, t0=0.0, f0=3.1, amp=0.3, noise=0.1, hetero=False, seed=1):
    rng = np.random.RandomState(seed)
    t = np.sort(rng.rand(N)) * T + t0
    dy = noise * np.ones(N)
    if hetero:
        dy = noise * np.exp(0.7 * rng.randn(N))
    y = 12.0 + amp * np.cos(2 * np.pi * f0 * t - 0.3) + dy * rng.randn(N)
    return t, y, dy

def gls_numpy(t, y, dy, freqs, floating_mean=True):
    """Zechmeister & Kurster 2009 GLS, float64, P = 1 - chi2/chi2_0."""
    t = np.asarray(t, float); y = np.asarray(y, float); dy = np.asarray(dy, float)
    w = dy ** -2; w /= w.sum()
    ybar = np.dot(w, y)
    yc = y - ybar
    YY = np.dot(w, yc ** 2)
    P = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        ph = 2 * np.pi * f * t
        c = np.cos(ph); s = np.sin(ph)
        C = np.dot(w, c); S = np.dot(w, s)
        YC = np.dot(w, yc * c); YS = np.dot(w, yc * s)
        CC = np.dot(w, c * c); SS = np.dot(w, s * s); CS = np.dot(w, c * s)
        if floating_mean:
            YC -= 0.0  # yc already centered: sum w yc cos - ybar_c*C, ybar_c = 0
            CC -= C * C; SS -= S * S; CS -= C * S
        D = CC * SS - CS ** 2
        P[i] = (SS * YC ** 2 + CC * YS ** 2 - 2 * CS * YC * YS) / (YY * D)
    return P

def gls_numpy_fast(t, y, dy, freqs, floating_mean=True, chunk=2000):
    t = np.asarray(t, float); y = np.asarray(y, float); dy = np.asarray(dy, float)
    w = dy ** -2; w /= w.sum()
    ybar = np.dot(w, y); yc = y - ybar; YY = np.dot(w, yc ** 2)
    out = np.empty(len(freqs))
    for a in range(0, len(freqs), chunk):
        f = np.asarray(freqs[a:a + chunk])
        ph = 2 * np.pi * np.outer(f, t)
        c = np.cos(ph); s = np.sin(ph)
        C = c @ w; S = s @ w
        YC = c @ (w * yc); YS = s @ (w * yc)
        CC = (c * c) @ w; SS = (s * s) @ w; CS = (c * s) @ w
        if floating_mean:
            CC = CC - C * C; SS = SS - S * S; CS = CS - C * S
        D = CC * SS - CS ** 2
        out[a:a + chunk] = (SS * YC ** 2 + CC * YS ** 2 - 2 * CS * YC * YS) / (YY * D)
    return out

def report(name, ref, got, freqs=None):
    ref = np.asarray(ref, float); got = np.asarray(got, float)
    d = np.abs(ref - got)
    i = np.argmax(d)
    ir, ig = np.argmax(ref), np.argmax(got)
    s = "%-60s maxabs=%.3e (at idx %d/%d ref=%.4f got=%.4f) medabs=%.2e peak_err=%.2e argmax_same=%s" % (
        name, d.max(), i, len(d), ref[i], got[i], np.median(d), abs(ref[ir] - got[ir]), ir == ig)
    if freqs is not None:
        s += " fref=%.5f fgot=%.5f" % (freqs[ir], freqs[ig])
    print(s, flush=True)
    return d.max()
