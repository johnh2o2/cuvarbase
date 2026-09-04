"""Shared harness pieces for the cross-cutting cuvarbase audit."""
import sys, time, traceback, warnings, json
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning, module='cuvarbase.nufft_lrt')

P_TR = 3.3      # injected transit period (d)
Q_TR = 0.05
DEPTH = 0.02
P_SIN = 2.7
SIGMA = 0.005


def make_lc(N=600, T=100.0, seed=1, transit=True, sinus=True):
    rng = np.random.RandomState(seed)
    t = np.sort(rng.uniform(0, T, N))
    y = np.ones(N)
    if transit:
        ph = (t / P_TR) % 1.0
        y -= DEPTH * (ph < Q_TR)
    if sinus:
        y += 0.01 * np.sin(2 * np.pi * t / P_SIN)
    y += SIGMA * rng.randn(N)
    dy = SIGMA * np.ones(N) * rng.uniform(0.8, 1.2, N)
    return t, y, dy


def bls_freqs(nf=2000, fmin=0.1, fmax=3.0):
    return np.linspace(fmin, fmax, nf)


def ls_freqs(nf=2000, df=1.0 / 500.0, k0=50):
    return df * (k0 + np.arange(nf))


def metrics(a, b):
    """Compare two 1-D result arrays."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    out = {}
    if a.shape != b.shape:
        out['shape'] = '%s vs %s' % (a.shape, b.shape)
        return out
    out['nan_a'] = int(np.sum(~np.isfinite(a)))
    out['nan_b'] = int(np.sum(~np.isfinite(b)))
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        out['maxabs'] = float('nan')
        return out
    d = np.abs(a[m] - b[m])
    out['maxabs'] = float(d.max())
    scale = max(np.abs(a[m]).max(), 1e-30)
    out['maxrel'] = float(d.max() / scale)
    if m.sum() > 2 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
        out['corr'] = float(np.corrcoef(a[m], b[m])[0, 1])
    out['argmax_same'] = bool(np.argmax(np.where(m, a, -np.inf)) ==
                              np.argmax(np.where(m, b, -np.inf)))
    out['bitwise'] = bool(np.array_equal(a.view(np.uint8) if a.dtype == b.dtype else a, b) if a.shape == b.shape else False)
    return out


def run_safely(fn, *args, **kwargs):
    t0 = time.time()
    try:
        r = fn(*args, **kwargs)
        return ('ok', r, time.time() - t0, None)
    except BaseException as e:  # noqa
        tb = traceback.format_exc().strip().splitlines()
        return ('raise', None, time.time() - t0,
                '%s: %s' % (type(e).__name__, str(e).splitlines()[0][:200] if str(e) else ''))


def fmt(m):
    if m is None:
        return '-'
    if 'shape' in m:
        return 'SHAPE ' + m['shape']
    s = 'maxabs=%.3g' % m.get('maxabs', float('nan'))
    if 'maxrel' in m:
        s += ' rel=%.2g' % m['maxrel']
    if 'corr' in m:
        s += ' corr=%.6f' % m['corr']
    s += ' argmax=%s' % ('same' if m.get('argmax_same') else 'DIFF')
    if m.get('nan_a') or m.get('nan_b'):
        s += ' nan=%d/%d' % (m['nan_a'], m['nan_b'])
    return s
