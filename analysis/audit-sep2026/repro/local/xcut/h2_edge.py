"""(d) tiny N, (e) NaN/inf/zero dy, (f) frequency-grid edge cases."""
import sys, json, warnings
import numpy as np
sys.path.insert(0, '/workspace/scratch/xcut')
from audit_common import make_lc, metrics, run_safely, fmt
import entry_points as E
warnings.simplefilter('ignore')
t, y, dy = make_lc()
K = 137  # index of the poisoned point

def classify(st, r, err, ref=None):
    if st != 'ok':
        return 'RAISE %s' % err
    a = np.asarray(r, dtype=np.float64).ravel()
    nn = int(np.sum(~np.isfinite(a)))
    s = 'ok n=%d nonfinite=%d' % (a.size, nn)
    if a.size and nn == a.size:
        s = 'ALL-NONFINITE n=%d' % a.size
    if ref is not None and a.size == np.asarray(ref).size:
        s += ' | vs point-dropped ref: ' + fmt(metrics(ref, a))
    return s

names = sys.argv[1:] or list(E.ALL)
res = {}
for name in names:
    fn = E.ALL[name]
    print('\n=== %s' % name)
    res[name] = {}
    # ---- (d) tiny N
    for n in (2, 3, 5, 10):
        tt, yy, ddy = make_lc(N=n, seed=3)
        st, r, _, err = run_safely(fn, tt, yy, ddy)
        c = classify(st, r, err)
        res[name]['N=%d' % n] = c
        print('  N=%-3d %s' % (n, c))
    # ---- (e) poisoned inputs; reference = same data with point K dropped
    keep = np.ones(len(t), bool); keep[K] = False
    st, ref, _, err = run_safely(fn, t[keep], y[keep], dy[keep])
    ref = ref if st == 'ok' else None
    poisons = {
        'dy0':    (t, y, np.where(np.arange(len(t)) == K, 0.0, dy)),
        'dyNaN':  (t, y, np.where(np.arange(len(t)) == K, np.nan, dy)),
        'yNaN':   (t, np.where(np.arange(len(t)) == K, np.nan, y), dy),
        'yInf':   (t, np.where(np.arange(len(t)) == K, np.inf, y), dy),
        'tNaN':   (np.where(np.arange(len(t)) == K, np.nan, t), y, dy),
        'y_const': (t, np.ones_like(y), dy),
        'dy_neg':  (t, y, -dy),
        'len_mismatch': (t, y[:-1], dy),
        'N=0': (t[:0], y[:0], dy[:0]),
        'N=1': (t[:1], y[:1], dy[:1]),
    }
    for pn, (tt, yy, ddy) in poisons.items():
        st, r, _, err = run_safely(fn, tt, yy, ddy)
        c = classify(st, r, err, ref if pn in ('dy0', 'dyNaN', 'yNaN', 'yInf', 'tNaN') else None)
        res[name][pn] = c
        print('  %-12s %s' % (pn, c))

json.dump(res, open('/workspace/scratch/xcut/h2_out.json', 'w'), indent=1, default=str)
