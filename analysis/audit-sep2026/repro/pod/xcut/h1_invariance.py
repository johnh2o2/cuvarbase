"""(a) BJD shift, (b) float32 inputs, (c) unsorted/duplicate t, (g) dy scaling."""
import sys, json, warnings
import numpy as np
sys.path.insert(0, '/workspace/scratch/xcut')
from audit_common import make_lc, metrics, run_safely, fmt
import entry_points as E

warnings.simplefilter('ignore')
t, y, dy = make_lc()
rng = np.random.RandomState(7)
perm = rng.permutation(len(t))
t_dup = t.copy(); t_dup[100:110] = t_dup[99]   # 11 identical timestamps

VARIANTS = {
    'bjd_frac': lambda: (t + 2457000.5, y, dy),
    'bjd_int':  lambda: (t + 2460000.0, y, dy),
    'f32':      lambda: (t.astype(np.float32), y.astype(np.float32), dy.astype(np.float32)),
    'f32_bjd':  lambda: ((t + 2457000.5).astype(np.float32), y.astype(np.float32), dy.astype(np.float32)),
    'unsorted': lambda: (t[perm], y[perm], dy[perm]),
    'dup_t':    lambda: (t_dup, y, dy),
    'dy_x7':    lambda: (t, y, dy * 7.0),
    'dy_const': lambda: (t, y, np.full_like(dy, dy.mean())),
    'list_in':  lambda: (list(t), list(y), list(dy)),
    'y_int_dy_int': lambda: (t, np.round(1000 * y).astype(np.int64), np.ones(len(t), dtype=np.int64)),
}

names = sys.argv[1:] or list(E.ALL)
out = {}
for name in names:
    fn = E.ALL[name]
    st, base, dt, err = run_safely(fn, t, y, dy)
    print('\n=== %s  [base: %s %.2fs%s]' % (name, st, dt, ('  ' + err) if err else ''))
    if st != 'ok':
        out[name] = {'base': err}
        continue
    out[name] = {}
    for vn, mk in VARIANTS.items():
        tt, yy, ddy = mk()
        st2, r, dt2, err2 = run_safely(fn, tt, yy, ddy)
        if st2 != 'ok':
            print('  %-10s RAISE %s' % (vn, err2))
            out[name][vn] = {'raise': err2}
            continue
        m = metrics(base, r)
        # bitwise equality on float32-castable results
        try:
            m['bitwise'] = bool(np.array_equal(np.asarray(base), np.asarray(r)))
        except Exception:
            pass
        out[name][vn] = m
        print('  %-10s %s%s' % (vn, fmt(m), '  BITWISE' if m.get('bitwise') else ''))
    # determinism: same call twice
    st3, r3, _, _ = run_safely(fn, t, y, dy)
    if st3 == 'ok':
        m = metrics(base, r3); m['bitwise'] = bool(np.array_equal(np.asarray(base), np.asarray(r3)))
        out[name]['repeat'] = m
        print('  %-10s %s%s' % ('repeat', fmt(m), '  BITWISE' if m['bitwise'] else ''))

json.dump(out, open('/workspace/scratch/xcut/h1_out.json', 'w'), indent=1, default=str)
