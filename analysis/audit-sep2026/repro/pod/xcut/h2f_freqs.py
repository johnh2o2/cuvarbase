"""(f) frequency / period grid edge cases for the entry points that accept a grid."""
import sys, json, warnings
import numpy as np
sys.path.insert(0, '/workspace/scratch/xcut')
from audit_common import make_lc, metrics, run_safely, fmt, ls_freqs
import entry_points as E
warnings.simplefilter('ignore')
t, y, dy = make_lc()

def classify(st, r, err):
    if st != 'ok':
        return 'RAISE %s' % err
    a = np.asarray(r, dtype=np.float64).ravel()
    nn = int(np.sum(~np.isfinite(a)))
    return 'ok n=%d nonfinite=%d vals=%s' % (a.size, nn, np.array2string(a[:4], precision=4))

F = E.FREQS[:300]
LF = ls_freqs(300)
grid_eps = {
    'bls_fast': ('freqs', F), 'bls_std': ('freqs', F), 'bls_custom': ('freqs', F),
    'sparse_gpu': ('freqs', F), 'sparse_cpu': ('freqs', F[:40]), 'bls_batch': ('freqs', F),
    'tls_fast': ('periods', E.PERIODS), 'tls_legacy': ('periods', E.PERIODS), 'tls_batch': ('periods', E.PERIODS),
    'ls': ('freqs', LF), 'ls_dirsum': ('freqs', LF), 'ls_cufinufft': ('freqs', LF), 'ls_batched': ('freqs', LF),
    'ce': ('freqs', F), 'ce_fast': ('freqs', F), 'ce_weighted': ('freqs', F),
    'pdm_linterp': ('freqs', F), 'pdm_step_fast': ('freqs', F), 'pdm_tophat': ('freqs', F[:50]),
}
names = sys.argv[1:] or list(grid_eps)
res = {}
for name in names:
    fn = E.ALL[name]; key, g = grid_eps[name]
    print('\n=== %s (%s)' % (name, key))
    res[name] = {}
    variants = {
        'empty': g[:0],
        'single': g[:1],
        'two': g[:2],
        'float32': g.astype(np.float32),
        'unsorted': g[::-1].copy(),
        'shuffled': np.random.RandomState(0).permutation(g),
        'contains0': np.concatenate([[0.0], g[1:]]),
        'negative': np.concatenate([[-g[1]], g[1:]]),
        'list': list(g),
        'nan': np.concatenate([[np.nan], g[1:]]),
        'huge': np.concatenate([g[:-1], [1e7]]),
    }
    st0, base, _, _ = run_safely(fn, t, y, dy, **{key: g})
    for vn, gv in variants.items():
        st, r, _, err = run_safely(fn, t, y, dy, **{key: gv})
        c = classify(st, r, err)
        if st == 'ok' and st0 == 'ok' and vn in ('float32', 'list'):
            c += ' | vs base: ' + fmt(metrics(base, r))
        if st == 'ok' and st0 == 'ok' and vn in ('unsorted',):
            try:
                c += ' | reversed-vs-base: ' + fmt(metrics(base, np.asarray(r)[::-1]))
            except Exception:
                pass
        if st == 'ok' and st0 == 'ok' and vn == 'shuffled':
            perm = np.random.RandomState(0).permutation(len(g))
            try:
                c += ' | unshuffled-vs-base: ' + fmt(metrics(base, np.asarray(r)[np.argsort(perm)]))
            except Exception:
                pass
        res[name][vn] = c
        print('  %-10s %s' % (vn, c))
json.dump(res, open('/workspace/scratch/xcut/h2f_out.json', 'w'), indent=1, default=str)
