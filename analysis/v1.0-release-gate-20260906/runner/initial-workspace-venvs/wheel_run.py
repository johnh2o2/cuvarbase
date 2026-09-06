import numpy as np
import cuvarbase
from cuvarbase.bls import eebls_transit
from cuvarbase.tls import tls_search_batch
from cuvarbase.lombscargle import lomb_scargle_simple
assert cuvarbase.__file__.startswith('/workspace/venvs/wheel-test/')
print('Installed wheel:', cuvarbase.__file__)
rng = np.random.default_rng(1)
t = np.sort(rng.uniform(0, 30, 2000))
dy = np.full(t.size, 1e-3)
y = 1 - 0.01 * (((t - 3.0) % 2.5) < 0.1) + dy * rng.standard_normal(t.size)
f, p, _ = eebls_transit(t, y, dy)
assert len(f) == len(p) > 0 and np.all(np.isfinite(p))
print('eebls_transit best period:', 1 / f[p.argmax()])
r = tls_search_batch([(t, y, dy)])[0]
assert np.isfinite(r['period']) and r['period'] > 0
print('tls_search_batch:', r['period'])
f, p = lomb_scargle_simple(t, y, dy)
assert len(f) == len(p) > 0 and np.all(np.isfinite(p))
print('lomb_scargle_simple:', len(p))
