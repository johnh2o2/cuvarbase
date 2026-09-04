import time, json, os, sys
import numpy as np
import pycuda.driver as cuda
import pycuda.autoprimaryctx  # noqa
from cuvarbase.bls_frequencies import keplerian_freq_grid

SURVEYS = {
    'ZTF':  dict(ndata=150,   baseline=730.0,  pmin=0.5, pmax=100.0),
    'HAT':  dict(ndata=6000,  baseline=3650.0, pmin=0.5, pmax=100.0),
    'TESS': dict(ndata=20000, baseline=27.0,   pmin=0.5, pmax=13.5),
}

def make_lc(cfg, seed):
    rng = np.random.RandomState(seed)
    n, T = cfg['ndata'], cfg['baseline']
    t = np.sort(rng.uniform(0, T, n))
    period, q0, depth = 2.5271, 0.035, 0.01
    y = np.ones(n); y[((t % period)/period) < q0] -= depth
    y += 0.002*rng.randn(n); dy = np.full(n, 0.002)
    return t, y, dy

def grid_for(cfg):
    f, q = keplerian_freq_grid(cfg['pmin'], cfg['pmax'], cfg['baseline'], oversampling=2, return_qvals=True)
    return f.astype(np.float64), 0.5*q, 2.0*q

def sync(): cuda.Context.synchronize()

def med(fn, runs=5, warm=1):
    for _ in range(warm): fn()
    ts = []
    for _ in range(runs):
        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter()-t0)
    return float(np.median(ts)), ts

def throttle():
    try:
        d = dict(l.split() for l in open('/sys/fs/cgroup/cpu.stat'))
        return int(d.get('nr_throttled',0)), int(d.get('throttled_usec',0))
    except Exception:
        return 0, 0

class KProxy:
    """Wrap a prepared pycuda Function: count launches + CUDA-event GPU time."""
    def __init__(self, name, func, log):
        self.name, self.func, self.log = name, func, log
    def _rec(self, stream, call, args, kw):
        e0, e1 = cuda.Event(), cuda.Event()
        e0.record(stream) if stream is not None else e0.record()
        r = call(*args, **kw)
        e1.record(stream) if stream is not None else e1.record()
        self.log.append((self.name, e0, e1)); return r
    def prepared_call(self, *args, **kw):
        return self._rec(None, self.func.prepared_call, args, kw)
    def prepared_async_call(self, *args, **kw):
        stream = args[2] if len(args) > 2 and isinstance(args[2], cuda.Stream) else None
        return self._rec(stream, self.func.prepared_async_call, args, kw)

def proxied(functions, log):
    return {k: KProxy(k, v, log) for k, v in functions.items()}

def summarize_log(log):
    sync(); out = {}
    for name, e0, e1 in log:
        out.setdefault(name, [0, 0.0]); out[name][0] += 1; out[name][1] += e0.time_till(e1)
    log.clear(); return {k: dict(launches=v[0], gpu_ms=v[1]) for k, v in out.items()}
