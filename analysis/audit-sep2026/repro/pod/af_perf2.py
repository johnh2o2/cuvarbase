"""(a) PDM fast vs standard kernels across sizes; (b) eebls_gpu (eebls_transit default for
ndata>=500) vs eebls_gpu_fast, compile excluded; (c) sparse_bls_gpu per-call compile vs
precompiled kernel. Shared GPU: 5x medians, relative only."""
import numpy as np, time, warnings
warnings.simplefilter('ignore')
import pycuda.driver as cuda
from cuvarbase.pdm import PDMAsyncProcess
from cuvarbase.bls import eebls_gpu, eebls_gpu_fast, compile_bls, sparse_bls_gpu, compile_sparse_bls, transit_autofreq, q_transit
rng = np.random.RandomState(2)
def med(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); cuda.Context.synchronize(); ts.append(time.perf_counter()-t0)
    return 1e3*np.median(ts)
pdm = PDMAsyncProcess()
for n, nf in [(300, 100000), (1000, 100000), (10000, 10000), (50000, 20000)]:
    t = np.sort(rng.rand(n)*365.); y = 0.3*np.sin(2*np.pi*t/2.3) + 0.2*rng.randn(n); dy = 0.2*np.ones(n); freqs = np.linspace(0.1, 20., nf)
    for kind in ('binned_linterp', 'binned_linterp_fast', 'binned_step', 'binned_step_fast'):
        pdm.run([(t, y, dy)], freqs=freqs, kind=kind); pdm.finish()
    row = {kind: med(lambda: (pdm.run([(t, y, dy)], freqs=freqs, kind=kind), pdm.finish())) for kind in ('binned_linterp', 'binned_linterp_fast', 'binned_step', 'binned_step_fast')}
    print("PDM n=%6d nf=%6d: " % (n, nf) + "  ".join("%s=%.1fms" % (k, v) for k, v in row.items()))
# (b) BLS default (solutions) path vs fast path, compile excluded
n = 6000; t = np.sort(rng.rand(n)*365.); y = 1 - 0.01*((((t-0.3)/2.5) % 1) < 0.02) + 0.003*rng.randn(n); dy = 0.003*np.ones(n)
freqs, qvals = transit_autofreq(t, fmin=0.1, fmax=2.0, qmin_fac=0.5)
qmins = 0.5*qvals; qmaxes = 2.0*qvals
print("BLS grid: nfreq=%d, qmin range %.4f..%.4f" % (len(freqs), qmins.min(), qmins.max()))
fns = compile_bls()
t_gpu = med(lambda: eebls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxes, functions=fns), reps=3)
t_fast = med(lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxes), reps=5)
t_gpu_c = med(lambda: eebls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxes), reps=3)
print("eebls_gpu (eebls_transit default path) precompiled: %.0f ms | with its per-call compile_bls: %.0f ms | eebls_gpu_fast (cached kernels): %.0f ms  -> default path is %.0fx slower kernel-wise, %.0fx end-to-end" % (t_gpu, t_gpu_c, t_fast, t_gpu/t_fast, t_gpu_c/t_fast))
p1, s1 = eebls_gpu(t, y, dy, freqs, qmin=qmins, qmax=qmaxes, functions=fns); p2 = eebls_gpu_fast(t, y, dy, freqs, qmin=qmins, qmax=qmaxes)
print("   argmax agree: %s (f=%.4f vs %.4f), peak %.4f vs %.4f" % (np.argmax(p1) == np.argmax(p2), freqs[np.argmax(p1)], freqs[np.argmax(p2)], p1.max(), p2.max()))
# (c) sparse path
n = 300; ts = np.sort(rng.rand(n)*365.); ys = 1 - 0.01*((((ts-0.3)/2.5) % 1) < 0.02) + 0.003*rng.randn(n); dys = 0.003*np.ones(n)
fr = freqs[:20000]; qm = qmins[:20000]; qM = qmaxes[:20000]
k = compile_sparse_bls(block_size=64)
t_pre = med(lambda: sparse_bls_gpu(ts, ys, dys, fr, qmin=qm, qmax=qM, kernel=k))
t_def = med(lambda: sparse_bls_gpu(ts, ys, dys, fr, qmin=qm, qmax=qM))
print("sparse_bls_gpu ndata=300 nf=20000: precompiled kernel %.0f ms | default (compile per call) %.0f ms -> %.1fx" % (t_pre, t_def, t_def/t_pre))
