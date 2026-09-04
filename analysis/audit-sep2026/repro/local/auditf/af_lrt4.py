"""NUFFT-LRT per-template cost with a CORRECT reuse prototype (grid zeroed per template)."""
import numpy as np, time, warnings
warnings.simplefilter('ignore')
from cuvarbase.cunfft import nfft_adjoint_async
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
rng = np.random.RandomState(5)
n = 5000; t = np.sort(rng.rand(n)*100.); nf = 2*n
lrt = NUFFTLRTAsyncProcess(sigma=2)
tmpl = lrt._generate_template(t, 3.3, 0.0, 0.15, 1.0); tmpl -= tmpl.mean()
a = np.asarray(lrt.compute_nufft(t, tmpl, nf)).copy()
nproc = lrt.nufft_proc
mem = nproc.allocate([(t.astype(np.float32), tmpl.astype(np.float32), nf)])[0]
def reuse(tm):
    mem.ghat_g.fill(0, stream=mem.stream)
    mem.y[:] = tm.astype(np.float32)
    nfft_adjoint_async(mem, nproc.function_tuple, block_size=nproc.block_size)
    mem.stream.synchronize(); return mem.ghat_c
b = reuse(tmpl).copy()
print("reuse (zeroed grid) vs default path: max|d|=%.2e, max|a|=%.1f -> relative %.1e" % (np.max(np.abs(a-b)), np.max(np.abs(a)), np.max(np.abs(a-b))/np.max(np.abs(a))))
def med(fn, reps=40):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    return 1e3*np.median(ts)
print("per-template ms: default compute_nufft %.2f | reuse %.2f | template generation %.3f  (shared GPU; relative)" % (med(lambda: lrt.compute_nufft(t, tmpl, nf)), med(lambda: reuse(tmpl)), med(lambda: lrt._generate_template(t, 3.3, 0.0, 0.15, 1.0))))
# end-to-end run() for a small grid: how much of it is compute_nufft?
periods = np.linspace(3.0, 3.6, 10); epochs = np.linspace(0, 3.3, 20)
t_run = med(lambda: lrt.run(t, 1 + 0.003*rng.randn(n), periods, np.array([0.15]), epochs=epochs), 5)
print("run(): %d templates -> %.0f ms = %.2f ms/template" % (len(periods)*len(epochs), t_run, t_run/(len(periods)*len(epochs))))
