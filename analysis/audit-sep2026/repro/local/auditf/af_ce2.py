"""CE per-call host overhead breakdown at 1e4 x 1e5, and the float32-freqs API trap."""
import numpy as np, time, warnings
warnings.simplefilter('ignore')
import pycuda.driver as cuda
from cuvarbase.ce import ConditionalEntropyAsyncProcess
rng = np.random.RandomState(2)
n, nf = 10000, 100000
t = np.sort(rng.rand(n)*365.); y = 0.3*np.sin(2*np.pi*t/2.3) + 0.2*rng.randn(n); dy = 0.2*np.ones(n)
freqs = np.linspace(0.1, 20., nf)
def med(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); cuda.Context.synchronize(); ts.append(time.perf_counter()-t0)
    return 1e3*np.median(ts)
for fast in (False, True):
    ce = ConditionalEntropyAsyncProcess(use_fast=fast)
    ce.run([(t, y, dy)], freqs=freqs); ce.finish()
    t_comp = med(lambda: ce._compile_and_prepare_functions())
    t_alloc = med(lambda: ce.allocate([(t, y, dy)], freqs=[freqs]))
    mem = ce.allocate([(t, y, dy)], freqs=[freqs]); mem[0].transfer_freqs_to_gpu()
    t_run_mem = med(lambda: (ce.run([(t, y, dy)], memory=mem, freqs=[freqs]), ce.finish()))
    t_run = med(lambda: (ce.run([(t, y, dy)], freqs=freqs), ce.finish()))
    t_kern = med(lambda: (ce.call_func(mem[0], ce.function_tuple, block_size=256, transfer_to_device=False, transfer_to_host=False), ce.finish()))
    print("use_fast=%-5s run() wall %.0f ms | of which: _compile_and_prepare_functions (called EVERY run) %.0f ms, allocate() %.0f ms, kernels-only %.1f ms ; run(memory=prealloc) %.0f ms (still recompiles)"
          % (fast, t_run, t_comp, t_alloc, t_kern, t_run_mem))
    print("   bins_g allocated: %d MB (%s by the fast path)" % (mem[0].bins_g.nbytes // 2**20, 'UNUSED' if fast else 'used'))
ce = ConditionalEntropyAsyncProcess()
try:
    ce.run([(t, y, dy)], freqs=freqs.astype(np.float32)); ce.finish(); print("float32 freqs array: OK")
except Exception as e:
    print("float32 freqs array for a single LC -> %s: %s" % (type(e).__name__, str(e)[:90]))
