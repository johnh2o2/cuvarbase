"""Infrastructure: compile (SourceModule) counts per call for every entry point, pinned vs
pageable H2D sources, per-call wall time of the eebls_transit default path, device kwarg."""
import numpy as np, time, json, warnings
warnings.simplefilter('ignore')
import pycuda.driver as cuda, pycuda.compiler as pc, pycuda.gpuarray as gpuarray

ncomp = {'n': 0, 'time': 0.0}
_orig_init = pc.SourceModule.__init__
def _init(self, *a, **k):
    t0 = time.perf_counter(); r = _orig_init(self, *a, **k); ncomp['n'] += 1; ncomp['time'] += time.perf_counter() - t0; return r
pc.SourceModule.__init__ = _init
h2d = []
_orig_set = gpuarray.GPUArray.set_async
def _set(self, ary, stream=None, **kw):
    b = getattr(ary, 'base', None)
    h2d.append('pinned' if isinstance(b, cuda.PagelockedHostAllocation) else ('pageable:' + type(b).__name__))
    return _orig_set(self, ary, stream=stream, **kw)
gpuarray.GPUArray.set_async = _set

from cuvarbase.pdm import PDMAsyncProcess
from cuvarbase.ce import ConditionalEntropyAsyncProcess
from cuvarbase.lombscargle import LombScargleAsyncProcess
from cuvarbase.cunfft import NFFTAsyncProcess
from cuvarbase.bls import eebls_gpu_fast, eebls_transit, sparse_bls_gpu, eebls_gpu, eebls_gpu_batch
from cuvarbase.tls import tls_search_gpu
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess

rng = np.random.RandomState(1)
def lc(n, T=30.):
    t = np.sort(rng.rand(n)*T); y = 1 + 0.01*np.sin(2*np.pi*t/1.7) + 0.005*rng.randn(n); dy = 0.005*np.ones(n)
    return t, y, dy
freqs = np.linspace(0.2, 5., 3000)
res = {}
def measure(label, fn, reps=3):
    out = []
    for i in range(reps):
        ncomp['n'] = 0; ncomp['time'] = 0.; h2d.clear()
        t0 = time.perf_counter(); fn(); wall = time.perf_counter() - t0
        out.append((ncomp['n'], ncomp['time'], wall, sorted(set(h2d))))
    print("%-48s compiles/call: %s  compile-time: %s  wall: %s  H2D sources: %s" %
          (label, [o[0] for o in out], ["%.0fms" % (1e3*o[1]) for o in out], ["%.0fms" % (1e3*o[2]) for o in out], out[-1][3]))
    res[label] = [[o[0], o[1], o[2], o[3]] for o in out]

t, y, dy = lc(1000)
pdm = PDMAsyncProcess()
measure('PDM run (binned_linterp)', lambda: (pdm.run([(t, y, dy)], freqs=freqs), pdm.finish()))
measure('PDM run nbins=20 (new nbins)', lambda: (pdm.run([(t, y, dy)], freqs=freqs, nbins=20), pdm.finish()))
measure('PDM run nbins=10 again', lambda: (pdm.run([(t, y, dy)], freqs=freqs, nbins=10), pdm.finish()))
ce = ConditionalEntropyAsyncProcess()
measure('CE run', lambda: (ce.run([(t, y, dy)], freqs=freqs), ce.finish()))
measure('CE batched_run_const_nfreq (10 LCs, bs=5)', lambda: ce.batched_run_const_nfreq([lc(1000) for _ in range(10)], batch_size=5, freqs=freqs))
ls = LombScargleAsyncProcess()
lsf = (1./(5*30.))*(30+np.arange(20000))
measure('LS run', lambda: (ls.run([(t, y, dy)], freqs=lsf), ls.finish()))
measure('LS batched_run_const_nfreq (10 LCs)', lambda: ls.batched_run_const_nfreq([lc(1000) for _ in range(10)], freqs=lsf))
nf = NFFTAsyncProcess()
measure('NFFT run', lambda: (nf.run([(t, y, 2000)]), nf.finish()))
measure('eebls_gpu_fast', lambda: eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.1))
measure('eebls_transit default (ndata=1000 -> eebls_gpu)', lambda: eebls_transit(t, y, dy))
measure('eebls_transit use_fast=True', lambda: eebls_transit(t, y, dy, use_fast=True))
measure('eebls_gpu (solutions path)', lambda: eebls_gpu(t, y, dy, freqs[:500], qmin=0.01, qmax=0.1))
ts, ys, dys = lc(200)
measure('eebls_transit default (ndata=200 -> sparse GPU)', lambda: eebls_transit(ts, ys, dys))
measure('sparse_bls_gpu', lambda: sparse_bls_gpu(ts, ys, dys, freqs[:500]))
measure('eebls_gpu_batch (4 LCs)', lambda: eebls_gpu_batch([lc(1000) for _ in range(4)], freqs, qmin=0.01, qmax=0.1))
measure('tls_search_gpu (fast default)', lambda: tls_search_gpu(t, y, dy, period_min=0.5, period_max=5.0))
lrt = NUFFTLRTAsyncProcess(sigma=2)
measure('NUFFT-LRT run (5 periods x 1 dur x 8 epochs)', lambda: lrt.run(t, y, np.linspace(1.5, 2.0, 5), np.array([0.1]), epochs=np.linspace(0, 1.5, 8)))

# device kwarg
p2 = PDMAsyncProcess(device=1)
print("PDMAsyncProcess(device=1): self.device=%r ; active context device=%s (device kwarg is never consulted; only CUDA_DEVICE env at first use)" % (p2.device, cuda.Context.get_device().name()))
print("device count on pod:", cuda.Device.count())
# stream usage: how many streams does a 1-LC PDM/CE/LS run create?
print("streams held: pdm=%d ce=%d ls=%d" % (len(pdm.streams), len(ce.streams), len(ls.streams)))
json.dump(res, open('/workspace/scratch/af_infra.json', 'w'), indent=1)
