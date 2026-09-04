"""Stage fractions and launch counts at 1e4 points x 1e5 freqs: PDM kinds, CE default vs
fast, CE fast grid-size sensitivity. Shared GPU: 5x medians, relative comparisons only."""
import numpy as np, time, json, warnings
warnings.simplefilter('ignore')
import pycuda.driver as cuda
from cuvarbase.pdm import PDMAsyncProcess
from cuvarbase.ce import ConditionalEntropyAsyncProcess

rng = np.random.RandomState(2)
n, nf = 10000, 100000
t = np.sort(rng.rand(n)*365.); y = 0.3*np.sin(2*np.pi*t/2.3) + 0.2*rng.randn(n); dy = 0.2*np.ones(n)
freqs = np.linspace(0.1, 20., nf)
res = {}
launches = []
_orig = cuda.Function.prepared_async_call
def _rec(self, grid, block, stream, *args, **kw):
    e0 = cuda.Event(); e1 = cuda.Event()
    e0.record(stream); r = _orig(self, grid, block, stream, *args, **kw); e1.record(stream)
    launches.append((tuple(grid), tuple(block), kw.get('shared_size', 0), e0, e1)); return r
cuda.Function.prepared_async_call = _rec

def bench(label, fn, reps=5):
    walls, kern, nl = [], [], []
    for i in range(reps):
        launches.clear(); t0 = time.perf_counter(); fn(); cuda.Context.synchronize(); walls.append(time.perf_counter()-t0)
        kern.append(sum(e1.time_since(e0) for _,_,_,e0,e1 in launches)/1e3); nl.append(len(launches))
    w = np.median(walls); k = np.median(kern)
    grids = sorted(set((l[0], l[1], l[2]) for l in launches))
    print("%-46s wall %.1f ms (kernel %.1f ms = %.0f%%, host+copies %.1f ms) launches=%d grid/block/shmem=%s" % (label, 1e3*w, 1e3*k, 100*k/w, 1e3*(w-k), int(np.median(nl)), grids[:3]))
    res[label] = dict(wall_ms=1e3*w, kernel_ms=1e3*k, launches=int(np.median(nl)), grids=[list(map(str, g)) for g in grids[:3]])

pdm = PDMAsyncProcess()
for kind in ['binned_linterp', 'binned_linterp_fast', 'binned_step', 'binned_step_fast']:
    pdm.run([(t, y, dy)], freqs=freqs, kind=kind); pdm.finish()  # warm (compile)
    bench('PDM %s 1e4x1e5' % kind, lambda: (pdm.run([(t, y, dy)], freqs=freqs, kind=kind), pdm.finish()))
for bs in (64, 128):
    bench('PDM binned_linterp block_size=%d' % bs, lambda: (pdm.run([(t, y, dy)], freqs=freqs, kind='binned_linterp', block_size=bs), pdm.finish()))
    bench('PDM binned_linterp_fast block_size=%d' % bs, lambda: (pdm.run([(t, y, dy)], freqs=freqs, kind='binned_linterp_fast', block_size=bs), pdm.finish()))
# batched path (memory reuse?)
bench('PDM batched_run_const_nfreq 5 LCs bs=5', lambda: pdm.batched_run_const_nfreq([(t, y, dy)]*5, batch_size=5, freqs=freqs), reps=3)

ce = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5)
ce.run([(t, y, dy)], freqs=freqs); ce.finish()
bench('CE default (use_fast=False) 1e4x1e5', lambda: (ce.run([(t, y, dy)], freqs=freqs), ce.finish()))
cef = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=5, use_fast=True)
cef.run([(t, y, dy)], freqs=freqs); cef.finish()
bench('CE use_fast (auto grid) 1e4x1e5', lambda: (cef.run([(t, y, dy)], freqs=freqs), cef.finish()))
nsm = cuda.Context.get_device().get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
for nb in (nsm*4, nsm*16):
    bench('CE use_fast force_nblocks=%d 1e4x1e5' % nb, lambda: (cef.run([(t, y, dy)], freqs=freqs, force_nblocks=nb), cef.finish()))
# ndata=2000: data fits shared memory -> grid = floor(2*shmem_lim/shmem)
t2, y2, dy2 = t[:2000], y[:2000], dy[:2000]
bench('CE use_fast ndata=2000 (auto grid, data in shmem)', lambda: (cef.run([(t2, y2, dy2)], freqs=freqs), cef.finish()))
bench('CE use_fast ndata=2000 shmem_lc=False (auto grid)', lambda: (cef.run([(t2, y2, dy2)], freqs=freqs, shmem_lc=False), cef.finish()))
for nb in (nsm*4, nsm*16):
    bench('CE use_fast ndata=2000 force_nblocks=%d' % nb, lambda: (cef.run([(t2, y2, dy2)], freqs=freqs, force_nblocks=nb), cef.finish()))
bench('CE default ndata=2000', lambda: (ce.run([(t2, y2, dy2)], freqs=freqs), ce.finish()))
# parity of fast vs default at 1e4x1e5
r1 = ce.run([(t, y, dy)], freqs=freqs); ce.finish(); p1 = np.array(r1[0][1]).copy()
r2 = cef.run([(t, y, dy)], freqs=freqs, force_nblocks=nsm*16); cef.finish(); p2 = np.array(r2[0][1]).copy()
print("CE fast(force_nblocks) vs default: max|d|=%.2e argmin %d/%d" % (np.max(np.abs(p1-p2)), p1.argmin(), p2.argmin()))
json.dump(res, open('/workspace/scratch/af_timing.json', 'w'), indent=1)
