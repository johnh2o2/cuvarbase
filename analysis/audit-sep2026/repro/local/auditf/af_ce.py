"""CE audit: mag-bin index overflow (max point -> bin == mag_bins), reference parity
(standard / fast / weighted), and the fast-kernel grid size."""
import numpy as np, time, sys, json
from scipy.special import ndtr
import pycuda.driver as cuda
from cuvarbase.ce import ConditionalEntropyAsyncProcess
from cuvarbase.memory import ConditionalEntropyMemory

rng = np.random.RandomState(7)

def make_lc(n=1500, T=200., f=0.7, amp=0.3, sig=0.1):
    t = np.sort(rng.rand(n) * T)
    y = amp * np.sin(2*np.pi*f*t) + sig * rng.randn(n)
    dy = sig * np.ones(n)
    return t, y, dy

def gpu_bins_and_phase(t, y, f, nphase, nmag):
    """Replicate the GPU's binning in float32: t,y mean-subtracted, y scaled to [0,1],
    m = floor(y*nmag) (NO clamp, as in ce_memory.setdata), phase bin from float32 product."""
    t = (t - t.mean()); y = (y - y.mean())
    t32 = t.astype(np.float32); y32 = y.astype(np.float32)
    yscale = (y32.max() - y32.min()); y0 = y32.min()
    ys = (y32 - y0) / yscale
    m = np.floor(ys * nmag).astype(np.int64)  # <- max point gives nmag
    ft = (t32 * np.float32(f)).astype(np.float32)
    ph = ft - np.floor(ft)
    n = (np.floor(ph * nphase).astype(np.int64)) % nphase
    return n, m

def ce_from_hist(H, nmag):
    dm = 1.0 / nmag
    Nphi = H.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = H * np.log(dm * Nphi[:, None] / H)
    term[H == 0] = 0
    return term.sum() / H.sum()

def ref_clamped(t, y, freqs, nphase, nmag):
    out = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        n, m = gpu_bins_and_phase(t, y, f, nphase, nmag)
        m = np.minimum(m, nmag - 1)
        H = np.zeros((nphase, nmag)); np.add.at(H, (n, m), 1)
        out[i] = ce_from_hist(H, nmag)
    return out

def ref_emulate_bug_nonfast(t, y, freqs, nphase, nmag):
    """Emulate histogram_data_count's flat index offset + n*NMAG + m with m == nmag:
    it lands in (n+1, 0) of the same frequency, or bin (0,0) of the NEXT frequency
    when n == nphase-1 (or past the end of the array for the last frequency)."""
    nf = len(freqs); NB = nphase * nmag
    flat = np.zeros(nf * NB + NB)  # +NB slack for the final overflow
    for i, f in enumerate(freqs):
        n, m = gpu_bins_and_phase(t, y, f, nphase, nmag)
        idx = i * NB + n * nmag + m
        np.add.at(flat, idx, 1)
    out = np.empty(nf)
    for i in range(nf):
        H = flat[i*NB:(i+1)*NB].reshape(nphase, nmag)
        out[i] = ce_from_hist(H, nmag)
    return out

def ref_weighted(t, y, dy, freqs, nphase, nmag, max_phi=3.):
    t = (t - t.mean()); y = (y - y.mean())
    t32 = t.astype(np.float32); y32 = y.astype(np.float32); dy32 = dy.astype(np.float32)
    yscale = (y32.max() - y32.min()); y0 = y32.min()
    Y = (y32 - y0) / yscale; DY = dy32 / yscale
    out = np.empty(len(freqs))
    dm = 1.0 / nmag
    for i, f in enumerate(freqs):
        ft = (t32 * np.float32(f)).astype(np.float32)
        ph = ft - np.floor(ft)
        n0 = (np.floor(ph * nphase).astype(np.int64)) % nphase
        m0 = (Y * nmag).astype(np.int64)
        H = np.zeros((nphase, nmag))
        for m in range(nmag):
            z = m / nmag - Y
            keep = ~((np.abs(z) > max_phi * DY) & (m != m0))
            zmax = z + 1.0 / nmag
            w = ndtr(zmax / DY) - ndtr(z / DY)
            np.add.at(H[:, m], n0[keep], w[keep])
        pphi = H.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = H * np.log(dm * pphi[:, None] / H)
        term[(H <= 0) | (pphi[:, None] <= 1e-10)] = 0
        out[i] = term.sum() / H.sum()
    return out

res = {}
t, y, dy = make_lc()
nphase, nmag = 10, 5
freqs = np.linspace(0.3, 1.2, 300)

# ---- 1. setdata bin index overflow
mem = ConditionalEntropyMemory(phase_bins=nphase, mag_bins=nmag)
mem.setdata(t, y, dy=dy)
res['setdata_max_bin_index'] = int(mem.y.max())
res['setdata_count_at_nmag'] = int(np.sum(mem.y == nmag))
print("setdata: max mag-bin index =", mem.y.max(), "(valid range 0..%d);" % (nmag-1),
      "points with index == mag_bins:", np.sum(mem.y == nmag))

# ---- 2. non-fast path: read back the histogram
proc = ConditionalEntropyAsyncProcess(phase_bins=nphase, mag_bins=nmag)
mems = proc.allocate([(t, y, dy)], freqs=[freqs])
mems[0].transfer_freqs_to_gpu()
r = proc.run([(t, y, dy)], memory=mems, freqs=[freqs]); proc.finish()
ce_gpu = np.array(r[0][1], dtype=np.float64)
bins = mems[0].bins_g.get().reshape(len(freqs), nphase, nmag)
per_f = bins.sum(axis=(1, 2))
print("non-fast: per-frequency histogram totals: min=%d max=%d (ndata=%d)" % (per_f.min(), per_f.max(), len(t)))
res['nonfast_hist_totals'] = dict(min=int(per_f.min()), max=int(per_f.max()), ndata=len(t),
                                  n_freq_total_lt_ndata=int(np.sum(per_f < len(t))),
                                  n_freq_total_gt_ndata=int(np.sum(per_f > len(t))))
ref_c = ref_clamped(t, y, freqs, nphase, nmag)
ref_b = ref_emulate_bug_nonfast(t, y, freqs, nphase, nmag)
print("non-fast GPU vs clamped reference: max|d|=%.3e   vs bug-emulating reference: max|d|=%.3e"
      % (np.max(np.abs(ce_gpu - ref_c)), np.max(np.abs(ce_gpu - ref_b))))
res['nonfast_vs_clamped_maxabs'] = float(np.max(np.abs(ce_gpu - ref_c)))
res['nonfast_vs_bugemul_maxabs'] = float(np.max(np.abs(ce_gpu - ref_b)))
res['nonfast_argmin_gpu'] = int(np.argmin(ce_gpu)); res['nonfast_argmin_ref'] = int(np.argmin(ref_c))

# ---- 3. fast path (shared-memory histogram; index nmag*nphase aliases block_bin_phi[0])
procf = ConditionalEntropyAsyncProcess(phase_bins=nphase, mag_bins=nmag, use_fast=True)
r = procf.run([(t, y, dy)], freqs=[freqs]); procf.finish()
ce_fast = np.array(r[0][1], dtype=np.float64)
print("fast GPU vs clamped reference: max|d|=%.3e ; fast vs non-fast: max|d|=%.3e"
      % (np.max(np.abs(ce_fast - ref_c)), np.max(np.abs(ce_fast - ce_gpu))))
res['fast_vs_clamped_maxabs'] = float(np.max(np.abs(ce_fast - ref_c)))
res['fast_vs_nonfast_maxabs'] = float(np.max(np.abs(ce_fast - ce_gpu)))

# how large can the error get?  small ndata makes one misplaced point matter
t2, y2, dy2 = make_lc(n=60)
freqs2 = np.linspace(0.3, 1.2, 200)
procs = ConditionalEntropyAsyncProcess(phase_bins=nphase, mag_bins=nmag)
r = procs.run([(t2, y2, dy2)], freqs=[freqs2]); procs.finish()
ce60 = np.array(r[0][1], dtype=np.float64)
ref60 = ref_clamped(t2, y2, freqs2, nphase, nmag)
print("ndata=60 non-fast GPU vs clamped ref: max|d|=%.3e, median|d|=%.3e, CE range=%.3f..%.3f"
      % (np.max(np.abs(ce60-ref60)), np.median(np.abs(ce60-ref60)), ref60.min(), ref60.max()))
res['n60_maxabs'] = float(np.max(np.abs(ce60-ref60))); res['n60_range'] = [float(ref60.min()), float(ref60.max())]

# ---- 4. weighted path vs reference
procw = ConditionalEntropyAsyncProcess(phase_bins=nphase, mag_bins=nmag, weighted=True)
r = procw.run([(t, y, dy)], freqs=[freqs]); procw.finish()
ce_w = np.array(r[0][1], dtype=np.float64)
refw = ref_weighted(t, y, dy, freqs, nphase, nmag)
print("weighted GPU vs reference: max|d|=%.3e (CE range %.3f..%.3f); argmin gpu/ref: %d/%d"
      % (np.max(np.abs(ce_w - refw)), refw.min(), refw.max(), np.argmin(ce_w), np.argmin(refw)))
res['weighted_maxabs'] = float(np.max(np.abs(ce_w - refw)))

# ---- 5. fast-kernel grid size instrumentation
launches = []
_orig = cuda.Function.prepared_async_call
def _rec(self, grid, block, stream, *args, **kw):
    launches.append((tuple(grid), tuple(block), kw.get('shared_size', 0)))
    return _orig(self, grid, block, stream, *args, **kw)
cuda.Function.prepared_async_call = _rec
dev = cuda.Context.get_device()
nsm = dev.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
for n in (500, 2000, 10000):
    tt, yy, dd = make_lc(n=n)
    fr = np.linspace(0.3, 5.0, 100000)
    launches.clear()
    r = procf.run([(tt, yy, dd)], freqs=[fr]); procf.finish()
    print("fast CE ndata=%d nf=%d: launches=%s  (SMs on device: %d)" % (n, len(fr), launches, nsm))
    res['fast_grid_n%d' % n] = [list(map(str, l)) for l in launches]
cuda.Function.prepared_async_call = _orig
json.dump(res, open('/workspace/scratch/af_ce.json', 'w'), indent=1)
