"""Batch BLS parity: eebls_gpu_batch (full_bls_batch / _fused) vs
eebls_gpu_fast / eebls_gpu on identical inputs; ragged batches; BJD; astropy
peak cross-check; memory-reuse stale-data probe."""
import numpy as np, warnings, time
import pycuda.driver as cuda
from cuvarbase.bls import eebls_gpu_batch, eebls_gpu_fast, eebls_gpu, single_bls
from cuvarbase.memory.bls_memory import BLSBatchMemory
warnings.simplefilter('ignore')


def lc(N, seed, f=0.7, q=0.03, delta=0.02, baseline=365.0, bjd=0.0):
    r = np.random.RandomState(seed)
    t = np.sort(baseline * r.rand(N))
    y = 1.0 - delta * (((t * f) % 1.0) < q) + 0.01 * r.randn(N)
    dy = 0.01 * (0.7 + 0.6 * r.rand(N))
    return t + bjd, y, dy


def stat(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = np.abs(a - b)
    return "max|d|=%.2e rel=%.2e corr=%.6f argmax %s" % (
        d.max(), d.max() / max(b.max(), 1e-12), np.corrcoef(a, b)[0, 1],
        'SAME' if np.argmax(a) == np.argmax(b) else 'DIFF')


freqs = np.linspace(0.1, 2.0, 3000)
kw = dict(qmin=0.01, qmax=0.2, dlogq=0.3)

print("=== 1. single-LC parity batch vs fast, all noverlap/dphi, ndata sweep")
for N in (150, 2000, 20000):
    t, y, dy = lc(N, N)
    for nov in (1, 2, 3, 4, 8):
        for dphi in (0.0, 0.25):
            pf = eebls_gpu_fast(t, y, dy, freqs, noverlap=nov, dphi=dphi, **kw)
            pb = eebls_gpu_batch([(t, y, dy)], freqs, noverlap=nov, dphi=dphi, **kw)[0]
            print("  N=%5d nov=%d dphi=%.2f  %s" % (N, nov, dphi, stat(pb, pf)))

print("=== 2. ragged batch (each LC vs its own single-LC fast run), BJD LC included")
lcs = [lc(150, 1), lc(777, 2), lc(2000, 3, bjd=2457000.5), lc(4321, 4), lc(20000, 5), lc(64, 6), lc(63, 7)]
for nov, dphi in ((2, 0.0), (3, 0.0), (4, 0.25)):
    pbs = eebls_gpu_batch(lcs, freqs, noverlap=nov, dphi=dphi, **kw)
    for (t, y, dy), pb in zip(lcs, pbs):
        pf = eebls_gpu_fast(t, y, dy, freqs, noverlap=nov, dphi=dphi, **kw)
        print("  nov=%d dphi=%.2f N=%5d  %s" % (nov, dphi, len(t), stat(pb, pf)))
# chunked ragged batch: max_batch_lcs=3 -> sorted-by-ndata grouping
pbs2 = eebls_gpu_batch(lcs, freqs, noverlap=2, max_batch_lcs=3, **kw)
pbs1 = eebls_gpu_batch(lcs, freqs, noverlap=2, **kw)
print("  chunked(max_batch_lcs=3) vs single chunk: " + "; ".join("%.1e" % np.abs(a - b).max() for a, b in zip(pbs1, pbs2)))

print("=== 3. memory reuse: big LC then small LC in same slot (stale-tail probe)")
mem = BLSBatchMemory(20000, 2, len(freqs), stream=cuda.Stream())
big = lc(20000, 11); small = lc(100, 12)
_ = eebls_gpu_batch([big, big], freqs, memory=mem, **kw)
got = eebls_gpu_batch([small, small], freqs, memory=mem, **kw)
ref = eebls_gpu_batch([small], freqs, **kw)[0]
print("  slot0 %s | slot1 %s" % (stat(got[0], ref), stat(got[1], ref)))
# fewer LCs than allocated + n_lcs_active path
got = eebls_gpu_batch([small], freqs, memory=mem, **kw)
print("  1 LC in 2-slot mem: %s" % stat(got[0], ref))

print("=== 4. batch vs eebls_gpu (standard path, matched noverlap=3, dlogq=0.2)")
for N in (150, 2000):
    t, y, dy = lc(N, N)
    ps, sols = eebls_gpu(t, y, dy, freqs, noverlap=3, dlogq=0.2, qmin=0.01, qmax=0.2)
    pb = eebls_gpu_batch([(t, y, dy)], freqs, noverlap=3, dlogq=0.2, qmin=0.01, qmax=0.2)[0]
    pf = eebls_gpu_fast(t, y, dy, freqs, noverlap=3, dlogq=0.2, qmin=0.01, qmax=0.2)
    print("  N=%d batch vs eebls_gpu: %s | fast vs eebls_gpu: %s" % (N, stat(pb, ps), stat(pf, ps)))
    print("     eebls_gpu peak f=%.5f  batch peak f=%.5f  (injected 0.7)" % (freqs[np.argmax(ps)], freqs[np.argmax(pb)]))

print("=== 5. astropy BoxLeastSquares peak cross-check (independent reference)")
from astropy.timeseries import BoxLeastSquares
for N in (150, 2000):
    t, y, dy = lc(N, N)
    bls = BoxLeastSquares(t, y, dy)
    P = 1.0 / freqs[::-1]
    durs = np.array([0.03, 0.05, 0.08, 0.12]) * (1 / 0.7)
    res = bls.power(P, durs, objective='snr')
    pa = res.power[::-1]
    pb = eebls_gpu_batch([(t, y, dy)], freqs, noverlap=3, **kw)[0]
    print("  N=%d astropy peak f=%.5f  batch peak f=%.5f  corr(batch, astropy-snr)=%.4f" % (
        N, freqs[np.argmax(pa)], freqs[np.argmax(pb)], np.corrcoef(pa, pb)[0, 1]))
