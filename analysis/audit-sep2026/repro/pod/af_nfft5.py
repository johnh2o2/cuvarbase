"""(1) NFFTAsyncProcess.run(memory=) never zeroes the grid -> accumulation across calls;
(2) immediate host read races the async D2H copy; (3) double-path component replica with a
properly zeroed grid."""
import numpy as np, warnings
warnings.simplefilter('ignore')
import pycuda.driver as cuda
from cuvarbase.cunfft import NFFTAsyncProcess, nfft_adjoint_async
rng = np.random.RandomState(4)
n0, nf = 20000, 40000
t = np.sort(rng.rand(n0)*100.); y = rng.randn(n0)
proc = NFFTAsyncProcess(sigma=2, m=8)
fresh = np.asarray(proc.run([(t, y, nf)])[0]); cuda.Context.synchronize(); fresh = fresh.copy()
mem = proc.allocate([(t, y, nf)])[0]
proc.run([(t, y, nf)], memory=[mem]); mem.stream.synchronize(); r1 = mem.ghat_c.copy()
proc.run([(t, y, nf)], memory=[mem]); mem.stream.synchronize(); r2 = mem.ghat_c.copy()
mem.ghat_g.fill(0); proc.run([(t, y, nf)], memory=[mem]); mem.stream.synchronize(); r3 = mem.ghat_c.copy()
print("(1) reuse: run#1 vs fresh max|d|=%.2e ; run#2 (same memory, no zeroing) vs fresh: %.2e ; run#3 after ghat_g.fill(0): %.2e  (|ghat| ~ %.1f)"
      % (np.max(np.abs(r1-fresh)), np.max(np.abs(r2-fresh)), np.max(np.abs(r3-fresh)), np.median(np.abs(fresh))))
bad = 0; N = 40
for k in range(N):
    mem.ghat_g.fill(0); mem.stream.synchronize(); mem.ghat_c[:] = 0
    out = proc.run([(t, y, nf)], memory=[mem])[0]
    snap = np.array(out, copy=True); mem.stream.synchronize(); done = np.array(out, copy=True)
    if not np.array_equal(snap, done): bad += 1
print("(2) immediate read differs from post-sync value in %d/%d runs (pinned host buffer, no sync in run())" % (bad, N))
# (3) component replica, grid zeroed
n0, nf, T = 2000, 4000, 100.
t = np.sort(rng.rand(n0)*T); y = rng.randn(n0)
def replica_grid(t, y, nf, sigma, m):
    t = t.astype(np.float64); y = y.astype(np.float64)
    ng = int(sigma*nf); b = 2*sigma*m/((2*sigma-1)*np.pi)
    x0, xf = t.min(), t.max(); xval = (t - x0)/(xf - x0)
    xg = m + (ng*xval - np.floor(ng*xval))
    q1 = np.exp(-xg*xg/b)/np.sqrt(b*np.pi); q2 = np.exp(2*xg/b); q3 = np.exp(-np.arange(2*m+1)**2/b)
    u = np.floor(ng*xval - m).astype(np.int64)
    grid = np.zeros(ng); Q = q1.copy()
    for k in range(2*m+1):
        np.add.at(grid, (u + k) % ng, Q*q3[k]*y); Q = Q*q2
    return grid, b, ng, x0, xf
def replica_full(grid, b, ng, nf, x0, xf, spp=1.0, f0=0.0):
    G = np.fft.ifft(grid)*ng; k = np.arange(nf); sT = spp*(xf-x0); n0_ = (x0/sT)*ng; k0 = f0*sT
    theta = 2*np.pi*n0_*(k0+k)/ng; khat = np.pi*(k0+k)/ng
    return G[:nf]*np.exp(1j*theta)*np.exp(b*khat*khat)
ks = np.arange(0, nf//2, 25)
ex = np.array([np.sum(y*np.exp(2j*np.pi*k*t/(t.max()-t.min()))) for k in ks]); l1 = np.sum(np.abs(y))
for dbl in (False, True):
    p = NFFTAsyncProcess(use_double=dbl, sigma=2, m=8)
    m_ = p.allocate([(t, y, nf)])[0]
    p.run([(t, y, nf)], memory=[m_]); m_.stream.synchronize()
    m_.ghat_g.fill(0); m_.stream.synchronize()
    g_gpu = nfft_adjoint_async(m_, p.function_tuple, just_return_gridded_data=True, block_size=256)
    gr, b, ng, x0, xf = replica_grid(t, y, nf, 2, 8)
    m_.ghat_g.fill(0); m_.stream.synchronize()
    p.run([(t, y, nf)], memory=[m_]); m_.stream.synchronize(); ghat = m_.ghat_c.copy()
    rep = replica_full(gr, b, ng, nf, x0, xf); rep_g = replica_full(g_gpu.astype(np.float64), b, ng, nf, x0, xf)
    print("(3) use_double=%-5s grid: GPU vs f64 replica max|d|=%.2e (max|grid|=%.2f) | ghat: GPU vs exact %.2e ; replica(f64 grid) vs exact %.2e ; FFT+normalize of the GPU grid vs exact %.2e ; GPU vs replica-on-GPU-grid %.2e"
          % (dbl, np.max(np.abs(g_gpu-gr)), np.max(np.abs(gr)), np.max(np.abs(ghat[ks]-ex))/l1, np.max(np.abs(rep[ks]-ex))/l1, np.max(np.abs(rep_g[ks]-ex))/l1, np.max(np.abs(ghat[ks]-rep_g[ks]))/l1))
