"""Component test of the double-precision NFFT path: gridded data vs a float64 numpy replica
of precompute_psi + fast_gaussian_grid; then FFT+normalize replica vs GPU output."""
import numpy as np, warnings
warnings.simplefilter('ignore')
from cuvarbase.cunfft import NFFTAsyncProcess, nfft_adjoint_async
rng = np.random.RandomState(4)
n0, nf, T = 2000, 4000, 100.
t = np.sort(rng.rand(n0)*T); y = rng.randn(n0)
def replica_grid(t, y, nf, sigma, m, dtype):
    t = t.astype(dtype); y = y.astype(dtype)
    ng = int(sigma*nf); b = dtype(2*sigma*m/((2*sigma-1)*np.pi))
    x0, xf = t.min(), t.max()
    xval = (t - x0)/(xf - x0)
    xg = m + (ng*xval - np.floor(ng*xval))
    q1 = np.exp(-xg*xg/b)/np.sqrt(b*np.pi); q2 = np.exp(2*xg/b); q3 = np.exp(-np.arange(2*m+1)**2/b)
    u = np.floor(ng*xval - m).astype(np.int64)
    grid = np.zeros(ng, dtype=np.float64)
    Q = q1.copy()
    for k in range(2*m+1):
        np.add.at(grid, (u + k) % ng, Q*q3[k]*y); Q = Q*q2
    return grid, b, ng, x0, xf
def replica_full(grid, b, ng, nf, x0, xf, spp=1.0, f0=0.0):
    G = np.fft.ifft(grid) * ng     # cufft inverse is unnormalized
    k = np.arange(nf); sT = spp*(xf-x0); n0 = (x0/sT)*ng; k0 = f0*sT
    theta = 2*np.pi*n0*(k0+k)/ng; khat = np.pi*(k0+k)/ng
    return G[:nf]*np.exp(1j*theta)*np.exp(b*khat*khat)
for dbl in (False, True):
    proc = NFFTAsyncProcess(use_double=dbl, sigma=2, m=8)
    mem = proc.allocate([(t, y, nf)])[0]
    proc.run([(t, y, nf)], memory=[mem]); mem.stream.synchronize()  # compile
    g_gpu = nfft_adjoint_async(mem, proc.function_tuple, just_return_gridded_data=True, block_size=256)
    gr64, b, ng, x0, xf = replica_grid(t, y, nf, 2, 8, np.float64)
    gr32, *_ = replica_grid(t, y, nf, 2, 8, np.float32)
    print("use_double=%s: GPU grid vs float64 replica: max|d|=%.2e (max|grid|=%.2f); vs float32-input replica: %.2e" % (dbl, np.max(np.abs(g_gpu - gr64)), np.max(np.abs(gr64)), np.max(np.abs(g_gpu - gr32))))
    proc.run([(t, y, nf)], memory=[mem]); mem.stream.synchronize(); ghat = mem.ghat_c.copy()
    rep = replica_full(gr64, b, ng, nf, x0, xf)
    rep_gpu_grid = replica_full(g_gpu.astype(np.float64), b, ng, nf, x0, xf)
    ks = np.arange(0, nf//2)
    ex = np.array([np.sum(y*np.exp(2j*np.pi*k*t/(xf-x0))) for k in ks[::50]])
    print("   final ghat: GPU vs replica(f64 grid): max|d|/||y||_1=%.2e ; GPU vs replica(GPU grid): %.2e ; replica(f64) vs exact DFT: %.2e ; GPU vs exact: %.2e"
          % (np.max(np.abs(ghat[ks]-rep[ks]))/np.sum(np.abs(y)), np.max(np.abs(ghat[ks]-rep_gpu_grid[ks]))/np.sum(np.abs(y)),
             np.max(np.abs(rep[ks[::50]]-ex))/np.sum(np.abs(y)), np.max(np.abs(ghat[ks[::50]]-ex))/np.sum(np.abs(y))))
