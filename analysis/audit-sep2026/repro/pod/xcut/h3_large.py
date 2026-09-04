"""(h) very large N / nfreq: memory, launch bounds, shared-memory limits, message quality."""
import sys, warnings, time
import numpy as np
sys.path.insert(0, '/workspace/scratch/xcut')
from audit_common import make_lc, run_safely
warnings.simplefilter('ignore')
import pycuda.driver as cuda
from cuvarbase import bls as B
from cuvarbase import tls as TL
from cuvarbase.lombscargle import LombScargleAsyncProcess
from cuvarbase.ce import ConditionalEntropyAsyncProcess
from cuvarbase.pdm import PDMAsyncProcess

def rep(label, st, r, dt, err):
    extra = ''
    if st == 'ok':
        a = np.asarray(r if not isinstance(r, (tuple, list, dict)) else (r[0] if isinstance(r, (tuple, list)) else r.get('chi2')), dtype=np.float64).ravel()
        extra = 'n=%d nonfinite=%d' % (a.size, int(np.sum(~np.isfinite(a))))
    print('%-55s %-6s %6.1fs %s %s' % (label, st, dt, extra, err or ''))
    print('   free GPU mem: %.2f GB' % (cuda.mem_get_info()[0] / 1e9))

which = sys.argv[1]
if which == 'bls_bigN':
    for N in (int(2e5), int(1e6)):
        t, y, dy = make_lc(N=N, T=1000.0)
        f = np.linspace(0.1, 3.0, 2000)
        rep('eebls_gpu_fast N=%d nf=2000' % N, *run_safely(B.eebls_gpu_fast, t, y, dy, f, qmin=0.01, qmax=0.2))
        rep('eebls_gpu_batch N=%d nf=2000' % N, *run_safely(B.eebls_gpu_batch, [(t, y, dy)], f, qmin=0.01, qmax=0.2))
        rep('eebls_gpu N=%d nf=2000' % N, *run_safely(B.eebls_gpu, t, y, dy, f, qmin=0.01, qmax=0.2))
elif which == 'bls_bignf':
    t, y, dy = make_lc(N=2000, T=1000.0)
    f = np.linspace(0.05, 10.0, int(1e6))
    rep('eebls_gpu_fast N=2000 nf=1e6', *run_safely(B.eebls_gpu_fast, t, y, dy, f, qmin=0.01, qmax=0.2))
    rep('eebls_gpu_batch N=2000 nf=1e6', *run_safely(B.eebls_gpu_batch, [(t, y, dy)], f, qmin=0.01, qmax=0.2))
    rep('eebls_gpu N=2000 nf=1e6', *run_safely(B.eebls_gpu, t, y, dy, f, qmin=0.01, qmax=0.2))
    rep('eebls_gpu_fast qmin=1e-5 (shared mem limit)', *run_safely(B.eebls_gpu_fast, t, y, dy, f[:100], qmin=1e-5, qmax=0.2))
    rep('eebls_gpu qmin=1e-5', *run_safely(B.eebls_gpu, t, y, dy, f[:10], qmin=1e-5, qmax=0.2))
elif which == 'sparse_big':
    for N in (600, 1500, 3000, 6000, int(2e4)):
        t, y, dy = make_lc(N=N, T=100.0)
        rep('sparse_bls_gpu N=%d nf=50' % N, *run_safely(B.sparse_bls_gpu, t, y, dy, np.linspace(0.1, 3.0, 50)))
elif which == 'tls_big':
    for N in (int(2e5), int(1e6)):
        t, y, dy = make_lc(N=N, T=1000.0)
        rep('tls_search_gpu (fast) N=%d nper=500' % N, *run_safely(TL.tls_search_gpu, t, y, dy, periods=np.linspace(2, 6, 500)))
    t, y, dy = make_lc(N=20000, T=100.0)
    rep('tls_search_gpu legacy N=20000', *run_safely(TL.tls_search_gpu, t, y, dy, periods=np.linspace(2, 6, 50), use_fast=False))
    t, y, dy = make_lc(N=2000, T=1000.0)
    rep('tls_search_gpu fast N=2000 nper=1e6', *run_safely(TL.tls_search_gpu, t, y, dy, periods=np.linspace(0.5, 100, int(1e6))))
elif which == 'ls_big':
    proc = LombScargleAsyncProcess()
    for N, nf in ((int(2e5), 2000), (int(1e6), 2000), (2000, int(1e6)), (int(1e6), int(1e6))):
        t, y, dy = make_lc(N=N, T=1000.0)
        f = (1.0 / 5000.0) * (100 + np.arange(nf))
        def go():
            r = proc.run([(t, y, dy)], freqs=[f]); proc.finish(); return np.copy(r[0][1])
        rep('LS N=%d nf=%d' % (N, nf), *run_safely(go))
    t, y, dy = make_lc(N=int(1e6), T=1000.0); f = (1.0 / 5000.0) * (100 + np.arange(2000))
    def go2():
        r = proc.run([(t, y, dy)], freqs=[f], use_fft=False); proc.finish(); return np.copy(r[0][1])
    rep('LS dirsum N=1e6 nf=2000', *run_safely(go2))
elif which == 'ce_big':
    for N, nf, fast in ((int(2e5), 2000, False), (int(1e6), 2000, False), (2000, int(1e6), False), (int(2e5), 2000, True), (2000, int(1e6), True), (int(1e6), int(1e6), False)):
        proc = ConditionalEntropyAsyncProcess(use_fast=fast)
        t, y, dy = make_lc(N=N, T=1000.0)
        f = np.linspace(0.05, 10.0, nf)
        def go():
            r = proc.run([(t, y, dy)], freqs=[f]); proc.finish(); return np.copy(r[0][1])
        rep('CE fast=%s N=%d nf=%d run()' % (fast, N, nf), *run_safely(go))
        if not fast:
            def go3():
                r = proc.large_run([(t, y, dy)], freqs=[f]); return np.copy(r[0][1])
            rep('CE N=%d nf=%d large_run()' % (N, nf), *run_safely(go3))
elif which == 'pdm_big':
    proc = PDMAsyncProcess()
    for N, nf, kind in ((int(2e5), 2000, 'binned_linterp'), (int(1e6), 2000, 'binned_linterp_fast'), (2000, int(1e6), 'binned_linterp_fast'), (int(1e6), 2000, 'binned_linterp')):
        t, y, dy = make_lc(N=N, T=1000.0)
        f = np.linspace(0.05, 10.0, nf)
        def go():
            r = proc.run([(t, y, dy)], freqs=f, kind=kind); proc.finish(); return np.copy(r[0][1])
        rep('PDM %s N=%d nf=%d' % (kind, N, nf), *run_safely(go))
elif which == 'lrt_big':
    from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
    p = NUFFTLRTAsyncProcess()
    for N in (int(2e5), int(1e6)):
        t, y, dy = make_lc(N=N, T=1000.0)
        rep('NUFFT-LRT N=%d 5 periods x 2 dur' % N, *run_safely(lambda: p.run(t, y, np.linspace(3, 4, 5), np.array([0.1, 0.2]))))
