import sys, os, warnings, time, threading
import numpy as np
sys.path.insert(0, '/workspace/scratch/xcut')
from audit_common import make_lc, metrics, fmt, run_safely, ls_freqs
warnings.simplefilter('ignore')
from cuvarbase.base import ensure_context
ensure_context()
import pycuda.driver as cuda
which = sys.argv[1]
t, y, dy = make_lc()
F = np.linspace(0.1, 3.0, 1500)

if which == 'bls_std_alloc':
    from cuvarbase import bls as B
    free0 = cuda.mem_get_info()[0]
    print('free before: %.2f GB' % (free0 / 1e9))
    # replicate eebls_gpu's sizing
    nbins_tot_max = B.count_tot_nbins(int(np.floor(1 / 0.2)), int(np.ceil(1 / 0.01)), 0.2)
    mem_per_f = 4 * 5 * nbins_tot_max * 3 * 4
    fbs = int((0.9 * free0 - 600 * 3 * 4 - 300 * 5 * 4) / mem_per_f)
    print('eebls_gpu default sizing for nf=300: nbins_tot_max=%d -> freq_batch_size=%d (grid arrays = %.2f GB), len(freqs)=300' % (nbins_tot_max, fbs, 4 * 5 * fbs * nbins_tot_max * 3 * 4 / 1e9))
    fns = B.compile_bls()
    for kw in ({}, {'freq_batch_size': 300}, {'max_memory': int(2e8)}):
        ts = []
        for i in range(3):
            t0 = time.time(); r = B.eebls_gpu(t, y, dy, F[:300], qmin=0.01, qmax=0.2, functions=fns, **kw); cuda.Context.synchronize(); ts.append(time.time() - t0)
        print('eebls_gpu %-28s call times: %s  free after: %.2f GB' % (kw, ' '.join('%.3f' % x for x in ts), cuda.mem_get_info()[0] / 1e9))
    r0 = B.eebls_gpu(t, y, dy, F[:300], qmin=0.01, qmax=0.2, functions=fns)[0]
    r1 = B.eebls_gpu(t, y, dy, F[:300], qmin=0.01, qmax=0.2, functions=fns, freq_batch_size=300)[0]
    r2 = B.eebls_gpu(t, y, dy, F[:300], qmin=0.01, qmax=0.2, functions=fns, freq_batch_size=37)[0]
    print('results default vs fbs=300: %s ; vs fbs=37: %s' % (fmt(metrics(r0, r1)), fmt(metrics(r0, r2))))
    tt, yy, ddy = make_lc(N=2000, T=1000.0)
    f = np.linspace(0.05, 10.0, int(1e6))
    st, r, dt, err = run_safely(B.eebls_gpu, tt, yy, ddy, f, qmin=0.01, qmax=0.2, functions=fns)
    print('eebls_gpu nf=1e6 default max_memory: %s (%.1fs) free=%.2f GB' % (('RAISE ' + err) if st != 'ok' else 'ok', dt, cuda.mem_get_info()[0] / 1e9))
    st, r, dt, err = run_safely(B.eebls_gpu, tt, yy, ddy, f, qmin=0.01, qmax=0.2, functions=fns, max_memory=int(4e9))
    print('eebls_gpu nf=1e6 max_memory=4GB: %s (%.1fs)' % (('RAISE ' + err) if st != 'ok' else 'ok', dt))
    # custom too
    st, r, dt, err = run_safely(B.eebls_gpu_custom, t, y, dy, F[:100], np.array([0.02, 0.05]), np.linspace(0, 1, 50), functions=fns)
    nq, nphi = 2, 50
    fbs = int((0.9 * free0) / (4 * 5 * nq * nphi * 4))
    print('eebls_gpu_custom: %s (%.2fs); its freq_batch_size sizing for 100 freqs would be %d' % (('RAISE ' + err) if st != 'ok' else 'ok', dt, fbs))

elif which == 'ls_sigma_fix':
    from cuvarbase.lombscargle import LombScargleAsyncProcess
    from astropy.timeseries import LombScargle
    T = t.max() - t.min(); df = 1.0 / (5 * T)
    for fmin, fmax in ((1.5, 3.0), (2.0, 3.0), (2.5, 3.0), (5.0, 10.0), (0.1, 3.0)):
        k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df)); f = df * (k0 + np.arange(nf))
        a = LombScargle(t, y, dy).power(f)
        row = []
        for sigma in (4, 8, 16, 32):
            p = LombScargleAsyncProcess(sigma=sigma)
            g = np.copy(p.run([(t, y, dy)], freqs=[f])[0][1]); p.finish()
            m = metrics(a, g); row.append('sigma=%d: rel=%.1e' % (sigma, m['maxrel']))
        need = 2 * (k0 + nf) / nf
        print('fmin=%g fmax=%g k0/nf=%.2f (sigma needed for (k0+nf) <= sigma*nf/2: >= %.1f) | %s' % (fmin, fmax, k0 / nf, need, ' | '.join(row)))
    # nharmonics=2 at moderate k0
    fmin, fmax = 0.5, 3.0
    k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df)); f = df * (k0 + np.arange(nf))
    for sigma in (4, 8):
        p = LombScargleAsyncProcess(nharmonics=2, sigma=sigma)
        g = np.copy(p.run([(t, y, dy)], freqs=[f])[0][1]); p.finish()
        gd = np.copy(p.run([(t, y, dy)], freqs=[f], use_fft=False)[0][1]); p.finish()
        print('nharmonics=2 fmin=0.5 fmax=3 sigma=%d: NFFT vs direct-sum: %s' % (sigma, fmt(metrics(gd, g))))

elif which == 'fap_mh':
    from cuvarbase.lombscargle import LombScargleAsyncProcess, fap_baluev
    f = ls_freqs(1500)
    p = LombScargleAsyncProcess(nharmonics=2)
    bf, sig = p.batched_run_const_nfreq([(t, y, dy)], freqs=f, only_return_best_freqs=True)
    g = np.copy(p.run([(t, y, dy)], freqs=[f])[0][1]); p.finish()
    z = g.max()
    fap3 = fap_baluev(t, dy, z, f.max())
    fap5 = fap_baluev(t, dy, z, f.max(), d_K=5)
    print('nharmonics=2: best_freq=%.4f reported significance=%.6g -> implied FAP=%.3g ; fap_baluev(d_K=3)=%.3g ; fap_baluev(d_K=2H+1=5)=%.3g' % (bf[0], sig[0], 1 - sig[0], fap3, fap5))
    # also astropy FAP for the H=1 case as a sanity reference
    from astropy.timeseries import LombScargle
    p1 = LombScargleAsyncProcess()
    g1 = np.copy(p1.run([(t, y, dy)], freqs=[f])[0][1]); p1.finish()
    ls = LombScargle(t, y, dy)
    print('H=1: cuvarbase fap_baluev(z=%.4f)=%.3g vs astropy false_alarm_probability(method=baluev)=%.3g' % (g1.max(), fap_baluev(t, dy, g1.max(), f.max()), ls.false_alarm_probability(g1.max(), method='baluev', minimum_frequency=f.min(), maximum_frequency=f.max())))

elif which == 'thread_push':
    from cuvarbase import bls as B
    ref = B.eebls_gpu_fast(t, y, dy, F, qmin=0.01, qmax=0.2)
    out = {}
    def work(i, push):
        try:
            if push:
                ensure_context().context.push()
            r = B.eebls_gpu_fast(t, y, dy, F, qmin=0.01, qmax=0.2)
            out[i] = fmt(metrics(ref, r))
            if push:
                cuda.Context.pop()
        except BaseException as e:
            out[i] = 'RAISE %s: %s' % (type(e).__name__, str(e).splitlines()[0][:120])
    for push in (False, True):
        ths = [threading.Thread(target=work, args=(i, push)) for i in range(3)]
        [th.start() for th in ths]; [th.join() for th in ths]
        print('worker threads push=%s: %s' % (push, out))
    print('main after: %s' % fmt(metrics(ref, B.eebls_gpu_fast(t, y, dy, F, qmin=0.01, qmax=0.2))))

elif which == 'nfft_epoch_fix':
    # Would epoch-subtracting t on the host fix NFFTAsyncProcess/NUFFT-LRT float32 at BJD? (magnitudes)
    from cuvarbase.cunfft import NFFTAsyncProcess
    proc = NFFTAsyncProcess()
    g0 = np.abs(proc.run([(t, y, 512)])[0]); proc.finish()
    tb = t + 2457000.5
    g1 = np.abs(proc.run([(tb - np.floor(tb.min()), y, 512)])[0]); proc.finish()
    print('NFFT float32 |ghat|: t vs (t_bjd - floor(min)): %s' % fmt(metrics(g0, g1)))
    T = t.max() - t.min(); k = np.arange(256)
    ref = np.abs(np.array([np.sum(y * np.exp(2j * np.pi * (kk / T) * t)) for kk in k]))
    print('NFFT float32 |ghat| at t (0..100) vs exact DFT (256 modes): %s' % fmt(metrics(ref, g0[:256])))
    print('NFFT float32 |ghat| at t+1000.5 vs exact DFT (256 modes): %s' % fmt(metrics(ref, np.abs(proc.run([(t + 1000.5, y, 512)])[0])[:256])))
