import sys, os, warnings, time
import numpy as np
sys.path.insert(0, '/workspace/scratch/xcut')
from audit_common import make_lc, metrics, fmt, run_safely, ls_freqs
warnings.simplefilter('ignore')
import pycuda.driver as cuda
which = sys.argv[1]
t, y, dy = make_lc()
F = np.linspace(0.1, 3.0, 1500)

if which == 'prealloc_race':
    from cuvarbase.ce import ConditionalEntropyAsyncProcess
    from cuvarbase.lombscargle import LombScargleAsyncProcess
    tB, yB, dyB = make_lc(N=900, seed=2, transit=False)
    tC, yC, dyC = make_lc(N=300, seed=5)
    for fast in (False, True):
        proc = ConditionalEntropyAsyncProcess(use_fast=fast)
        fB = np.copy(proc.run([(tB, yB, dyB)], freqs=[F])[0][1]); proc.finish()
        fC = np.copy(proc.run([(tC, yC, dyC)], freqs=[F])[0][1]); proc.finish()
        proc.preallocate(max_nobs=900, freqs=F, nlcs=1)
        for m in proc.memory: m.transfer_freqs_to_gpu()
        r = proc.run([(tB, yB, dyB)], freqs=[F]); proc.finish(); rB_early = np.copy(r[0][1]); cuda.Context.synchronize(); rB_late = np.copy(r[0][1])
        r = proc.run([(tC, yC, dyC)], freqs=[F]); proc.finish(); rC_early = np.copy(r[0][1]); cuda.Context.synchronize(); rC_late = np.copy(r[0][1])
        print('CE fast=%s preallocate(streams=None)+manual freq upload: B read after finish(): %s | after Context.synchronize(): %s' % (fast, fmt(metrics(fB, rB_early)), fmt(metrics(fB, rB_late))))
        print('                                                          C read after finish(): %s | after Context.synchronize(): %s' % (fmt(metrics(fC, rC_early)), fmt(metrics(fC, rC_late))))
        # with explicit streams
        proc.preallocate(max_nobs=900, freqs=F, nlcs=1, streams=[cuda.Stream()])
        for m in proc.memory: m.transfer_freqs_to_gpu()
        r = proc.run([(tC, yC, dyC)], freqs=[F]); proc.finish(); rC_s = np.copy(r[0][1])
        print('   with explicit streams=[Stream()] (not proc.streams): C after finish(): %s' % fmt(metrics(fC, rC_s)))
        proc.memory = None
    f = ls_freqs(1500)
    proc = LombScargleAsyncProcess()
    fB = np.copy(proc.run([(tB, yB, dyB)], freqs=[f])[0][1]); proc.finish()
    fC = np.copy(proc.run([(tC, yC, dyC)], freqs=[f])[0][1]); proc.finish()
    proc.preallocate(max_nobs=900, nlcs=1, freqs=f)
    r = proc.run([(tB, yB, dyB)], freqs=[f]); proc.finish(); e = np.copy(r[0][1]); cuda.Context.synchronize(); l = np.copy(r[0][1])
    print('LS preallocate(streams=None): B after finish(): %s | after ctx sync: %s' % (fmt(metrics(fB, e)), fmt(metrics(fB, l))))
    r = proc.run([(tC, yC, dyC)], freqs=[f]); proc.finish(); e = np.copy(r[0][1]); cuda.Context.synchronize(); l = np.copy(r[0][1])
    print('LS preallocate(streams=None): C after finish(): %s | after ctx sync: %s' % (fmt(metrics(fC, e)), fmt(metrics(fC, l))))

elif which == 'ce_spill2':
    from cuvarbase.ce import ConditionalEntropyAsyncProcess
    from cuvarbase.utils import normalize_light_curves
    def ce_ref_global(t, y, freqs, nphase=10, nmag=5):
        """standard (global-histogram) kernel model: max point in last phase bin spills into
        the NEXT frequency's (phase 0, mag 0) bin; otherwise into (own phase+1, mag 0)."""
        y = np.asarray(y, dtype=np.float32); t = np.asarray(t, dtype=np.float32)
        yn = (y - y.min()) / (y.max() - y.min())
        mbin = np.floor(yn * nmag).astype(np.int64)
        Hs = []
        incoming = 0
        for i, f in enumerate(freqs):
            ph = (t * np.float32(f)) % np.float32(1.0)
            pbin = (np.floor(ph * nphase).astype(np.int64)) % nphase
            H = np.zeros((nphase, nmag))
            H[0, 0] += incoming; incoming = 0
            for pb, mb in zip(pbin, mbin):
                if mb < nmag:
                    H[pb, mb] += 1
                elif pb + 1 < nphase:
                    H[pb + 1, 0] += 1
                else:
                    incoming = 1
            Hs.append(H)
        out = np.zeros(len(freqs))
        for i, H in enumerate(Hs):
            Nphi = H.sum(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                term = np.where(H > 0, H * np.log((1.0 / nmag) * Nphi[:, None] / H), 0.0)
            out[i] = term.sum() / H.sum()
        return out
    (tn, yn, _), = normalize_light_curves([(t, y, dy)])
    freqs = F[:300]
    proc = ConditionalEntropyAsyncProcess(use_fast=False)
    g = np.copy(proc.run([(t, y, dy)], freqs=[freqs])[0][1]); proc.finish()
    print('standard kernel vs global-spill model: %s' % fmt(metrics(ce_ref_global(tn, yn, freqs), g)))
    tt, yy, ddy = make_lc(N=5, seed=3)
    (tn5, yn5, _), = normalize_light_curves([(tt, yy, ddy)])
    g5 = np.copy(proc.run([(tt, yy, ddy)], freqs=[freqs])[0][1]); proc.finish()
    print('N=5 standard kernel vs global-spill model: %s' % fmt(metrics(ce_ref_global(tn5, yn5, freqs), g5)))

elif which == 'ls_k0':
    from cuvarbase.lombscargle import LombScargleAsyncProcess
    from astropy.timeseries import LombScargle
    proc = LombScargleAsyncProcess()
    T = t.max() - t.min()
    df = 1.0 / (5 * T)
    print('baseline T=%.1f d, df=%.3g (5 samples/peak)' % (T, df))
    for fmin, fmax in ((0.002, 3.0), (0.1, 3.0), (0.5, 3.0), (1.0, 3.0), (1.5, 3.0), (2.0, 3.0), (2.5, 3.0), (1.0, 1.5), (5.0, 10.0)):
        k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df))
        f = df * (k0 + np.arange(nf))
        g = np.copy(proc.run([(t, y, dy)], freqs=[f])[0][1]); proc.finish()
        a = LombScargle(t, y, dy).power(f)
        m = metrics(a, g)
        print('fmin=%-5g fmax=%-4g k0=%-5d nf=%-5d k0/nf=%.2f (k0+nf)/(sigma*nf)=%.2f  maxabs=%.2e rel=%.2e corr=%.5f argmax=%s' % (
            fmin, fmax, k0, nf, k0 / nf, (k0 + nf) / (4.0 * nf), m['maxabs'], m['maxrel'], m.get('corr', float('nan')), 'same' if m['argmax_same'] else 'DIFF'))
    # does autoadjust / sigma help?  (LombScargleAsyncProcess(sigma=...))
    for sigma in (4, 8, 16):
        proc2 = LombScargleAsyncProcess(sigma=sigma)
        fmin, fmax = 1.5, 3.0
        k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df)); f = df * (k0 + np.arange(nf))
        g = np.copy(proc2.run([(t, y, dy)], freqs=[f])[0][1]); proc2.finish()
        a = LombScargle(t, y, dy).power(f); m = metrics(a, g)
        print('sigma=%d fmin=1.5 fmax=3: maxabs=%.2e rel=%.2e corr=%.5f' % (sigma, m['maxabs'], m['maxrel'], m.get('corr', float('nan'))))
    # the direct-sum and cufinufft backends on the failing grid
    fmin, fmax = 2.0, 3.0
    k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df)); f = df * (k0 + np.arange(nf))
    a = LombScargle(t, y, dy).power(f)
    g = np.copy(proc.run([(t, y, dy)], freqs=[f], use_fft=False)[0][1]); proc.finish()
    print('direct sums on fmin=2,fmax=3: %s' % fmt(metrics(a, g)))
    p3 = LombScargleAsyncProcess(use_cufinufft=True)
    g = np.copy(p3.run([(t, y, dy)], freqs=[f])[0][1]); p3.finish()
    print('cufinufft on fmin=2,fmax=3: %s' % fmt(metrics(a, g)))
    # the 0.5% floor at k0 small: sigma / m dependence
    for sigma, m_ in ((4, 8), (4, 12), (2, 8), (8, 8)):
        p4 = LombScargleAsyncProcess(sigma=sigma, m=m_)
        f = df * (50 + np.arange(1024))
        g = np.copy(p4.run([(t, y, dy)], freqs=[f])[0][1]); p4.finish()
        a = LombScargle(t, y, dy).power(f); mm = metrics(a, g)
        print('sigma=%d m=%d k0=50 nf=1024: maxabs=%.2e rel=%.2e' % (sigma, m_, mm['maxabs'], mm['maxrel']))

elif which == 'perf_compile':
    from cuvarbase import bls as B
    import time
    for name, fn, args, kw in (
        ('eebls_gpu (std)', B.eebls_gpu, (t, y, dy, F[:300]), dict(qmin=0.01, qmax=0.2)),
        ('eebls_gpu_custom', B.eebls_gpu_custom, (t, y, dy, F[:100], np.array([0.02, 0.05]), np.linspace(0, 1, 50)), {}),
        ('sparse_bls_gpu', B.sparse_bls_gpu, (t, y, dy, F[:100]), {}),
        ('eebls_gpu_fast (cached)', B.eebls_gpu_fast, (t, y, dy, F[:300]), dict(qmin=0.01, qmax=0.2)),
    ):
        ts = []
        for i in range(4):
            t0 = time.time(); fn(*args, **kw); cuda.Context.synchronize(); ts.append(time.time() - t0)
        print('%-24s call times (s): %s' % (name, ' '.join('%.3f' % x for x in ts)))
    fns = B.compile_bls()
    ts = []
    for i in range(3):
        t0 = time.time(); B.eebls_gpu(t, y, dy, F[:300], qmin=0.01, qmax=0.2, functions=fns); cuda.Context.synchronize(); ts.append(time.time() - t0)
    print('%-24s call times (s): %s' % ('eebls_gpu functions=', ' '.join('%.3f' % x for x in ts)))
    k = B.compile_sparse_bls(block_size=64)
    ts = []
    for i in range(3):
        t0 = time.time(); B.sparse_bls_gpu(t, y, dy, F[:100], kernel=k); cuda.Context.synchronize(); ts.append(time.time() - t0)
    print('%-24s call times (s): %s' % ('sparse_bls_gpu kernel=', ' '.join('%.3f' % x for x in ts)))

elif which == 'perf_host':
    from cuvarbase.lombscargle import LombScargleAsyncProcess
    from cuvarbase.memory.lombscargle_memory import weights
    import time
    N = int(1e6)
    tt, yy, ddy = make_lc(N=N, T=1000.0)
    w = np.power(ddy, -2)
    t0 = time.time(); s = sum(w); t1 = time.time(); s2 = np.sum(w); t2 = time.time()
    print('N=1e6: builtin sum(w)=%.3fs  np.sum=%.4fs' % (t1 - t0, t2 - t1))
    t0 = time.time(); a = min(tt); b = max(tt); t1 = time.time(); a2 = tt.min(); b2 = tt.max(); t2 = time.time()
    print('N=1e6: builtin min/max(t)=%.3fs  np.min/max=%.4fs' % (t1 - t0, t2 - t1))
    proc = LombScargleAsyncProcess()
    f = (1.0 / 5000.0) * (100 + np.arange(2000))
    proc.run([(tt, yy, ddy)], freqs=[f]); proc.finish()
    ts = []
    for i in range(3):
        t0 = time.time(); proc.run([(tt, yy, ddy)], freqs=[f]); proc.finish(); ts.append(time.time() - t0)
    print('LS run() N=1e6 nf=2000 wall (s): %s' % ' '.join('%.3f' % x for x in ts))
    # profile host pieces
    import cProfile, pstats, io
    pr = cProfile.Profile(); pr.enable(); proc.run([(tt, yy, ddy)], freqs=[f]); proc.finish(); pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(14); print('\n'.join(s.getvalue().splitlines()[:30]))

elif which == 'bls_std_oom':
    from cuvarbase import bls as B
    tt, yy, ddy = make_lc(N=2000, T=1000.0)
    f = np.linspace(0.05, 10.0, int(1e6))
    print('free before: %.2f GB' % (cuda.mem_get_info()[0] / 1e9))
    st, r, dt, err = run_safely(B.eebls_gpu, tt, yy, ddy, f, qmin=0.01, qmax=0.2)
    print('eebls_gpu nf=1e6 default max_memory: %s (%.1fs)' % (('RAISE ' + err) if st != 'ok' else 'ok', dt))
    st, r, dt, err = run_safely(B.eebls_gpu, tt, yy, ddy, f, qmin=0.01, qmax=0.2, max_memory=int(4e9))
    print('eebls_gpu nf=1e6 max_memory=4GB: %s (%.1fs)' % (('RAISE ' + err) if st != 'ok' else 'ok', dt))
