import sys, os, warnings, subprocess
import numpy as np
sys.path.insert(0, '/workspace/scratch/xcut')
from audit_common import make_lc, metrics, fmt, run_safely, ls_freqs
warnings.simplefilter('ignore')
which = sys.argv[1]
t, y, dy = make_lc()
F = np.linspace(0.1, 3.0, 1500)

if which == 'ce_prealloc':
    from cuvarbase.ce import ConditionalEntropyAsyncProcess
    tB, yB, dyB = make_lc(N=900, seed=2, transit=False)
    tC, yC, dyC = make_lc(N=300, seed=5)
    for fast in (False, True):
        proc = ConditionalEntropyAsyncProcess(use_fast=fast)
        fB = np.copy(proc.run([(tB, yB, dyB)], freqs=[F])[0][1]); proc.finish()
        fC = np.copy(proc.run([(tC, yC, dyC)], freqs=[F])[0][1]); proc.finish()
        proc.preallocate(max_nobs=900, freqs=F, nlcs=1)
        rB = np.copy(proc.run([(tB, yB, dyB)], freqs=[F])[0][1]); proc.finish()
        rC = np.copy(proc.run([(tC, yC, dyC)], freqs=[F])[0][1]); proc.finish()
        print('fast=%s after preallocate(): B vs fresh %s | std(rB)=%.3g ; C vs fresh %s | std(rC)=%.3g' % (fast, fmt(metrics(fB, rB)), rB.std(), fmt(metrics(fC, rC)), rC.std()))
        print('   freqs_g on device after preallocate: first 3 = %s' % proc.memory[0].freqs_g.get()[:3])
        for m in proc.memory:
            m.transfer_freqs_to_gpu()
        rB2 = np.copy(proc.run([(tB, yB, dyB)], freqs=[F])[0][1]); proc.finish()
        rC2 = np.copy(proc.run([(tC, yC, dyC)], freqs=[F])[0][1]); proc.finish()
        print('   after manual transfer_freqs_to_gpu(): B %s ; C %s' % (fmt(metrics(fB, rB2)), fmt(metrics(fC, rC2))))

elif which == 'ce_spill':
    from cuvarbase.ce import ConditionalEntropyAsyncProcess
    from cuvarbase.utils import normalize_light_curves
    def ce_ref(t, y, freqs, nphase=10, nmag=5, mode='clip'):
        y = np.asarray(y, dtype=np.float32); t = np.asarray(t, dtype=np.float32)
        yn = (y - y.min()) / (y.max() - y.min())
        mbin = np.floor(yn * nmag).astype(np.int64)      # max point -> nmag (out of range)
        out = np.zeros(len(freqs))
        for i, f in enumerate(freqs):
            ph = (t * np.float32(f)) % np.float32(1.0)
            pbin = (np.floor(ph * nphase).astype(np.int64)) % nphase
            H = np.zeros((nphase, nmag))
            extra_phase0 = 0
            for pb, mb in zip(pbin, mbin):
                if mb < nmag:
                    H[pb, mb] += 1
                elif mode == 'clip':
                    H[pb, nmag - 1] += 1
                elif mode == 'spill':
                    # flat index pb*nmag + nmag == (pb+1)*nmag + 0
                    if pb + 1 < nphase:
                        H[pb + 1, 0] += 1
                    else:
                        extra_phase0 += 1   # fast kernel: lands in block_bin_phi[0]
                elif mode == 'drop':
                    pass
            Nphi = H.sum(axis=1)
            if mode == 'spill':
                Nphi = Nphi.copy(); Nphi[0] += extra_phase0
            with np.errstate(divide='ignore', invalid='ignore'):
                term = np.where(H > 0, H * np.log((1.0 / nmag) * Nphi[:, None] / H), 0.0)
            out[i] = term.sum() / (H.sum() + (extra_phase0 if mode == 'spill' else 0))
        return out
    (tn, yn, _), = normalize_light_curves([(t, y, dy)])
    freqs = F[:300]
    refs = {m: ce_ref(tn, yn, freqs, mode=m) for m in ('clip', 'spill', 'drop')}
    for fast in (False, True):
        proc = ConditionalEntropyAsyncProcess(use_fast=fast)
        g = np.copy(proc.run([(t, y, dy)], freqs=[freqs])[0][1]); proc.finish()
        for m, r in refs.items():
            print('fast=%s GPU vs %-5s reference: %s' % (fast, m, fmt(metrics(r, g))))
    # the phase bin of the max point: spill affects freqs where the max point falls in the last phase bin
    imax = np.argmax(yn)
    ph = (np.float32(tn[imax]) * freqs.astype(np.float32)) % np.float32(1)
    last = np.floor(ph * 10) == 9
    proc = ConditionalEntropyAsyncProcess(use_fast=False)
    g = np.copy(proc.run([(t, y, dy)], freqs=[freqs])[0][1]); proc.finish()
    d = np.abs(g - refs['spill'])
    print('std kernel: max|GPU-spill_ref| where max point in LAST phase bin (%d freqs) = %.3g ; elsewhere = %.3g' % (last.sum(), d[last].max(), d[~last].max()))
    print('N=5 case: GPU vs refs:')
    tt, yy, ddy = make_lc(N=5, seed=3)
    (tn5, yn5, _), = normalize_light_curves([(tt, yy, ddy)])
    for fast in (False, True):
        proc = ConditionalEntropyAsyncProcess(use_fast=fast)
        g = np.copy(proc.run([(tt, yy, ddy)], freqs=[freqs])[0][1]); proc.finish()
        for m in ('clip', 'spill', 'drop'):
            print('   fast=%s vs %-5s: %s' % (fast, m, fmt(metrics(ce_ref(tn5, yn5, freqs, mode=m), g))))

elif which == 'tls_batch_consistency':
    from cuvarbase import tls as TL
    P = np.linspace(2, 6, 400)
    tB, yB, dyB = make_lc(N=900, seed=2, transit=False)
    tC, yC, dyC = make_lc(N=300, seed=5)
    lcs = [(t, y, dy), (tB, yB, dyB), (tC, yC, dyC), (t, y, dy)]
    singles = [TL.tls_search_batch([lc], periods=P, return_arrays=True)[0] for lc in lcs[:3]] + [None]
    singles[3] = singles[0]
    b1 = TL.tls_search_batch(lcs, periods=P, return_arrays=True)
    b2 = TL.tls_search_batch(lcs, periods=P, return_arrays=True)
    for i in range(4):
        print('LC%d batch vs single chi2: %s | batch twice: maxabs=%.3g | SDE single/b1/b2 = %.4f/%.4f/%.4f | period %.5f/%.5f/%.5f' % (
            i, fmt(metrics(singles[i]['chi2'], b1[i]['chi2'])), np.nanmax(np.abs(b1[i]['chi2'] - b2[i]['chi2'])),
            singles[i]['SDE'], b1[i]['SDE'], b2[i]['SDE'], singles[i]['period'], b1[i]['period'], b2[i]['period']))
    # run-to-run for a single LC
    r = [TL.tls_search_batch([lcs[0]], periods=P, return_arrays=True)[0] for _ in range(5)]
    print('single LC 5 runs: chi2 max spread = %.3g, SDE values = %s, t0 = %s' % (
        max(np.nanmax(np.abs(a['chi2'] - r[0]['chi2'])) for a in r), [round(a['SDE'], 4) for a in r], [round(a['t0_phase'], 6) for a in r]))

elif which == 'nanfreq':
    from cuvarbase import bls as B
    ref = B.eebls_gpu_fast(t, y, dy, F, qmin=0.01, qmax=0.2)
    case = sys.argv[2]
    if case == 'nan_freq':
        st, r, _, err = run_safely(B.eebls_gpu_fast, t, y, dy, np.array([np.nan, 0.5]), qmin=0.01, qmax=0.2)
    elif case == 'nan_q':
        st, r, _, err = run_safely(B.eebls_gpu_fast, t, y, dy, F[:2], qmin=np.array([np.nan, 0.01]), qmax=np.array([np.nan, 0.2]))
    elif case == 'nan_both_N3':
        t3, y3, d3 = make_lc(N=3, seed=3)
        st, r, _, err = run_safely(B.eebls_gpu_fast, t3, y3, d3, np.array([np.nan]), qmin=np.array([np.nan]), qmax=np.array([np.nan]))
    elif case == 'nan_t_transit':
        tt = t.copy(); tt[137] = np.nan
        st, r, _, err = run_safely(B.eebls_transit_gpu, tt, y, dy, use_fast=True)
    elif case == 'nan_t_fast':
        tt = t.copy(); tt[137] = np.nan
        st, r, _, err = run_safely(B.eebls_gpu_fast, tt, y, dy, F, qmin=0.01, qmax=0.2)
    elif case == 'qmax_gt1':
        st, r, _, err = run_safely(B.eebls_gpu_fast, t, y, dy, F, qmin=0.01, qmax=5.0)
    elif case == 'nbins0_zero':
        st, r, _, err = run_safely(B.eebls_gpu_fast, t, y, dy, F, qmin=0.01, qmax=np.inf)
    print('%s: %s' % (case, ('RAISE ' + err) if st != 'ok' else 'ok nonfinite=%d vs ref %s' % (int(np.sum(~np.isfinite(r))), fmt(metrics(ref, r)) if len(r) == len(ref) else 'n=%d' % len(r))))
    st, r, _, err = run_safely(B.eebls_gpu_fast, t, y, dy, F, qmin=0.01, qmax=0.2)
    print('   subsequent valid call: %s' % (('RAISE ' + err) if st != 'ok' else fmt(metrics(ref, r))))

elif which == 'ls_accuracy':
    from cuvarbase.lombscargle import LombScargleAsyncProcess
    from astropy.timeseries import LombScargle
    for dbl in (False, True):
        proc = LombScargleAsyncProcess(use_double=dbl)
        for k0 in (1, 50, 500, 5000):
            for nf in (64, 256, 1024, 4096):
                df = 1.0 / 500.0
                f = df * (k0 + np.arange(nf))
                g = np.copy(proc.run([(t, y, dy)], freqs=[f])[0][1]); proc.finish()
                a = LombScargle(t, y, dy).power(f)
                m = metrics(a, g)
                print('double=%s k0=%-5d nf=%-5d  vs astropy: maxabs=%.2e rel=%.2e corr=%.6f' % (dbl, k0, nf, m['maxabs'], m['maxrel'], m.get('corr', float('nan'))))

elif which == 'ls_dy0':
    from cuvarbase.lombscargle import LombScargleAsyncProcess
    from astropy.timeseries import LombScargle
    proc = LombScargleAsyncProcess()
    f = ls_freqs(1500)
    K = 137
    for name, (tt, yy, ddy) in {'dy0': (t, y, np.where(np.arange(600) == K, 0.0, dy)), 'yNaN': (t, np.where(np.arange(600) == K, np.nan, y), dy), 'dy_tiny': (t, y, np.where(np.arange(600) == K, 1e-9, dy))}.items():
        g = np.copy(proc.run([(tt, yy, ddy)], freqs=[f])[0][1]); proc.finish()
        keep = np.arange(600) != K
        a = LombScargle(t[keep], y[keep], dy[keep]).power(f)
        print('%s: GPU LS power range [%.3g, %.3g], vs astropy(point dropped): %s' % (name, g.min(), g.max(), fmt(metrics(a, g))))

elif which == 'context2':
    p = subprocess.run([sys.executable, '-c', 'import cuvarbase.bls, cuvarbase.lombscargle, cuvarbase.ce, cuvarbase.pdm, cuvarbase.tls\nimport cuvarbase.base.context as c\nprint("ctx created at import:", c._autoctx is not None)'], cwd='/workspace/cuvarbase', capture_output=True, text=True, timeout=300)
    print('import-time context: rc=%d stdout=%s stderr=%s' % (p.returncode, p.stdout.strip(), p.stderr.strip()[-300:]))
    code = ('import numpy as np, os, sys\nfrom cuvarbase.bls import eebls_gpu_fast\n'
            'rng=np.random.RandomState(0); t=np.sort(rng.uniform(0,100,600)); y=1+0.01*rng.randn(600); dy=0.01*np.ones(600)\nF=np.linspace(0.1,3,200)\n'
            'r0=eebls_gpu_fast(t,y,dy,F)\npid=os.fork()\n'
            'if pid==0:\n    try:\n        r=eebls_gpu_fast(t,y,dy,F); print("child ok maxdiff", np.abs(r-r0).max(), flush=True)\n'
            '    except BaseException as e:\n        print("child RAISE", type(e).__name__, str(e).splitlines()[0][:150], flush=True)\n    os._exit(0)\n'
            'os.waitpid(pid,0)\nprint("parent after fork maxdiff", np.abs(eebls_gpu_fast(t,y,dy,F)-r0).max(), flush=True)')
    p = subprocess.run([sys.executable, '-c', code], cwd='/workspace/cuvarbase', capture_output=True, text=True, timeout=300)
    print('fork after context: rc=%d stdout=%s stderr=%s' % (p.returncode, p.stdout.strip(), p.stderr.strip()[-300:]))
    # multiprocessing spawn (the sane way)
    code2 = ('import numpy as np, multiprocessing as mp\ndef w(_):\n    from cuvarbase.bls import eebls_gpu_fast\n    t=np.linspace(0,100,600); y=1+0.01*np.sin(t); dy=0.01*np.ones(600)\n    return float(eebls_gpu_fast(t,y,dy,np.linspace(0.1,3,200)).max())\n'
             'if __name__=="__main__":\n    with mp.get_context("spawn").Pool(2) as p: print("spawn pool:", p.map(w,[0,1]))')
    open('/workspace/scratch/xcut/_spawn.py', 'w').write(code2)
    p = subprocess.run([sys.executable, '/workspace/scratch/xcut/_spawn.py'], cwd='/workspace/cuvarbase', capture_output=True, text=True, timeout=300)
    print('spawn pool: rc=%d stdout=%s stderr=%s' % (p.returncode, p.stdout.strip(), p.stderr.strip()[-200:]))

elif which == 'sparse_limit':
    from cuvarbase import bls as B
    import pycuda.driver as cuda
    dev = cuda.Context.get_device() if False else None
    for N in (1500, 2000, 2500, 3000):
        tt, yy, ddy = make_lc(N=N)
        st, r, _, err = run_safely(B.sparse_bls_gpu, tt, yy, ddy, F[:20])
        n_pow2 = 1
        while n_pow2 < N: n_pow2 *= 2
        smem = (3 * n_pow2 + 2 * N + 3 * 64) * 4
        print('N=%d shared=%d bytes: %s' % (N, smem, ('RAISE ' + err) if st != 'ok' else 'ok'))
    # eebls_transit auto path uses sparse below 500 only; what does a user get calling eebls_transit(use_sparse=True) at N=3000?
    tt, yy, ddy = make_lc(N=3000)
    st, r, _, err = run_safely(B.eebls_transit, tt, yy, ddy, use_sparse=True)
    print('eebls_transit(use_sparse=True, N=3000): %s' % (('RAISE ' + err) if st != 'ok' else 'ok'))
