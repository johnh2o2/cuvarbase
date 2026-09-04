"""(i) state leakage across calls / memory reuse, (j) batch determinism, base/context behaviour."""
import sys, warnings, os, threading, subprocess
import numpy as np
sys.path.insert(0, '/workspace/scratch/xcut')
from audit_common import make_lc, metrics, fmt, run_safely, ls_freqs
warnings.simplefilter('ignore')

which = sys.argv[1]
tA, yA, dyA = make_lc(N=600, seed=1)
tB, yB, dyB = make_lc(N=900, seed=2, transit=False)
tC, yC, dyC = make_lc(N=300, seed=5)
F = np.linspace(0.1, 3.0, 1500)

if which == 'bls_memory':
    from cuvarbase import bls as B
    from cuvarbase.bls import BLSMemory
    fresh = {k: B.eebls_gpu_fast(*d, F, qmin=0.01, qmax=0.2) for k, d in dict(A=(tA, yA, dyA), B=(tB, yB, dyB), C=(tC, yC, dyC)).items()}
    # reuse one memory object sized for the largest LC
    mem = BLSMemory(max_ndata=900, max_nfreqs=1500)
    for k, d in (('A', (tA, yA, dyA)), ('B', (tB, yB, dyB)), ('C', (tC, yC, dyC)), ('A', (tA, yA, dyA))):
        st, r, _, err = run_safely(B.eebls_gpu_fast, *d, F, qmin=0.01, qmax=0.2, memory=mem)
        print('memory reuse, LC %s: %s' % (k, ('RAISE ' + err) if st != 'ok' else fmt(metrics(fresh[k], r))))
    # memory reuse with different conventions and without re-transfer
    st, r, _, err = run_safely(B.eebls_gpu_fast, tB, yB, dyB, F, qmin=0.01, qmax=0.2, memory=mem, transfer_to_device=False, convention='snr')
    print('memory reuse transfer_to_device=False after loading A, passing B arrays, snr: %s' % (('RAISE ' + err) if st != 'ok' else ('n=%d, vs snr(A)=%s' % (len(r), fmt(metrics(B.convert_bls_power(fresh['A'], yA, dyA, 'snr'), r))))))
    # memory reuse with a SMALLER freq grid than allocation
    st, r, _, err = run_safely(B.eebls_gpu_fast, tA, yA, dyA, F[:700], qmin=0.01, qmax=0.2, memory=mem)
    print('memory reuse with nf=700 < alloc 1500: %s' % (('RAISE ' + err) if st != 'ok' else ('n=%d vs fresh[:700]: %s' % (len(r), fmt(metrics(fresh['A'][:700], r[:700]))))))
    # memory reuse with LARGER ndata than allocation
    tD, yD, dyD = make_lc(N=1200, seed=9)
    st, r, _, err = run_safely(B.eebls_gpu_fast, tD, yD, dyD, F, qmin=0.01, qmax=0.2, memory=mem)
    print('memory reuse with ndata=1200 > alloc 900: %s' % (('RAISE ' + err) if st != 'ok' else 'ok (silent?) n=%d' % len(r)))
    # memory reuse with LARGER nf than allocation
    st, r, _, err = run_safely(B.eebls_gpu_fast, tA, yA, dyA, np.linspace(0.1, 3.0, 2500), qmin=0.01, qmax=0.2, memory=mem)
    print('memory reuse with nf=2500 > alloc 1500: %s' % (('RAISE ' + err) if st != 'ok' else 'ok (silent?) n=%d' % len(r)))
    # chi2_0 / convention after reuse
    r1 = B.eebls_gpu_fast(tA, yA, dyA, F, qmin=0.01, qmax=0.2, convention='snr')
    r2 = B.eebls_gpu_fast(tA, yA, dyA, F, qmin=0.01, qmax=0.2, memory=BLSMemory(600, 1500), convention='snr')
    print('snr convention fresh vs memory=: %s' % fmt(metrics(r1, r2)))
    # kernel cache: functions= from a different block size
    fn64 = B._get_cached_kernels(64, False, ['full_bls_no_sol', 'full_bls_no_sol_fused'])
    st, r, _, err = run_safely(B.eebls_gpu_fast, tA, yA, dyA, F, qmin=0.01, qmax=0.2, functions=fn64)
    print('functions compiled for block 64 launched with default block 256: %s' % (('RAISE ' + err) if st != 'ok' else fmt(metrics(fresh['A'], r))))
    st, r, _, err = run_safely(B.eebls_gpu_fast, tA, yA, dyA, F, qmin=0.01, qmax=0.2, functions=fn64, block_size=64)
    print('functions for block 64 with block_size=64: %s' % (('RAISE ' + err) if st != 'ok' else fmt(metrics(fresh['A'], r))))

elif which == 'bls_batch_det':
    from cuvarbase import bls as B
    fA = B.eebls_gpu_fast(tA, yA, dyA, F, qmin=0.01, qmax=0.2)
    fB = B.eebls_gpu_fast(tB, yB, dyB, F, qmin=0.01, qmax=0.2)
    fC = B.eebls_gpu_fast(tC, yC, dyC, F, qmin=0.01, qmax=0.2)
    lcs = [(tA, yA, dyA), (tB, yB, dyB), (tC, yC, dyC), (tA, yA, dyA)]
    r1 = B.eebls_gpu_batch(lcs, F, qmin=0.01, qmax=0.2)
    r2 = B.eebls_gpu_batch(lcs, F, qmin=0.01, qmax=0.2)
    for i, (ref, nm) in enumerate(((fA, 'A'), (fB, 'B'), (fC, 'C'), (fA, 'A'))):
        print('batch[%d]=%s vs single fast: %s | batch run twice bitwise: %s' % (i, nm, fmt(metrics(ref, r1[i])), np.array_equal(r1[i], r2[i])))
    r3 = B.eebls_gpu_batch(lcs, F, qmin=0.01, qmax=0.2, max_batch_lcs=1)
    print('max_batch_lcs=1 vs 4: %s' % [fmt(metrics(a, b)) for a, b in zip(r1, r3)])
    # memory= reuse across calls with different LC counts
    from cuvarbase.memory.bls_memory import BLSBatchMemory
    import pycuda.driver as cuda
    mem = BLSBatchMemory(900, 4, 1500, stream=cuda.Stream())
    r4 = B.eebls_gpu_batch(lcs, F, qmin=0.01, qmax=0.2, memory=mem)
    r5 = B.eebls_gpu_batch(lcs[:2], F, qmin=0.01, qmax=0.2, memory=mem)
    r6 = B.eebls_gpu_batch(lcs, F[:800], qmin=0.01, qmax=0.2, memory=mem)
    print('memory reuse 4 LCs: %s' % [fmt(metrics(a, b)) for a, b in zip(r1, r4)])
    print('memory reuse then 2 LCs: %s' % [fmt(metrics(a, b)) for a, b in zip(r1[:2], r5)])
    print('memory reuse then nf=800: %s' % [fmt(metrics(a[:800], b)) for a, b in zip(r1, r6)])

elif which == 'ls_state':
    from cuvarbase.lombscargle import LombScargleAsyncProcess
    f = ls_freqs(1500)
    proc = LombScargleAsyncProcess()
    def one(d):
        r = proc.run([d], freqs=[f]); proc.finish(); return np.copy(r[0][1])
    fresh = {k: one(d) for k, d in dict(A=(tA, yA, dyA), B=(tB, yB, dyB), C=(tC, yC, dyC)).items()}
    seq = [('A', (tA, yA, dyA)), ('B', (tB, yB, dyB)), ('C', (tC, yC, dyC)), ('A', (tA, yA, dyA))]
    r = proc.batched_run_const_nfreq([d for _, d in seq], freqs=f, batch_size=1)
    print('batched bs=1 vs fresh:', [fmt(metrics(fresh[k], p)) for (k, _), (_, p) in zip(seq, r)])
    r4 = proc.batched_run_const_nfreq([d for _, d in seq], freqs=f, batch_size=4)
    print('batched bs=4 vs fresh:', [fmt(metrics(fresh[k], p)) for (k, _), (_, p) in zip(seq, r4)])
    print('bs=1 vs bs=4 bitwise:', [np.array_equal(a[1], b[1]) for a, b in zip(r, r4)])
    # run() with several LCs at once (multi-stream)
    rr = proc.run([d for _, d in seq], freqs=[f] * 4); proc.finish()
    print('run() 4 LCs multi-stream vs fresh:', [fmt(metrics(fresh[k], np.copy(p))) for (k, _), (_, p) in zip(seq, rr)])
    # returned arrays alias pinned buffers? modify after next run
    rr = proc.run([(tA, yA, dyA)], freqs=[f]); proc.finish(); pa = rr[0][1]
    rr2 = proc.run([(tB, yB, dyB)], freqs=[f]); proc.finish()
    print('run() result array from call 1 still equals fresh A after call 2 (aliasing check): %s' % fmt(metrics(fresh['A'], pa)))
    # preallocate path with an LC bigger than max_nobs
    proc.preallocate(max_nobs=700, nlcs=1, freqs=f)
    st, r, _, err = run_safely(one, (tB, yB, dyB))
    print('preallocate(max_nobs=700) then N=900: %s' % (('RAISE ' + err) if st != 'ok' else ('ok n=%d vs fresh: %s' % (len(r), fmt(metrics(fresh['B'], r))))))
    proc.memory = None

elif which == 'ce_pdm_state':
    from cuvarbase.ce import ConditionalEntropyAsyncProcess
    from cuvarbase.pdm import PDMAsyncProcess
    for fast in (False, True):
        proc = ConditionalEntropyAsyncProcess(use_fast=fast)
        def one(d):
            r = proc.run([d], freqs=[F]); proc.finish(); return np.copy(r[0][1])
        fresh = {k: one(d) for k, d in dict(A=(tA, yA, dyA), B=(tB, yB, dyB), C=(tC, yC, dyC)).items()}
        seq = [('A', (tA, yA, dyA)), ('B', (tB, yB, dyB)), ('C', (tC, yC, dyC)), ('A', (tA, yA, dyA))]
        r = proc.batched_run_const_nfreq([d for _, d in seq], freqs=F, batch_size=4)
        print('CE fast=%s batched bs=4 vs fresh:' % fast, [fmt(metrics(fresh[k], p)) for (k, _), (_, p) in zip(seq, r)])
        r1 = proc.batched_run_const_nfreq([d for _, d in seq], freqs=F, batch_size=1)
        print('CE fast=%s batched bs=1 vs bs=4 bitwise:' % fast, [np.array_equal(a[1], b[1]) for a, b in zip(r, r1)])
        rr = proc.run([d for _, d in seq], freqs=[F] * 4); proc.finish()
        print('CE fast=%s run() 4 LCs vs fresh:' % fast, [fmt(metrics(fresh[k], np.copy(p))) for (k, _), (_, p) in zip(seq, rr)])
        # preallocate then run with smaller LCs (buffered transfer reuse): stale tail?
        proc.preallocate(max_nobs=900, freqs=F, nlcs=1)
        rB = one((tB, yB, dyB)); rC = one((tC, yC, dyC))
        print('CE fast=%s preallocate(900): B then C(300): C vs fresh: %s' % (fast, fmt(metrics(fresh['C'], rC))))
        proc.memory = None
    proc = PDMAsyncProcess()
    def one(d, kind='binned_linterp'):
        r = proc.run([d], freqs=F, kind=kind); proc.finish(); return np.copy(r[0][1])
    fresh = {k: one(d) for k, d in dict(A=(tA, yA, dyA), B=(tB, yB, dyB), C=(tC, yC, dyC)).items()}
    seq = [('A', (tA, yA, dyA)), ('B', (tB, yB, dyB)), ('C', (tC, yC, dyC)), ('A', (tA, yA, dyA))]
    r = proc.batched_run_const_nfreq([d for _, d in seq], freqs=F, batch_size=4)
    print('PDM batched bs=4 vs fresh:', [fmt(metrics(fresh[k], p)) for (k, _), (_, p) in zip(seq, r)])
    rr = proc.run([d for _, d in seq], freqs=F); proc.finish()
    print('PDM run() 4 LCs vs fresh:', [fmt(metrics(fresh[k], np.copy(p))) for (k, _), (_, p) in zip(seq, rr)])
    # gpu_data reuse from allocate() with a different LC
    gd, pc = proc.allocate([(tA, yA, dyA)], freqs=[F])
    st, r, _, err = run_safely(lambda: (proc.run([(tB, yB, dyB)], gpu_data=gd, pow_cpus=pc, freqs=F), proc.finish())[0])
    print('PDM gpu_data allocated for N=600 reused for N=900: %s' % (('RAISE ' + err) if st != 'ok' else 'ok (silent) vs fresh B: ' + fmt(metrics(fresh['B'], np.copy(r[0][1])))))

elif which == 'tls_state':
    from cuvarbase import tls as TL
    P = np.linspace(2, 6, 400)
    fresh = {k: TL.tls_search_gpu(*d, periods=P)['chi2'] for k, d in dict(A=(tA, yA, dyA), B=(tB, yB, dyB), C=(tC, yC, dyC)).items()}
    seq = [('A', (tA, yA, dyA)), ('B', (tB, yB, dyB)), ('C', (tC, yC, dyC)), ('A', (tA, yA, dyA))]
    r = TL.tls_search_batch([d for _, d in seq], periods=P, return_arrays=True)
    print('TLS batch(4) vs single:', [fmt(metrics(fresh[k], p['chi2'])) for (k, _), p in zip(seq, r)])
    r2 = TL.tls_search_batch([d for _, d in seq], periods=P, return_arrays=True)
    print('TLS batch twice bitwise chi2:', [np.array_equal(a['chi2'], b['chi2']) for a, b in zip(r, r2)], 'SDE equal:', [a['SDE'] == b['SDE'] for a, b in zip(r, r2)])
    # legacy memory reuse
    from cuvarbase.tls import TLSMemory
    mem = TLSMemory(900, 400)
    fl = {k: TL.tls_search_gpu(*d, periods=P, use_fast=False)['chi2'] for k, d in dict(A=(tA, yA, dyA), B=(tB, yB, dyB)).items()}
    for k, d in seq[:3]:
        st, rr, _, err = run_safely(TL.tls_search_gpu, *d, periods=P, use_fast=False, memory=mem)
        print('TLS legacy memory reuse LC %s: %s' % (k, ('RAISE ' + err) if st != 'ok' else fmt(metrics(fl.get(k, TL.tls_search_gpu(*d, periods=P, use_fast=False)['chi2']), rr['chi2']))))

elif which == 'threads':
    from cuvarbase import bls as B
    ref = B.eebls_gpu_fast(tA, yA, dyA, F, qmin=0.01, qmax=0.2)
    out = {}
    def work(i):
        try:
            r = B.eebls_gpu_fast(tA, yA, dyA, F, qmin=0.01, qmax=0.2)
            out[i] = fmt(metrics(ref, r))
        except BaseException as e:
            out[i] = 'RAISE %s: %s' % (type(e).__name__, str(e).splitlines()[0][:150] if str(e) else '')
    ths = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [th.start() for th in ths]; [th.join() for th in ths]
    for i in sorted(out):
        print('thread %d: %s' % (i, out[i]))
    # main thread still works?
    st, r, _, err = run_safely(B.eebls_gpu_fast, tA, yA, dyA, F, qmin=0.01, qmax=0.2)
    print('main thread after worker threads: %s' % (('RAISE ' + err) if st != 'ok' else fmt(metrics(ref, r))))

elif which == 'sticky':
    from cuvarbase import bls as B
    import pycuda.driver as cuda
    ref = B.eebls_gpu_fast(tA, yA, dyA, F, qmin=0.01, qmax=0.2)
    for bad in ({'qmin': np.nan}, {'qmin': 0.0}, {'qmin': -0.01}, {'qmax': 0.0}, {'qmin': 0.3, 'qmax': 0.2}):
        kw = dict(qmin=0.01, qmax=0.2); kw.update(bad)
        st, r, _, err = run_safely(B.eebls_gpu_fast, tA, yA, dyA, F, **kw)
        print('eebls_gpu_fast %s: %s' % (bad, ('RAISE ' + err) if st != 'ok' else ('ok n=%d nonfinite=%d vs ref: %s' % (len(r), int(np.sum(~np.isfinite(r))), fmt(metrics(ref, r))))))
        st, r, _, err = run_safely(B.eebls_gpu_fast, tA, yA, dyA, F, qmin=0.01, qmax=0.2)
        print('   ...subsequent valid call: %s' % (('RAISE ' + err) if st != 'ok' else fmt(metrics(ref, r))))
        if st != 'ok':
            break

elif which == 'sticky_nanfreq':
    from cuvarbase import bls as B
    ref = B.eebls_gpu_fast(tA, yA, dyA, F, qmin=0.01, qmax=0.2)
    st, r, _, err = run_safely(B.eebls_transit_gpu, *make_lc(N=3, seed=3), use_fast=True)
    print('eebls_transit_gpu N=3 use_fast: %s' % (('RAISE ' + err) if st != 'ok' else 'ok n=%d' % len(r[1])))
    st, r, _, err = run_safely(B.eebls_gpu_fast, tA, yA, dyA, F, qmin=0.01, qmax=0.2)
    print('   ...subsequent valid call: %s' % (('RAISE ' + err) if st != 'ok' else fmt(metrics(ref, r))))
    from cuvarbase.base import ensure_context
    st, r, _, err = run_safely(ensure_context)
    print('   ensure_context after sticky error: %s' % (('RAISE ' + err) if st != 'ok' else 'returns cached module (no recovery attempted)'))
    st, r, _, err = run_safely(lambda: B.transit_autofreq(make_lc(N=3, seed=3)[0]))
    print('transit_autofreq N=3 -> qvals: %s' % (('RAISE ' + err) if st != 'ok' else str(r[1][:5])))
    st, r, _, err = run_safely(lambda: B.fmin_transit(make_lc(N=3, seed=3)[0]))
    print('fmin_transit N=3 -> %s' % (('RAISE ' + err) if st != 'ok' else r))

elif which == 'context':
    # CUDA_DEVICE handling and two processes
    env = dict(os.environ, CUDA_DEVICE='7')
    p = subprocess.run([sys.executable, '-c', 'import numpy as np\nfrom cuvarbase.bls import eebls_gpu_fast\nprint(eebls_gpu_fast(np.linspace(0,10,50), np.ones(50), np.ones(50), np.linspace(0.1,1,10))[:2])'], cwd='/workspace/cuvarbase', env=env, capture_output=True, text=True, timeout=300)
    print('CUDA_DEVICE=7 (nonexistent): rc=%d\n stdout=%s\n stderr(tail)=%s' % (p.returncode, p.stdout.strip(), '\n'.join(p.stderr.strip().splitlines()[-3:])))
    env = dict(os.environ, CUDA_DEVICE='0')
    p = subprocess.run([sys.executable, '-c', 'import cuvarbase.base as b\nm=b.ensure_context()\nm2=b.ensure_context()\nprint(m is m2, m.device.name(), m.context is m2.context)\nimport pycuda.driver as cuda\nprint("current ctx is primary:", cuda.Context.get_current() is not None)'], cwd='/workspace/cuvarbase', env=env, capture_output=True, text=True, timeout=300)
    print('ensure_context idempotence: rc=%d stdout=%s stderr=%s' % (p.returncode, p.stdout.strip(), p.stderr.strip()[-300:]))
    # import without GPU touching: does `import cuvarbase.bls` create a context?
    p = subprocess.run([sys.executable, '-c', 'import cuvarbase.bls, cuvarbase.lombscargle, cuvarbase.ce, cuvarbase.pdm, cuvarbase.tls\nimport cuvarbase.base as b\nprint("ctx created at import:", b._autoctx is not None)'], cwd='/workspace/cuvarbase', capture_output=True, text=True, timeout=300)
    print('import-time context: rc=%d stdout=%s stderr=%s' % (p.returncode, p.stdout.strip(), p.stderr.strip()[-300:]))
    # two processes concurrently
    code = 'import numpy as np\nfrom cuvarbase.bls import eebls_gpu_fast\nt=np.linspace(0,100,600); y=np.ones(600); dy=np.ones(600)\nr=eebls_gpu_fast(t,y,dy,np.linspace(0.1,3,2000))\nprint(np.sum(np.isfinite(r)))'
    ps = [subprocess.Popen([sys.executable, '-c', code], cwd='/workspace/cuvarbase', stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(3)]
    for i, pp in enumerate(ps):
        o, e = pp.communicate(timeout=300)
        print('proc %d rc=%d out=%s err=%s' % (i, pp.returncode, o.strip(), e.strip()[-200:]))
    # fork after context creation
    p = subprocess.run([sys.executable, '-c', 'import numpy as np, os\nfrom cuvarbase.bls import eebls_gpu_fast\nt=np.linspace(0,100,600); y=np.ones(600); dy=np.ones(600)\nr=eebls_gpu_fast(t,y,dy,np.linspace(0.1,3,200))\npid=os.fork()\nif pid==0:\n    try:\n        r=eebls_gpu_fast(t,y,dy,np.linspace(0.1,3,200)); print("child ok", np.isfinite(r).all())\n    except BaseException as e:\n        print("child RAISE", type(e).__name__, str(e).splitlines()[0][:120])\n    os._exit(0)\nos.waitpid(pid,0)\nprint("parent after fork ok", np.isfinite(eebls_gpu_fast(t,y,dy,np.linspace(0.1,3,200))).all())'], cwd='/workspace/cuvarbase', capture_output=True, text=True, timeout=300)
    print('fork after context: rc=%d stdout=%s stderr=%s' % (p.returncode, p.stdout.strip(), p.stderr.strip()[-300:]))
