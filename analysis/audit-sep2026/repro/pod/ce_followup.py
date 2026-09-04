import sys, time
import numpy as np
from cuvarbase.ce import ConditionalEntropyAsyncProcess, conditional_entropy_fast
from cuvarbase.utils import normalize_light_curves
sys.path.insert(0, '/workspace/scratch')
from ce_audit import make, run_gpu, run_gpu_with_bins, hist_ref, ce_from_hist, prep

section = sys.argv[1]

if section == 'stdvsfast':
    print("=== standard_ce vs ce_classical_fast on identical data (both mis-bin the y==1 point differently) ===")
    for seed in range(3):
        t, y, err = make(300, 20.0, 1.3, 10 + seed)
        freqs = np.linspace(0.1, 3.0, 400)
        s = run_gpu(ConditionalEntropyAsyncProcess(), t, y, err, freqs)
        f = run_gpu(ConditionalEntropyAsyncProcess(use_fast=True), t, y, err, freqs)
        ref = ce_from_hist(hist_ref(t, y, freqs, 10, 5, clip=True), 5)
        # correct-hist variant: what if we clip y to (1 - eps)?
        yc = y.copy(); 
        print("seed %d: max|standard-fast|=%.3e  max|standard-ref|=%.3e max|fast-ref|=%.3e  (values ~%.2f)" % (seed, np.max(np.abs(s - f)), np.max(np.abs(s - ref)), np.max(np.abs(f - ref)), ref.mean()))
    # N=3 case
    r = np.random.RandomState(0); t = np.sort(r.rand(3) * 20); y = r.randn(3); freqs = np.linspace(0.1, 3.0, 50)
    s = run_gpu(ConditionalEntropyAsyncProcess(), t, y, np.ones(3), freqs); f = run_gpu(ConditionalEntropyAsyncProcess(use_fast=True), t, y, np.ones(3), freqs)
    print("N=3: standard=%s fast=%s" % (np.unique(np.round(s, 4)), np.unique(np.round(f, 4))))

if section == 'balanced':
    print("=== balanced_magbins: constructor kwarg vs run kwarg ===")
    t, y, err = make(400, 20.0, 1.3, 4); freqs = np.linspace(0.1, 3.0, 400)
    p_ctor = ConditionalEntropyAsyncProcess(balanced_magbins=True)
    print("proc has attribute balanced_magbins:", hasattr(p_ctor, 'balanced_magbins'))
    g_ctor = run_gpu(p_ctor, t, y, err, freqs)
    g_plain = run_gpu(ConditionalEntropyAsyncProcess(), t, y, err, freqs)
    g_run = run_gpu(ConditionalEntropyAsyncProcess(), t, y, err, freqs, balanced_magbins=True)
    print("max|ctor(balanced=True) - plain|=%.3e   max|run(balanced=True) - plain|=%.3e" % (np.max(np.abs(g_ctor - g_plain)), np.max(np.abs(g_run - g_plain))))
    # memory object flag
    proc = ConditionalEntropyAsyncProcess(balanced_magbins=True)
    mems = proc.allocate(normalize_light_curves([(t, y, err)]), freqs=[freqs])
    print("memory.balanced_magbins via allocate() on a proc constructed with balanced_magbins=True:", mems[0].balanced_magbins)
    # reference for constdpdm: balanced bins + mag_bwf widths
    proc = ConditionalEntropyAsyncProcess()
    proc.run([(t, y, err)], freqs=freqs); proc.finish()
    data = normalize_light_curves([(t, y, err)])
    mems = proc.allocate(data, freqs=[freqs], balanced_magbins=True); mems[0].transfer_freqs_to_gpu()
    res = proc.run([(t, y, err)], memory=mems, freqs=[freqs], balanced_magbins=True); proc.finish()
    g = np.copy(res[0][1]); mem = mems[0]
    ybins = mem.y[:mem.n0].astype(int); bwf = mem.mag_bwf.astype(np.float64)
    t32, _, _ = prep(t, y)
    H = np.zeros((len(freqs), 10, 5))
    for i, f in enumerate(freqs.astype(np.float32)):
        ph = t32 * f; ph = (ph - np.floor(ph)).astype(np.float64); n0 = np.floor(10 * ph).astype(int) % 10
        np.add.at(H, (i, n0, ybins), 1)
    Nphi = H.sum(axis=2, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(H > 0, H * np.log(bwf[None, None, :] * Nphi / np.where(H > 0, H, 1)), 0)
    ref = term.sum(axis=(1, 2)) / H.sum(axis=(1, 2))
    print("constdpdm_ce vs numpy(balanced bins, mag_bwf widths): max|d|=%.3e ; mag_bwf=%s sum=%.3f (bin counts %s)" % (np.max(np.abs(g - ref)), np.array2string(bwf, precision=3), bwf.sum(), np.bincount(ybins)))
    # zero-width bins: quantized y
    yq = np.round(y * 2) / 2
    mems = proc.allocate(normalize_light_curves([(t, yq, err)]), freqs=[freqs], balanced_magbins=True); mems[0].transfer_freqs_to_gpu()
    res = proc.run([(t, yq, err)], memory=mems, freqs=[freqs], balanced_magbins=True); proc.finish(); g = np.copy(res[0][1])
    print("quantized y (%d distinct values): mag_bwf=%s -> CE finite=%s n(-inf)=%d n(nan)=%d" % (len(np.unique(yq)), np.array2string(mems[0].mag_bwf, precision=3), np.all(np.isfinite(g)), np.isinf(g).sum(), np.isnan(g).sum()))
    # use_fast + balanced via run kwarg
    gf = run_gpu(ConditionalEntropyAsyncProcess(use_fast=True), t, y, err, freqs, balanced_magbins=True)
    gs = run_gpu(ConditionalEntropyAsyncProcess(), t, y, err, freqs, balanced_magbins=True)
    print("use_fast=True + balanced_magbins(run kwarg): max|fast - standard(constdpdm)|=%.3e (no error raised)" % np.max(np.abs(gf - gs)))
    # compute_log_prob + balanced
    gl = run_gpu(ConditionalEntropyAsyncProcess(compute_log_prob=True), t, y, err, freqs, balanced_magbins=True)
    print("compute_log_prob=True + balanced_magbins: max|result - constdpdm|=%.3e (log_prob silently ignored if 0)" % np.max(np.abs(gl - gs)))

if section == 'accum':
    print("=== set_data=False: histogram accumulation across runs ===")
    r = np.random.RandomState(0); t = np.sort(r.rand(60) * 20); freqs = np.linspace(0.1, 3.0, 50)
    proc = ConditionalEntropyAsyncProcess(); d = [(t, r.randn(60), np.ones(60))]
    proc.run(d, freqs=freqs); proc.finish()
    mems = proc.allocate(normalize_light_curves(d), freqs=[freqs]); mems[0].transfer_freqs_to_gpu()
    for k in range(4):
        res = proc.run(d, memory=mems, freqs=[freqs], set_data=(k == 0)); proc.finish()
        b = mems[0].bins_g.get()
        print("run %d set_data=%s: total bins=%d (N*nf=%d), max bin=%d, ce[0]=%.5f" % (k, k == 0, b.sum(), 60 * 50, b.max(), res[0][1][0]))

if section == 'grid':
    print("=== ce_classical_faster grid-size heuristic: grid=floor(2*shmem_lim/shmem) ===")
    import pycuda.driver as cuda
    from cuvarbase.core import ensure_context
    dev = ensure_context().device
    nsm = dev.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT); shl = dev.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
    print("device SMs=%d shmem_lim=%d" % (nsm, shl))
    for nd in [300, 1000, 2000]:
        t, y, err = make(nd, 20.0, 1.3, 7); freqs = np.linspace(0.1, 20.0, 200000)
        shmem = 8 * 50 + 4 * 10 + 8 * nd
        print("ndata=%d: shmem/block=%d B -> default grid=%d blocks" % (nd, shmem, int(np.floor(2 * shl / shmem))))
        proc = ConditionalEntropyAsyncProcess(use_fast=True); run_gpu(proc, t, y, err, freqs[:100])
        procs = ConditionalEntropyAsyncProcess(); run_gpu(procs, t, y, err, freqs[:100])
        def tm(fn):
            best = 1e9
            for k in range(7):
                t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
            return best
        t_std = tm(lambda: run_gpu(procs, t, y, err, freqs))
        t_def = tm(lambda: run_gpu(proc, t, y, err, freqs))
        out = ["standard=%.4f" % t_std, "fast(default grid)=%.4f" % t_def]
        ref = run_gpu(proc, t, y, err, freqs)
        for nb in [128, 512, 2048, 8192]:
            tt = tm(lambda: run_gpu(proc, t, y, err, freqs, force_nblocks=nb))
            same = np.max(np.abs(run_gpu(proc, t, y, err, freqs, force_nblocks=nb) - ref))
            out.append("fast(nblocks=%d)=%.4f[d=%.0e]" % (nb, tt, same))
        print("   " + "  ".join(out))

if section == 'san_align':
    # to be run under compute-sanitizer
    r = np.random.RandomState(0); t = np.sort(r.rand(200) * 20); y = r.randn(200); freqs = np.linspace(0.1, 3.0, 8)
    proc = ConditionalEntropyAsyncProcess(phase_bins=5, mag_bins=4, use_double=True, use_fast=True)
    try:
        g = run_gpu(proc, t, y, np.ones(200), freqs, shmem_lc=False); print("OK", g)
    except Exception as e:
        print("EXC", type(e).__name__, e)

if section == 'san_oob':
    # standard path: make the y-max point land in phase bin NPHASE-1 at the LAST frequency -> write to bins[nf*NPHASE*NMAG]
    r = np.random.RandomState(0); t = np.sort(r.rand(100) * 20); y = r.randn(100)
    imax = np.argmax(y)
    t32 = (t - t.mean()).astype(np.float32)
    f = 1.0
    # search for a last frequency where the max point's phase is in [0.9,1)
    for f in np.linspace(1.0, 3.0, 20001):
        ph = t32[imax] * np.float32(f); ph = ph - np.floor(ph)
        if ph >= 0.92 and ph < 0.98:
            break
    freqs = np.array([0.5, 0.7, f])
    print("last freq %.6f puts ymax point at phase %.4f (bin %d of 10)" % (f, ph, int(ph * 10)))
    proc = ConditionalEntropyAsyncProcess()
    mems = proc.allocate(normalize_light_curves([(t, y, np.ones(100))]), freqs=[freqs]); mems[0].transfer_freqs_to_gpu()
    proc.run([(t, y, np.ones(100))], freqs=freqs); proc.finish()
    res = proc.run([(t, y, np.ones(100))], memory=mems, freqs=[freqs]); proc.finish()
    b = mems[0].bins_g.get(); print("bins total=%d (expected %d); per-freq sums=%s" % (b.sum(), 300, b.reshape(3, -1).sum(axis=1)))
