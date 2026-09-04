import sys, time
import numpy as np
from cuvarbase.ce import ConditionalEntropyAsyncProcess
from cuvarbase.utils import normalize_light_curves
sys.path.insert(0, '/workspace/scratch')
from ce_audit import make, run_gpu
section = sys.argv[1]
if section == 'zerowidth':
    t, y, err = make(400, 20.0, 1.3, 4); freqs = np.linspace(0.1, 3.0, 100)
    y3 = np.round(y)  # few distinct values
    proc = ConditionalEntropyAsyncProcess()
    proc.run([(t, y3, err)], freqs=freqs); proc.finish()
    mems = proc.allocate(normalize_light_curves([(t, y3, err)]), freqs=[freqs], balanced_magbins=True); mems[0].transfer_freqs_to_gpu()
    res = proc.run([(t, y3, err)], memory=mems, freqs=[freqs], balanced_magbins=True); proc.finish(); g = np.copy(res[0][1])
    print("y with %d distinct values, balanced_magbins=True: mag_bwf=%s -> CE finite=%s n(-inf)=%d n(nan)=%d; sample=%s" % (len(np.unique(y3)), np.array2string(mems[0].mag_bwf, precision=3), np.all(np.isfinite(g)), np.isinf(g).sum(), np.isnan(g).sum(), g[:3]))
if section == 'grid2':
    import pycuda.driver as cuda
    t, y, err = make(300, 20.0, 1.3, 7); freqs = np.linspace(0.1, 20.0, 200000)
    proc = ConditionalEntropyAsyncProcess(use_fast=True); run_gpu(proc, t, y, err, freqs[:100])
    procs = ConditionalEntropyAsyncProcess(); run_gpu(procs, t, y, err, freqs[:100])
    def tm(fn, n=15):
        best = 1e9
        for k in range(n):
            t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
        return best
    for rnd in range(2):
        out = ["standard=%.4f" % tm(lambda: run_gpu(procs, t, y, err, freqs)), "fast(default=34 blocks)=%.4f" % tm(lambda: run_gpu(proc, t, y, err, freqs))]
        for nb in [64, 128, 256, 512, 1024]:
            out.append("nblocks=%d:%.4f" % (nb, tm(lambda: run_gpu(proc, t, y, err, freqs, force_nblocks=nb))))
        print("ndata=300 nf=200000 round %d: " % rnd + "  ".join(out))
    # ndata=1500
    t, y, err = make(1500, 20.0, 1.3, 7)
    for rnd in range(2):
        out = ["standard=%.4f" % tm(lambda: run_gpu(procs, t, y, err, freqs)), "fast(default=%d blocks)=%.4f" % (int(np.floor(2 * 49152 / (440 + 8 * 1500))), tm(lambda: run_gpu(proc, t, y, err, freqs)))]
        for nb in [64, 128, 256, 512, 1024]:
            out.append("nblocks=%d:%.4f" % (nb, tm(lambda: run_gpu(proc, t, y, err, freqs, force_nblocks=nb))))
        print("ndata=1500 nf=200000 round %d: " % rnd + "  ".join(out))
