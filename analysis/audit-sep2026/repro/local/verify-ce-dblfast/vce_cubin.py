"""Two-stage: 'build' compiles the (orig|fixed) CE kernel to a cubin (run WITHOUT sanitizer);
'run' loads it via module_from_file and runs ConditionalEntropyAsyncProcess (run UNDER sanitizer).
Usage: python vce_cubin.py {build|run} {orig|fixed} PB MB shmem_lc use_double use_fast ndata"""
import sys, subprocess, numpy as np
stage, mode, PB, MB = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
shl, dbl, fast, N = bool(int(sys.argv[5])), bool(int(sys.argv[6])), bool(int(sys.argv[7])), int(sys.argv[8])
import cuvarbase.ce as cemod
from cuvarbase.utils import find_kernel, _module_reader
cubin = '/workspace/scratch/ce_%s_%d_%d_%d.cubin' % (mode, PB, MB, dbl)
if stage == 'build':
    src = open(find_kernel('ce')).read()
    if mode == 'fixed':
        old = "unsigned int r = ((nmag * nphase + nphase) * sizeof(unsigned int)) % sizeof(FLT);\n\tFLT * Hc = (FLT *)&block_bin_phi[nphase + r];"
        assert src.count(old) == 2
        src = src.replace(old, "unsigned int r = (((nmag * nphase + nphase) * sizeof(unsigned int)) % sizeof(FLT)) / sizeof(unsigned int);\n\tFLT * Hc = (FLT *)&block_bin_phi[nphase + r];")
    d = dict(NPHASE=PB, NMAG=MB, PHASE_OVERLAP=0, MAG_OVERLAP=0)
    if dbl: d['DOUBLE_PRECISION'] = None
    cu = '/workspace/scratch/ce_%s_%d_%d_%d.cu' % (mode, PB, MB, dbl)
    open(cu, 'w').write(_module_reader.__globals__['_module_reader'](cu, cpp_defs=d) if False else None or '')
    # _module_reader reads a file path: write raw src first, then expand
    open(cu, 'w').write(src)
    txt = _module_reader(cu, cpp_defs=d)
    open(cu, 'w').write('extern "C" {\n' + txt + '\n}\n')
    subprocess.check_call(['nvcc', '-cubin', '-arch=sm_89', '--use_fast_math', '-lineinfo', '-o', cubin, cu])
    print("built", cubin); sys.exit(0)
import pycuda.driver as drv
cemod.SourceModule = lambda txt, options=None: drv.module_from_file(cubin)
rng = np.random.RandomState(0)
t = np.sort(rng.rand(N) * 20); y = 12 + 0.3*np.cos(2*np.pi*t*1.7) + 0.05*rng.randn(N); dy = 0.05*np.ones(N)
freqs = np.linspace(0.1, 3.0, 256)
rkw = dict(shmem_lc=shl) if fast else {}
p = cemod.ConditionalEntropyAsyncProcess(phase_bins=PB, mag_bins=MB, use_double=dbl, use_fast=fast)
res = p.run([(t, y, dy)], freqs=freqs, **rkw); p.finish()
print("OK mode=%s PB=%d MB=%d shmem_lc=%d double=%d fast=%d ndata=%d  min=%.5f argmin=%d" % (mode, PB, MB, shl, dbl, fast, N, res[0][1].min(), np.argmin(res[0][1])))
