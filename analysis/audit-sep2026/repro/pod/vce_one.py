"""Run ONE CE config in a fresh process. Usage: python vce_one.py {orig|fixed} PB MB shmem_lc use_double use_fast"""
import sys, numpy as np
mode, PB, MB, shl, dbl, fast = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4])), bool(int(sys.argv[5])), bool(int(sys.argv[6]))
import cuvarbase.ce as cemod
from cuvarbase.utils import find_kernel
if mode == 'fixed':
    src = open(find_kernel('ce')).read()
    old = "unsigned int r = ((nmag * nphase + nphase) * sizeof(unsigned int)) % sizeof(FLT);\n\tFLT * Hc = (FLT *)&block_bin_phi[nphase + r];"
    assert src.count(old) == 2
    src = src.replace(old, "unsigned int r = (((nmag * nphase + nphase) * sizeof(unsigned int)) % sizeof(FLT)) / sizeof(unsigned int);\n\tFLT * Hc = (FLT *)&block_bin_phi[nphase + r];")
    open('/workspace/scratch/ce_fixed.cu', 'w').write(src)
    cemod.find_kernel = lambda name: '/workspace/scratch/ce_fixed.cu'
rng = np.random.RandomState(0); N = int(sys.argv[7]) if len(sys.argv) > 7 else 200
t = np.sort(rng.rand(N) * 20); y = 12 + 0.3*np.cos(2*np.pi*t*1.7) + 0.05*rng.randn(N); dy = 0.05*np.ones(N)
freqs = np.linspace(0.1, 3.0, 256)
rkw = dict(shmem_lc=shl) if fast else {}
p = cemod.ConditionalEntropyAsyncProcess(phase_bins=PB, mag_bins=MB, use_double=dbl, use_fast=fast)
res = p.run([(t, y, dy)], freqs=freqs, **rkw); p.finish()
print("OK mode=%s PB=%d MB=%d shmem_lc=%d double=%d fast=%d  min=%.5f argmin=%d" % (mode, PB, MB, shl, dbl, fast, res[0][1].min(), np.argmin(res[0][1])))
