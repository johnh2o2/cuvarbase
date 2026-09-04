"""Minimal repro of CE use_double+use_fast misaligned shared memory, and test of the element-offset fix.
Usage: python vce_dblfast.py {orig|fixed} PB MB [shmem_lc 0/1]
Runs in a fresh process each time (a crash kills the CUDA context)."""
import sys, os, re, numpy as np
mode, PB, MB = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
shl = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True
import cuvarbase.ce as cemod
from cuvarbase.utils import find_kernel
src_path = find_kernel('ce')
if mode == 'fixed':
    src = open(src_path).read()
    old = "unsigned int r = ((nmag * nphase + nphase) * sizeof(unsigned int)) % sizeof(FLT);\n\tFLT * Hc = (FLT *)&block_bin_phi[nphase + r];"
    assert src.count(old) == 2, src.count(old)
    new = "unsigned int r = (((nmag * nphase + nphase) * sizeof(unsigned int)) % sizeof(FLT)) / sizeof(unsigned int);\n\tFLT * Hc = (FLT *)&block_bin_phi[nphase + r];"
    src = src.replace(old, new)
    patched = '/workspace/scratch/ce_fixed.cu'
    open(patched, 'w').write(src)
    cemod.find_kernel = lambda name: patched if name == 'ce' else find_kernel(name)
rng = np.random.RandomState(0); N = 200
t = np.sort(rng.rand(N) * 20); y = 12 + 0.3*np.cos(2*np.pi*t*1.7) + 0.05*rng.randn(N); dy = 0.05*np.ones(N)
freqs = np.linspace(0.1, 3.0, 256)
def run(**kw):
    rkw = {}
    if kw.get('use_fast'): rkw['shmem_lc'] = shl
    p = cemod.ConditionalEntropyAsyncProcess(phase_bins=PB, mag_bins=MB, **kw)
    res = p.run([(t, y, dy)], freqs=freqs, **rkw); p.finish()
    return res[0][1]
rbytes = ((MB*PB + PB) * 4) % 8
print("mode=%s PB=%d MB=%d shmem_lc=%s r_bytes=%d" % (mode, PB, MB, shl, rbytes))
ref = run(use_double=True, use_fast=False)               # standard double kernel (no shared Hc)
f32 = run(use_double=False, use_fast=True)
print("  float32 fast vs double std: max|d|=%.2e" % np.max(np.abs(f32 - ref)))
d = run(use_double=True, use_fast=True)
print("  double fast vs double std:  max|d|=%.2e  argmax match=%s" % (np.max(np.abs(d - ref)), np.argmin(d) == np.argmin(ref)))
