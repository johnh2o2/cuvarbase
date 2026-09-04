"""Kernel prototype: replace the two shared-memory float atomicAdds per point (each an ATOMS.CAST.SPIN loop on sm_89)
with ONE 64-bit CAS loop on a packed (yw, w) float2 bin. Same float arithmetic per bin; compare outputs + event timing."""
import time, json, sys
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from prof_common import *
from cuvarbase.utils import find_kernel, _module_reader
from cuvarbase.bls import BLSMemory, _function_signatures, _get_cached_kernels, eebls_gpu_fast

src = _module_reader(find_kernel('bls'), cpp_defs=dict(BLOCK_SIZE=256))
i0 = src.index('__global__ void full_bls_no_sol_fused('); i1 = src.index('// needs ndata * nfreq threads', i0)
fused = src[i0:i1]
p64 = fused.replace('full_bls_no_sol_fused(', 'full_bls_no_sol_fused_p64(')
p64 = p64.replace('''	float *fine_yw = sh;
	float *fine_w = (float *)&sh[hist_size];
	float *best_bls = (float *)&sh[2 * hist_size];''', '''	float2 *fine = (float2 *) sh;                 // packed (yw, w) per fine bin; 8-byte aligned
	float *best_bls = (float *)&sh[2 * hist_size];''')
p64 = p64.replace('''			fine_yw[k] = 0.f;
			fine_w[k] = 0.f;''', '''			fine[k] = make_float2(0.f, 0.f);''')
p64 = p64.replace('''			atomicAdd(&(fine_yw[j]), yw[k]);
			atomicAdd(&(fine_w[j]), w[k]);''', '''			atomicAdd2(&(fine[j]), yw[k], w[k]);''')
p64 = p64.replace('''					thread_yw += fine_yw[idx];
					thread_w += fine_w[idx];''', '''					float2 fb = fine[idx];
					thread_yw += fb.x;
					thread_w += fb.y;''')
assert p64.count('atomicAdd2') == 1 and 'fine_yw' not in p64
helper = r'''
__device__ __forceinline__ void atomicAdd2(float2 *addr, float a, float b){
	unsigned long long *p = (unsigned long long *) addr;
	unsigned long long old = *p, assumed;
	do {
		assumed = old;
		float2 v = *reinterpret_cast<float2 *>(&assumed);
		v.x += a; v.y += b;
		old = atomicCAS(p, assumed, *reinterpret_cast<unsigned long long *>(&v));
	} while (assumed != old);
}
'''
mod = SourceModule(src + helper + p64, options=['--use_fast_math'])
k_new = mod.get_function('full_bls_no_sol_fused_p64').prepare(_function_signatures['full_bls_no_sol_fused'])
k_old = _get_cached_kernels(256, False, ['full_bls_no_sol', 'full_bls_no_sol_fused'])['full_bls_no_sol_fused']
out = {}
for name in ('ZTF', 'TESS', 'HAT'):
    cfg = SURVEYS[name]; ndata = cfg['ndata']; freqs, qmins, qmaxs = grid_for(cfg); nfreq = len(freqs); t, y, dy = make_lc(cfg, 1000)
    mem = BLSMemory(ndata, nfreq); mem.setdata(t, y, dy, qmin=qmins, qmax=qmaxs, freqs=freqs, transfer=True); sync()
    max_nbins = int(np.max(mem.nbinsf[:nfreq])); hist = 2*max_nbins; mem_req = (256 + 2*hist)*4
    def launch(k):
        e0, e1 = cuda.Event(), cuda.Event(); e0.record()
        k.prepared_call((min(nfreq, 5000), 1), (256, 1, 1), mem.t_g.ptr, mem.yw_g.ptr, mem.w_g.ptr, mem.bls_g.ptr, mem.freqs_g.ptr, mem.nbins0_g.ptr, mem.nbinsf_g.ptr,
                        np.uint32(ndata), np.uint32(nfreq), np.uint32(0), np.uint32(hist), np.uint32(2), np.float32(0.3), np.float32(0.0), np.uint32(0), shared_size=mem_req)
        e1.record(); e1.synchronize(); return e0.time_till(e1)
    R = out[name] = dict(shared_bytes=mem_req)
    for lab, k in (('stock', k_old), ('packed64', k_new)):
        launch(k); ts = [launch(k) for _ in range(9)]; R[lab+'_ms'] = dict(min=float(np.min(ts)), med=float(np.median(ts)))
        R[lab+'_bls'] = mem.bls_g.get() / mem.yy
    a, b = R.pop('stock_bls'), R.pop('packed64_bls')
    R['speedup_min'] = R['stock_ms']['min'] / R['packed64_ms']['min']
    R['parity'] = dict(max_abs_diff=float(np.max(np.abs(a-b))), argmax_same=bool(np.argmax(a) == np.argmax(b)), corr=float(np.corrcoef(a, b)[0,1]), n_bitdiff=int(np.sum(a != b)))
    # run-to-run nondeterminism of the stock kernel itself, for scale
    launch(k_old); c = mem.bls_g.get()/mem.yy; launch(k_old); d = mem.bls_g.get()/mem.yy
    R['stock_run_to_run'] = dict(max_abs_diff=float(np.max(np.abs(c-d))), n_bitdiff=int(np.sum(c != d)))
    print(name, json.dumps(R, indent=1), flush=True)
    # ALSO: time-sorted (no scatter) input, to see how the packed CAS behaves under conflicts
json.dump(out, open('/workspace/scratch/packed_proto.json', 'w'), indent=1, default=str)
