"""Sparse-BLS pair decode: stock O(N) per-pair while-loop decode vs incremental decode. Bit-parity + timing."""
import time, json, sys, re
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from prof_common import *
from cuvarbase.utils import find_kernel, _module_reader
from cuvarbase.bls import compile_sparse_bls, sparse_bls_gpu, sparse_bls_cpu

src = _module_reader(find_kernel('sparse_bls'), cpp_defs=dict(BLOCK_SIZE=64))
i0 = src.index('// Step 7'); i1 = src.index('// Step 8')
NEW = r'''// Step 7 (PATCHED): incremental pair decode -- O(N + pairs/blockDim) per thread instead of O(N * pairs/blockDim)
        float thread_max_bls = 0.f;
        float thread_q = 0.f;
        float thread_phi0 = 0.f;
        unsigned int N = ndata;
        unsigned int total_nonwrap = N * (N + 1) / 2;
        unsigned int total_wrap = N * (N - 1) / 2;
        {
            unsigned int i = 0, idx = tid;
            while (i < N && idx >= (N - i)) { idx -= (N - i); i++; }
            for (unsigned int p = tid; p < total_nonwrap; p += blockDim.x) {
                unsigned int j = i + 1 + idx;
                float phi0 = sh_phi[i];
                float q = (j < N) ? (0.5f * (sh_phi[j] + sh_phi[j-1]) - phi0) : (sh_phi[N - 1] - phi0 + 1e-7f);
                if (!(q <= 0.f || q < qmin_f || q > qmax_f)) {
                    unsigned int last = (j < N) ? j - 1 : N - 1;
                    float W = (i == 0) ? sh_cumsum_w[last] : sh_cumsum_w[last] - sh_cumsum_w[i - 1];
                    float YW = (i == 0) ? sh_cumsum_yw[last] : sh_cumsum_yw[last] - sh_cumsum_yw[i - 1];
                    YW -= ybar * W;
                    float bls = bls_power(YW, W, YY, ignore_negative_delta_sols);
                    if (bls > thread_max_bls) { thread_max_bls = bls; thread_q = q; thread_phi0 = phi0; }
                }
                idx += blockDim.x;
                while (i < N && idx >= (N - i)) { idx -= (N - i); i++; }
            }
        }
        {
            unsigned int i = 1, idx = tid;
            while (i < N && idx >= i) { idx -= i; i++; }
            for (unsigned int p = tid; p < total_wrap; p += blockDim.x) {
                unsigned int k = idx;
                float phi0 = sh_phi[i];
                float q = (k > 0) ? ((1.f - phi0) + 0.5f * (sh_phi[k-1] + sh_phi[k])) : (1.f - phi0 + 1e-7f);
                if (!(q <= 0.f || q < qmin_f || q > qmax_f)) {
                    float W = sh_cumsum_w[N - 1] - sh_cumsum_w[i - 1];
                    float YW = sh_cumsum_yw[N - 1] - sh_cumsum_yw[i - 1];
                    if (k > 0) { W += sh_cumsum_w[k - 1]; YW += sh_cumsum_yw[k - 1]; }
                    YW -= ybar * W;
                    float bls = bls_power(YW, W, YY, ignore_negative_delta_sols);
                    if (bls > thread_max_bls) { thread_max_bls = bls; thread_q = q; thread_phi0 = phi0; }
                }
                idx += blockDim.x;
                while (i < N && idx >= i) { idx -= i; i++; }
            }
        }

        '''
patched_src = src[:i0] + NEW + src[i1:]
mod = SourceModule(patched_src, options=['--use_fast_math'])
k_new = mod.get_function('sparse_bls_kernel')
k_old = compile_sparse_bls(block_size=64)

def run(kern, t, y, dy, freqs, qmn, qmx, bs, ndata):
    tt = (t - np.floor(t.min())).astype(np.float32)
    t_g = gpuarray.to_gpu(tt); y_g = gpuarray.to_gpu(y.astype(np.float32)); dy_g = gpuarray.to_gpu(dy.astype(np.float32))
    f_g = gpuarray.to_gpu(freqs.astype(np.float32)); a_g = gpuarray.to_gpu(qmn.astype(np.float32)); b_g = gpuarray.to_gpu(qmx.astype(np.float32))
    nf = len(freqs); p_g = gpuarray.zeros(nf, np.float32); q_g = gpuarray.zeros(nf, np.float32); ph_g = gpuarray.zeros(nf, np.float32)
    n_pow2 = 1
    while n_pow2 < ndata: n_pow2 *= 2
    shm = (3*n_pow2 + 2*ndata + 3*bs)*4
    def launch():
        kern(t_g, y_g, dy_g, f_g, a_g, b_g, np.uint32(ndata), np.uint32(nf), np.uint32(0), p_g, q_g, ph_g,
             block=(bs,1,1), grid=(min(nf,65535),1), shared=shm)
    m, allt = med(launch, 5)
    return m, p_g.get(), q_g.get(), ph_g.get(), allt

out = {}
cfg = SURVEYS['ZTF']; freqs, qmins, qmaxs = grid_for(cfg)
for ndata in (150, 300, 499):
    c = dict(cfg); c['ndata'] = ndata; t, y, dy = make_lc(c, 11)
    for bounds_lab, (qa, qb) in (('keplerian', (qmins, qmaxs)), ('unbounded', (np.zeros_like(qmins), np.full_like(qmaxs, 0.5)))):
        for bs in (64, 128, 256):
            ko = compile_sparse_bls(block_size=bs); kn = SourceModule(patched_src.replace('#define BLOCK_SIZE 64', f'#define BLOCK_SIZE {bs}'), options=['--use_fast_math']).get_function('sparse_bls_kernel')
            mo, po, qo, pho, _ = run(ko, t, y, dy, freqs, qa, qb, bs, ndata)
            mn, pn, qn, phn, _ = run(kn, t, y, dy, freqs, qa, qb, bs, ndata)
            key = f'N={ndata} {bounds_lab} bs={bs}'
            out[key] = dict(stock_ms=mo*1e3, patched_ms=mn*1e3, speedup=mo/mn,
                            bit_identical=bool(np.array_equal(po, pn) and np.array_equal(qo, qn) and np.array_equal(pho, phn)),
                            max_abs_diff=float(np.max(np.abs(po-pn))), peak_P=float(1/freqs[np.argmax(pn)]))
            print(key, out[key], flush=True)
# CPU reference parity for one config (subset of freqs)
c = dict(cfg); c['ndata'] = 150; t, y, dy = make_lc(c, 11); sub = freqs[::200]
pc, sc = sparse_bls_cpu(t, y, dy, sub, qmin=qmins[::200], qmax=qmaxs[::200])
mo, po, qo, pho, _ = run(k_new, t, y, dy, sub, qmins[::200], qmaxs[::200], 64, 150)
out['cpu_vs_patched_gpu'] = dict(max_abs_diff=float(np.max(np.abs(pc - po))), corr=float(np.corrcoef(pc, po)[0,1]), argmax_same=bool(np.argmax(pc)==np.argmax(po)))
print(out['cpu_vs_patched_gpu'])
json.dump(out, open('/workspace/scratch/sparse_proto.json','w'), indent=1)
