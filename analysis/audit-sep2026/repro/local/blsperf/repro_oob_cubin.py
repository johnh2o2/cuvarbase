"""Same reproducer, but loads a pre-built cubin (no nvcc subprocess) so it can run under compute-sanitizer."""
import sys, os, subprocess, numpy as np
import pycuda.driver as cuda, pycuda.autoprimaryctx  # noqa
from cuvarbase.bls import eebls_gpu, count_tot_nbins, _function_signatures, _all_function_names
from cuvarbase.utils import find_kernel, _module_reader
from cuvarbase.bls_frequencies import keplerian_freq_grid
ndata = int(sys.argv[1]) if len(sys.argv) > 1 else 600
maxmem = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5e9
cubin = '/workspace/scratch/bls_externc.cubin'
if not os.path.exists(cubin):
    src = 'extern "C" {\n' + _module_reader(find_kernel('bls'), cpp_defs=dict(BLOCK_SIZE=256)) + '\n}\n'
    open('/workspace/scratch/bls_externc.cu', 'w').write(src)
    subprocess.check_call(['nvcc', '-arch=sm_89', '--use_fast_math', '-cubin', '-o', cubin, '/workspace/scratch/bls_externc.cu'])
mod = cuda.module_from_file(cubin)
# compile-free stand-in for gpuarray.zeros (pycuda's fill() would JIT an elementwise kernel -> nvcc under the sanitizer fails)
import cuvarbase.bls as B
class _Z:
    def __init__(self, n, dtype=np.float32):
        self.dtype = np.dtype(dtype); self.n = int(n); self.nbytes = self.n*self.dtype.itemsize
        self.gpudata = cuda.mem_alloc(max(self.nbytes, 4)); self.ptr = int(self.gpudata); cuda.memset_d32(self.gpudata, 0, self.n)
    def fill(self, v, stream=None): cuda.memset_d32_async(self.gpudata, 0, self.n, stream)
    def get(self): a = np.empty(self.n, self.dtype); cuda.memcpy_dtoh(a, self.gpudata); return a
class _NS: pass
ns = _NS(); ns.zeros = lambda n, dtype=np.float32: _Z(n, dtype); ns.to_gpu = B.gpuarray.to_gpu
B.gpuarray = ns
fr = {n: mod.get_function(n).prepare(_function_signatures[n]) for n in _all_function_names if n != 'full_bls_no_sol_optimized'}
f, q = keplerian_freq_grid(0.5, 100., 3650., oversampling=2, return_qvals=True)
f = f.astype(np.float64)[:20000]; qmins = (0.5*q[:20000]).astype(np.float64); qmaxs = (2.0*q[:20000]).astype(np.float64)
rng = np.random.RandomState(0); t = np.sort(rng.uniform(0, 3650., ndata)); y = 1 + 0.002*rng.randn(ndata); dy = np.full(ndata, 0.002)
try:
    p, sols = eebls_gpu(t, y, dy, f, qmin=qmins, qmax=qmaxs, functions=fr, max_memory=maxmem)
    cuda.Context.synchronize(); print("call returned; max power", float(np.max(p)))
except Exception as e:
    print("EXCEPTION:", repr(e))
