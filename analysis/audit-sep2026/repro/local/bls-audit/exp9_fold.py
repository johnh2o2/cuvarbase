"""Exp 9: what exactly does the compiled kernel's fold compute at large t*f?
Compile the kernel's own mod1(t*f) and the bin index expression with the
same flags (--use_fast_math) and compare with numpy float32 variants."""
import sys
sys.path.insert(0, '/workspace/scratch/blsaudit')
import numpy as np
import pycuda.autoprimaryctx  # noqa
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from blsref import make_data, ref_fast
from cuvarbase.bls import eebls_gpu_fast, single_bls
from cuvarbase.utils import subtract_epoch

src = r"""
__device__ float mod1(float a){ return a - floorf(a); }
__device__ int mod(int a, int b){ int r = a % b; return (r < 0) ? r + b : r; }
__global__ void fold(const float* t, float f0, int nbf, float dphi, int noverlap, float* phi, int* b_multi, int* j_fused, int n){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n){
        float p = mod1(t[k] * f0);
        phi[k] = p;
        b_multi[k] = mod((int) floorf(((float) nbf) * p - dphi), nbf);
        float u = ((float) nbf) * p - 0.f;
        j_fused[k] = mod((int) floorf(((float) noverlap) * u), nbf * noverlap);
    }
}
"""
mod_fast = SourceModule(src, options=['--use_fast_math'])
mod_nofma = SourceModule(src, options=['--use_fast_math', '--fmad=false'])
mod_plain = SourceModule(src)

for label, t0 in [("t0=0", 0.0), ("BJD", 2455000.5)]:
    t, y, dy = make_data(ndata=5000, baseline=3650., freq=6.3, q=0.06, phi0=0.9, snr=15., seed=5, t0=t0)
    ts, epoch = subtract_epoch(t)
    t32 = ts.astype(np.float32)
    f = 6.3 + 0.00013 * 65
    f32 = np.float32(f)
    nbf = 100
    # numpy variants
    a = np.float32(t32 * f32)
    ph_np = a - np.floor(a)                                   # plain float32 (single_bls model)
    exact_prod = t32.astype(np.float64) * np.float64(f32)     # exact product of the float32 inputs
    ph_fma = (exact_prod - np.floor(a).astype(np.float64)).astype(np.float32)  # fma(t,f,-floor(round(t*f)))
    ph_64 = (ts * f) % 1.0
    t_g = gpuarray.to_gpu(t32)
    for mname, m in [("fast_math", mod_fast), ("fast_math+fmad=false", mod_nofma), ("plain", mod_plain)]:
        fn = m.get_function("fold")
        phi_g = gpuarray.zeros(len(t32), np.float32)
        b_g = gpuarray.zeros(len(t32), np.int32)
        j_g = gpuarray.zeros(len(t32), np.int32)
        fn(t_g, f32, np.int32(nbf), np.float32(0.5), np.int32(2), phi_g, b_g, j_g, np.int32(len(t32)),
           block=(256, 1, 1), grid=((len(t32) + 255) // 256, 1))
        phi = phi_g.get()
        print("%s [%s]: gpu phase == np float32 mod1: %d/%d;  == fma-variant: %d/%d;  max|gpu - f64 fold| = %.2e; max|np32 - f64| = %.2e; max|fma - f64| = %.2e; #phi<0 or >=1: %d"
              % (label, mname, (phi == ph_np).sum(), len(phi), (phi == ph_fma).sum(), len(phi),
                 np.abs(((phi - ph_64) + 0.5) % 1 - 0.5).max(), np.abs(((ph_np - ph_64) + 0.5) % 1 - 0.5).max(),
                 np.abs(((ph_fma - ph_64) + 0.5) % 1 - 0.5).max(), int(((phi < 0) | (phi >= 1)).sum())))
    # distinct float32 phase values (phase quantum)
    print("   distinct float32 phases (np mod1): %d of %d; ulp(t*f max=%.0f) = %.2e cycles"
          % (len(np.unique(ph_np)), len(ph_np), (t32 * f32).max(), np.spacing(np.float32((t32 * f32).max()))))
