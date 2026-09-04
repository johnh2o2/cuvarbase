"""Verifier for topic nfft-absolute-time (auditor id 31).
Isolate: is |ghat| corruption at BJD from (a) float32 cast of t in NFFTMemory.fromdata
or (b) float32 phase factors in normalize()?"""
import sys, warnings
import numpy as np
sys.path.insert(0, '/workspace/scratch/xcut')
from audit_common import make_lc, metrics, fmt
warnings.simplefilter('ignore')
from cuvarbase.cunfft import NFFTAsyncProcess

t, y, dy = make_lc()
NF = 512
K = np.arange(NF // 2)   # accurate band

def exact(tt, yy, rel=True):
    tt = np.asarray(tt, dtype=np.float64)
    T = tt.max() - tt.min()
    tr = tt - tt.min() if rel else tt
    return np.array([np.sum(yy * np.exp(2j * np.pi * (kk / T) * tr)) for kk in K])

def rep(label, a, b):
    print('%-70s %s' % (label, fmt(metrics(a, b))))

p32 = NFFTAsyncProcess()
p64 = NFFTAsyncProcess(use_double=True)
def run(p, tt):
    g = p.run([(np.asarray(tt, dtype=np.float64), y, NF)])[0]; p.finish(); return g[:NF//2]

ref_rel = exact(t, y)          # magnitudes of relative-time transform = magnitudes of absolute-time transform
for off in (0.0, 1000.5, 2457000.5):
    tb = t + off
    g32 = run(p32, tb); g64 = run(p64, tb)
    print('\n=== offset %.1f  float32 spacing of t: %.3g d ===' % (off, float(np.spacing(np.float32(tb.max())))))
    rep('|ghat| f32 vs exact (offset-invariant magnitudes)', np.abs(g32), np.abs(ref_rel))
    rep('|ghat| f64 vs exact', np.abs(g64), np.abs(ref_rel))
    # hypothesis (a): the float32-quantised times explain the f32 result
    tq = tb.astype(np.float32).astype(np.float64)
    ref_q = exact(tq, y)
    rep('|ghat| f32 vs exact DFT of float32-QUANTISED times', np.abs(g32), np.abs(ref_q))
    # phases: absolute-t convention claimed by nufft_lrt.compute_nufft comment
    ref_abs = exact(tb, y, rel=False)
    ph32 = np.angle(g32 * np.conj(ref_abs)); ph64 = np.angle(g64 * np.conj(ref_abs))
    print('   phase err vs ABSOLUTE-t DFT: f32 max=%.3g rad rms=%.3g | f64 max=%.3g rad rms=%.3g' % (
        np.abs(ph32).max(), np.sqrt(np.mean(ph32**2)), np.abs(ph64).max(), np.sqrt(np.mean(ph64**2))))
    # phases vs exact DFT of quantised times with absolute phase (isolates phase-factor error from quantisation)
    ref_qabs = exact(tq, y, rel=False)
    phq = np.angle(g32 * np.conj(ref_qabs))
    print('   f32 phase err vs ABSOLUTE-t DFT of quantised times: max=%.3g rad rms=%.3g' % (np.abs(phq).max(), np.sqrt(np.mean(phq**2))))
    # fix candidate: host float64 epoch subtraction
    tf = tb - np.floor(tb.min())
    gf = run(p32, tf)
    rep('|ghat| f32 with host floor(min(t)) subtracted vs exact', np.abs(gf), np.abs(ref_rel))
    # what does |ghat| at 0 offset vs BJD look like (auditor metric)?
    if off > 0:
        rep('|ghat| f32 t vs t+offset (auditor metric)', np.abs(run(p32, t)), np.abs(g32))

# (b) direct check: does the float32 normalize phase factor change magnitudes? Compare
# complex ratio magnitude |g32(t+off)| / |g32(t)| where quantisation is negligible (off=1000.5)
# -> already covered above by 'vs exact'. Additionally, isolate __cosf/__sinf at 1e7 rad.
import pycuda.autoprimaryctx  # noqa
import pycuda.driver as cuda, pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
src = r"""
__global__ void mag(float *th, float *out, int n){
  int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float c=cos(th[i]), s=sin(th[i]); out[i]=sqrtf(c*c+s*s);} }
"""
mod = SourceModule(src, options=['--use_fast_math'])
th = (2*np.pi*2457000.5*np.linspace(0, 2.56, 4096)).astype(np.float32)
th_g = gpuarray.to_gpu(th); out = gpuarray.zeros(len(th), np.float32)
mod.get_function('mag')(th_g, out, np.int32(len(th)), block=(256,1,1), grid=(16,1,1))
o = out.get()
print('\nfast-math |cos+i sin| at theta up to %.3g rad: min=%.6f max=%.6f' % (th.max(), o.min(), o.max()))
