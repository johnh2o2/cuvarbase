import numpy as np
from scipy.special import ndtr
import cuvarbase.ce as cemod
from cuvarbase.ce import ConditionalEntropyAsyncProcess
from cuvarbase.utils import normalize_light_curves
import pycuda.compiler

def make(ndata, baseline, f0, seed, noise, amp=1.0):
    r = np.random.RandomState(seed)
    t = np.sort(r.rand(ndata))*baseline
    y = 12 + amp*np.sin(2*np.pi*f0*t) + 0.3*amp*np.sin(4*np.pi*f0*t) + noise*r.randn(ndata)
    return t, y, noise*np.ones(ndata)
def gpu_bins(proc, t, y, dy, freqs):
    proc.run([(t, y, dy)], freqs=freqs); proc.finish()
    data = normalize_light_curves([(t, y, dy)])
    mems = proc.allocate(data, freqs=[freqs]); mems[0].transfer_freqs_to_gpu()
    res = proc.run([(t, y, dy)], memory=mems, freqs=[freqs]); proc.finish()
    return np.copy(res[0][1]), mems[0].bins_g.get().reshape(len(freqs), proc.phase_bins, proc.mag_bins)
def exact_hist(t, y, dy, freqs, PB, MB):
    t = np.asarray(t, np.float64); y = np.asarray(y, np.float64)
    t = (t - t.mean()).astype(np.float32); y = (y - y.mean()).astype(np.float32)
    yscale = y.max()-y.min(); Y = ((y-y.min())/yscale).astype(np.float64); DY = (dy.astype(np.float32)/yscale).astype(np.float64)
    m = np.arange(MB); P = ndtr(((m+1)/MB - Y[:,None])/DY[:,None]) - ndtr((m/MB - Y[:,None])/DY[:,None])
    H = np.zeros((len(freqs), PB, MB))
    for i, f in enumerate(freqs.astype(np.float32)):
        ph = t*np.float32(f); ph = (ph-np.floor(ph)).astype(np.float64)
        n0 = (np.floor(PB*ph).astype(int)) % PB
        np.add.at(H, (i, n0), P)
    return H, DY.mean()*MB

# instrument: which source is being compiled?
_SM = pycuda.compiler.SourceModule
class SM(_SM):
    def __init__(self, src, *a, **k):
        print("   [compiling kernel: %s]" % ("FIXED" if "WHOLE bin" in src else "ORIG"))
        super().__init__(src, *a, **k)
cemod.SourceModule = SM
orig_find = cemod.find_kernel
fixed_find = lambda name: '/workspace/scratch/ce_wasym_fixed_v21.cu' if name == 'ce' else orig_find(name)

fr = np.linspace(0.1, 3.0, 40)
for MB in [5, 10]:
    for noise in [0.05, 0.15, 0.4]:
        t, y, dy = make(300, 20.0, 1.3, 3, noise)
        He, sig_bw = exact_hist(t, y, dy, fr, 10, MB)
        out = {}
        for label, fk in [('FIXED', fixed_find), ('ORIG', orig_find)]:
            cemod.find_kernel = fk
            p = ConditionalEntropyAsyncProcess(phase_bins=10, mag_bins=MB, weighted=True, max_phi=3.0)
            ce, b = gpu_bins(p, t, y, dy, fr)
            out[label] = (ce, b)
        def wce(Hw):
            Nphi = Hw.sum(axis=2, keepdims=True); dm = 1/MB
            with np.errstate(divide='ignore', invalid='ignore'):
                term = np.where((Hw > 0) & (Nphi > 1e-10), Hw*np.log(dm*Nphi/np.where(Hw > 0, Hw, 1)), 0)
            return term.sum(axis=(1, 2))/Hw.sum(axis=(1, 2))
        print("MB=%2d noise=%.2f (sigma=%.2f binwidths): bins max|FIXED-exact|=%.2e  max|ORIG-exact|=%.2e | mass/pt FIXED %.4f ORIG %.4f exact %.4f | CE max|FIXED-exactCE|=%.2e max|ORIG-exactCE|=%.2e"
              % (MB, noise, sig_bw, np.abs(out['FIXED'][1]-He).max(), np.abs(out['ORIG'][1]-He).max(),
                 out['FIXED'][1][0].sum()/300, out['ORIG'][1][0].sum()/300, He[0].sum()/300,
                 np.abs(out['FIXED'][0]-wce(He)).max(), np.abs(out['ORIG'][0]-wce(He)).max()))
