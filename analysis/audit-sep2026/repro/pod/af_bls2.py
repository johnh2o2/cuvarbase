"""BLS float32 fold, harsher case: f=20/d, q=0.01, T=3650 d (ulp(t*f) ~ 0.5 fine bins)."""
import numpy as np, warnings
warnings.simplefilter('ignore')
from cuvarbase.bls import eebls_gpu_fast
rng = np.random.RandomState(9)
P = 0.05; f = 1./P; q = 0.01; depth = 0.01; sig = 0.003; n = 8000
ph = rng.rand(n); ysig = 1.0 - depth*(np.abs(ph - 0.37) < q/2) + sig*rng.randn(n); dys = sig*np.ones(n)
for Tb in (5., 365., 3650.):
    cyc = rng.randint(0, int(Tb/P), n); tt = cyc*P + ph*P; o = np.argsort(tt); tt = tt[o]; yy = ysig[o]
    pg = eebls_gpu_fast(tt, yy, dys, np.array([f]), qmin=0.005, qmax=0.05, dlogq=0.3, noverlap=2)
    print("T=%6.0f d f=%g/d q=%.3f: gpu power@ftrue=%.5f  (float32 ulp(t*f)@max = %.1e phase = %.2f fine bins of %.4f)" % (Tb, f, q, pg[0], np.spacing(np.float32(Tb*f)), np.spacing(np.float32(Tb*f))/(0.005/2), 0.005/2))
