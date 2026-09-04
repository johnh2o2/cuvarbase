"""BLS: float32 fold precision at long baselines (same phases, different baseline) vs a
float64 replica of the binned fused-noverlap algorithm; astropy cross-check."""
import numpy as np, json, warnings
warnings.simplefilter('ignore')
from cuvarbase.bls import eebls_gpu_fast, sparse_bls_cpu
from astropy.timeseries import BoxLeastSquares

rng = np.random.RandomState(9)
res = {}

def ref_fused(t, y, dy, f, qmin, qmax, dlogq=0.3, noverlap=2, dtype=np.float64):
    """float64 replica of full_bls_no_sol_fused for one frequency."""
    t = t - np.floor(t.min())
    w = dy**-2.; w /= w.sum(); ybar = np.sum(y*w); yw = (y - ybar)*w; YY = np.sum(w*(y-ybar)**2)
    nbf = int(1./qmin); nb0 = int(1./qmax); mbw = -(-nbf // nb0)
    nfine = nbf*noverlap
    ph = t.astype(dtype)*dtype(f); ph = ph - np.floor(ph)
    j = (np.floor(noverlap*(nbf*ph)).astype(int)) % nfine
    FY = np.bincount(j, weights=yw, minlength=nfine); FW = np.bincount(j, weights=w, minlength=nfine)
    best = 0.
    def dnb(m): return 1 if dlogq < 0 else max(1, int(np.floor(dlogq*m)))
    for jj in range(nfine):
        sy = 0.; sw = 0.; f_m0 = 0; m = 1
        while m < mbw:
            f_m = m*noverlap
            for u in range(f_m0, f_m):
                idx = jj + u
                if idx >= nfine: idx -= nfine
                sy += FY[idx]; sw += FW[idx]
            f_m0 = f_m
            if 1e-10 < sw < 1 - 1e-4:
                b = sy*sy/(sw*(1-sw))
                if b > best: best = b
            m += dnb(m)
    return best / YY

P = 0.5; f = 1./P; q = 0.03; depth = 0.01; sig = 0.003; n = 6000
ph = rng.rand(n)
ysig = 1.0 - depth*(np.abs(ph - 0.37) < q/2) + sig*rng.randn(n); dys = sig*np.ones(n)
qmin, qmax = 0.01, 0.1
for Tb in (5., 50., 500., 3650.):
    cyc = rng.randint(0, int(Tb/P), n); tt = cyc*P + ph*P; o = np.argsort(tt); tt = tt[o]; yy = ysig[o]
    pg = eebls_gpu_fast(tt, yy, dys, np.array([f, f*1.001, f*0.999]), qmin=qmin, qmax=qmax, dlogq=0.3, noverlap=2)
    pr = ref_fused(tt, yy, dys, f, qmin, qmax)
    pr32 = ref_fused(tt, yy, dys, f, qmin, qmax, dtype=np.float32)
    print("T=%6.0f d (f=%g/d, q=%.2f): power@ftrue gpu=%.5f  ref f64=%.5f  ref(float32 fold)=%.5f  | float32 ulp(t*f)@max: %.1e phase = %.2f fine bins"
          % (Tb, f, q, pg[0], pr, pr32, np.spacing(np.float32(Tb*f)), np.spacing(np.float32(Tb*f))*int(1/qmin)*2))
    res['T%d' % int(Tb)] = dict(gpu=float(pg[0]), ref64=float(pr), ref32=float(pr32))

# astropy cross-check at a matched single duration on a modest baseline
tt = np.sort(rng.rand(3000)*30.); yy = 1 - depth*((((tt-0.3)/P) % 1) < q) + sig*rng.randn(3000); dd = sig*np.ones(3000)
freqs = np.linspace(1.5, 2.5, 2000)
pg = eebls_gpu_fast(tt, yy, dd, freqs, qmin=q/2, qmax=q*2, dlogq=0.1, noverlap=4)
bls = BoxLeastSquares(tt, yy, dd)
pa = bls.power(1/freqs, q*P, objective='likelihood', oversample=20)
chi2_0 = np.sum(((yy - np.average(yy, weights=dd**-2))/dd)**2)
pa_chi2ratio = 2*pa.power/chi2_0   # astropy loglik = 0.5*chi2_0*P/(1-r) ~ chi2_0*P/2 for r<<1
i = np.argmax(pg); j = np.argmax(pa_chi2ratio)
print("astropy cross-check: gpu peak %.4f @f=%.4f ; astropy(loglik->chi2ratio) peak %.4f @f=%.4f ; corr(gpu,astropy)=%.4f" % (pg[i], freqs[i], pa_chi2ratio[j], freqs[j], np.corrcoef(pg, pa_chi2ratio)[0,1]))
res['astropy'] = dict(gpu_peak=float(pg[i]), f_gpu=float(freqs[i]), astropy_peak=float(pa_chi2ratio[j]), f_ast=float(freqs[j]), corr=float(np.corrcoef(pg, pa_chi2ratio)[0,1]))
json.dump(res, open('/workspace/scratch/af_bls.json', 'w'), indent=1)
