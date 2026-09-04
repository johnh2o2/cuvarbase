"""Double-precision NFFT / LS accuracy, measured with an explicit sync (no read race)."""
import numpy as np, warnings, json
warnings.simplefilter('ignore')
import pycuda.driver as cuda
from cuvarbase.cunfft import NFFTAsyncProcess
from cuvarbase.lombscargle import LombScargleAsyncProcess
from astropy.timeseries import LombScargle
rng = np.random.RandomState(4)
res = {}
def exact(t, y, ks, T): return np.array([np.sum(y*np.exp(2j*np.pi*k*t/T)) for k in ks])
for n0, nf, T in [(2000, 4000, 100.), (6000, 894250, 3650.)]:
    t = np.sort(rng.rand(n0)*T); y = rng.randn(n0); l1 = np.sum(np.abs(y))
    ks = np.unique(np.concatenate([np.arange(0, nf//2, max(1, nf//13)), [nf//4, nf//2-1]]))
    ex = exact(t, y, ks, t.max()-t.min())
    for dbl in (False, True):
        for sigma in (2, 4):
            p = NFFTAsyncProcess(use_double=dbl, sigma=sigma, m=8)
            gA = np.asarray(p.run([(t, y, nf)])[0]).copy(); p.finish()          # copy BEFORE finish (harness style A)
            gB = p.run([(t, y, nf)])[0]; cuda.Context.synchronize(); gB = np.asarray(gB).copy()  # sync then copy
            eA = np.max(np.abs(gA[ks]-ex))/l1; eB = np.max(np.abs(gB[ks]-ex))/l1
            print("NFFT n0=%d nf=%d sigma=%d use_double=%-5s: err(copy-before-finish)=%.2e  err(sync-then-copy)=%.2e" % (n0, nf, sigma, dbl, eA, eB))
            res['nfft_n%d_nf%d_s%d_d%d' % (n0, nf, sigma, dbl)] = [float(eA), float(eB)]
# LS double vs float at a modest grid and at survey scale vs astropy
def ls_cmp(n, T, fsig, fmin, fmax, stride):
    t = np.sort(rng.rand(n)*T); y = 10 + 0.5*np.sin(2*np.pi*fsig*t+0.3) + 0.3*rng.randn(n); dy = 0.3*(0.5+rng.rand(n))
    df = 1./(5*T); k0 = max(1, int(round(fmin/df))); nfr = int(fmax/df)-k0; freqs = df*(k0+np.arange(nfr))
    idx = np.unique(np.concatenate([np.arange(0, nfr, stride), np.argmin(np.abs(freqs-fsig))+np.arange(-3, 4)]))
    pa = LombScargle(t, y, dy, fit_mean=True, center_data=True).power(freqs[idx], method='cython')
    ipk = np.argmax(pa)
    for dbl in (False, True):
        proc = LombScargleAsyncProcess(use_double=dbl)
        r = proc.run([(t, y, dy)], freqs=freqs); proc.finish(); p = np.array(r[0][1], dtype=np.float64)
        d = np.abs(p[idx]-pa)
        print("LS n=%d T=%g nf=%d k0=%d use_double=%-5s: max|d|=%.2e p95=%.2e peak rel err=%.2e (astropy %.4f gpu %.4f)" % (n, T, nfr, k0, dbl, d.max(), np.percentile(d, 95), abs(p[idx][ipk]-pa[ipk])/pa[ipk], pa[ipk], p[idx][ipk]))
        res['ls_n%d_d%d' % (n, dbl)] = dict(maxabs=float(d.max()), peakrel=float(abs(p[idx][ipk]-pa[ipk])/pa[ipk]))
ls_cmp(1000, 365., 18.0, 0.1, 20., 7)
ls_cmp(6000, 3650., 40.0, 1.0, 50., 61)
json.dump(res, open('/workspace/scratch/af_dbl.json', 'w'), indent=1)
