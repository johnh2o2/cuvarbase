"""How much does the systematics power inflate the estimated PSD across the band?"""
import warnings, importlib.util, numpy as np
warnings.filterwarnings('ignore')
spec = importlib.util.spec_from_file_location('harness', '/workspace/cuvarbase/scripts/nufft_lrt_validation.py')
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess, _smoothed_periodogram
rng = np.random.RandomState(11)
t = H.make_times(rng); n = len(t); nf = 2*n
sw = 3e-3; amps = np.array([6., 3., 6.])*sw
M, V, mu, cov = H.build_basis_from_population(rng, t, 90., sw, sw, 0.8, amps)
proc = NUFFTLRTAsyncProcess()
def psd(x):
    Y = proc.compute_nufft(t, (x-x.mean()).astype(np.float32), nf)
    return _smoothed_periodogram((np.abs(Y)**2).astype(np.float32), 5)
r = []
for i in range(10):
    noise = sw*rng.randn(n) + H.ou_noise(rng, t, sw, 0.8)
    y = 1 + noise + M @ (rng.randn(3)*amps)
    c, *_ = np.linalg.lstsq(V, y-y.mean(), rcond=None)
    pe, po, pr = psd(y - V@mu), psd(noise), psd(y - V@c)
    r.append((np.median(pe/po), np.percentile(pe/po, 90), np.median(pr/po), np.percentile(pr/po,90), np.mean(pe/po > 3)))
r = np.array(r)
print('PSD(y - V mu)/PSD(noise): median ratio %.1f, p90 ratio %.1f, frac bins >3x = %.2f' % (r[:,0].mean(), r[:,1].mean(), r[:,4].mean()))
print('PSD(OLS residual)/PSD(noise): median ratio %.2f, p90 ratio %.2f' % (r[:,2].mean(), r[:,3].mean()))
# frequency dependence: ratio in low-k (k<=20), mid and high thirds
noise = sw*rng.randn(n) + H.ou_noise(rng, t, sw, 0.8); y = 1+noise+M@(rng.randn(3)*amps)
q = psd(y-V@mu)/psd(noise)
print('ratio by band: k<20 %.1f | k in [20,400) %.1f | k in [400,800) %.1f | k>=800 %.1f' % (np.median(q[:20]), np.median(q[20:400]), np.median(q[400:800]), np.median(q[800:])))
