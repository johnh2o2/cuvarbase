import sys, warnings; sys.path.insert(0, 'scripts'); warnings.simplefilter('ignore')
import numpy as np, nufft_lrt_validation as H
rng = np.random.RandomState(20260711)
t = H.make_times(rng, 'ground', 90.0, 600)
for seed in (7, 8):
    r = H.snr_calibration(np.random.RandomState(seed), t, {}, n=1000)
    print('seed %d: n=%d mean=%.3f (SE %.3f) std=%.3f' % (seed, r['n'], r['mean'], r['std']/np.sqrt(r['n']), r['std']), flush=True)
# same with sigma=2 (pre-fix NFFT default) for comparison
r = H.snr_calibration(np.random.RandomState(7), t, {'sigma': 2.0}, n=1000)
print('sigma=2 seed 7: mean=%.3f (SE %.3f) std=%.3f' % (r['mean'], r['std']/np.sqrt(r['n']), r['std']), flush=True)
# and the sign-flip check: statistic must be exactly odd in y
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
proc = NUFFTLRTAsyncProcess()
y = 1e-3 * np.random.RandomState(1).randn(len(t))
a = proc.run(t, y, np.array([3.7]), durations=np.array([0.15]), epochs=np.array([0.0]))[0, 0, 0]
b = proc.run(t, -y, np.array([3.7]), durations=np.array([0.15]), epochs=np.array([0.0]))[0, 0, 0]
print('odd check: S(y)=%.6f S(-y)=%.6f sum=%.2e' % (a, b, a + b))
import sys, warnings; sys.path.insert(0, 'scripts'); warnings.simplefilter('ignore')
import numpy as np, nufft_lrt_validation as H
rng = np.random.RandomState(20260711)
t = H.make_times(rng, 'ground', 90.0, 600)
r = H.snr_calibration(np.random.RandomState(20260711 + 100), t, {}, n=200)
print('campaign seed, n=200: mean=%.3f std=%.3f' % (r['mean'], r['std']), flush=True)
r = H.snr_calibration(np.random.RandomState(20260711 + 100), t, {}, n=5000)
v = r
print('campaign seed, n=5000: mean=%.3f (SE %.3f) std=%.3f' % (r['mean'], r['std']/np.sqrt(5000), r['std']), flush=True)
