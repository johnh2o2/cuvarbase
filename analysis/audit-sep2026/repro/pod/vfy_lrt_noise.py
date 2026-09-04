import warnings, numpy as np
warnings.simplefilter("ignore")
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
rng = np.random.default_rng(1); N, T = 600, 60.0
t = np.sort(rng.uniform(0, T, N)); P0, dur, depth, sig, e0 = 5.3, 0.22, 0.01, 0.003, 1.2
ph = np.fmod(t - e0, P0) / P0; ph[ph < 0] += 1; ph[ph > 0.5] -= 1
y = 1.0 - depth * (np.abs(ph) <= dur / (2 * P0)) + sig * rng.standard_normal(N)
periods = np.round(np.arange(4.0, 6.61, 0.1), 3); durations = np.array([0.15, 0.22]); epochs = np.linspace(0, P0, 16, endpoint=False)
p = NUFFTLRTAsyncProcess()
for off in (0.0, 10.3):
    a = p.run(t+off, y, periods, durations, epochs=epochs+off)
    b = p.run(t+off, y, periods, durations, epochs=epochs+off)
    print("run-to-run (unpatched float32) off=%g: rel=%.2e identical=%s" % (off, np.abs(a-b).max()/np.abs(a).max(), np.array_equal(a, b)))
off = 10.3
a = p.run(t+off, y, periods, durations, epochs=epochs+off)
b = p.run((t+off)-10.0, y, periods, durations, epochs=(epochs+off)-10.0)
print("fix effect at off=10.3 (subtract 10 in float64 before cast): rel=%.2e" % (np.abs(a-b).max()/np.abs(a).max()))
# epochs=None path at BJD (single epoch 0.0 absolute): is it even meaningful?
a = p.run(t, y, periods, durations)
b = p.run(t+2457000.0, y, periods, durations)
print("epochs=None: off=0 vs off=2457000 (float32): corr=%.3f rel=%.2e" % (np.corrcoef(a.ravel(), b.ravel())[0,1], np.abs(a-b).max()/np.abs(a).max()))
