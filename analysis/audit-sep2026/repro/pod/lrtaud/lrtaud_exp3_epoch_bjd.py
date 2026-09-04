"""epochs=None default; BJD-scale times."""
import numpy as np, sys
sys.path.insert(0, '/workspace/scratch/lrtaud')
from lrt_common import *
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess

proc = NUFFTLRTAsyncProcess()
rng = np.random.RandomState(2)
t = make_times(rng); n = len(t)
P, dur, depth = 5.3, 0.22, 0.01
periods = np.exp(np.linspace(np.log(2), np.log(18), 32)); periods[np.argmin(np.abs(periods-P))] = P
ip = int(np.argmin(np.abs(periods-P)))
for ep in (0.0, P/2):
    y = 1 + 3e-3*rng.randn(n) + box(t, P, ep, dur, depth)
    s0 = proc.run(t, y, periods, durations=np.array([dur]))          # epochs=None
    print('injected epoch=%.2f  epochs=None: shape=%s best P=%.3f (true %.1f)  SNR@true=%.2f  max=%.2f' % (
        ep, s0.shape, periods[s0[:,0].argmax()], P, s0[ip,0], s0.max()))
    epochs = np.linspace(0, 1, 48, endpoint=False)
    best = [proc.run(t, y, np.array([p]), durations=np.array([dur]), epochs=epochs*p).max() for p in periods]
    print('     with 48-epoch grid: best P=%.3f  SNR@true=%.2f  max=%.2f' % (periods[int(np.argmax(best))], best[ip], max(best)))

# BJD offsets
y = 1 + 3e-3*rng.randn(n) + box(t, P, 1.7, dur, depth)
epochs = np.linspace(0, P, 48, endpoint=False)
ref = proc.run(t, y, np.array([P]), durations=np.array([dur]), epochs=epochs)[0,0]
for off in (0.0, 2000.0, 2457000.0):
    s = proc.run(t + off, y, np.array([P]), durations=np.array([dur]), epochs=epochs + off)[0,0]
    tm = proc._generate_template((t+off).astype(np.float32), P, 1.7+off, dur, 1.0)
    print('t offset %10.1f: max SNR=%.2f (local %.2f)  corr vs local=%.4f  in-transit pts in template=%d (local %d)' % (
        off, s.max(), ref.max(), np.corrcoef(s, ref)[0,1], (tm<0).sum(), (proc._generate_template(t.astype(np.float32), P, 1.7, dur, 1.0)<0).sum()))
    if off == 2457000.0:
        proc64 = NUFFTLRTAsyncProcess(use_double=True)
        s64 = proc64.run(t + off, y, np.array([P]), durations=np.array([dur]), epochs=epochs + off)[0,0]
        print('   use_double=True at offset %.0f: max SNR=%.2f corr vs local=%.4f' % (off, s64.max(), np.corrcoef(s64, ref)[0,1]))
