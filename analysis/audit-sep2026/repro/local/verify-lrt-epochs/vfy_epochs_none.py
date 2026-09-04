"""Verifier: epochs=None period search vs epoch grid, random injected epochs;
plus shipped-example configuration re-run with an epoch grid."""
import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/scratch')
from lrt_common import make_times, box
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess

proc = NUFFTLRTAsyncProcess()
rng = np.random.RandomState(7)
t = make_times(rng); n = len(t)
P, dur, depth = 5.3, 0.22, 0.01
periods = np.exp(np.linspace(np.log(2), np.log(18), 32)); periods[np.argmin(np.abs(periods-P))] = P
ip = int(np.argmin(np.abs(periods-P)))
hits_none = hits_grid = 0; ntrial = 12
for k in range(ntrial):
    ep = rng.uniform(0, P)
    y = 1 + 3e-3*rng.randn(n) + box(t, P, ep, dur, depth)
    s0 = proc.run(t, y, periods, durations=np.array([dur]))
    bp0 = periods[s0[:,0].argmax()]
    epochs = np.linspace(0, 1, 48, endpoint=False)
    best = [proc.run(t, y, np.array([p]), durations=np.array([dur]), epochs=epochs*p).max() for p in periods]
    bpg = periods[int(np.argmax(best))]
    hits_none += (bp0 == P); hits_grid += (bpg == P)
    print('epoch=%.2f (phase %.2f): epochs=None bestP=%.3f SNR@true=%6.2f max=%5.2f | grid bestP=%.3f SNR@true=%5.2f' % (
        ep, ep/P, bp0, s0[ip,0], s0.max(), bpg, best[ip]))
print('RECOVERY: epochs=None %d/%d ; epoch grid %d/%d' % (hits_none, ntrial, hits_grid, ntrial))

# shipped example configuration (examples/nufft_lrt_example.py)
print('--- shipped example config, seed 42 ---')
np.random.seed(42)
t = np.sort(np.random.uniform(0, 20, 200))
tp, td, te, dp = 3.5, 0.3, 0.5, 0.02
y = 1.0 + box(t, tp, te, td, dp) + 0.01*np.random.randn(200)  # same shape as example generator
periods = np.linspace(2.0, 5.0, 50); durations = np.linspace(0.1, 0.5, 10)
snr = proc.run(t, y, periods, durations=durations)
bi = np.unravel_index(np.argmax(snr), snr.shape)
print('epochs=None: best P=%.2f dur=%.2f SNR=%.2f ; SNR at nearest true P/dur = %.2f' % (
    periods[bi[0]], durations[bi[1]], snr[bi], snr[np.argmin(abs(periods-tp)), np.argmin(abs(durations-td))]))
best = np.zeros((len(periods), len(durations)))
for i, p in enumerate(periods):
    eps = np.linspace(0, p, 40, endpoint=False)
    best[i] = proc.run(t, y, np.array([p]), durations=durations, epochs=eps).max(axis=2)[0]
bi = np.unravel_index(np.argmax(best), best.shape)
print('40-epoch grid: best P=%.2f dur=%.2f SNR=%.2f' % (periods[bi[0]], durations[bi[1]], best[bi]))
