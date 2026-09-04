"""Exp 9c: characterize the rare fused-kernel vs float64-binned-reference disagreements at large t*f."""
import sys
sys.path.insert(0, '/workspace/scratch/blsaudit')
import numpy as np
from blsref import make_data, ref_fast, _prep, fold, m_sequence, bls_value
from cuvarbase.bls import eebls_gpu_fast

t, y, dy = make_data(ndata=5000, baseline=3650., freq=6.3, q=0.06, phi0=0.9, snr=15., seed=5)
T = t.max() - t.min()
freqs = np.arange(6.2995, 6.3005, 0.25 * 0.02 / T)
kw = dict(qmin=1e-2, qmax=0.5, dlogq=0.3)
gf = eebls_gpu_fast(t, y, dy, freqs, noverlap=2, **kw)
r2 = ref_fast(t, y, dy, freqs, noverlap=2, f32fold=True, **kw)
d = np.abs(gf - r2)
bad = np.where(d > 1e-6)[0]
print("nfreqs=%d; #freqs |gpu-ref32|>1e-6: %d; worst i=%d f=%.6f gpu=%.6f ref=%.6f" % (len(freqs), len(bad), np.argmax(d), freqs[np.argmax(d)], gf[np.argmax(d)], r2[np.argmax(d)]))
# for the worst frequency: how many points have nbf*phi within 1 float32 ulp of an integer (bin edge)?
ts, epoch, w, yw, yy = _prep(t, y, dy)
for i in bad[:5]:
    f32 = np.float32(freqs[i])
    ph = fold(ts, f32, True)
    x64 = 100.0 * ph
    x32 = (np.float32(100.0) * ph.astype(np.float32)).astype(np.float64)
    ne = int((np.floor(x64) != np.floor(x32)).sum())
    print("  i=%d: points where floor(100*phi) differs float32 vs float64: %d (of %d); gpu=%.6f ref=%.6f" % (i, ne, len(ph), gf[i], r2[i]))
# reference with float32 bin arithmetic
def ref_f32bin(i):
    f32 = np.float32(freqs[i]); ph = fold(ts, f32, True).astype(np.float32)
    nbf = 100; best = 0.
    for s in range(2):
        dp = np.float32(s / 2.)
        b = np.floor(np.float32(nbf) * ph - dp).astype(np.int64) % nbf
        hyw = np.bincount(b, weights=yw, minlength=nbf); hw = np.bincount(b, weights=w, minlength=nbf)
        cyw = np.concatenate(([0.], np.cumsum(np.concatenate((hyw, hyw))))); cw = np.concatenate(([0.], np.cumsum(np.concatenate((hw, hw)))))
        n = np.arange(nbf)
        for m in m_sequence(nbf, 2, 0.3, f32=True):
            p = bls_value(cyw[n + m] - cyw[n], cw[n + m] - cw[n], False)
            best = max(best, p.max())
    return best / yy
for i in bad[:5]:
    print("  i=%d: ref with float32 bin arithmetic = %.6f (gpu %.6f)" % (i, ref_f32bin(i), gf[i]))
