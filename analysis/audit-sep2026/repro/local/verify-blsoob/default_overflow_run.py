"""Run the DEFAULT eebls_transit path (no freq_batch_size, no max_memory) on a
TESS-like 2-min cadence one-year stitched light curve (260K points) and compare
against the same call with an explicit safe freq_batch_size."""
import numpy as np, time
import pycuda.driver as cuda, pycuda.autoprimaryctx  # noqa
from cuvarbase.bls import eebls_transit, compile_bls
rng = np.random.RandomState(3)
t = np.arange(262800) * 2.0 / 1440.   # 365 d at 2-min cadence
P, q, depth = 3.7, 0.02, 0.01
phase = (t / P) % 1
y = 1.0 - depth * (phase < q) + 0.002 * rng.randn(len(t))
dy = np.full(len(t), 0.002)
fr = compile_bls()
t0 = time.time()
freqs, p_auto, sols = eebls_transit(t, y, dy, functions=fr)
cuda.Context.synchronize(); ta = time.time() - t0
print("ndata=%d nfreq=%d  auto: %.1fs  max=%.4f at P=%.4f  #zero=%d  #>1=%d" % (len(t), len(freqs), ta, p_auto.max(), 1/freqs[np.argmax(p_auto)], (p_auto == 0).sum(), (p_auto > 1).sum()))
t0 = time.time()
freqs2, p_safe, sols2 = eebls_transit(t, y, dy, functions=fr, freq_batch_size=8000)
cuda.Context.synchronize(); tb = time.time() - t0
print("freq_batch_size=8000: %.1fs  max=%.4f at P=%.4f  #zero=%d" % (tb, p_safe.max(), 1/freqs2[np.argmax(p_safe)], (p_safe == 0).sum()))
print("corr(auto, safe)=%.4f  max|diff|=%.3e" % (np.corrcoef(p_auto, p_safe)[0, 1], np.abs(p_auto - p_safe).max()))
