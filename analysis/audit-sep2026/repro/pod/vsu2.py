"""Isolate the centering effect GPU-vs-GPU (same kernel, same sort, same q bounds): current wrapper vs float64-centered wrapper."""
import sys, numpy as np
sys.path.insert(0, '/workspace/scratch')
import pycuda.autoprimaryctx  # noqa
from cuvarbase.bls import sparse_bls_gpu, compile_sparse_bls, q_transit
from sparse_exp import ref_sparse
kern = compile_sparse_bls(block_size=64)

def centered(t, y, dy, freqs, **kw):
    y = np.asarray(y, dtype=np.float64); w = np.asarray(dy, dtype=np.float64) ** -2
    return sparse_bls_gpu(t, y - np.sum(w * y) / np.sum(w), dy, freqs, **kw)

def make(N, seed, base=365.0, ybar=12.0, depth=5e-3, sig=5e-3, f=1.37, q=0.02):
    r = np.random.RandomState(seed)
    t = np.sort(base * r.rand(N)); ph = (t * f) % 1
    y = ybar - depth * (ph < q) + sig * r.randn(N)
    dy = sig * (0.7 + 0.6 * r.rand(N))
    return t, y, dy

df = 0.02 / 365 / 4
freqs = 1.37 + df * np.arange(-200, 201); qv = q_transit(freqs); kw = dict(qmin=qv * 0.5, qmax=qv * 2.0, kernel=kern)
print("=== E. N=200 mag 12, injected 5 mmag transit at 1.37/d, 20 seeds: current vs float64-centered (same kernel)")
n_arg = 0; n_far = 0; peak_rel = []; grid_rel = []; frac = []
for seed in range(20):
    t, y, dy = make(200, seed)
    pc = sparse_bls_gpu(t, y, dy, freqs, **kw)[0]; pf = centered(t, y, dy, freqs, **kw)[0]
    ic, i_f = np.argmax(pc), np.argmax(pf)
    rel = (pc - pf) / np.maximum(pf, 1e-12)
    n_arg += ic != i_f; n_far += abs(ic - i_f) > 4
    peak_rel.append(rel[i_f]); grid_rel.append(np.abs(rel).max()); frac.append(np.mean(np.abs(rel) > 1e-3))
    if ic != i_f:
        print("   seed %2d: argmax moved by %+d grid steps (df=q/4 -> %.2f peak widths); power at fixed peak: current %.5f fixed %.5f; current's own peak %.5f" % (seed, ic - i_f, (ic - i_f) / 4.0, pc[i_f], pf[i_f], pc[ic]))
print("  argmax differs in %d/20 seeds (moved >1 peak width in %d) | peak-power rel err: median %.1e max %.1e | grid max|rel|: median %.1e max %.1e | frac of grid with |rel|>1e-3: %.2f"
      % (n_arg, n_far, np.median(np.abs(peak_rel)), np.max(np.abs(peak_rel)), np.median(grid_rel), np.max(grid_rel), np.mean(frac)))

print("\n=== F. pure noise (no transit), N=200 mag 12: false-alarm-level statistic max(power) over 401 freqs, 20 seeds")
mc, mf = [], []
for seed in range(20):
    t, y, dy = make(200, 100 + seed, depth=0.0)
    mc.append(sparse_bls_gpu(t, y, dy, freqs, **kw)[0].max()); mf.append(centered(t, y, dy, freqs, **kw)[0].max())
mc, mf = np.array(mc), np.array(mf)
print("  max power: current mean %.5f, fixed mean %.5f, mean rel diff %+.2e, max |rel diff| %.2e" % (mc.mean(), mf.mean(), np.mean((mc - mf) / mf), np.max(np.abs(mc - mf) / mf)))

print("\n=== G. magnitude scale dependence (same LC shifted by a constant), N=300, seed 0: max|rel| current-vs-fixed over grid")
for off in (0.0, 1.0, 12.0, 20.0, 1e4):
    t, y, dy = make(300, 0, ybar=off)
    pc = sparse_bls_gpu(t, y, dy, freqs, **kw)[0]; pf = centered(t, y, dy, freqs, **kw)[0]
    print("  offset %8.1f: max|rel| %.1e, peak rel %+.1e, argmax same %s" % (off, (np.abs(pc - pf) / np.maximum(pf, 1e-12)).max(), (pc[np.argmax(pf)] - pf.max()) / pf.max(), np.argmax(pc) == np.argmax(pf)))
