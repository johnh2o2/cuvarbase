"""Independent minimal reproduction: default CE path, brightest-point bin index."""
import numpy as np
import cuvarbase.ce as ce
from cuvarbase.memory import ce_memory as cem

def cpu_ce(t, y, freqs, PB, MB, PO=0, MO=0, clip=True):
    """Plain CE reference (Graham+2013 form used by standard_ce)."""
    t = np.float32(t - t.mean()); y = np.float32(y - y.mean())   # run() -> normalize_light_curves
    yy = (np.float32(y) - np.float32(y.min())) / np.float32(y.max() - y.min())
    m = np.floor(yy * np.float32(MB)).astype(int)
    if clip:
        m = np.minimum(m, MB - 1)
    out = np.empty(len(freqs))
    for k, f in enumerate(freqs):
        ft = np.float32(t) * np.float32(f)          # float32 like the kernel
        ph = ft - np.floor(ft)
        n = (np.float32(ph) * np.float32(PB)).astype(np.int32) % PB
        H = np.zeros((PB, MB))
        for dn in range(PO + 1):
            for dm in range(MO + 1):
                mm = m - dm
                ok = mm >= 0
                np.add.at(H, ((n[ok] - dn) % PB, mm[ok]), 1)
        N = H.sum()
        Hphi = H.sum(axis=1)
        p = H / N
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(H > 0, p * np.log(Hphi[:, None] / N / p), 0.0)  # = p log(p(phi)/p(phi,m)) -> CE = sum p log(p(phi)/p)
        out[k] = term.sum() + np.log(1.0 / MB)
    return out

rng = np.random.RandomState(1)
for N in (5, 60, 500):
    t = np.sort(rng.uniform(0, 30, N)).astype(np.float32)
    y = (0.3*np.sin(2*np.pi*t/1.7) + 0.1*rng.randn(N)).astype(np.float32)
    dy = np.full(N, 0.1, np.float32)
    freqs = np.linspace(0.05, 3.0, 200).astype(np.float32)

    # 1. Index range on default path
    proc = ce.ConditionalEntropyAsyncProcess()      # defaults: mag_bins=5, phase_bins=10, use_fast=False
    mem = cem.ConditionalEntropyMemory(phase_bins=proc.phase_bins, mag_bins=proc.mag_bins,
                                       phase_overlap=proc.phase_overlap, mag_overlap=proc.mag_overlap,
                                       weighted=False)
    mem.setdata(t, y)
    print(f"N={N}: setdata y index max={mem.y[:N].max()} (valid 0..{proc.mag_bins-1}); #==mag_bins: {(mem.y[:N]==proc.mag_bins).sum()}")

    # 2. GPU standard vs fast vs CPU reference
    r_std = proc.run([(t, y, dy)], freqs=[freqs])[0][1]
    proc.finish()
    procf = ce.ConditionalEntropyAsyncProcess(use_fast=True)
    r_fast = procf.run([(t, y, dy)], freqs=[freqs])[0][1]
    procf.finish()
    ref = cpu_ce(t, y, freqs, 10, 5)
    print(f"   GPU std vs ref(clip): max|d|={np.abs(r_std-ref).max():.3e}  argmin same={np.argmin(r_std)==np.argmin(ref)}")
    print(f"   GPU fast vs ref(clip): max|d|={np.abs(r_fast-ref).max():.3e} argmin same={np.argmin(r_fast)==np.argmin(ref)}")
    print(f"   GPU std vs fast: max|d|={np.abs(r_std-r_fast).max():.3e}")
    # frequency-order dependence (std kernel spills into next frequency)
    r_std_rev = proc.run([(t, y, dy)], freqs=[freqs[::-1].copy()])[0][1][::-1]
    proc.finish()
    print(f"   GPU std reversed-freq-grid vs forward: max|d|={np.abs(r_std_rev-r_std).max():.3e}")

    # 3. Proposed fix, monkeypatched (scratch only)
    orig = cem.ConditionalEntropyMemory.setdata
    def patched(self, t_, y_, **kw):
        orig(self, t_, y_, **kw)
        if not self.weighted and not self.balanced_magbins:
            if self.buffered_transfer:
                self.y[:self.n0] = np.minimum(self.y[:self.n0], self.mag_bins - 1)
            else:
                self.y = np.minimum(self.y, self.mag_bins - 1).astype(self.ytype)
    cem.ConditionalEntropyMemory.setdata = patched
    try:
        p2 = ce.ConditionalEntropyAsyncProcess()
        r_std_fix = p2.run([(t, y, dy)], freqs=[freqs])[0][1]; p2.finish()
        p3 = ce.ConditionalEntropyAsyncProcess(use_fast=True)
        r_fast_fix = p3.run([(t, y, dy)], freqs=[freqs])[0][1]; p3.finish()
    finally:
        cem.ConditionalEntropyMemory.setdata = orig
    print(f"   FIXED std vs ref: max|d|={np.abs(r_std_fix-ref).max():.3e}; FIXED fast vs ref: {np.abs(r_fast_fix-ref).max():.3e}; FIXED std vs fast: {np.abs(r_std_fix-r_fast_fix).max():.3e}")
