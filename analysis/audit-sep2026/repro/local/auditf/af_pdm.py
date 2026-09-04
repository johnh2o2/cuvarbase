"""PDM audit: parity of all 8 kinds vs float64 references, float32 fold precision at long
baselines, ptxas local-memory report."""
import numpy as np, time, sys, json, subprocess, os
from cuvarbase.pdm import PDMAsyncProcess, pdm2_cpu, binless_pdm_cpu
from cuvarbase.utils import weights

rng = np.random.RandomState(11)

def ref_binned(t, y, w, freqs, nbins, linterp, dtype=np.float64):
    """Float64 replica of the kernel algorithm (power = 1 - var_model/var_tot)."""
    t = (t - t.mean()).astype(dtype); y = (y - y.mean()).astype(dtype); w = w.astype(dtype)
    ybar = np.dot(w, y); var_tot = np.dot(w, (y - ybar)**2)
    out = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        ph = t * dtype(f); ph = ph - np.floor(ph)
        pn = ph * nbins
        b = np.floor(pn).astype(int) % nbins
        W = np.bincount(b, weights=w, minlength=nbins)
        S = np.bincount(b, weights=w*y, minlength=nbins)
        means = np.where(W > 1e-10, S / np.where(W > 0, W, 1), 0.)
        if linterp:
            alpha = pn - np.floor(pn) - 0.5
            b0 = np.where(alpha < 0, b - 1, b); b1 = np.where(alpha < 0, b, b + 1)
            b0[b0 < 0] += nbins; b1[b1 >= nbins] -= nbins
            alpha = np.where(alpha < 0, alpha + 1, alpha)
            y0 = (1 - alpha) * means[b0] + alpha * means[b1]
        else:
            y0 = means[b]
        out[i] = 1 - np.dot(w, (y - y0)**2) / var_tot
    return out

res = {}
proc = PDMAsyncProcess()

# ---- 1. parity: binned kinds vs float64 reference (ndata=2000, T=100 d)
n = 2000; T = 100.
t = np.sort(rng.rand(n) * T); y = 0.5*np.sin(2*np.pi*2.37*t) + 0.2*rng.randn(n); dy = 0.2*np.ones(n)
w = weights(dy)
freqs = np.linspace(0.1, 10, 4000)
for kind in ['binned_linterp', 'binned_linterp_fast', 'binned_step', 'binned_step_fast']:
    r = proc.run([(t, y, dy)], freqs=freqs, kind=kind, nbins=10); proc.finish()
    p = np.array(r[0][1], dtype=np.float64)
    ref = ref_binned(t, y, w, freqs, 10, 'linterp' in kind)
    d = np.abs(p - ref)
    print("%-22s max|d|=%.2e  mean|d|=%.2e  peak gpu=%.5f ref=%.5f argmax %d/%d" % (kind, d.max(), d.mean(), p.max(), ref.max(), p.argmax(), ref.argmax()))
    res['parity_' + kind] = dict(maxabs=float(d.max()), meanabs=float(d.mean()), argmax_gpu=int(p.argmax()), argmax_ref=int(ref.argmax()))
# fast vs non-fast pairwise
for a, b in [('binned_linterp', 'binned_linterp_fast'), ('binned_step', 'binned_step_fast')]:
    ra = proc.run([(t, y, dy)], freqs=freqs, kind=a, nbins=10); proc.finish(); pa = np.array(ra[0][1]).copy()
    rb = proc.run([(t, y, dy)], freqs=freqs, kind=b, nbins=10); proc.finish(); pb = np.array(rb[0][1]).copy()
    print("%s vs %s: max|d|=%.2e" % (a, b, np.max(np.abs(pa - pb))))

# ---- 2. binless kinds vs float64 CPU reference (small)
n2 = 80
t2 = np.sort(rng.rand(n2) * 20.); y2 = 0.5*np.sin(2*np.pi*1.3*t2) + 0.2*rng.randn(n2); dy2 = 0.2*np.ones(n2)
w2 = weights(dy2); f2 = np.linspace(0.2, 3, 60)
for kind, tophat in [('binless_tophat', True), ('binless_tophat_fast', True), ('binless_gauss', False), ('binless_gauss_fast', False)]:
    r = proc.run([(t2, y2, dy2)], freqs=f2, kind=kind, dphi=0.05); proc.finish()
    p = np.array(r[0][1], dtype=np.float64)
    ref = np.array(binless_pdm_cpu(t2, y2, w2, f2, dphi=0.05, tophat=tophat))
    d = np.abs(p - ref)
    print("%-22s max|d|=%.2e mean|d|=%.2e argmax %d/%d" % (kind, d.max(), d.mean(), p.argmax(), ref.argmax()))
    res['parity_' + kind] = dict(maxabs=float(d.max()), meanabs=float(d.mean()))

# ---- 3. float32 fold precision vs baseline (same phases, long vs short baseline)
P = 0.1; ftrue = 1. / P
nn = 5000
ph = rng.rand(nn)
ysig = 0.5*np.sin(2*np.pi*ph) + 0.1*rng.randn(nn); dys = 0.1*np.ones(nn)
for Tb in (10., 100., 1000., 3650.):
    cyc = rng.randint(0, int(Tb / P), nn)
    tt = np.sort(cyc * P + ph * P)
    order = np.argsort(cyc * P + ph * P); yy = ysig[order]
    fgrid = np.array([ftrue, ftrue*0.5, ftrue*2])
    refv = ref_binned(tt, yy, weights(dys), fgrid, 10, True)
    r = proc.run([(tt, yy, dys)], freqs=fgrid, kind='binned_linterp', nbins=10); proc.finish(); pg = np.array(r[0][1]).copy()
    r = proc.run([(tt, yy, dys)], freqs=fgrid, kind='binned_linterp_fast', nbins=10); proc.finish(); pf = np.array(r[0][1]).copy()
    print("baseline T=%6.0f d, f=%g/d: power@ftrue ref(f64)=%.4f gpu=%.4f fast=%.4f | @2f ref=%.4f gpu=%.4f  (ulp of float32 t*f near max: %.2e)"
          % (Tb, ftrue, refv[0], pg[0], pf[0], refv[2], pg[2], np.spacing(np.float32(Tb/2*ftrue))))
    res['precision_T%d' % int(Tb)] = dict(ref=float(refv[0]), gpu=float(pg[0]), fast=float(pf[0]), ref2f=float(refv[2]), gpu2f=float(pg[2]))

# ---- 4. ptxas local-memory report for the PDM kernels
src = open('/workspace/cuvarbase/cuvarbase/kernels/pdm.cu').read().replace('//INSERT_NBINS_HERE', '#define NBINS 10')
open('/workspace/scratch/af_pdm_nb10.cu', 'w').write(src)
out = subprocess.run(['nvcc', '-arch=sm_89', '--use_fast_math', '-Xptxas', '-v', '-c', '-o', '/dev/null', '/workspace/scratch/af_pdm_nb10.cu'],
                     capture_output=True, text=True)
lines = [l for l in out.stderr.splitlines() if 'Function properties' in l or 'spill' in l or 'Used' in l or 'Compiling entry' in l]
print("\n".join(lines))
res['ptxas'] = lines
json.dump(res, open('/workspace/scratch/af_pdm.json', 'w'), indent=1)
