"""CE: independent CPU reference (proper mag binning) vs GPU standard/fast kernels;
LS: small-nf sentinel and non-uniform-grid silent evaluation."""
import sys, warnings
import numpy as np
sys.path.insert(0, '/workspace/scratch/xcut')
from audit_common import make_lc, metrics, fmt, ls_freqs
warnings.simplefilter('ignore')
from cuvarbase.ce import ConditionalEntropyAsyncProcess
from cuvarbase.utils import normalize_light_curves

def ce_cpu(t, y, freqs, nphase=10, nmag=5, clip=True):
    # replicate ConditionalEntropyMemory.setdata binning, with/without clipping the max point
    y = np.asarray(y, dtype=np.float32); t = np.asarray(t, dtype=np.float32)
    y0 = y.min(); ys = y.max() - y.min()
    yn = (y - y0) / ys
    mbin = np.floor(yn * nmag).astype(np.int64)
    if clip:
        mbin = np.minimum(mbin, nmag - 1)
    out = np.zeros(len(freqs))
    for i, f in enumerate(freqs):
        ph = (t * np.float32(f)) % np.float32(1.0)
        pbin = (np.floor(ph * nphase).astype(np.int64)) % nphase
        H = np.zeros((nphase, nmag + 1))
        np.add.at(H, (pbin, mbin), 1)
        H = H[:, :nmag]
        Nphi = H.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(H > 0, H * np.log((1.0 / nmag) * Nphi / H), 0.0)
        out[i] = term.sum() / H.sum()
    return out

t, y, dy = make_lc(N=600)
freqs = np.linspace(0.1, 3.0, 300)
(tn, yn, dn), = normalize_light_curves([(t, y, dy)])

print('## CE: GPU vs CPU reference (max-y point clipped into top bin) and vs reference that drops the max point')
ref = ce_cpu(tn, yn, freqs)
keep = np.arange(len(y)) != np.argmax(y)
ref_drop = ce_cpu(tn[keep], yn[keep], freqs)
for use_fast in (False, True):
    proc = ConditionalEntropyAsyncProcess(use_fast=use_fast)
    r = proc.run([(t, y, dy)], freqs=[freqs]); proc.finish(); g = np.copy(r[0][1])
    print('use_fast=%s vs clipped-ref: %s' % (use_fast, fmt(metrics(ref, g))))
    print('use_fast=%s vs ref-with-max-point-dropped: %s' % (use_fast, fmt(metrics(ref_drop, g))))
    # perturb the max point so it is no longer at exactly 1.0 after normalization
    y2 = y.copy(); y2[np.argmax(y)] = y2[np.argmax(y)] - 1e-6
    r2 = proc.run([(t, y2, dy)], freqs=[freqs]); proc.finish(); g2 = np.copy(r2[0][1])
    ref2 = ce_cpu(tn, (yn - (yn.max()-yn.min()) * 0 + np.where(np.arange(len(y))==np.argmax(y), -1e-6, 0)), freqs)
    print('use_fast=%s max-point-nudged-by-1e-6 vs ref (nudged): %s' % (use_fast, fmt(metrics(ref2, g2))))
    print('use_fast=%s effect of the 1e-6 nudge on GPU output: %s' % (use_fast, fmt(metrics(g, g2))))

print('\n## CE standard kernel: frequency-order dependence (last-frequency spill)')
proc = ConditionalEntropyAsyncProcess()
fA = freqs.copy(); fB = np.concatenate([freqs[1:], freqs[:1]])
gA = np.copy(proc.run([(t, y, dy)], freqs=[fA])[0][1]); proc.finish()
gB = np.copy(proc.run([(t, y, dy)], freqs=[fB])[0][1]); proc.finish()
d = gA[1:] - gB[:-1]
print('max |CE(f_i) in order A - CE(f_i) in order B| = %.3g at %d freqs differing (>1e-7)' % (np.abs(d).max(), int((np.abs(d) > 1e-7).sum())))
print('position of freqs[0] in order B is last: CE diff for it = %.3g' % (gA[0] - gB[-1]))

print('\n## CE with N small: which point is lost?')
for N in (5, 10, 20, 50):
    tt, yy, ddy = make_lc(N=N, seed=3)
    (tn2, yn2, _), = normalize_light_curves([(tt, yy, ddy)])
    ref = ce_cpu(tn2, yn2, freqs[:50]); refd = ce_cpu(tn2[np.arange(N)!=np.argmax(yy)], yn2[np.arange(N)!=np.argmax(yy)], freqs[:50])
    for use_fast in (False, True):
        p = ConditionalEntropyAsyncProcess(use_fast=use_fast)
        g = np.copy(p.run([(tt, yy, ddy)], freqs=[freqs[:50]])[0][1]); p.finish()
        print('N=%d fast=%s  vs clipped-ref %s | vs dropped-ref %s' % (N, use_fast, fmt(metrics(ref, g)), fmt(metrics(refd, g))))

print('\n## LS: small nf (NFFT path) sentinel; direct sums as reference')
from cuvarbase.lombscargle import LombScargleAsyncProcess
from astropy.timeseries import LombScargle
proc = LombScargleAsyncProcess()
for nf in (2, 3, 4, 6, 8, 12, 16, 24, 32, 64):
    f = ls_freqs(nf)
    g = np.copy(proc.run([(t, y, dy)], freqs=[f])[0][1]); proc.finish()
    gd = np.copy(proc.run([(t, y, dy)], freqs=[f], use_fft=False)[0][1]); proc.finish()
    a = LombScargle(t, y, dy).power(f)
    print('nf=%-3d nfft: %s | dirsum vs astropy: %s | nfft vs astropy: %s' % (nf, np.array2string(g[:3], precision=4), fmt(metrics(a, gd)), fmt(metrics(a, g))))

print('\n## LS: non-uniform grid that passes check_k0 is silently evaluated on the implied uniform grid')
f = ls_freqs(300)
fbad = f.copy(); fbad[2:] = f[2] + 3 * (f[2:] - f[2])   # spacing triples after the 2nd point
g = np.copy(proc.run([(t, y, dy)], freqs=[fbad])[0][1]); proc.finish()
gu = np.copy(proc.run([(t, y, dy)], freqs=[f])[0][1]); proc.finish()
a = LombScargle(t, y, dy).power(fbad)
print('returned power vs astropy at the USER grid: %s' % fmt(metrics(a, g)))
print('returned power vs GPU power at the IMPLIED uniform grid df*(k0+arange): %s' % fmt(metrics(gu, g)))
