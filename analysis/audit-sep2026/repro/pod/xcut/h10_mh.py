import sys, warnings
import numpy as np
sys.path.insert(0, '/workspace/scratch/xcut')
from audit_common import make_lc, metrics, fmt
warnings.simplefilter('ignore')
from cuvarbase.lombscargle import LombScargleAsyncProcess, lomb_scargle_direct_sums
from cuvarbase.utils import normalize_light_curves, weights
t, y, dy = make_lc()
T = t.max() - t.min(); df = 1.0 / (5 * T)
(tc, yc, _), = normalize_light_curves([(t, y, dy)])
w = weights(dy); ybar = np.dot(w, yc); yw = w * (yc - ybar); YY = np.dot(w, (yc - ybar) ** 2)
for H in (1, 2, 3):
    p = LombScargleAsyncProcess(nharmonics=H)
    for fmin, fmax in ((0.1, 3.0), (0.5, 3.0), (1.0, 3.0), (1.5, 3.0)):
        k0 = int(round(fmin / df)); nf = int(round((fmax - fmin) / df)); f = df * (k0 + np.arange(nf))
        g = np.copy(p.run([(t, y, dy)], freqs=[f])[0][1]); p.finish()
        sub = slice(0, nf, 7)
        ref = lomb_scargle_direct_sums(tc, yw, w, f[sub], YY, nharms=H)
        print('H=%d fmin=%g fmax=%g k0/nf=%.2f: GPU NFFT vs CPU direct sums (nharms=%d): %s' % (H, fmin, fmax, k0 / nf, H, fmt(metrics(ref, g[sub]))))
    if H > 1:
        f = df * (50 + np.arange(1000))
        a = np.copy(p.run([(t, y, dy)], freqs=[f], use_fft=False)[0][1]); p.finish()
        ref1 = lomb_scargle_direct_sums(tc, yw, w, f[::7], YY, nharms=1)
        refH = lomb_scargle_direct_sums(tc, yw, w, f[::7], YY, nharms=H)
        print('   use_fft=False with nharmonics=%d: vs H=1 direct sums %s | vs H=%d direct sums %s' % (H, fmt(metrics(ref1, a[::7])), H, fmt(metrics(refH, a[::7]))))
        a = np.asarray(p.run([(t, y, dy)], freqs=[f], python_dir_sums=True)[0][1]); p.finish()
        print('   python_dir_sums=True with nharmonics=%d: vs H=1 direct sums %s | vs H=%d %s' % (H, fmt(metrics(ref1, a[::7])), H, fmt(metrics(refH, a[::7]))))
