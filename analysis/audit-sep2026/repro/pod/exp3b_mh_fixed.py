import sys, time
import numpy as np
sys.path.insert(0, '/workspace/scratch')
from common import make_lc, report
from astropy.timeseries import LombScargle
import patches
patches.apply(F=True, P=True, B=True)
from cuvarbase.lombscargle import LombScargleAsyncProcess
exec(open('/workspace/scratch/exp3_mhgls.py').read().split('print("=== CPU lomb_scargle_direct_sums')[0].split('from cuvarbase.utils import normalize_light_curves')[1])
print("=== GPU multiharmonic with all three fixes applied (F+P+B), vs weighted lstsq ===")
for H in (1, 2, 3):
    ref = lstsq_power(t, y, dy, freqs, H)
    for dbl in (True, False):
        proc = LombScargleAsyncProcess(use_double=dbl, sigma=4, m=8, autoset_m=False, nharmonics=H)
        r = proc.run([(t, y, dy)], freqs=freqs); proc.finish()
        report("FIXED GPU H=%d dbl=%s" % (H, dbl), ref, np.array(r[0][1][:nf], float), freqs)
        del proc
print("=== window=True with fixes: ratio to astropy LS of ones ===")
t1, y1, dy1 = make_lc(N=300, T=365.0, f0=3.1, hetero=True, seed=6)
fr = (1.0 / (5 * 365.0)) * (1 + np.arange(20000))
proc = LombScargleAsyncProcess(use_double=True, sigma=4, m=8, autoset_m=False)
r = proc.run([(t1, y1, dy1)], freqs=fr, window=True); proc.finish()
gw = np.array(r[0][1][:len(fr)], float)
refW = LombScargle(t1, np.ones_like(y1), dy1, fit_mean=False, center_data=False).power(fr, method='cython')
report("window vs 4 * astropy LS(ones, fit_mean=False, center_data=False)", 4 * refW, gw)
print("ratio percentiles (1,50,99):", np.percentile(gw / refW, [1, 50, 99]))
