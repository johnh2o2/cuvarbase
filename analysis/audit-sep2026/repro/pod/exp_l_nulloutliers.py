"""Experiment L: dissect the high-SDE outliers among 400 pure-noise LCs (batch, Keplerian window)."""
import numpy as np, sys, time, warnings
sys.path.insert(0, '/workspace/scratch')
from audit_common import *
from cuvarbase.tls import tls_search_batch, tls_search_gpu
from cuvarbase import tls_grids, tls_stats
warnings.simplefilter('ignore')
base, sig, cad = 60.0, 1e-3, 30.0
N = 400
lcs = [noise_lc(base, cad, sig, seed=1000 + i) for i in range(N)]
periods = tls_grids.period_grid_ofir(lcs[0][0])
res = tls_search_batch(lcs, periods=periods, return_arrays=True)
sde = np.array([r['SDE'] for r in res])
order = np.argsort(-sde)
n = len(periods)
print("top-10 null SDEs:")
for i in order[:10]:
    r = res[i]; ip = int(np.nanargmax(r['power']))
    print("  seed=%d SDE=%.2f raw=%.2f P=%.4f (grid idx %d/%d) dur=%.3f d (q=%.4f) depth=%.5f n_transits=%d SNR=%.1f chi2_min=%.1f coarse-min=%.1f" % (
        1000 + i, r['SDE'], r['SDE_raw'], r['period'], ip, n, r['duration'], r['duration'] / r['period'], r['depth'], r['n_transits'], r['SNR'], r['chi2_min'], np.nanmin(r['chi2'])))
# period distribution of null peaks
pk = np.array([r['period'] for r in res])
hist, edges = np.histogram(np.log10(pk), bins=12, range=(np.log10(periods.min()), np.log10(periods.max())))
print("null-peak period histogram (log bins):", list(zip(np.round(10**edges[:-1], 2), hist)))
print("fraction of null peaks with P > 20 d (only 2-3 transits):", np.mean(pk > 20), "; grid fraction of periods > 20 d:", np.mean(periods > 20))
# worst LC: compare with t0_oversample=33 and reference, and an SDE computed with the peak-region masked
i = order[0]; t, y, dy = lcs[i]
r33 = tls_search_gpu(t, y, dy, periods=periods, t0_oversample=33.0)
print("worst LC seed=%d: fast3 SDE=%.2f, fast33 SDE=%.2f P33=%.4f" % (1000 + i, sde[i], r33['SDE'], r33['period']))
r = res[i]; chi2 = r['chi2']; ip = int(np.nanargmax(r['power']))
print("  power around peak:", np.round(r['power'][max(0, ip-4):ip+5] / np.nanstd(r['power']), 1))
print("  chi2 around peak:", np.round(chi2[max(0, ip-4):ip+5] - np.nanmin(chi2), 1))
ref, dt = run_ref(t, y, dy, threads=24)
print("  reference: SDE=%.2f P=%.4f depth=%.5f dur=%.3f [%ds]" % (ref.SDE, ref.period, 1 - ref.depth, ref.duration, dt))
# in-transit points at the cuvarbase solution
ph = ((t - r['T0']) / r['period'] + 0.5) % 1 - 0.5
intr = np.abs(ph * r['period']) < r['duration'] / 2
print("  n in-transit points at cuvarbase solution: %d ; their mean flux-1 = %.5f (%.1f sigma/sqrt(n))" % (intr.sum(), (y[intr] - 1).mean(), (y[intr] - 1).mean() / (sig / np.sqrt(max(intr.sum(), 1)))))
