"""Experiment C: null (pure-noise) SDE distribution; FAP calibration; edge-peak fraction."""
import numpy as np, sys, json, time
sys.path.insert(0, '/workspace/scratch')
from audit_common import *
from cuvarbase.tls import tls_search_batch, tls_search_gpu
from cuvarbase import tls_grids, tls_stats
import transitleastsquares as tlsref
from transitleastsquares.stats import FAP as refFAP

base, sig, cad = 60.0, 1e-3, 30.0
N = 400
lcs = [noise_lc(base, cad, sig, seed=1000 + i) for i in range(N)]
periods = tls_grids.period_grid_ofir(lcs[0][0])
print("ndata=%d nperiods=%d" % (len(lcs[0][0]), len(periods)))
t1 = time.time()
res = tls_search_batch(lcs, periods=periods, return_arrays=True)
print("batch of %d null LCs: %.1fs" % (N, time.time() - t1))
sde = np.array([r['SDE'] for r in res]); sde_raw = np.array([r['SDE_raw'] for r in res])
fap = np.array([r['FAP'] for r in res]); nf = np.array([r['n_failed_periods'] for r in res])
argmax = np.array([int(np.nanargmax(r['power'])) for r in res])
n = len(periods)
edge = ((argmax < 45) | (argmax >= n - 45)).mean()
print("null SDE (batch, Keplerian window): mean=%.2f std=%.2f  q50=%.2f q90=%.2f q99=%.2f max=%.2f" % (sde.mean(), sde.std(), *np.percentile(sde, [50, 90, 99]), sde.max()))
print("null SDE_raw: mean=%.2f q90=%.2f q99=%.2f" % (sde_raw.mean(), *np.percentile(sde_raw, [90, 99])))
print("frac SDE>5.7: %.3f  >7: %.3f  >8.3: %.3f  (ref table: 0.1 / 0.01 / 0.001)" % ((sde > 5.7).mean(), (sde > 7).mean(), (sde > 8.3).mean()))
print("cuvarbase 'empirical' FAP: median=%.3g; frac FAP<0.1: %.3f  FAP<0.01: %.3f" % (np.median(fap), (fap < 0.1).mean(), (fap < 0.01).mean()))
print("peak within 45 grid points of an edge: %.3f (uniform expectation %.3f)" % (edge, 90.0 / n))
print("n_failed periods: mean=%.1f max=%d" % (nf.mean(), nf.max()))
# reference FAP for these SDE values
rf = np.array([refFAP(s) for s in sde], dtype=object)
print("reference FAP table applied to cuvarbase SDEs: frac 'None'(>0.1)=%.3f  frac<0.01=%.3f" % (np.mean([x is None for x in rf]), np.mean([(x is not None and x < 0.01) for x in rf])))

# same null LCs, standard (fixed) window and t0_oversample=33 for a subset
sub = 100
r_std = [tls_search_gpu(*lcs[i], periods=periods, return_arrays=True) for i in range(sub)]
sde_std = np.array([r['SDE'] for r in r_std])
r_33 = [tls_search_gpu(*lcs[i], periods=periods, t0_oversample=33.0) for i in range(sub)]
sde_33 = np.array([r['SDE'] for r in r_33])
print("null SDE std-window(fast3): mean=%.2f q90=%.2f q99=%.2f | std-window t0_os=33: mean=%.2f q90=%.2f q99=%.2f" % (sde_std.mean(), *np.percentile(sde_std, [90, 99]), sde_33.mean(), *np.percentile(sde_33, [90, 99])))
# SR/SDE formula check: recompute SDE with reference formula (SR=min/chi2, running_median edge-extension) on cuvarbase chi2
from transitleastsquares.stats import spectra
sde_reffmla = []
for r in res[:sub]:
    chi2 = np.asarray(r['chi2'])
    SR, power_raw, power, SDE_raw, SDE = spectra(chi2[np.isfinite(chi2)], 3)
    sde_reffmla.append(SDE)
sde_reffmla = np.array(sde_reffmla)
print("cuvarbase chi2 -> reference spectra(): mean=%.2f q90=%.2f q99=%.2f ; cuvarbase own: mean=%.2f q90=%.2f q99=%.2f ; corr=%.3f  mean abs diff=%.2f" % (sde_reffmla.mean(), *np.percentile(sde_reffmla, [90, 99]), sde[:sub].mean(), *np.percentile(sde[:sub], [90, 99]), corr(sde_reffmla, sde[:sub]), np.mean(np.abs(sde_reffmla - sde[:sub]))))
# reference on 24 null LCs (own grid, default settings)
nref = 24
sde_ref = []; t1 = time.time()
for i in range(nref):
    r, dt = run_ref(*lcs[i], threads=24)
    sde_ref.append(r.SDE)
sde_ref = np.array(sde_ref)
print("reference null SDE (%d LCs, %.0fs, %d periods): mean=%.2f std=%.2f max=%.2f | cuvarbase batch same LCs: mean=%.2f std=%.2f max=%.2f | fast std-window: mean=%.2f" % (nref, time.time() - t1, len(r.periods), sde_ref.mean(), sde_ref.std(), sde_ref.max(), sde[:nref].mean(), sde[:nref].std(), sde[:nref].max(), sde_std[:nref].mean()))
print("per-LC (ref, cuv_batch, cuv_std):", [(round(a,2), round(b,2), round(c,2)) for a, b, c in zip(sde_ref, sde[:nref], sde_std[:nref])])
