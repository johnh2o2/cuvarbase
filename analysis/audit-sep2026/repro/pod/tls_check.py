"""TLS fast-path correctness checks: vs transitleastsquares, exact numpy chi2 at the reported solution, T0 reconstruction, BJD parity."""
import warnings, time; warnings.filterwarnings('ignore')
import numpy as np
from cuvarbase import tls, tls_models, tls_grids, tls_stats

rng = np.random.RandomState(7)
ndata = 1310; cad = 30. / 60 / 24; noise = 1e-3
t = np.arange(ndata) * cad + 0.137
P_inj, depth_inj, t0_inj = 7.7, 0.005, 2.31
q_inj = tls_grids.q_transit(P_inj); dur_inj = q_inj * P_inj
# limb-darkened injection via batman
import batman
pr = batman.TransitParams(); pr.t0 = t0_inj; pr.per = P_inj; pr.rp = np.sqrt(depth_inj); pr.a = 1.0 / (np.pi * q_inj); pr.inc = 90; pr.ecc = 0; pr.w = 90; pr.limb_dark = 'quadratic'; pr.u = [0.4804, 0.1867]
y = batman.TransitModel(pr, t).light_curve(pr) + noise * rng.randn(ndata)
dy = np.full(ndata, noise)

r = tls.tls_transit(t, y, dy, R_star=1.0, M_star=1.0, period_min=0.6, period_max=13.7)
T0 = r['T0'] if 'T0' in r else None
print("GPU TLS: P=%.5f (inj %.5f)  t0_phase=%.4f  T0_abs=%s  dur=%.4f (inj %.4f)  depth=%.5f (inj %.5f)  SDE=%.2f  chi2_min=%.3f  nper=%d"
      % (r['period'], P_inj, r['T0'], r.get('T0'), r['duration'], dur_inj, r['depth'], depth_inj, r['SDE'], r['chi2_min'], len(r['periods'])))
# tls_search_gpu 'T0' key is the phase (legacy dict); reconstruct absolute mid-transit
epoch = np.floor(t.min()); T0abs = epoch + r['T0'] * r['period']
print("  reconstructed T0_abs = %.4f ; injected t0 mod P = %.4f ; (T0_abs - t0_inj) mod P = %.4f d (frac of dur %.2f)"
      % (T0abs, t0_inj % P_inj, ((T0abs - t0_inj + 0.5 * P_inj) % P_inj) - 0.5 * P_inj, (((T0abs - t0_inj + 0.5 * P_inj) % P_inj) - 0.5 * P_inj) / dur_inj))

# exact numpy chi2 at the reported (P, t0, dur, depth) using the same template table
Ttab, S1, S2 = tls_models.generate_template_tables()
def exact_chi2(P, t0ph, dur, depth):
    ph = ((t - epoch) / P) % 1.0
    rel = ph - t0ph; rel -= np.rint(rel)
    c = rel / (0.5 * dur / P)
    Tv = np.interp(c, np.linspace(-1, 1, len(Ttab)), Ttab, left=0, right=0)
    model = 1 - depth * Tv
    return np.sum((y - model) ** 2 / (dy ** 2 + 1e-10))
def exact_best_depth(P, t0ph, dur):
    ph = ((t - epoch) / P) % 1.0; rel = ph - t0ph; rel -= np.rint(rel); c = rel / (0.5 * dur / P)
    Tv = np.interp(c, np.linspace(-1, 1, len(Ttab)), Ttab, left=0, right=0)
    a = (1 - y) / (dy**2 + 1e-10); b = 1 / (dy**2 + 1e-10)
    num = np.sum(a * Tv); den = np.sum(b * Tv * Tv); return num / den, num * num / den
chi2_0 = np.sum((1 - y) ** 2 / (dy ** 2 + 1e-10))
c_exact = exact_chi2(r['period'], r['T0'], r['duration'], r['depth'])
d_opt, score_opt = exact_best_depth(r['period'], r['T0'], r['duration'])
print("  exact numpy chi2 at reported solution = %.3f  vs reported chi2_min = %.3f  (diff %.3e; chi2_0=%.3f); optimal depth at that (t0,dur) = %.5f (reported %.5f)"
      % (c_exact, r['chi2_min'], c_exact - r['chi2_min'], chi2_0, d_opt, r['depth']))
# coarse spectrum value at best period vs brute-force exact scan over the SAME coarse (dur, t0) grid
ib = int(np.argmin(np.nan_to_num(r['chi2'], nan=np.inf)))
Pb = r['periods'][ib]
qv = tls_grids.q_transit(np.float64(Pb)); qs = np.exp(np.linspace(np.log(0.5 * qv), np.log(2 * qv), 15))
best = np.inf
for q in qs:
    n_t0 = int(np.clip(np.ceil(3.0 / q), 30, 20000))
    for j in range(n_t0):
        _, sc = exact_best_depth(Pb, j / n_t0, q * Pb)
        best = min(best, chi2_0 - sc)
print("  coarse spectrum chi2 at best period %.5f: GPU binned = %.3f ; exact per-point scan on same trial grid = %.3f ; diff = %.3f (= %.2e of chi2_0)"
      % (Pb, r['chi2'][ib], best, r['chi2'][ib] - best, (r['chi2'][ib] - best) / chi2_0))

# BJD parity
rb = tls.tls_transit(t + 2457000.0, y, dy, R_star=1.0, M_star=1.0, period_min=0.6, period_max=13.7)
print("BJD parity: P %.6f vs %.6f ; SDE %.3f vs %.3f ; max|chi2 diff| = %.3e ; T0 phase %.5f vs %.5f"
      % (r['period'], rb['period'], r['SDE'], rb['SDE'], np.nanmax(np.abs(r['chi2'] - rb['chi2'])), r['T0'], rb['T0']))

# reference transitleastsquares on the same grid (its own grid), compare recovered params
from transitleastsquares import transitleastsquares
t0w = time.time()
model = transitleastsquares(t, y, dy)
res = model.power(R_star=1, M_star=1, period_min=0.6, period_max=13.7, oversampling_factor=3, use_threads=8, show_progress_bar=False)
print("reference TLS (%.1fs): P=%.5f T0=%.4f dur=%.4f depth=%.5f SDE=%.2f" % (time.time() - t0w, res.period, res.T0, res.duration, 1 - res.depth, res.SDE))
print("  |T0_gpu - T0_ref| mod P = %.4f d" % (abs(((T0abs - res.T0 + 0.5 * P_inj) % P_inj) - 0.5 * P_inj)))

# null light curve: SDE distribution sanity (5 draws) and n_failed
sdes = []
for i in range(5):
    yn = 1 + noise * rng.randn(ndata)
    rn = tls.tls_transit(t, yn, dy, period_min=0.6, period_max=13.7)
    sdes.append(rn['SDE'])
print("null SDEs (5 draws):", np.round(sdes, 2))

# batch vs single parity with mixed lengths
lcs = [(t, y, dy), (t[:900], y[:900], dy[:900]), (t + 2457000.0, y, dy)]
rb = tls.tls_search_batch(lcs, period_min=0.6, period_max=13.7, return_arrays=True)
print("batch: periods %s ; SDE %s ; max|chi2[0]-chi2[2]| = %.3e" % ([round(x['period'], 5) for x in rb], [round(x['SDE'], 3) for x in rb], np.nanmax(np.abs(rb[0]['chi2'] - rb[2]['chi2']))))
