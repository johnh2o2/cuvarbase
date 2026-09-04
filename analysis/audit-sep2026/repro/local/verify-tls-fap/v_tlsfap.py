"""Verifier: is the TLS 'FAP' calibrated?  Null (pure-noise) LCs -> SDE -> FAP, versus empirical null exceedance."""
import numpy as np, sys, time, warnings
sys.path.insert(0, '/workspace/scratch')
from audit_common import noise_lc, run_ref
from cuvarbase.tls import tls_search_batch
from cuvarbase import tls_grids, tls_stats
warnings.simplefilter('ignore')

def null_run(base, cad, sig, N, seed0, periods=None, label=''):
    lcs = [noise_lc(base, cad, sig, seed=seed0 + i) for i in range(N)]
    if periods is None:
        periods = tls_grids.period_grid_ofir(lcs[0][0])
    t1 = time.time()
    res = tls_search_batch(lcs, periods=periods)
    dt = time.time() - t1
    sde = np.array([r['SDE'] for r in res]); fap = np.array([r['FAP'] for r in res])
    print("[%s] ndata=%d nperiods=%d N=%d  %.1fs" % (label, len(lcs[0][0]), len(periods), N, dt))
    print("   null SDE: mean=%.2f std=%.2f q50=%.2f q90=%.2f q99=%.2f max=%.2f" % (sde.mean(), sde.std(), *np.percentile(sde, [50, 90, 99]), sde.max()))
    print("   frac SDE>7: %.3f   >9: %.3f" % ((sde > 7).mean(), (sde > 9).mean()))
    for a in (0.1, 0.01, 1e-3, 1e-4):
        print("   frac null LCs with cuvarbase FAP < %g : %.3f   (calibrated value would be %g)" % (a, (fap < a).mean(), a))
    # empirical null exceedance vs formula, at a few SDE levels
    for s in (7.0, 8.0, 9.0, 10.0):
        print("   SDE=%.0f: formula FAP=%.2e  empirical null P(SDE>=%.0f)=%.3f" % (s, tls_stats.false_alarm_probability(s), s, (sde >= s).mean()))
    return lcs, periods, sde, fap

# 1. auditor's configuration
lcs, periods, sde, fap = null_run(60.0, 30.0, 1e-3, 400, 1000, label='auditor cfg 60d/30min/Ofir')
# 2. same LCs, coarser grid (every 8th Ofir period)
null_run(60.0, 30.0, 1e-3, 400, 1000, periods=periods[::8], label='same LCs, Ofir[::8]')
# 3. TESS-sector-like 27 d
null_run(27.0, 30.0, 1e-3, 400, 5000, label='27d/30min/Ofir')
# 4. 1-yr 30-min
null_run(365.0, 30.0, 1e-3, 100, 9000, label='365d/30min/Ofir')

# 5. reference package on a few of the auditor's null LCs (own grid, defaults)
n = 6
print("reference transitleastsquares on %d of the null LCs:" % n)
for i in range(n):
    r, dt = run_ref(*lcs[i], threads=16)
    print("   seed=%d ref SDE=%.2f (nper=%d, %.0fs)  ref FAP=%s | cuvarbase SDE=%.2f FAP=%.2e" % (1000 + i, r.SDE, len(r.periods), dt, r.FAP, sde[i], fap[i]))
