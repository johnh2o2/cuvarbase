"""Experiment G (CPU): period grid vs reference; duration grid; t0 grid resolution; edge handling."""
import numpy as np, sys
sys.path.insert(0, '/workspace/scratch')
from audit_common import *
from cuvarbase import tls_grids
from transitleastsquares.grid import period_grid, duration_grid, T14
for span, R, M in [(60.0, 1.0, 1.0), (1400.0, 1.0, 1.0), (27.4, 0.3, 0.3), (90.0, 3.4, 1.9)]:
    t = np.array([0.0, span])
    pc = tls_grids.period_grid_ofir(t, R_star=R, M_star=M)
    pr = np.sort(period_grid(R, M, span))
    print("span=%6.1f R=%.1f M=%.1f: cuv n=%d [%.4f..%.3f]  ref n=%d [%.4f..%.3f]  max rel diff (matched by index)=%.2e" % (span, R, M, len(pc), pc.min(), pc.max(), len(pr), pr.min(), pr.max(), np.max(np.abs(pc[:min(len(pc),len(pr))]/pr[:min(len(pc),len(pr))]-1))))
# user limits -> few periods
t = np.array([0.0, 60.0])
pc = tls_grids.period_grid_ofir(t, period_min=5.0, period_max=5.05)
pr = period_grid(1, 1, 60.0, period_min=5.0, period_max=5.05)
print("narrow user range [5,5.05]: cuv n=%d, ref n=%d (ref falls back to a 100-pt default grid when <100)" % (len(pc), len(pr)))
pc = tls_grids.period_grid_ofir(t, period_min=40.0, period_max=100.0)
print("user range beyond span/2 [40,100] on a 60-d span: cuv n=%d [%.2f..%.2f] (linspace fallback)" % (len(pc), pc.min(), pc.max()))
# Ofir grid spacing vs duration: phase shift across the baseline between adjacent periods, in units of the narrowest searched duration (0.5 q_kep)
span = 1400.0
pc = tls_grids.period_grid_ofir(np.array([0, span]))
q = tls_grids.q_transit(pc)
dphi = span * np.abs(np.diff(1/pc))  # phase drift across the baseline
ratio = dphi / (q[:-1])
print("Ofir grid (OS=3, 1400 d): drift-between-adjacent-periods / q_kep: min=%.3f med=%.3f max=%.3f  -> / (0.5 q_kep) max=%.3f" % (ratio.min(), np.median(ratio), ratio.max(), 2*ratio.max()))
# t0 grid: n_t0 = ceil(3/q); phase stride q/3 -> max epoch error q/6 duration; refinement halfwidth
for q in [0.2, 0.1, 0.05, 0.01, 0.002, 0.0005, 0.0001]:
    print("q=%.4f: n_t0=%d stride/q=%.3f (needs %d bins; cap 8192 -> smear %.2f)" % (q, tls_grids.t0_grid_size(q), 1.0/tls_grids.t0_grid_size(q)/q, 3/q, max(1, 3/q/8192)))
# duration grid (standard fixed vs keplerian) and reference
for P in [0.5, 1, 3, 10, 30, 100, 365, 700]:
    qk = tls_grids.q_transit(P); 
    print("P=%5.0f: q_kep(Sun,1Re)=%.5f window [%.5f,%.5f]; ref window [%.5f,%.5f]; std window [0.005,0.15] %s" % (P, qk, 0.5*qk, 2*qk, T14(0.13, 0.1, P, small=True), T14(3.5, 1.0, P, small=False), 'covers' if 0.005 <= qk <= 0.15 else 'MISSES q_kep'))
d, c = tls_grids.duration_grid([1.0, 10.0]); print("duration_grid P=1,10: counts", c, d[0][:3], d[1][-2:])
try:
    tls_grids.duration_grid([1.0], R_planet_min=5, R_planet_max=0.5)
except Exception as e: print("duration_grid with R_planet_min>R_planet_max ->", type(e).__name__, e)
print("period_grid_ofir with baseline < 2 Roche periods (1 d):", tls_grids.period_grid_ofir(np.array([0, 1.0]))[:3], "...")
