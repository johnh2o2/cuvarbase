"""Exp 8: Keplerian frequency-grid helpers vs physics and vs recovery."""
import sys
sys.path.insert(0, '/workspace/scratch/blsaudit')
import numpy as np
from blsref import make_data, exact_bls
from cuvarbase.bls import (transit_autofreq, q_transit, fmax_transit0, fmin_transit, freq_transit,
                           eebls_gpu_fast, eebls_transit)
from astropy import constants as c, units as u

# (a) fmax_transit0 and q_transit vs physics (circular orbit at rho_sun, b=0, Rp->0)
rho = c.M_sun / (4. / 3. * np.pi * c.R_sun ** 3)
fmax0 = np.sqrt(c.G * rho / (3 * np.pi)).to(1 / u.day).value
print("fmax_transit0 (rho=1) code=%.4f  physics=%.4f  (%.2f%%)" % (fmax_transit0(), fmax0, 100 * (fmax_transit0() / fmax0 - 1)))
for P in [1., 3., 10., 30.]:
    a = ((c.G * c.M_sun * (P * u.day) ** 2 / (4 * np.pi ** 2)) ** (1. / 3.)).to(u.R_sun).value
    q_phys = np.arcsin(1. / a) / np.pi  # T14/P for b=0, k->0
    print("  P=%5.1f d: a/R*=%.2f  q_kep(code)=%.5f  q_phys=%.5f  T14=%.2f h" % (P, a, q_transit(1. / P), q_phys, q_phys * P * 24))

# (b) grid density: recovery of a signal injected halfway between grid points
t, y, dy = make_data(ndata=3000, baseline=120., freq=0.5, q=0.02, phi0=0.3, snr=20, seed=31)
freqs, q0 = transit_autofreq(t, qmin_fac=0.5, fmin=0.45, fmax=0.55)
print("autofreq: fmin_transit=%.4f (2/T=%.4f), nfreqs in [0.45,0.55] = %d, df at 0.5 = %.3e = q0/(%.2f T)"
      % (fmin_transit(t), 2 / (t.max() - t.min()), len(freqs), freqs[1] - freqs[0],
         q_transit(0.5) / ((freqs[1] - freqs[0]) * (t.max() - t.min()))))
i = int(np.argmin(np.abs(freqs - 0.5)))
for off_label, ftrue in [("on-grid", freqs[i]), ("mid-grid", 0.5 * (freqs[i] + freqs[i + 1]))]:
    tt, yy, ddy = make_data(ndata=3000, baseline=120., freq=ftrue, q=q_transit(ftrue), phi0=0.3, snr=20, seed=31)
    pf = eebls_gpu_fast(tt, yy, ddy, freqs, qmin=0.5 * q0, qmax=2 * q0)
    ex = exact_bls(tt, yy, ddy, ftrue, q_transit(ftrue), 0.3)
    print("  %s f=%.6f: exact=%.4f grid max=%.4f (%.0f%%) at f=%.6f" % (off_label, ftrue, ex, pf.max(), 100 * pf.max() / ex, freqs[np.argmax(pf)]))
    for spp in [1, 2, 5]:
        fr, qq = transit_autofreq(tt, qmin_fac=0.5, fmin=0.45, fmax=0.55, samples_per_peak=spp)
        pf = eebls_gpu_fast(tt, yy, ddy, fr, qmin=0.5 * qq, qmax=2 * qq)
        print("     samples_per_peak=%d: nfreqs=%d grid max=%.4f (%.0f%% of exact)" % (spp, len(fr), pf.max(), 100 * pf.max() / ex))

# (c) default fmin/fmax for eebls_transit at a few baselines & ndata
for ndata, T in [(300, 100.), (1200, 200.), (18000, 27.), (60000, 1400.)]:
    tt = np.sort(T * np.random.RandomState(1).rand(ndata))
    fr, qq = transit_autofreq(tt, qmin_fac=0.5)
    print("ndata=%d T=%.0f: fmin=%.5f (P_max=%.1f d; 2/T -> %.1f d; min_obs -> %.5f) fmax=%.3f (P_min=%.2f h) nfreqs=%d"
          % (ndata, T, fr[0], 1 / fr[0], T / 2, freq_transit(5. / ndata), fr[-1], 24 / fr[-1], len(fr)))
