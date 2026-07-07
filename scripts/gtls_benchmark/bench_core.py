"""Apples-to-apples GTLS vs cuvarbase (TLS + BLS) — shared, GPU-independent core.

This module holds everything that does NOT touch a GPU: light-curve
injection (with a Keplerian-consistent duration so BOTH search grids bracket
the true transit), the single shared Ofir period grid fed to every method,
an identical-SDE recompute so all methods are scored by the same statistic,
and the timing bookkeeping. The GPU method calls live in the runner.

Design decisions (fairness):
- ONE light curve per baseline, fed to every method  -> identical SNR by
  construction (the comparison between methods can never differ in SNR).
- Injected transit uses Kepler's 3rd law for a(P) so its duration equals the
  physically expected Keplerian duration -> it lands inside cuvarbase's
  narrow 0.5-2x-Earth q band AND inside GTLS's wide duration grid. Neither
  method is handed a transit its grid cannot represent.
- ONE Ofir period grid (period_grid_ofir) is generated once and passed to
  gtls.power(periods=...), tls_search_batch(periods=...) and the BLS search,
  so the period axis is bit-identical across methods.
- SDE is recomputed with the SAME function on every method's chi2(P) spectrum.
"""
import importlib.util
import numpy as np

import os


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Prefer the installed package (pod / any env with cuvarbase importable);
# fall back to loading the single numpy-only module by path (local, no pycuda).
try:
    from cuvarbase import tls_grids
except Exception:
    _CUV = os.environ.get("CUVARBASE_DIR",
                          "/Users/johnhoffman/Documents/cuvarbase/cuvarbase")
    tls_grids = _load("tls_grids", _CUV + "/tls_grids.py")

# Physical constants (SI) for Kepler's third law
_G = 6.67430e-11
_MSUN = 1.98840e30
_RSUN = 6.95700e8
_SPD = 86400.0


def keplerian_a_over_Rstar(period_days, M_star=1.0, R_star=1.0):
    """a/R_star for a circular orbit from Kepler's third law."""
    P = period_days * _SPD
    a_m = (_G * M_star * _MSUN * P**2 / (4.0 * np.pi**2))**(1.0 / 3.0)
    return a_m / (R_star * _RSUN)


def transit_snr(depth, noise, period, duration_days, baseline_days, cadence_days):
    """Total transit SNR ~ (depth/noise) * sqrt(N_in_transit_total)."""
    n_transits = max(1, int(np.floor(baseline_days / period)))
    pts_per_transit = duration_days / cadence_days
    n_in = n_transits * pts_per_transit
    return depth / noise * np.sqrt(max(n_in, 1.0))


def make_lc(baseline_days, cadence_days, period, depth, noise, seed,
            M_star=1.0, R_star=1.0, u=(0.4804, 0.1867), inject=True):
    """Regular-cadence LC with an optional batman limb-darkened transit whose
    a/R_star follows Kepler's 3rd law (=> physical Keplerian duration).

    batman is only available on the GPU pod; imported lazily so this module
    loads locally for grid/SDE checks.
    """
    import batman
    rng = np.random.RandomState(seed)
    n = int(round(baseline_days / cadence_days))
    t = np.arange(n) * cadence_days
    y = 1.0 + rng.randn(n) * noise
    dy = np.full(n, noise, dtype=float)
    meta = dict(ndata=n, period=period, depth=depth, noise=noise,
                baseline=baseline_days, cadence=cadence_days)
    if inject:
        a = keplerian_a_over_Rstar(period, M_star, R_star)
        t0 = 0.35 * period + t.min()
        pm = batman.TransitParams()
        pm.t0 = t0
        pm.per = period
        pm.rp = float(np.sqrt(depth))       # depth ~ (Rp/Rs)^2
        pm.a = float(a)
        pm.inc = 90.0
        pm.ecc = 0.0
        pm.w = 90.0
        pm.u = list(u)
        pm.limb_dark = "quadratic"
        m = batman.TransitModel(pm, t)
        y = y + (m.light_curve(pm) - 1.0)
        # physical T14 (edge-on) for bookkeeping / grid-bracket check
        b = 0.0
        T14 = period / np.pi * np.arcsin(
            1.0 / a * np.sqrt((1.0 + pm.rp)**2 - b**2))
        meta.update(t0=t0, a_over_Rstar=a, T14_days=float(T14),
                    q_true=float(T14 / period),
                    snr=float(transit_snr(depth, noise, period, T14,
                                          baseline_days, cadence_days)))
    return t, y, dy, meta


def shared_period_grid(t, oversampling_factor=3, period_min=0.6,
                       period_max=None):
    """The single Ofir grid every method searches (Pmax defaults to S/2)."""
    return tls_grids.period_grid_ofir(
        t, R_star=1.0, M_star=1.0, oversampling_factor=oversampling_factor,
        period_min=period_min, period_max=period_max)


def recompute_sde_from_sr(sr, periods, oversampling_factor=3):
    """The one identical SDE routine. Takes a signal-residue spectrum SR(P)
    (large = better fit) already ascending in period. SDE = detrended-SR peak
    z-score with a median filter (window = OS*30, TLS convention, odd, capped
    at 91). Every method is scored through THIS function so the statistic is
    identical; only the spectrum differs."""
    from scipy.signal import medfilt
    sr = np.asarray(sr, dtype=float)
    p = np.asarray(periods, dtype=float)
    ok = np.isfinite(sr) & np.isfinite(p)
    sr, p = sr[ok], p[ok]
    if sr.size < 5:
        return dict(SDE=0.0, best_period=float("nan"), depth_snr=0.0)
    w = min(int(oversampling_factor * 30), 91)
    if w % 2 == 0:
        w += 1
    trend = medfilt(sr, kernel_size=w) if (3 <= w < len(sr)) \
        else np.zeros_like(sr)
    resid = sr - trend
    sd = resid.std()
    SDE = resid / sd if sd > 0 else resid * 0.0
    ibest = int(np.argmax(SDE))
    return dict(SDE=float(SDE[ibest]), best_period=float(p[ibest]),
                sr_max=float(np.nanmax(sr)))


def recompute_sde(chi2, periods, oversampling_factor=3):
    """Score a chi2(period) spectrum: SR = 1 - chi2/chi2_null(=max), then the
    identical SDE routine above."""
    chi2 = np.asarray(chi2, dtype=float)
    ok = np.isfinite(chi2) & (chi2 < 1e29)
    c = chi2[ok]
    p = np.asarray(periods, dtype=float)[ok]
    if c.size < 5:
        return dict(SDE=0.0, best_period=float("nan"), depth_snr=0.0)
    chi2_null = np.nanmax(c)
    out = recompute_sde_from_sr(1.0 - c / chi2_null, p, oversampling_factor)
    out["depth_snr"] = float(np.sqrt(max(chi2_null - np.nanmin(c), 0.0)))
    out["chi2_min"] = float(np.nanmin(c))
    return out


def recovered(p_found, p_true, tol=0.02):
    for k in (1.0, 2.0, 0.5, 3.0, 1 / 3.0):
        if abs(p_found - k * p_true) / (k * p_true) < tol:
            return True
    return False


if __name__ == "__main__":
    # Local self-test of the fairness invariants (no GPU, no batman needed
    # for the grid parts).
    for base in (200, 1500, 3000):
        cad = 30.0 / 60 / 24
        t = np.arange(int(base / cad)) * cad
        pg = shared_period_grid(t)
        # pick a period giving >=3 transits even at the shortest baseline
        P = 8.13
        a = keplerian_a_over_Rstar(P)
        T14 = P / np.pi * np.arcsin(1.0 / a * np.sqrt((1 + 0.05)**2))
        q = T14 / P
        # is q inside cuvarbase's 0.5-2x Earth band at this period?
        _, _, qvals = tls_grids.duration_grid_keplerian(
            np.array([P]), 1.0, 1.0, 1.0, n_durations=15)
        band = (0.5 * qvals[0], 2.0 * qvals[0])
        print(f"base={base:5d} Npg={len(pg):7d} P={P} T14={T14*24:.2f}h "
              f"q_true={q:.4f} cuvar_band=[{band[0]:.4f},{band[1]:.4f}] "
              f"in_band={band[0] <= q <= band[1]}")
