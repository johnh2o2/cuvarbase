import numpy as np, warnings, time, sys
warnings.simplefilter('ignore')
import batman

G = 6.67430e-11; R_sun = 6.957e8; M_sun = 1.9884e30; R_earth = 6.371e6

def a_over_rs(P_days, R_star=1.0, M_star=1.0):
    P = P_days * 86400.0
    a = (G * M_star * M_sun * P**2 / (4 * np.pi**2))**(1/3)
    return a / (R_star * R_sun)

def t14_days(P_days, rp, R_star=1.0, M_star=1.0, b=0.0):
    ars = a_over_rs(P_days, R_star, M_star)
    inc = np.degrees(np.arccos(b / ars))
    x = np.sqrt((1 + rp)**2 - b**2) / ars / np.sin(np.radians(inc))
    return P_days / np.pi * np.arcsin(np.clip(x, 0, 1))

def make_lc(P, rp, t0, baseline=60.0, cadence_min=30.0, sigma=1e-3, seed=0,
            R_star=1.0, M_star=1.0, b=0.0, u=(0.4804, 0.1867), limb_dark='quadratic',
            t_start=0.0, random_times=False):
    rng = np.random.RandomState(seed)
    n = int(baseline * 1440 / cadence_min)
    if random_times:
        t = np.sort(rng.rand(n) * baseline) + t_start
    else:
        t = t_start + np.arange(n) * cadence_min / 1440.0
    p = batman.TransitParams()
    p.t0 = t0; p.per = P; p.rp = rp; p.a = a_over_rs(P, R_star, M_star)
    p.inc = np.degrees(np.arccos(b / p.a)); p.ecc = 0.0; p.w = 90.0
    p.u = list(u); p.limb_dark = limb_dark
    m = batman.TransitModel(p, t)
    f = m.light_curve(p)
    y = f + sigma * rng.randn(n)
    dy = sigma * np.ones(n)
    true_depth = 1.0 - f.min()
    return t, y, dy, dict(true_depth=true_depth, t14=t14_days(P, rp, R_star, M_star, b), a=p.a, inc=p.inc)

def noise_lc(baseline=60.0, cadence_min=30.0, sigma=1e-3, seed=0):
    rng = np.random.RandomState(seed)
    n = int(baseline * 1440 / cadence_min)
    t = np.arange(n) * cadence_min / 1440.0
    y = 1 + sigma * rng.randn(n)
    return t, y, sigma * np.ones(n)

def run_ref(t, y, dy, period_min=None, period_max=None, threads=16, **kw):
    import transitleastsquares as tls
    model = tls.transitleastsquares(t, y, dy)
    args = dict(show_progress_bar=False, use_threads=threads, verbose=False)
    if period_min is not None: args['period_min'] = period_min
    if period_max is not None: args['period_max'] = period_max
    args.update(kw)
    t1 = time.time(); r = model.power(**args); dt = time.time() - t1
    return r, dt

def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1]
