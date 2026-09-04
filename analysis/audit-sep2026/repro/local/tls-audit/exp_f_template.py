"""Experiment F (CPU): template shape vs reference; depth bias; table consistency; trapezoid fallback."""
import numpy as np, sys
sys.path.insert(0, '/workspace/scratch')
from audit_common import *
from cuvarbase import tls_models
from transitleastsquares.transit import reference_transit
import transitleastsquares.tls_constants as C

n = 1001
x = np.linspace(-1, 1, n)
T_cuv = tls_models.generate_transit_template(n_template=n)
T_ref = 1.0 - reference_transit(samples=n, per=C.DEFAULT_PERIOD, rp=C.DEFAULT_RP, a=C.DEFAULT_A, inc=C.DEFAULT_INC, ecc=0, w=90, u=C.DEFAULT_U, limb_dark='quadratic')
T_trap = tls_models._trapezoid_template(n)
def ingress_frac(T):  # fraction of half-width until template reaches 0.9 of max
    i = np.argmax(T > 0.9); return (x[i] + 1) / 2
print("template: cuv(rp=0.1,a=15,inc=90) vs ref(rp=0.03,a=23.1,inc=89.21): rms diff=%.4f max diff=%.4f; ingress frac (to 0.9 depth) cuv=%.3f ref=%.3f trap=%.3f; mean T: cuv=%.4f ref=%.4f trap=%.4f; mean T^2: cuv=%.4f ref=%.4f trap=%.4f" % (
    np.sqrt(np.mean((T_cuv - T_ref)**2)), np.max(np.abs(T_cuv - T_ref)), ingress_frac(T_cuv), ingress_frac(T_ref), ingress_frac(T_trap), T_cuv.mean(), T_ref.mean(), T_trap.mean(), (T_cuv**2).mean(), (T_ref**2).mean(), (T_trap**2).mean()))
print("edges: cuv T[0]=%.2e T[-1]=%.2e T[mid]=%.5f ; symmetric max|T(x)-T(-x)|=%.2e" % (T_cuv[0], T_cuv[-1], T_cuv[n//2], np.max(np.abs(T_cuv - T_cuv[::-1]))))
# LS depth bias when the true signal shape differs from the template (dense uniform sampling, matched duration)
def ls_depth(signal, T): return np.sum(signal * T) / np.sum(T * T)
box = np.ones(n)
for name, sig in [('box', box), ('ref-shape(rp=0.03)', T_ref), ('cuv-shape', T_cuv)]:
    print("true signal %-20s unit max depth: LS depth with cuv template=%.3f, with ref template=%.3f, with trapezoid=%.3f; score ratio (cuv/ref template) = %.3f" % (
        name, ls_depth(sig, T_cuv), ls_depth(sig, T_ref), ls_depth(sig, T_trap), (np.sum(sig*T_cuv)**2/np.sum(T_cuv**2)) / (np.sum(sig*T_ref)**2/np.sum(T_ref**2))))
# a real batman transit for a small planet rp=0.02 (b=0) and big planet rp=0.15 -> in transit_coord units
import batman
for rp in [0.01, 0.03, 0.1, 0.15]:
    p = batman.TransitParams(); p.t0 = 0; p.per = 10.0; p.rp = rp; p.a = a_over_rs(10.0); p.inc = 90; p.ecc = 0; p.w = 90; p.u = C.DEFAULT_U; p.limb_dark = 'quadratic'
    t14 = t14_days(10.0, rp); tt = np.linspace(-t14/2, t14/2, n)
    f = batman.TransitModel(p, tt).light_curve(p); s = (1 - f)
    dmax = s.max()
    print("batman rp=%.2f: true max depth=%.5f ; LS depth (matched duration) cuv template=%.5f (%.3fx) ref template=%.5f (%.3fx); mean in-transit dimming=%.5f" % (rp, dmax, ls_depth(s, T_cuv), ls_depth(s, T_cuv)/dmax, ls_depth(s, T_ref), ls_depth(s, T_ref)/dmax, s.mean()))
# table consistency
T, S1, S2 = tls_models.generate_template_tables(n_table=1024)
xs = np.linspace(-1, 1, 1025)
Tf = tls_models.generate_transit_template(n_template=1025)
print("tables: T vs direct template max diff=%.2e ; S1[-1]=%.5f vs mean(T)*2=%.5f ; S2[-1]=%.5f vs mean(T^2)*2=%.5f ; S1 monotone=%s" % (np.max(np.abs(T - Tf)), S1[-1], 2*Tf.mean(), S2[-1], 2*(Tf**2).mean(), bool(np.all(np.diff(S1) >= 0))))
# limb darkening laws
for law, u in [('linear', [0.5]), ('quadratic', [0.4804, 0.1867]), ('nonlinear', [0.5, 0.1, 0.1, -0.1]), ('uniform', [])]:
    try:
        Tl = tls_models.generate_transit_template(n_template=201, limb_dark=law, u=u)
        print("law %-10s ok: mean T=%.4f" % (law, Tl.mean()))
    except Exception as e:
        print("law %-10s -> %s: %s" % (law, type(e).__name__, e))
    try:
        tls_models.validate_limb_darkening_coeffs(u, law); print("   validate ok")
    except Exception as e: print("   validate ->", type(e).__name__, e)
