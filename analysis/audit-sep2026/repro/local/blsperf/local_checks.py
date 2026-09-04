"""Local (no-GPU) checks: grid-generation cost and nbins rounding parity."""
import time, importlib.util, sys, types
import numpy as np

# load bls_frequencies without cuvarbase/__init__ (pycuda absent locally)
spec = importlib.util.spec_from_file_location(
    'bls_frequencies', '/Users/johnhoffman/Documents/cuvarbase/cuvarbase/bls_frequencies.py')
bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)

# copy of bls.transit_autofreq machinery (pure numpy) -----------------
def fmax_transit0(rho=1.): return 8.6307*np.sqrt(rho)
def q_transit(freq, rho=1.):
    f23 = np.minimum(1., np.power(freq/fmax_transit0(rho), 2./3.)); return np.arcsin(f23)/np.pi
def freq_transit(q, rho=1.): return fmax_transit0(rho)*(np.sin(np.pi*q)**1.5)
def fmin_transit(t, rho=1., min_obs_per_transit=5):
    return max(freq_transit(float(min_obs_per_transit)/len(t), rho), 2./(t.max()-t.min()))
def fmax_transit(rho=1., qmax=0.5): return min(fmax_transit0(rho), freq_transit(qmax, rho))
def transit_autofreq(t, fmin=None, fmax=None, samples_per_peak=2, rho=1., qmin_fac=0.2, qmax_fac=None):
    if qmax_fac is None: qmax_fac = 1./qmin_fac
    if fmin is None: fmin = fmin_transit(t, rho=rho)
    if fmax is None: fmax = fmax_transit(rho=rho, qmax=0.5/qmax_fac)
    T = t.max()-t.min(); freqs=[fmin]
    while freqs[-1] < fmax:
        df = qmin_fac*q_transit(freqs[-1], rho=rho)/(samples_per_peak*T); freqs.append(freqs[-1]+df)
    freqs = np.array(freqs); return freqs, q_transit(freqs, rho=rho)

def vectorized_autofreq(t, fmin, fmax, samples_per_peak=2, rho=1., qmin_fac=0.2):
    # closed-form: df/dn = qmin_fac*q(f)/(OS*T) -> integrate dn = OS*T/(qmin_fac) * df/q(f)
    # do it with a fine cumulative trapezoid in f (exact to the ODE, differs from the
    # Euler recursion by O(df^2)); this is only a cost estimate of the vectorized form
    T = t.max()-t.min()
    fgrid = np.linspace(fmin, fmax, 200000)
    inv = 1.0/q_transit(fgrid, rho)
    n = np.concatenate(([0.], np.cumsum(0.5*(inv[1:]+inv[:-1])*np.diff(fgrid)))) * samples_per_peak*T/qmin_fac
    N = int(np.floor(n[-1]))
    return np.interp(np.arange(N+1), n, fgrid)

for name, ndata, baseline in [('ZTF',150,730.), ('HAT',6000,3650.), ('TESS',20000,27.), ('Kepler',65000,1460.)]:
    rng = np.random.RandomState(0); t = np.sort(rng.uniform(0, baseline, ndata))
    t0=time.perf_counter(); f,q = transit_autofreq(t, qmin_fac=0.5); dt=time.perf_counter()-t0
    t0=time.perf_counter(); fv = vectorized_autofreq(t, fmin_transit(t), fmax_transit(qmax=0.25), qmin_fac=0.5); dtv=time.perf_counter()-t0
    print(f"{name:7s} transit_autofreq(eebls_transit default, qmin_fac=0.5): nfreq={len(f):7d}  {dt*1e3:8.1f} ms   (vectorized ODE form: {len(fv)} freqs, {dtv*1e3:.1f} ms)")
    t0=time.perf_counter(); fk,qk = bf.keplerian_freq_grid(0.5, min(100., baseline/2), baseline, oversampling=2, return_qvals=True); dtk=time.perf_counter()-t0
    print(f"        keplerian_freq_grid(0.5d..{min(100., baseline/2)}d): nfreq={len(fk):7d}  {dtk*1e3:8.1f} ms")
    # nbins rounding parity between BLSMemory (float64 1/qmin) and BLSBatchMemory (float32 1/qmin)
    qmin = (0.5*qk).astype(np.float64); qmax=(2.0*qk).astype(np.float64)
    nbf64 = (np.ones(len(fk))/qmin).astype(np.uint32); nb064=(np.ones(len(fk))/qmax).astype(np.uint32)
    nbf32 = (1.0/np.asarray(qmin,dtype=np.float32)).astype(np.uint32); nb032=(1.0/np.asarray(qmax,dtype=np.float32)).astype(np.uint32)
    print(f"        nbins parity fast(f64) vs batch(f32): nbinsf differ at {np.sum(nbf64!=nbf32)} / {len(fk)} freqs, nbins0 differ at {np.sum(nb064!=nb032)}")
    # and for scalar defaults qmin=1e-2,qmax=0.5
q=np.float64(1e-2); print("scalar qmin=1e-2: f64 ->", int(1/q), " f32 ->", int(np.float32(1.0)/np.float32(q)))
for q in [0.01,0.02,0.03,0.05,0.07,0.1,0.001,0.003,0.005]:
    print(f"  qmin={q}: f64 nbins={int(1/np.float64(q))} f32 nbins={int(np.float32(1.0)/np.float32(q))}")
