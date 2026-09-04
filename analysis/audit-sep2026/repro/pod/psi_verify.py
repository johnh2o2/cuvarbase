"""Independent minimal reproduction for topic nfft-psi-table (verifier).
(a) default LombScargleAsyncProcess path: w-grid spectrum vs exact DFT, yw-grid vs exact DFT
(b) CPU emulation of the hypothesised root cause: each point's Gaussian is displaced by
    delta_i = frac(ng_yw*x_i) - frac(ng_w*x_i) cells on the w grid -> exact DFT of w at
    displaced positions should match the device w-spectrum to roundoff.
(c) candidate fix: own psi for the w grid.
(d) k0 = 0 grid (fmin = 0) to show it is not a k0 artefact.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.timeseries import LombScargle
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0
from cuvarbase.utils import normalize_light_curves

def dft(t, c, freqs, chunk=4000):
    out = np.empty(len(freqs), complex)
    for a in range(0, len(freqs), chunk):
        ph = 2 * np.pi * np.outer(freqs[a:a + chunk], t)
        out[a:a + chunk] = (np.cos(ph) + 1j * np.sin(ph)) @ c
    return out

rng = np.random.RandomState(7)
N = 200; T = 200.0
t = np.sort(rng.rand(N) * T); dy = 0.1 * (1 + rng.rand(N))
y = 3.0 + 0.5 * np.cos(2 * np.pi * 2.37 * t + 1.0) + dy * rng.randn(N)
spp = 5.0

def run(freqs, use_double, fix=False, sigma=4, m=8, fast_grid=True):
    nf = len(freqs); k0 = get_k0(freqs); df = freqs[1] - freqs[0]
    (tn, yn, dyn), = normalize_light_curves([(t, y, dy)])
    proc = LombScargleAsyncProcess(use_double=use_double, sigma=sigma, m=m, autoset_m=False)
    mem = proc.allocate([(tn, yn, dyn)], nfreqs=[nf], k0s=[k0])
    if fix:
        mem[0].nfft_mem_w.precomp_psi = True
        mem[0].nfft_mem_w.allocate_precomp_psi(n0=N)
    r = proc.run([(t, y, dy)], memory=mem, freqs=freqs, fast_grid=fast_grid)
    proc.finish()
    p = np.array(r[0][1][:nf], float)
    mw, myw = mem[0].nfft_mem_w, mem[0].nfft_mem_yw
    SW = mw.ghat_g.get().astype(complex); SYW = myw.ghat_g.get().astype(complex)
    w = dyn ** -2; w /= w.sum(); ybar = np.dot(w, yn); yw = w * (yn - ybar)
    nmw = 2 * nf + k0; SW = SW[:nmw]; modes_w = k0 + np.arange(nmw); modes_yw = k0 + np.arange(nf)
    # scaled coordinate exactly as the kernel does (spp*(xf-x0) normalisation)
    xg = (tn - tn.min()) * df   # kernel: (t-x0)/(spp*(xf-x0)) with spp = 1/(df*(xf-x0))
    ng_yw, ng_w = myw.n, mw.n
    # hypothesised displacement (in cells of the w grid) from using yw-grid psi on the w grid
    delta = (np.mod(ng_yw * xg, 1.0) - np.mod(ng_w * xg, 1.0)) / ng_w
    # note: kernel freq of mode k on grid ng is k/ng cycles per unit xg; t-domain freq = k*df
    ew = np.abs(SW - dft(tn, w, modes_w * df)).max()
    eyw = np.abs(SYW[:nf] - dft(tn, yw, modes_yw * df)).max()
    # displaced positions: xg + delta -> in t units: tn + delta*spp*(tmax-tmin)
    t_disp = tn + delta / df
    ew_model = np.abs(SW - dft(t_disp, w, modes_w * df)).max()
    ref = LombScargle(t, y, dy).power(freqs, method='cython')
    d = p - ref
    return dict(ng_yw=ng_yw, ng_w=ng_w, k0=k0, ew=ew, eyw=eyw, ew_model=ew_model,
                pmax=np.abs(d).max(), rel_peak=abs(d[np.argmax(ref)]) / ref.max(),
                ndelta=np.sum(np.abs(delta) > 1e-12), max_delta_cells=np.abs(delta * ng_w).max())

def show(lbl, r):
    print("%-44s k0=%d ng_yw=%d ng_w=%d | max|Sw-exact|=%.2e  model(displaced)=%.2e  max|Syw-exact|=%.2e | power max|d|=%.2e rel@peak=%.2e | pts displaced=%d max|delta|=%.2f cells"
          % (lbl, r['k0'], r['ng_yw'], r['ng_w'], r['ew'], r['ew_model'], r['eyw'], r['pmax'], r['rel_peak'], r['ndelta'], r['max_delta_cells']))

df = 1.0 / (spp * T)
for lbl, freqs in (("fmin=df (k0=1)", df * (1 + np.arange(int(10.0 / df)))),
                   ("fmin=100df (k0=100)", df * (100 + np.arange(int(10.0 / df))))):
    print("=== grid:", lbl, "nf=%d fmax=%.1f" % (len(freqs), freqs.max()))
    for ud in (True, False):
        show("  dbl=%s shipped (shared psi)" % ud, run(freqs, ud))
        show("  dbl=%s own psi for w grid (fix)" % ud, run(freqs, ud, fix=True))
    show("  dbl=True fast_grid=False (no psi)", run(freqs, True, fast_grid=False))
