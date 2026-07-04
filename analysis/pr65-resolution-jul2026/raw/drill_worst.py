"""Drill into the worst frequency of test_standard qi=0 pi=0:
compare single_bls's fold vs the GPU kernel's fold (with and without
FMA contraction of mod1) for the flip point, and reproduce p_gpu/p_cpu.
"""
import numpy as np
from sim_standard import data, subtract_epoch

f32 = np.float32

q_values = np.logspace(-1.5, np.log10(0.1), num=100)
phi_values = np.linspace(0, 1, int(np.ceil(2. / min(q_values))))
q = q_values[0]
phi = phi_values[0]
freq = 1.0

t, y, dy = data(snr=10, q=q, phi0=phi, freq=freq, baseline=365., tshift=4.5)
df = min(q_values) / (10 * (max(t) - min(t)))
delta_f = 5 * df / freq
freqs = np.linspace(freq * (1 - delta_f), (1 + delta_f) * freq,
                    int(5. * 2 * delta_f * freq / df))

# worst frequency from the GPU diagnostic
fworst = 0.9999707539
iw = int(np.argmin(np.abs(freqs - fworst)))
fq = freqs[iw]
print("iw:", iw, "freq:", repr(fq))

t_sub, epoch = subtract_epoch(t)
t32 = t_sub.astype(f32)
f_32 = f32(fq)

# GPU solution from diagnostic: nb=22, s=1, jphi=21
nb, s, jphi = 22, 1, 21
q_sol = 1.0 / nb
phi_sub_store = f32((1.0 / nb) * (jphi + s * 0.5))     # store_best_sols
phi_orig = (phi_sub_store + epoch * fq) % 1.0          # eebls_gpu re-ref
phi_sub_single = (phi_orig - epoch * fq) % 1.0         # single_bls converts
print("phi_sub_store:", repr(phi_sub_store),
      "round-trip:", repr(f32(phi_sub_single)),
      "equal:", f32(phi_sub_single) == phi_sub_store)

# ---- single_bls membership ----
phc = t32 * f_32
phc = phc - f32(phi_sub_single)
phc = phc - np.floor(phc)
mask_cpu = phc < f32(q_sol)

# ---- GPU membership, plain float32 (no FMA) ----
u = t32 * f_32                       # float32 product
phi_pt = u - np.floor(u)             # mod1, rounded product
arg = f32(nb) * phi_pt - f32(s) * f32(0.5)
b_plain = np.floor(arg).astype(int) % nb
mask_gpu_plain = b_plain == jphi

# ---- GPU membership, FMA-contracted mod1: t*f0 - floorf(t*f0) ----
# fmaf(t, f0, -floorf(u)): exact product (f64) minus floor of the
# ROUNDED product, rounded once to f32.
exact = t32.astype(np.float64) * np.float64(f_32)   # exact for f32 inputs
phi_fma = (exact - np.floor(u.astype(np.float64))).astype(f32)
arg_fma = f32(nb) * phi_fma - f32(s) * f32(0.5)
b_fma = np.floor(arg_fma).astype(int) % nb
mask_gpu_fma = b_fma == jphi
# also FMA in the bin-index expression itself
arg_fma2 = (phi_fma.astype(np.float64) * nb - 0.5).astype(f32)
b_fma2 = np.floor(arg_fma2).astype(int) % nb
mask_gpu_fma2 = b_fma2 == jphi

for name, m in [("cpu(single_bls)", mask_cpu),
                ("gpu_plain", mask_gpu_plain),
                ("gpu_fma_mod1", mask_gpu_fma),
                ("gpu_fma_both", mask_gpu_fma2)]:
    print(f"{name:>16}: n_in_box={m.sum()}")

flips = np.where(mask_cpu != mask_gpu_fma)[0]
print("flips cpu vs gpu_fma_mod1:", flips)
for i in np.unique(np.concatenate([flips, [198]])):
    print(f" idx={i} t32={t32[i]!r} u={u[i]!r} phi_pt={phi_pt[i]!r} "
          f"phi_fma={phi_fma[i]!r} phc={phc[i]!r} "
          f"cpu={mask_cpu[i]} plain={mask_gpu_plain[i]} fma={mask_gpu_fma[i]}")

# ---- reproduce powers ----
w = np.power(dy, -2)
w /= np.sum(w)
ybar_h = np.dot(w, y)
YY_h = np.dot(w, (np.array(y) - ybar_h) ** 2)
yw = ((np.array(y) - ybar_h) * w).astype(f32)
w32 = np.asarray(w).astype(f32)

def binned_power(mask):
    W = f32(0.); YW = f32(0.)
    for i in np.where(mask)[0]:
        YW = f32(YW + yw[i]); W = f32(W + w32[i])
    val = f32(YW * YW / (W * (f32(1.) - W)))
    return float(val) / YY_h

print("binned power with cpu membership:  ", binned_power(mask_cpu))
print("binned power with plain membership:", binned_power(mask_gpu_plain))
print("binned power with fma membership:  ", binned_power(mask_gpu_fma))
print("expected p_gpu=0.2319  p_cpu(single_bls)=0.1962")

# single_bls value
w64 = np.power(dy, -2); w64 /= np.sum(w64.astype(f32))
ybar_s = np.dot(w64, np.asarray(y).astype(f32))
YY_s = np.dot(w64, (np.asarray(y).astype(f32) - ybar_s) ** 2)
W = np.sum(w64[mask_cpu])
YW = np.dot(w64[mask_cpu], np.asarray(y).astype(f32)[mask_cpu]) - ybar_s * W
print("single_bls at sol:", (YW ** 2) / (W * (1 - W)) / YY_s)
