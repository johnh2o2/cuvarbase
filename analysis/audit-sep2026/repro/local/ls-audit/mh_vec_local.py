import numpy as np, time
src = open('/Users/johnhoffman/Documents/cuvarbase/cuvarbase/lombscargle.py').read()
a = src.index('def mhdirect_sums'); b = src.index('def lomb_scargle_direct_sums')
ns = {}; exec("import numpy as np\n" + src[a:b], ns)
_mh_power_from_spectra = ns['_mh_power_from_spectra']

def mh_power_vectorized(sw, syw, k0, H, nf, YY):
    i = np.arange(nf)
    c = np.empty((2 * H + 1, nf)); s = np.empty((2 * H + 1, nf)); c[0] = 1; s[0] = 0
    for m in range(1, 2 * H + 1):
        v = sw[(m - 1) * k0 + m * i]; c[m] = v.real; s[m] = v.imag
    YC = np.empty((H, nf)); YS = np.empty((H, nf))
    for h in range(1, H + 1):
        v = syw[(h - 1) * k0 + h * i]; YC[h - 1] = v.real; YS[h - 1] = v.imag
    hs = np.arange(1, H + 1)
    n = hs[:, None]; m = hs[None, :]
    sgn = np.sign(n - m); sgn[n == m] = 1
    CC = 0.5 * (c[n + m] + c[abs(n - m)]) - c[1:H+1][:, None, :] * c[1:H+1][None, :, :]
    CS = 0.5 * (s[n + m] - sgn[..., None] * s[abs(n - m)]) - c[1:H+1][:, None, :] * s[1:H+1][None, :, :]
    SS = 0.5 * (c[abs(n - m)] - c[n + m]) - s[1:H+1][:, None, :] * s[1:H+1][None, :, :]
    A = np.empty((nf, 2 * H, 2 * H))
    A[:, :H, :H] = CC.transpose(2, 0, 1); A[:, :H, H:] = CS.transpose(2, 0, 1)
    A[:, H:, :H] = CS.transpose(2, 1, 0); A[:, H:, H:] = SS.transpose(2, 0, 1)
    bvec = np.concatenate([YC, YS]).T
    theta = np.linalg.solve(A, bvec[..., None])[..., 0]
    return np.einsum('ij,ij->i', theta, bvec) / YY   # b^T A^-1 b / YY

rng = np.random.RandomState(0)
for H in (2, 3):
    nf, k0 = 20000, 7
    sw = 0.05 * (rng.randn((2 * H - 1) * k0 + 2 * H * nf + 1) + 1j * rng.randn((2 * H - 1) * k0 + 2 * H * nf + 1))
    syw = 0.05 * (rng.randn((H - 1) * k0 + H * nf + 1) + 1j * rng.randn((H - 1) * k0 + H * nf + 1))
    t0 = time.time(); p0 = _mh_power_from_spectra(sw, syw, k0, H, nf, 1.0); t_loop = time.time() - t0
    t0 = time.time(); p1 = mh_power_vectorized(sw, syw, k0, H, nf, 1.0); t_vec = time.time() - t0
    print("H=%d nf=%d: host loop %.2f s (%.0f us/freq), vectorized %.4f s (%.2f us/freq), %.0fx; max|diff|=%.1e" % (H, nf, t_loop, 1e6 * t_loop / nf, t_vec, 1e6 * t_vec / nf, t_loop / t_vec, np.abs(p0 - p1).max()))
