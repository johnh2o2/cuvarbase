"""Sweep block_size x nbins for the coarse TLS kernel (kepler-4yr-like
config), reporting steady-state kernel-only times."""
import warnings
import time

warnings.filterwarnings('ignore')

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from cuvarbase import tls, tls_grids, tls_models
from cuvarbase.base import ensure_context

ensure_context()

ndata, nlc = 65440, 4
cad = 30. / 60 / 24
lcs = []
for i in range(nlc):
    rng = np.random.RandomState(1234 + i)
    t = np.arange(ndata) * cad
    y = 1.0 + rng.randn(ndata) * 6e-4
    lcs.append((t, y, np.full(ndata, 6e-4)))

periods = tls_grids.period_grid_ofir(
    lcs[0][0], R_star=1.0, M_star=1.0, oversampling_factor=3,
    period_min=0.6, period_max=500.)
periods32 = np.asarray(periods, np.float32)
_, _, qv = tls_grids.duration_grid_keplerian(
    np.asarray(periods, np.float64), R_star=1.0, M_star=1.0,
    R_planet=1.0, qmin_fac=0.5, qmax_fac=2.0, n_durations=15)
qmin = (qv * 0.5).astype(np.float32)
qmax = (qv * 2).astype(np.float32)
nperiods = len(periods32)

t_hi_c, t_lo_c, a_c, b_c, offs, lens, chi2_0, epochs, spans = \
    tls._preprocess_batch(lcs)
T_tab, S1_tab, S2_tab = tls_models.generate_template_tables()
periods_g = gpuarray.to_gpu(periods32)
qmin_g_ = gpuarray.to_gpu(qmin)
qmax_g_ = gpuarray.to_gpu(qmax)
S1_g = gpuarray.to_gpu(S1_tab)
S2_g = gpuarray.to_gpu(S2_tab)
thi_g = gpuarray.to_gpu(t_hi_c)
tlo_g = gpuarray.to_gpu(t_lo_c)
a_g = gpuarray.to_gpu(a_c)
b_g = gpuarray.to_gpu(b_c)
off_g = gpuarray.to_gpu(offs.astype(np.int32))
len_g = gpuarray.to_gpu(lens.astype(np.int32))
outn = nlc * nperiods
chi2_g = gpuarray.empty(outn, np.float32)
t0_g = gpuarray.empty(outn, np.float32)
dur_g = gpuarray.empty(outn, np.float32)
dep_g = gpuarray.empty(outn, np.float32)


map_g = gpuarray.to_gpu(np.arange(nperiods, dtype=np.int32))


def launch(k, bs, smem):
    k['search'](thi_g, tlo_g, a_g, b_g, off_g, len_g, periods_g,
                qmin_g_, qmax_g_, map_g, S1_g, S2_g,
                np.int32(nperiods), np.int32(nperiods), np.int32(15),
                chi2_g, t0_g, dur_g, dep_g,
                block=(bs, 1, 1), grid=(nperiods, nlc, 1), shared=smem)


ref_chi2 = None
for bs in (128, 256, 512):
    for nb in (4096, 8192):
        try:
            k = tls._get_cached_fast_kernels(bs, nb, 3.0)
            smem = tls._tls_fast_shared_size(bs, nb)
            launch(k, bs, smem)
            cuda.Context.synchronize()
            ts = []
            for _ in range(3):
                cuda.Context.synchronize()
                s = time.perf_counter()
                launch(k, bs, smem)
                cuda.Context.synchronize()
                ts.append(time.perf_counter() - s)
            med = sorted(ts)[1]
            c = chi2_g.get()[:nperiods]
            if ref_chi2 is None:
                ref_chi2 = c
                corr = 1.0
            else:
                ok = (c > 0) & (ref_chi2 > 0)
                corr = np.corrcoef(c[ok], ref_chi2[ok])[0, 1]
            print("bs=%d nbins=%d smem=%dKB: %.3f s (%.1f ms/LC) "
                  "scoremax=%.1f corr_vs_first=%.5f"
                  % (bs, nb, smem // 1024, med, 1000 * med / nlc,
                     np.nanmax(c), corr))
        except Exception as e:
            print("bs=%d nbins=%d FAIL: %r" % (bs, nb, e))
