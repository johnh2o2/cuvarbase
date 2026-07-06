"""Stage-level profiling + tuning sweeps for the fast TLS batch engine.

Times each stage of tls_search_batch separately (by monkeypatching /
re-implementing its flow), then sweeps block_size and nbins on the
kernel-dominated regimes to pick defaults.

Usage (on pod):
    python scripts/tls_profile_stages.py [--regime kepler-4yr] [--nlc 4]
"""
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np


def make_lcs(regime, nlc):
    cfgs = {
        'tess-ffi':   dict(ndata=1310, cadence=30. / 60 / 24, noise=1e-3,
                           pinj=7.7, depth=0.005, pmin=0.6, pmax=13.7),
        'k2':         dict(ndata=4320, cadence=30. / 60 / 24, noise=8e-4,
                           pinj=12.4, depth=0.004, pmin=0.6, pmax=45.),
        'tess-2min':  dict(ndata=19710, cadence=2. / 60 / 24, noise=2e-3,
                           pinj=7.7, depth=0.005, pmin=0.6, pmax=13.7),
        'tess-yr':    dict(ndata=16850, cadence=30. / 60 / 24, noise=1e-3,
                           pinj=21.7, depth=0.004, pmin=0.6, pmax=175.),
        'kepler-4yr': dict(ndata=65440, cadence=30. / 60 / 24, noise=6e-4,
                           pinj=41.3, depth=0.003, pmin=0.6, pmax=500.),
    }
    c = cfgs[regime]
    lcs = []
    for i in range(nlc):
        rng = np.random.RandomState(1234 + i)
        t = np.arange(c['ndata']) * c['cadence']
        y = 1.0 + rng.randn(c['ndata']) * c['noise']
        q = 0.0763 * c['pinj'] ** (-2.0 / 3.0)
        t0 = 0.3 * c['pinj']
        rel = np.abs(((t - t0 + 0.5 * c['pinj']) % c['pinj'])
                     - 0.5 * c['pinj'])
        y[rel < 0.5 * q * c['pinj']] -= c['depth']
        lcs.append((t, y, np.full(c['ndata'], c['noise'])))
    return lcs, c


def profile(regime, nlc, block_size=None, nbins=None, refine_top_k=200,
            n_durations=15):
    import pycuda.driver as cuda
    from cuvarbase import tls, tls_grids, tls_models

    lcs, c = make_lcs(regime, nlc)

    def sync():
        cuda.Context.synchronize()

    T = {}

    t0 = time.perf_counter()
    periods = tls_grids.period_grid_ofir(
        lcs[0][0], R_star=1.0, M_star=1.0, oversampling_factor=3,
        period_min=c['pmin'], period_max=c['pmax'])
    periods32 = np.asarray(periods, dtype=np.float32)
    _, _, qv = tls_grids.duration_grid_keplerian(
        np.asarray(periods, np.float64), R_star=1.0, M_star=1.0,
        R_planet=1.0, qmin_fac=0.5, qmax_fac=2.0, n_durations=n_durations)
    qmin, qmax = (qv * 0.5).astype(np.float32), (qv * 2).astype(np.float32)
    T['grid_gen'] = time.perf_counter() - t0
    nperiods = len(periods32)

    bs = block_size or tls._TLS_FAST_DEFAULT_BLOCK
    qmin_g = float(qmin.min())
    nb = nbins or tls._auto_nbins(qmin_g, 3.0, bs)

    t0 = time.perf_counter()
    kernels = tls._get_cached_fast_kernels(bs, nb, 3.0)
    T['compile_or_cache'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    T_tab, S1_tab, S2_tab = tls_models.generate_template_tables()
    T['template'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    t_hi_c, t_lo_c, a_c, b_c, offs, lens, chi2_0, epochs, spans = \
        tls._preprocess_batch(lcs)
    T['preprocess_cpu'] = time.perf_counter() - t0

    import pycuda.gpuarray as gpuarray
    t0 = time.perf_counter()
    periods_gpu = gpuarray.to_gpu(periods32)
    qmin_gpu = gpuarray.to_gpu(qmin)
    qmax_gpu = gpuarray.to_gpu(qmax)
    T_g = gpuarray.to_gpu(T_tab)
    S1_g = gpuarray.to_gpu(S1_tab)
    S2_g = gpuarray.to_gpu(S2_tab)
    thi_g = gpuarray.to_gpu(t_hi_c)
    tlo_g = gpuarray.to_gpu(t_lo_c)
    a_g = gpuarray.to_gpu(a_c)
    b_g = gpuarray.to_gpu(b_c)
    off_g = gpuarray.to_gpu(offs.astype(np.int32))
    len_g = gpuarray.to_gpu(lens.astype(np.int32))
    out_n = nlc * nperiods
    chi2_g = gpuarray.empty(out_n, np.float32)
    t0_g = gpuarray.empty(out_n, np.float32)
    dur_g = gpuarray.empty(out_n, np.float32)
    depth_g = gpuarray.empty(out_n, np.float32)
    sync()
    T['h2d_alloc'] = time.perf_counter() - t0

    smem = tls._tls_fast_shared_size(bs, nb)
    map_g = gpuarray.to_gpu(np.arange(nperiods, dtype=np.int32))
    t0 = time.perf_counter()
    kernels['search'](
        thi_g, tlo_g, a_g, b_g, off_g, len_g,
        periods_gpu, qmin_gpu, qmax_gpu, map_g, S1_g, S2_g,
        np.int32(nperiods), np.int32(nperiods), np.int32(n_durations),
        chi2_g, t0_g, dur_g, depth_g,
        block=(bs, 1, 1), grid=(nperiods, nlc, 1), shared=smem)
    sync()
    T['coarse_kernel'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    chi2_h = chi2_g.get().reshape(nlc, nperiods)
    T['d2h_chi2'] = time.perf_counter() - t0

    K = int(min(refine_top_k, max(16, nperiods // 10), nperiods))
    t0 = time.perf_counter()
    cand = np.empty((nlc, K), dtype=np.int32)
    for j in range(nlc):
        cand[j] = np.argpartition(-chi2_h[j], K)[:K] if K < nperiods \
            else np.arange(nperiods)
    cand_g = gpuarray.to_gpu(cand.ravel())
    rchi2_g = gpuarray.empty(nlc * K, np.float32)
    rt0_g = gpuarray.empty(nlc * K, np.float32)
    rdur_g = gpuarray.empty(nlc * K, np.float32)
    rdepth_g = gpuarray.empty(nlc * K, np.float32)
    dur_ratio = float(np.median(qmax / qmin))
    dur_span = dur_ratio ** (1.0 / (2.0 * (n_durations - 1)))
    t0_hw = min(3.0, max(0.75, 1.5 / (nb * qmin_g)))
    kernels['refine'](
        thi_g, tlo_g, a_g, b_g, off_g, len_g,
        periods_gpu, cand_g, T_g,
        np.int32(nperiods), np.int32(K),
        np.float32(dur_span), np.float32(t0_hw), np.float32(33.0),
        t0_g, dur_g,
        rchi2_g, rt0_g, rdur_g, rdepth_g,
        block=(bs, 1, 1), grid=(K, nlc, 1),
        shared=tls._tls_refine_shared_size(bs))
    sync()
    T['refine'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    from cuvarbase import tls_stats
    for j in range(nlc):
        row = chi2_h[j]
        valid = row > 0
        cv = (chi2_0[j] - row[valid].astype(np.float64))
        bi = int(np.argmin(cv))
        tls_stats.compute_all_statistics(cv, periods32[valid], bi,
                                         0.01, 0.1, 10)
        tls_stats.compute_period_uncertainty(periods32[valid], cv, bi)
    T['stats_cpu'] = time.perf_counter() - t0

    total = sum(T.values())
    print("\n%s nlc=%d nperiods=%d ndata=%d bs=%d nbins=%d K=%d"
          % (regime, nlc, nperiods, c['ndata'], bs, nb, K))
    for k, v in T.items():
        print("  %-18s %8.3f s  (%4.1f%%)  %7.2f ms/LC"
              % (k, v, 100 * v / total, 1000 * v / nlc))
    print("  %-18s %8.3f s            %7.2f ms/LC"
          % ('TOTAL', total, 1000 * total / nlc))
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', default='kepler-4yr')
    ap.add_argument('--nlc', type=int, default=4)
    ap.add_argument('--block-size', type=int, default=None)
    ap.add_argument('--nbins', type=int, default=None)
    ap.add_argument('--sweep', action='store_true',
                    help='sweep block_size x nbins on this regime '
                         '(reports steady-state coarse-kernel time)')
    args = ap.parse_args()

    if args.sweep:
        for bs in (64, 128, 256):
            for nb in (None, 2048, 4096):
                try:
                    profile(args.regime, args.nlc, block_size=bs,
                            nbins=nb)
                except Exception as exc:
                    print("bs=%d nbins=%s FAILED: %r" % (bs, nb, exc))
        return

    # steady-state: run twice (first pays compile), report second
    profile(args.regime, args.nlc, block_size=args.block_size,
            nbins=args.nbins)
    profile(args.regime, args.nlc, block_size=args.block_size,
            nbins=args.nbins)


if __name__ == '__main__':
    main()
