"""Matched-fidelity throughput: cuvarbase fast TLS at the default coarse
epoch grid (t0_oversample=3) vs a reference-matched grid (t0_oversample=33)
on the compute-heavy regimes. cuvarbase only, no reference (reference is
>15 min/LC on Kepler). Gives the matched-fidelity ms/LC for the
apples-to-apples GTLS comparison.
"""
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np

REGIMES = {
    'tess-yr':    dict(ndata=16850, cadence=30. / 60 / 24, noise=1e-3,
                       pinj=21.7, depth=0.004, pmin=0.6, pmax=175.),
    'kepler-4yr': dict(ndata=65440, cadence=30. / 60 / 24, noise=6e-4,
                       pinj=41.3, depth=0.003, pmin=0.6, pmax=500.),
}


def make_lc(c, seed):
    rng = np.random.RandomState(seed)
    t = np.arange(c['ndata']) * c['cadence']
    y = 1.0 + rng.randn(c['ndata']) * c['noise']
    q = 0.0763 * c['pinj'] ** (-2.0 / 3.0)
    t0 = 0.3 * c['pinj']
    rel = np.abs(((t - t0 + 0.5 * c['pinj']) % c['pinj']) - 0.5 * c['pinj'])
    y[rel < 0.5 * q * c['pinj']] -= c['depth']
    return t, y, np.full(c['ndata'], c['noise'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nlc', type=int, default=4)
    args = ap.parse_args()

    import pycuda.driver as cuda
    from cuvarbase.base import ensure_context
    from cuvarbase import tls_grids
    from cuvarbase.tls import tls_search_batch
    ensure_context()

    def timed(lcs, periods, os_):
        cuda.Context.synchronize()
        t0 = time.perf_counter()
        res = tls_search_batch(lcs, R_star=1.0, M_star=1.0, periods=periods,
                               t0_oversample=os_, refine_top_k=50,
                               return_arrays=False)
        cuda.Context.synchronize()
        ms = (time.perf_counter() - t0) / len(lcs) * 1000
        rec = sum(abs(r['period'] - c['pinj']) / c['pinj'] < 0.01
                  for r in res if 'error' not in r)
        return ms, rec

    print("\n%-12s %8s %10s %10s %8s   %s"
          % ("regime", "nperiods", "t0os=3 ms", "t0os=33 ms", "factor",
             "recov (3/33)"))
    print("-" * 74)
    for name, c in REGIMES.items():
        globals()['c'] = c
        lcs = [make_lc(c, 4000 + i) for i in range(args.nlc)]
        periods = tls_grids.period_grid_ofir(
            lcs[0][0], R_star=1.0, M_star=1.0, oversampling_factor=3,
            period_min=c['pmin'], period_max=c['pmax'])
        # warmups (compile both band sets)
        timed(lcs[:1], periods, 3.0)
        timed(lcs[:1], periods, 33.0)
        ms3, r3 = timed(lcs, periods, 3.0)
        ms33, r33 = timed(lcs, periods, 33.0)
        print("%-12s %8d %10.1f %10.1f %7.1fx   %d/%d, %d/%d"
              % (name, len(periods), ms3, ms33, ms33 / ms3,
                 r3, args.nlc, r33, args.nlc))


if __name__ == '__main__':
    main()
