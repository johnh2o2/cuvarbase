"""Experiment A: cuvarbase TLS vs reference transitleastsquares on identical
batman-injected transits, identical period grid (the reference's own)."""
import numpy as np, time, sys, json, warnings
sys.path.insert(0, '/workspace/scratch')
from audit_common import *
from cuvarbase.tls import tls_search_gpu, tls_search_batch
from cuvarbase import tls_grids
import transitleastsquares as tlsref

configs = [
    # (P, rp, b, baseline, sigma, label)
    (3.0,  0.10, 0.0, 60.0, 1e-3, 'P3_rp0.10_b0'),
    (3.0,  0.03, 0.0, 60.0, 5e-4, 'P3_rp0.03_b0'),
    (7.3,  0.05, 0.0, 60.0, 1e-3, 'P7.3_rp0.05_b0'),
    (12.0, 0.04, 0.0, 90.0, 1e-3, 'P12_rp0.04_b0'),
    (12.0, 0.04, 0.85, 90.0, 1e-3, 'P12_rp0.04_b0.85'),
    (1.3,  0.02, 0.0, 30.0, 3e-4, 'P1.3_rp0.02_b0'),
    (25.0, 0.06, 0.0, 120.0, 1e-3, 'P25_rp0.06_b0'),
]
out = []
for (P, rp, b, base, sig, label) in configs:
    t, y, dy, info = make_lc(P, rp, 0.37 * P, baseline=base, sigma=sig, b=b, seed=hash(label) % 1000)
    ref, dt_ref = run_ref(t, y, dy)
    periods = np.sort(np.asarray(ref.periods, dtype=np.float64))
    order = np.argsort(ref.periods)
    ref_chi2 = np.asarray(ref.chi2)[order]
    ref_SR = np.asarray(ref.SR)[order]
    ref_power = np.asarray(ref.power)[order]
    res = {}
    # 1) fast default (standard fixed q window, matches legacy 'standard' kernel)
    t1 = time.time(); r = tls_search_gpu(t, y, dy, periods=periods); res['fast_std'] = (r, time.time() - t1)
    # 2) fast, no refinement
    t1 = time.time(); r = tls_search_gpu(t, y, dy, periods=periods, refine_top_k=0); res['fast_std_norefine'] = (r, time.time() - t1)
    # 3) fast, t0_oversample=33 (reference-like epoch grid)
    t1 = time.time(); r = tls_search_gpu(t, y, dy, periods=periods, t0_oversample=33.0); res['fast_std_t033'] = (r, time.time() - t1)
    # 4) legacy exact per-point kernel (t0_oversample=3), only if ndata fits
    if len(t) <= 3400:
        t1 = time.time(); r = tls_search_gpu(t, y, dy, periods=periods, use_fast=False); res['legacy_std'] = (r, time.time() - t1)
        t1 = time.time(); r = tls_search_gpu(t, y, dy, periods=periods, use_fast=False, t0_oversample=33.0); res['legacy_std_t033'] = (r, time.time() - t1)
    # 5) batch API with default Keplerian window (survey default)
    t1 = time.time(); rb = tls_search_batch([(t, y, dy)], periods=periods, return_arrays=True)[0]; res['batch_kep'] = (rb, time.time() - t1)
    # 6) Keplerian tls_transit-like via tls_search_gpu with qmin/qmax
    q = tls_grids.q_transit(periods)
    t1 = time.time(); r = tls_search_gpu(t, y, dy, periods=periods, qmin=0.5*q, qmax=2*q); res['fast_kep'] = (r, time.time() - t1)

    row = dict(label=label, P=P, rp=rp, b=b, ndata=len(t), nperiods=len(periods),
               true_depth=info['true_depth'], true_t14=info['t14'],
               ref=dict(period=float(ref.period), depth=float(1 - ref.depth), duration=float(ref.duration),
                        SDE=float(ref.SDE), SDE_raw=float(ref.SDE_raw), SNR=float(ref.snr), FAP=float(ref.FAP) if ref.FAP is not None else None,
                        T0=float(ref.T0), time=dt_ref))
    for k, (r, dt) in res.items():
        chi2 = np.asarray(r['chi2'], float)
        row[k] = dict(period=float(r['period']), depth=float(r['depth']), duration=float(r['duration']),
                      SDE=float(r['SDE']), SDE_raw=float(r['SDE_raw']), SNR=float(r['SNR']), FAP=float(r['FAP']),
                      T0=float(r['T0']), time=dt, n_failed=int(r['n_failed_periods']),
                      corr_chi2=float(corr(chi2, ref_chi2)), corr_SR=float(corr(r['SR'], ref_SR)),
                      corr_power=float(corr(r['power'], ref_power)),
                      chi2_min=float(r['chi2_min']), chi2_spec_min=float(np.nanmin(chi2)))
    out.append(row)
    print(json.dumps(row, indent=None)); sys.stdout.flush()
json.dump(out, open('/workspace/scratch/exp_a.json', 'w'), indent=1)
print("\n==== SUMMARY ====")
print("%-18s %7s %7s | %-8s %-8s %-8s %-8s %-8s | %s" % ('label', 'trueDep', 'trueT14', 'ref', 'fast', 'fast33', 'legacy', 'batchK', 'corr_chi2(fast,fast33,batchK)'))
for row in out:
    def g(k, f): return ('%8.4g' % row[k][f]) if k in row else '   n/a  '
    print("%-18s %7.4f %7.4f | SDE  %s %s %s %s %s | %s %s %s" % (row['label'], row['true_depth'], row['true_t14'], '%8.3f' % row['ref']['SDE'], g('fast_std','SDE'), g('fast_std_t033','SDE'), g('legacy_std','SDE'), g('batch_kep','SDE'), g('fast_std','corr_chi2'), g('fast_std_t033','corr_chi2'), g('batch_kep','corr_chi2')))
    print("%-18s %7s %7s | dep  %s %s %s %s %s" % ('', '', '', '%8.4g' % row['ref']['depth'], g('fast_std','depth'), g('fast_std_t033','depth'), g('legacy_std','depth'), g('batch_kep','depth')))
    print("%-18s %7s %7s | dur  %s %s %s %s %s" % ('', '', '', '%8.4g' % row['ref']['duration'], g('fast_std','duration'), g('fast_std_t033','duration'), g('legacy_std','duration'), g('batch_kep','duration')))
    print("%-18s %7s %7s | per  %s %s %s %s %s" % ('', '', '', '%8.5g' % row['ref']['period'], g('fast_std','period'), g('fast_std_t033','period'), g('legacy_std','period'), g('batch_kep','period')))
    print("%-18s %7s %7s | SNR  %s %s %s %s %s" % ('', '', '', '%8.4g' % row['ref']['SNR'], g('fast_std','SNR'), g('fast_std_t033','SNR'), g('legacy_std','SNR'), g('batch_kep','SNR')))
    print("%-18s %7s %7s | FAP  %s %s %s %s %s" % ('', '', '', '%8.3g' % (row['ref']['FAP'] if row['ref']['FAP'] is not None else -1), g('fast_std','FAP'), g('fast_std_t033','FAP'), g('legacy_std','FAP'), g('batch_kep','FAP')))
    print("%-18s %7s %7s | T0   %s %s %s %s %s  (true t0=%.4f)" % ('', '', '', '%8.4f' % row['ref']['T0'], g('fast_std','T0'), g('fast_std_t033','T0'), g('legacy_std','T0'), g('batch_kep','T0'), 0.37*row['P']))
    print("%-18s %7s %7s | time %s %s %s %s %s" % ('', '', '', '%8.2f' % row['ref']['time'], g('fast_std','time'), g('fast_std_t033','time'), g('legacy_std','time'), g('batch_kep','time')))
