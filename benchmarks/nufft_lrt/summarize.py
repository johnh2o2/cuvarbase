"""Render the NUFFT-LRT validation JSON as tables (markdown or rst).

Usage:
    python benchmarks/nufft_lrt/summarize.py results.json [--rst]

Prints the null calibration, the protocol, one completeness table per
configuration (plus the arm cost), the epoch recovery of the arms that
report a best epoch, and -- when the JSON holds paired configurations
(``white`` / ``white_bjd``, ``red_sys`` / ``red_sys_nzm``, harness
version 2) -- the one-to-one comparison of their per-lightcurve
statistics. ``--rst`` emits reStructuredText for ``docs/source/nufft_lrt.rst``.
"""
import argparse
import json

import numpy as np


TITLES = {'white': 'White noise',
          'white_bjd': 'White noise, absolute times (BJD-scale, '
                       't + 2457000 d)',
          'red_1x': 'Red noise, sigma_red = sigma_white',
          'red_3x': 'Red noise, sigma_red = 3 sigma_white',
          'red_sys': 'Red noise + shared systematics '
                     '(PCA basis + population prior)',
          'red_sys_nzm': 'Red noise + shared systematics, '
                         'non-zero-mean basis columns'}
ARM_ORDER = ['lrt', 'lrt_auto', 'lrt_marg', 'lrt_seq', 'lrt_flat',
             'bls', 'tls']
ARM_LABEL = {'lrt': 'LRT (explicit epoch grid)',
             'lrt_auto': 'LRT, default path (epochs=None)',
             'lrt_marg': 'LRT Detector A (marginal)',
             'lrt_seq': 'LRT sequential cotrend',
             'lrt_flat': 'LRT, flat PSD',
             'bls': 'BLS (eebls_gpu_fast)',
             'tls': 'TLS (tls_search_batch, delta-chi2)'}
PAIRS = [('white', 'white_bjd'), ('red_sys', 'red_sys_nzm')]
# within-configuration arm contrasts the docs quote (A minus B)
CONTRASTS = [('lrt', 'bls'), ('lrt_auto', 'lrt'), ('lrt', 'lrt_flat'),
             ('lrt_marg', 'lrt_seq'), ('lrt_seq', 'bls'), ('lrt', 'tls')]
N_BOOT = 2000


def _arms(cfg):
    return sorted(cfg['methods'],
                  key=lambda n: ARM_ORDER.index(n) if n in ARM_ORDER else 99)


class Md:
    def h(self, text):
        return '### %s\n' % text

    def table(self, header, rows, title=None):
        out = []
        if title:
            out.append('**%s**\n' % title)
        out.append('| ' + ' | '.join(header) + ' |')
        out.append('|---|' + '---:|' * (len(header) - 1))
        for r in rows:
            out.append('| ' + ' | '.join(r) + ' |')
        return '\n'.join(out) + '\n'


class Rst:
    def h(self, text):
        return '%s\n%s\n' % (text, '-' * len(text))

    def table(self, header, rows, title=None):
        out = ['.. list-table::%s' % ((' ' + title) if title else ''),
               '   :header-rows: 1', '']
        for r in [header] + rows:
            out.append('   * - ' + r[0])
            out.extend('     - ' + c for c in r[1:])
        return '\n'.join(out) + '\n'


def _period_hit(p_found, p_true, tol=0.01):
    if p_found is None or not np.isfinite(p_found):
        return False
    return any(abs(p_found - x) / x < tol
               for x in (p_true, 2 * p_true, 0.5 * p_true))


def _detections(m, depth, p_true):
    """(stat, period_ok) arrays of one arm's injections at one depth."""
    inj = m['injections'][depth]
    stat = np.asarray(inj['stat'], float)
    ok = np.array([_period_hit(p, p_true) for p in inj['p_found']])
    return stat, ok


def cell_uncertainty(m, depth, p_true, rng):
    """1-sigma uncertainty of a completeness cell: the null-threshold
    sampling error (bootstrap of the null maxima, N_BOOT resamples,
    completeness re-evaluated at each resampled 95th percentile) and
    the binomial error (half-width of the z = 1 Wilson interval),
    added in quadrature. Returns (completeness, sigma)."""
    nulls = np.asarray(m['null_stats'], float)
    stat, ok = _detections(m, depth, p_true)
    n = len(stat)
    p = float(np.mean((stat > m['null_max_p95']) & ok))
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        thr = np.percentile(rng.choice(nulls, len(nulls), replace=True), 95)
        boot[b] = np.mean((stat > thr) & ok)
    s_thr = float(boot.std())
    z = 1.0
    s_bin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, float(np.hypot(s_thr, s_bin))


def completeness_table(fmt, cfg, rng):
    depths = sorted({d for m in cfg['methods'].values()
                     for d in m['completeness']}, key=float)
    p_true = cfg['config']['p_true']
    header = ['arm', 'null p95'] + ['depth %s' % d for d in depths] \
        + ['ms/search']
    rows = []
    for name in _arms(cfg):
        m = cfg['methods'][name]
        row = [ARM_LABEL.get(name, name), '%.3f' % m['null_max_p95']]
        for d in depths:
            if d not in m['completeness']:
                row.append('--')
                continue
            if 'injections' in m and 'null_stats' in m:
                p, sig = cell_uncertainty(m, d, p_true, rng)
                row.append('%.0f +- %.0f%%' % (100 * p, 100 * sig))
            else:
                row.append('%.0f%%' % (100 * m['completeness'][d]))
        sps = m.get('seconds_per_search')
        if sps is None:
            row.append('--')
        else:
            ms = 1e3 * sps
            row.append('%.0f' % ms if ms >= 100 else '%.3g' % ms)
        rows.append(row)
    return fmt.table(header, rows)


def contrast_table(fmt, cfg):
    """Paired (same-lightcurve) completeness differences A - B within a
    configuration: b = detected by A only, c = by B only, difference
    (b - c) / n with sigma sqrt(b + c) / n (McNemar), each arm at its own
    fixed null-p95 threshold."""
    depths = sorted({d for m in cfg['methods'].values()
                     for d in m['completeness']}, key=float)
    rows = []
    for a, b_ in CONTRASTS:
        if a not in cfg['methods'] or b_ not in cfg['methods']:
            continue
        ma, mb = cfg['methods'][a], cfg['methods'][b_]
        if 'injections' not in ma or 'injections' not in mb:
            continue
        row = ['%s - %s' % (a, b_)]
        for d in depths:
            if d not in ma['injections'] or d not in mb['injections']:
                row.append('--')
                continue
            da = np.asarray(ma['injections'][d]['detected'], bool)
            db = np.asarray(mb['injections'][d]['detected'], bool)
            n = len(da)
            bb, cc = int(np.sum(da & ~db)), int(np.sum(db & ~da))
            row.append('%+.0f +- %.0f%%' % (100.0 * (bb - cc) / n,
                                            100.0 * np.sqrt(bb + cc) / n))
        rows.append(row)
    if not rows:
        return ''
    return fmt.table(['A - B', *['depth %s' % d for d in depths]], rows)


def epoch_table(fmt, cfg):
    rows = []
    for name in _arms(cfg):
        m = cfg['methods'][name]
        er = m.get('epoch_recovery') or {}
        if not er:
            continue
        for d in sorted(er, key=float):
            e = er[d]
            rows.append([ARM_LABEL.get(name, name), d,
                         '%d' % e['n_detected'],
                         '%.0f%%' % (100 * e['frac_within_half_duration']),
                         '%.3f' % e['median_abs_error_d'],
                         '%.3f' % e['max_abs_error_d']])
    if not rows:
        return ''
    return fmt.table(['arm', 'depth', 'detections',
                      'same transit (within dur/2)',
                      'median abs. error (d)', 'max abs. error (d)'], rows)


def paired_stats(a, b):
    """Per-arm one-to-one comparison of two configurations that saw the
    same lightcurves: max relative difference of the per-search
    statistic (null + injections), how many injections found a
    different best period, and how many detection decisions differ
    (each configuration at its own null-p95 threshold)."""
    out = {}
    for name in a['methods']:
        if name not in b['methods']:
            continue
        ma, mb = a['methods'][name], b['methods'][name]
        sa = np.asarray(ma['null_stats'], float)
        sb = np.asarray(mb['null_stats'], float)
        pdiff, ddiff, n_inj = 0, 0, 0
        for d in ma['injections']:
            ia, ib = ma['injections'][d], mb['injections'][d]
            sa = np.concatenate([sa, np.asarray(ia['stat'], float)])
            sb = np.concatenate([sb, np.asarray(ib['stat'], float)])
            pdiff += sum(pa != pb for pa, pb in zip(ia['p_found'],
                                                     ib['p_found']))
            ddiff += sum(da != db for da, db in zip(ia['detected'],
                                                     ib['detected']))
            n_inj += len(ia['stat'])
        scale = np.maximum(np.abs(sa), np.abs(sb))
        scale[scale == 0] = 1.0
        rel = np.abs(sa - sb) / scale
        out[name] = dict(n=int(len(sa)), max_rel=float(rel.max()),
                         median_rel=float(np.median(rel)),
                         n_inj=n_inj, period_diff=int(pdiff),
                         decision_diff=int(ddiff),
                         p95_a=ma['null_max_p95'], p95_b=mb['null_max_p95'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--rst', action='store_true')
    args = ap.parse_args()
    fmt = Rst() if args.rst else Md()
    with open(args.path) as f:
        r = json.load(f)
    meta = r['meta']
    cfgs = {c['name']: c for c in r['configs']}

    label = []
    if meta.get('gpu'):
        label.append('GPU: %s' % ', '.join(np.atleast_1d(meta['gpu'])))
    if meta.get('git_sha'):
        label.append('commit: %s' % ', '.join(
            s[:9] for s in np.atleast_1d(meta['git_sha'])))
    if meta.get('date'):
        label.append('date: %s' % ', '.join(np.atleast_1d(meta['date'])))
    if label:
        print('; '.join(label) + '\n')

    cal = r.get('snr_calibration')
    if cal:
        print(fmt.h('LRT statistic null calibration (white noise, fixed '
                    'template)'))
        print('mean = %.3f, std = %.3f over %d realizations (the pre-fix '
              'Sep-2026 campaign, sigma = 2 NFFT: 0.007, 1.812). '
              'Calibration constant of this configuration, not a '
              'pass/fail check: the statistic is a whitened correlation, '
              'not N(0,1), because the NFFT modes of irregular sampling '
              'are not orthogonal; its null std depends on the sampling, '
              'nf and the PSD estimator. This is why the thresholds below '
              'are empirical null percentiles.\n'
              % (cal['mean'], cal['std'], cal['n']))

    print(fmt.h('Protocol'))
    print('%d-point ground-like irregular sampling over %.0f d; trial '
          'grid %d periods (injected P = %.2f d on-grid), box duration '
          '%.2f d; thresholds = 95th percentile of %d null search maxima; '
          'completeness over %d injections per depth, period hit within '
          '1%% (incl. 2:1 aliases). Depths are fractions of the flux; '
          'sigma_white = %g.\n'
          % (meta['ndata'], meta['baseline'], meta['n_periods'],
             meta['p_true'], meta['dur_true'], meta['n_null'],
             meta['n_inj'], meta['sigma_white']))

    rng = np.random.RandomState(0)
    print('Completeness cells are "p +- sigma" with sigma the quadrature '
          'sum of the null-threshold sampling error (bootstrap of the '
          'null maxima) and the binomial (Wilson, z = 1) error; the '
          'paired-difference rows use the same lightcurves for both '
          'arms (McNemar sigma = sqrt(b + c) / n, thresholds fixed).\n')
    for cfg in r['configs']:
        print(fmt.h(TITLES.get(cfg['name'], cfg['name'])))
        print(completeness_table(fmt, cfg, rng))
        ct = contrast_table(fmt, cfg)
        if ct:
            print('Paired completeness differences (A - B, same '
                  'lightcurves):\n')
            print(ct)
        et = epoch_table(fmt, cfg)
        if et:
            print('Epoch recovery among detections (arms that return a '
                  'best epoch; "same transit" = within half the injected '
                  'duration, which any correct-period detection meets; '
                  'the errors show the grid resolution):\n')
            print(et)
        print('(compute: %.0f s)\n' % cfg.get('wall_s', float('nan')))

    pairs = [(a, b) for a, b in PAIRS if a in cfgs and b in cfgs]
    if pairs:
        print(fmt.h('Paired configurations (same lightcurves)'))
        print('Each pair saw identical noise and injections (shared '
              'sub-seed) and differs only in the time origin '
              '(white / white_bjd: + 2457000 d, an integer, so every '
              'method\'s floor(min t)-anchored grid keeps its phase) or '
              'in the basis column offsets (red_sys / red_sys_nzm). '
              'Differences beyond float32 rounding would indicate a '
              'time-scale or centring defect.\n')
        for a, b in pairs:
            ps = paired_stats(cfgs[a], cfgs[b])
            if not ps:
                continue
            rows = []
            for name in sorted(ps, key=lambda n: ARM_ORDER.index(n)
                               if n in ARM_ORDER else 99):
                p = ps[name]
                rows.append([ARM_LABEL.get(name, name),
                             '%d' % p['n'],
                             '%.1e' % p['max_rel'],
                             '%.1e' % p['median_rel'],
                             '%d / %d' % (p['period_diff'], p['n_inj']),
                             '%d / %d' % (p['decision_diff'], p['n_inj']),
                             '%.3f / %.3f' % (p['p95_a'], p['p95_b'])])
            print(fmt.table(['arm', 'searches', 'max rel. diff',
                             'median rel. diff', 'best period differs',
                             'detection differs', 'null p95 (%s / %s)'
                             % (a, b)], rows,
                            title='%s vs %s' % (a, b)))


if __name__ == '__main__':
    main()
