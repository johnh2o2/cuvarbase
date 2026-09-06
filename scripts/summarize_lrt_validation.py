"""Render the NUFFT-LRT validation JSON as markdown tables.

Usage: python scripts/summarize_lrt_validation.py results.json
"""
import json
import sys


def main(path):
    with open(path) as f:
        r = json.load(f)

    cal = r['snr_calibration']
    print('### LRT statistic null calibration (white noise, fixed '
          'template)\n')
    print('mean = %.3f, std = %.3f over %d realizations. Calibration '
          'constant of this configuration, not a pass/fail check: the '
          'statistic is a whitened correlation, not N(0,1) -- its null '
          'std is expected to be well above 1 (~1.8-2.7 for the '
          'harness\'s ground sampling at nf = 2n) because the NFFT '
          'modes of irregular sampling are not orthogonal. This is why '
          'the thresholds below are empirical null percentiles.\n'
          % (cal['mean'], cal['std'], cal['n']))

    meta = r['meta']
    print('Protocol: %d-point ground-like irregular sampling over %.0f d; '
          'trial grid %d periods (injected P=%.2f d on-grid), box '
          'duration %.2f d; thresholds = 95th percentile of %d null '
          'search maxima; completeness over %d injections per depth, '
          'period hit within 1%% (incl. 2:1 aliases).\n'
          % (meta['ndata'], meta['baseline'], meta['n_periods'],
             meta['p_true'], meta['dur_true'], meta['n_null'],
             meta['n_inj']))

    for cfg in r['configs']:
        c = cfg['config']
        red = c['sigma_red'] / c['sigma_white']
        title = {'white': 'White noise',
                 'red_1x': 'Red noise, sigma_red = sigma_white',
                 'red_3x': 'Red noise, sigma_red = 3 sigma_white',
                 'red_sys': 'Red noise + shared systematics '
                            '(PCA basis + population prior)'}\
            .get(cfg['name'], cfg['name'])
        print('### %s\n' % title)
        depths = sorted({d for m in cfg['methods'].values()
                         for d in m['completeness']}, key=float)
        header = '| method | null p95 |' + ''.join(
            ' depth %s |' % d for d in depths)
        print(header)
        print('|---|---:|' + '---:|' * len(depths))
        order = ['lrt', 'lrt_marg', 'lrt_seq', 'lrt_flat', 'bls', 'tls']
        for name in sorted(cfg['methods'],
                           key=lambda n: order.index(n)
                           if n in order else 99):
            if name == 'ls':
                continue   # arm dropped from the analysis (Jul 11)
            m = cfg['methods'][name]
            row = '| %s | %.3f |' % (name, m['null_max_p95'])
            for d in depths:
                comp = m['completeness'].get(d)
                row += (' %.0f%% |' % (100 * comp)
                        if comp is not None else ' — |')
            print(row)
        print('\n(wall: %.0f s)\n' % cfg.get('wall_s', float('nan')))


if __name__ == '__main__':
    main(sys.argv[1])
