#!/usr/bin/env python3
"""Aggregate raw JSON from bench_v026_head_to_head.py runs into Markdown
tables (stdout). Pure-CPU post-processing; no pycuda required.

Usage: python3 scripts/summarize_v026_head_to_head.py <raw_json_dir>
"""
import glob
import json
import os
import sys

import numpy as np


def load_all(raw_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, '*.json'))):
        with open(path) as f:
            out[os.path.basename(path)[:-5]] = json.load(f)
    return out


def fmt_ms(s):
    if s is None:
        return 'n/a'
    return '%.1f ms' % (1e3 * s) if s < 1 else '%.3f s' % s


def get_row(data, key, label):
    d = data.get(key)
    if d is None:
        return None
    for r in d['rows']:
        if r['label'] == label:
            return r
    return None


def main():
    raw_dir = sys.argv[1]
    D = load_all(raw_dir)

    # ---- environments ----
    print('## Environments\n')
    envs = {}
    for k, d in D.items():
        env = d.get('env')
        if env:
            envs[env['cuvarbase']] = env
    for v, env in sorted(envs.items()):
        print('- **cuvarbase %s**: python %s, numpy %s, pycuda %s, %s, '
              'driver %s, %s' % (v, env['python'], env['numpy'],
                                 env['pycuda'], env.get('nvcc', '?'),
                                 env['cuda_driver_version'], env['gpu']))
    print()

    # ---- warm ----
    print('## Standard BLS, warm / steady state (median of 7, 2 warmups)\n')
    print('| config | 0.2.6 warm (functions= precompiled) | v1.0 noverlap=1 '
          '(apples-to-apples) | ratio | v1.0 noverlap=2 (default, '
          'correctness) | v1.0 optimized kernel (nov=1) |')
    print('|---|---|---|---|---|---|')
    for cfg in ['canonical', 'small', 'tess']:
        r026 = get_row(D, 'v026_warm_%s' % cfg, 'v026_fast_warm_precompiled')
        r10a = get_row(D, 'v10_warm_%s' % cfg, 'v10_fast_noverlap1')
        r10b = get_row(D, 'v10_warm_%s' % cfg, 'v10_fast_noverlap2_default')
        r10o = get_row(D, 'v10_warm_%s' % cfg, 'v10_fast_optimized_noverlap1')
        if r026 is None or r10a is None:
            continue
        ratio = r026['median_s'] / r10a['median_s']
        print('| %s | %s | %s | **%.2fx** | %s | %s |'
              % (cfg, fmt_ms(r026['median_s']), fmt_ms(r10a['median_s']),
                 ratio, fmt_ms(r10b['median_s']) if r10b else 'n/a',
                 fmt_ms(r10o['median_s']) if r10o else 'n/a'))
    print()

    # ---- cold ----
    print('## Standard BLS, cold / out-of-the-box (fresh process, compiler '
          'caches cleared)\n')
    print('| config | version | import+ctx | first call (incl. compile) | '
          'second call |')
    print('|---|---|---|---|---|')
    for cfg in ['canonical', 'small', 'tess']:
        for ver, key, label in [
                ('0.2.6', 'v026_cold_%s' % cfg, 'v026_fast_naive'),
                ('v1.0', 'v10_cold_%s' % cfg, 'v10_fast_noverlap1')]:
            r = get_row(D, key, label)
            if r is None:
                continue
            print('| %s | %s | %s | %s | %s |'
                  % (cfg, ver, fmt_ms(r['import_and_context_s']),
                     fmt_ms(r['first_call_s']), fmt_ms(r['second_call_s'])))
    print()

    # ---- loop ----
    print('## Naive per-lightcurve loop (product defaults, no functions= '
          'handle; fresh process)\n')
    print('| version | N LCs | first call | steady per-call median | loop '
          'total | extrapolated 100-LC (first + 99 x steady) | effective '
          'per-LC (100-LC) |')
    print('|---|---|---|---|---|---|---|')
    for ver, key, label in [
            ('0.2.6', 'v026_loop_canonical', 'v026_fast_naive_loop'),
            ('v1.0 (nov=1)', 'v10_loop_canonical', 'v10_fast_noverlap1_loop')]:
        r = get_row(D, key, label)
        if r is None:
            continue
        ext = r['extrapolated_100lc_s']
        print('| %s | %d | %s | %s | %s | %s | %s |'
              % (ver, r['nlc'], fmt_ms(r['first_call_s']),
                 fmt_ms(r['steady_per_call_median_s']),
                 fmt_ms(r['loop_total_s']), fmt_ms(ext), fmt_ms(ext / 100)))
    print()

    # ---- correctness / BJD ----
    print('## Correctness + BJD demo (injected transit, noverlap=1 both '
          'versions)\n')
    c026 = D.get('v026_correctness')
    c10 = D.get('v10_correctness')
    if c026 and c10:
        inj = c026['injection']
        print('Injected: f=%.6f /d (P=%.4f d), q=%.3f, depth=%.4f\n'
              % (inj['freq'], 1.0 / inj['freq'], inj['q'], inj['depth']))
        print('| version | timescale | peak freq (/d) | peak power | power @ '
              'injected freq | recovered? |')
        print('|---|---|---|---|---|---|')
        rows = {}
        for tag, d in [('0.2.6', c026), ('v1.0', c10)]:
            for r in d['rows']:
                rows[(tag, r['timescale'])] = r
                print('| %s | %s | %.6f | %.6g | %.6g | %s |'
                      % (tag, r['timescale'], r['peak_freq'],
                         r['peak_power'], r['power_at_injected_freq'],
                         'YES' if r['recovered'] else '**NO**'))
        print()
        # correlations
        p026 = np.array(rows[('0.2.6', 'near_zero')]['periodogram'])
        p10 = np.array(rows[('v1.0', 'near_zero')]['periodogram'])
        p026b = np.array(rows[('0.2.6', 'bjd')]['periodogram'])
        p10b = np.array(rows[('v1.0', 'bjd')]['periodogram'])
        print('Periodogram correlations:')
        print('- v1.0 vs 0.2.6, near-zero t (parity check): r = %.6f'
              % np.corrcoef(p026, p10)[0, 1])
        print('- v1.0: BJD vs near-zero (epoch fix works): r = %.6f'
              % np.corrcoef(p10b, p10)[0, 1])
        print('- 0.2.6: BJD vs near-zero (float32 fold degradation): '
              'r = %.6f' % np.corrcoef(p026b, p026)[0, 1])
        print()

    # ---- LS ----
    print('## Lomb-Scargle (process reused, warm; median of 7)\n')
    print('| config | 0.2.6 | v1.0 | ratio | peak freq agreement |')
    print('|---|---|---|---|---|')
    for cfg, k026, k10 in [
            ('ndata=10000, nf=5000', 'v026_ls_canonical', 'v10_ls_canonical'),
            ('ndata=3000, nf=100000', 'v026_ls_large', 'v10_ls_large')]:
        r026 = get_row(D, k026, 'ls_v026')
        r10 = get_row(D, k10, 'ls_v10')
        if r026 is None or r10 is None:
            continue
        agree = ('%.5f vs %.5f /d' % (r026['peak_freq'], r10['peak_freq']))
        print('| %s | %s | %s | %.2fx | %s |'
              % (cfg, fmt_ms(r026['median_s']), fmt_ms(r10['median_s']),
                 r026['median_s'] / r10['median_s'], agree))
    print()


if __name__ == '__main__':
    main()
