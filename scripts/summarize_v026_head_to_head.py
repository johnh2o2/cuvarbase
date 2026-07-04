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


def pooled_warm(data, key_base, label):
    """Pool timed samples across benchmark rounds (key_base, key_base_r2,
    ...) and return (pooled_median, iqr, n, n_rounds)."""
    times = []
    n_rounds = 0
    for suffix in ('', '_r2', '_r3', '_r4'):
        r = get_row(data, key_base + suffix, label)
        if r is not None:
            times.extend(r['times_s'])
            n_rounds += 1
    if not times:
        return None
    return (float(np.median(times)),
            [float(np.percentile(times, 25)),
             float(np.percentile(times, 75))],
            len(times), n_rounds)


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

    # ---- warm (pooled across all rounds) ----
    print('## Standard BLS, warm / steady state (pooled across rounds; '
          '7 timed / 2 warmups per round)\n')
    print('| config | 0.2.6 warm (functions= precompiled) | v1.0 noverlap=1 '
          '(apples-to-apples) | ratio | v1.0 noverlap=2 (default, '
          'correctness) | n samples (026/v10) |')
    print('|---|---|---|---|---|---|')
    for cfg in ['canonical', 'small', 'tess']:
        p026 = pooled_warm(D, 'v026_warm_%s' % cfg,
                           'v026_fast_warm_precompiled')
        p10a = pooled_warm(D, 'v10_warm_%s' % cfg, 'v10_fast_noverlap1')
        p10b = pooled_warm(D, 'v10_warm_%s' % cfg,
                           'v10_fast_noverlap2_default')
        if p026 is None or p10a is None:
            continue
        ratio = p026[0] / p10a[0]
        print('| %s | %s [%s, %s] | %s [%s, %s] | **%.2fx** | %s | %d/%d |'
              % (cfg, fmt_ms(p026[0]), fmt_ms(p026[1][0]),
                 fmt_ms(p026[1][1]), fmt_ms(p10a[0]), fmt_ms(p10a[1][0]),
                 fmt_ms(p10a[1][1]), ratio,
                 fmt_ms(p10b[0]) if p10b else 'n/a', p026[2], p10a[2]))
    print()

    # ---- decomposition ----
    have_decomp = any(k.startswith('decomp_') for k in D)
    if have_decomp:
        print('## Warm-call decomposition (TESS config, 15 reps, '
              'interleaved run order)\n')
        print('| variant | v1.0 (run 1) | 0.2.6 | v1.0 (run 2, drift '
              'check) |')
        print('|---|---|---|---|')
        names = {'A_full': 'A: product call (per-call compile path)',
                 'B_precompiled': 'B: functions= precompiled',
                 'C_mem_reuse': 'C: B + memory reused',
                 'D_kernel_only': 'D: C without H2D/D2H (kernel only)'}
        for key, label in names.items():
            vals = []
            for f in ['decomp_v10_tess', 'decomp_v026_tess',
                      'decomp_v10_tess2']:
                d = D.get(f)
                vals.append(fmt_ms(d['results'][key]['median_s'])
                            if d else 'n/a')
            print('| %s | %s | %s | %s |' % (label, vals[0], vals[1],
                                             vals[2]))
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
