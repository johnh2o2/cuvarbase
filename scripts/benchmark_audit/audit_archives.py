#!/usr/bin/env python3
"""Recompute historical claims and quantify the TLS grid limits, on CPU."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'analysis/benchmark-audit-20260906'
RAW = ROOT / 'benchmarks/results'
sys.path.insert(0, str(ROOT/'scripts/gtls_benchmark'))
import bench_core
from gtls_apples_bench import gtls_dur_window


def read(path):
    return json.loads(path.read_text())


def csv_out(name, rows):
    with (OUT/name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tls = RAW/'gtls_comparison_jul2026'
    cuv = read(tls/'results_cuv.json')['results']
    gtls = read(tls/'results_gtls.json')['results']
    gtls.update(read(tls/'results_gtls_skip8_big.json')['results'])
    rows = []
    for baseline, g in gtls.items():
        c = cuv[baseline]['methods']['cuv_tls_matched']
        g = g['methods']['gtls_skip8']
        periods = bench_core.shared_period_grid(np.arange(int(baseline)*48)/48)
        qmin, qmax = gtls_dur_window(periods)
        qs = np.exp(np.log(qmin)[:, None] + np.linspace(0, 1, 38)[None, :]
                    * np.log(qmax/qmin)[:, None])
        rows.append(dict(
            baseline_days=int(baseline), ndata=cuv[baseline]['meta']['ndata'],
            nperiods=c['n_periods'], cuvarbase_seconds=c['time_s'],
            gtls_seconds=g['time_s'], speedup=g['time_s']/c['time_s'],
            cuvarbase_repeats=len(c['times_s']), gtls_repeats=len(g['times_s']),
            historical_sde_relative_difference=c['sde_identical']/g['sde_identical']-1,
            input_noise_ppm=cuv[baseline]['meta']['noise']*1e6,
            fraction_periods_narrow_edge_underresolved=float(np.mean(8/qmin>8192)),
            max_bin_smear=float(np.max(8/qmin/8192)),
            fraction_periods_narrow_edge_epoch_capped=float(np.mean(8/qmin>20000)),
            fraction_duration_cells_epoch_capped=float(np.mean(8/qs>20000)),
            cuvarbase_source='benchmarks/results/gtls_comparison_jul2026/results_cuv.json',
            gtls_source='benchmarks/results/gtls_comparison_jul2026/' +
                        ('results_gtls.json' if int(baseline)<=1000 else 'results_gtls_skip8_big.json')))
    csv_out('historical_tls.csv', rows)

    rows = []
    for path in sorted((RAW/'by_gpu').glob('*.json')):
        d = read(path)
        r = next(x for x in d['results'] if x['algorithm']=='bls_standard')
        g, c = r['gpu']['cuvarbase_v1'], r['cpu']['astropy']
        rows.append(dict(gpu=d['system']['gpu_name'],
                         cuvarbase_ms=g['time_per_lc']*1000,
                         astropy_ms=c['time_per_lc']*1000,
                         speedup=c['total_time']/g['total_time'],
                         comparable_duration_grids=False,
                         source=str(path.relative_to(ROOT))))
    csv_out('historical_bls_cpu.csv', rows)

    rows=[]
    a=read(RAW/'bls_survey_speed_jul2026/raw/bench_baseline.json')['surveys']
    b=read(RAW/'bls_survey_speed_jul2026/raw/bench_base_envfix.json')['surveys']
    c=read(RAW/'bls_survey_speed_jul2026/raw/bench_opt4_chunk.json')['surveys']
    for name in a:
        def best(row):
            return min(v['per_lc_s'] for v in row['variants'].values() if 'per_lc_s' in v)
        rows.append(dict(survey=name, original_s=best(a[name]),
                         thread_pinned_s=best(b[name]), final_s=best(c[name]),
                         original_ratio=best(a[name])/best(c[name]),
                         thread_pinned_ratio=best(b[name])/best(c[name])))
    csv_out('historical_bls_optimization.csv',rows)
    paths = sorted(RAW.rglob('*.json')) + [ROOT/'scripts/gtls_benchmark/gtls_apples_bench.py',
                                           ROOT/'scripts/gtls_benchmark/bench_core.py']
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths}
    (OUT/'historical_sources_sha256.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Wrote historical CSVs and source hashes to', OUT)


if __name__ == '__main__':
    main()
