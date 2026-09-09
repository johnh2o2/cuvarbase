#!/usr/bin/env python3
"""Prepare reviewable release copy only from completed, verified benchmark artifacts."""
import argparse,hashlib,json,os,re,shutil,tempfile
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root.resolve();repo=Path.cwd()
    for name in ['provenance-verification.json','recovery_analysis.json','timing_analysis.json','component_analysis.json']:
        d=json.loads((r/name).read_text());assert d.get('complete',d.get('verification',{}).get('complete')),name
    timings=json.loads((r/'timing_analysis.json').read_text());rec=json.loads((r/'recovery_analysis.json').read_text())
    assert timings['verification']['arrays_verified'] and rec['verification']['arrays_verified']
    # Prepare all documents against the retained originals in an isolated directory.
    # Apply only after generation succeeds, and refuse to overwrite intervening edits.
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=r/'release-doc-preparation.json';previous=json.loads(receipt.read_text())['outputs'] if receipt.exists() else {}
    originals=json.loads((r/'claims-before.json').read_text())['files']
    for item in originals:
        source=r/'sources/claims-before'/item['path'];assert digest(source)==item['sha256']
        assert digest(repo/item['path']) in [item['sha256'],previous.get(item['path'])], 'Intervening document edit: '+item['path']
    staging=tempfile.TemporaryDirectory(prefix='cuvarbase-benchmark-release-');stage=Path(staging.name)
    for item in originals:
        target=stage/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(r/'sources/claims-before'/item['path'],target)
    os.chdir(stage)
    t={(v['profile'],v['method'],v['mode']):v for v in timings['timings']}
    comparisons={(v['profile'],v['v1'],v['comparator']):v for v in rec['comparisons']}
    profiles=['tess_200s','tess_gap','ztf'];names={'tess_200s':'TESS 200 s','tess_gap':'Separated TESS sectors','ztf':'ZTF g/r'}
    fresh=[t[p,'bls_pypi','fresh_grid']['seconds_per_source']/t[p,'bls_v1','fresh_grid']['seconds_per_source'] for p in profiles]
    tls=[t[p,'gtls','batch16']['seconds_per_source']/t[p,'tls_v1','batch16']['seconds_per_source'] for p in profiles]
    copy=['# Transit searches: measured speed and recovery','',
        'cuvarbase v1 reduces the cost of the transit-search stage. This experiment compares actual PyPI BLS, external CPU/GPU BLS, and GTLS using observed ZTF and TESS cadences with independent synthetic transit injections. It measures both one-source latency and throughput for 16 distinct sources.','',
        f'For a fresh native Keplerian grid plus BLS search, v1 is **{min(fresh):.1f}–{max(fresh):.1f}× faster than PyPI 0.2.5** on these three examples. The separated-sector TESS result supports the reported 5-point detection/false-positive criterion; the other PyPI comparisons remain inconclusive. TLS batch search time is **{min(tls):.1f}–{max(tls):.1f}× lower than public GTLS**, but **equivalent TLS detection sensitivity is not established** by this experiment.','',
        '![Transit search time and independently measured recovery](figures/transit_benchmarks_20260908.png)','',
        '[PDF figure](figures/transit_benchmarks_20260908.pdf) · [SVG figure](figures/transit_benchmarks_20260908.svg) · [Full experiment and evidence](../analysis/transit-recovery-20260908/README.md)','',
        '| Observing pattern | BLS batch: PyPI / v1 time | BLS recovery match | TLS batch: GTLS / v1 time | TLS recovery match |',
        '|---|---:|---|---:|---|']
    for p in profiles:
        b=t[p,'bls_pypi','batch16']['seconds_per_source']/t[p,'bls_v1','batch16']['seconds_per_source'];g=t[p,'gtls','batch16']['seconds_per_source']/t[p,'tls_v1','batch16']['seconds_per_source']
        verdict=lambda c:'Supported within 5 pp' if c['comparable_detection'] else 'Not established'
        copy.append(f"| {names[p]} | {b:.2f}× | {verdict(comparisons[p,'bls_v1_batch','bls_pypi'])} | {g:.2f}× | {verdict(comparisons[p,'tls_v1','gtls_batch'])} |")
    copy+=['',
        '“Supported” uses paired, nominal one-sided 95% bounds: detection-recovery loss below 5 percentage points and false-positive increase below 5 points. An unresolved comparison remains a timing observation. Native SDE values are not evidence of equivalent sensitivity. Each method has 128 independent calibration nulls, 128 held-out injections and 128 held-out nulls per cadence.','',
        'BLS gains come from fused phase histograms, vectorized host scans and grid construction, and amortizing work across a batch. Disabling fusion increases diagnostic API time by 1.35–1.57×; observation scattering does not demonstrate a benefit on these cases. Both releases receive warmed kernels and reusable PyPI memory. TLS combines a phase-binned search and exact refinement of selected candidates with fewer Python-to-GPU dispatches. Batching two GTLS host loops improves its diagnostic runtime by 1.4–8.2×. GTLS and cuvarbase are related template searches with different numerical objectives, sampling and refinement. The remaining speed gap is not a comparison of identical computations. Full component evidence is retained in the report; not every gain is a phase-5 change.','',
        'The A40 bundle costs $0.49/hour. Figure costs are linear projections of measured search throughput, excluding preprocessing, imports, I/O, idle time and candidate vetting. The full report gives CPU-only break-even prices rather than assuming an unmeasured CPU rental price. The tests use real observing times with controlled flux/noise, known band baselines and observable injected transits; they are not a catalog completeness estimate or a complete QLP pipeline benchmark.','',
        'The period grid and density prior follow the published [QLP search description](https://arxiv.org/abs/2302.01293), with a separate tuning stage. Actual PyPI cuvarbase 0.2.5 has no TLS implementation, so its upgrade comparison is BLS only. Astropy, periodfind and fBLS were screened as external CPU BLS candidates; periodfind supplies the external GPU BLS comparison. “Best” means the strongest successfully tested setting in this campaign, not a universal ranking.','',
        'The earlier 30–171× equal-SDE TLS headline, thousands-fold CPU-TLS claim, and 257–354× Astropy-BLS headline are superseded as release advertising by this report. The [provenance audit](../analysis/benchmark-audit-20260906/README.md) explains their original arithmetic and limitations; historical measurements remain available for inspection.']
    Path('docs/TRANSIT_BENCHMARKS.md').write_text('\n'.join(copy)+'\n')
    folder=Path('docs/figures');folder.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:shutil.copyfile(r/f'benchmark_story.{ext}',folder/f'transit_benchmarks_20260908.{ext}')
    p=Path('README.md');text=p.read_text();lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('- **Transit Least Squares is 30-171x'):
            lines[i]='- **GPU transit searches with measured speed and recovery.** Compare v1 BLS with actual PyPI 0.2.5 and tested CPU/GPU alternatives, and v1 TLS with GTLS, on ZTF and TESS cadences. [One figure, recovery qualifications, and search-cost estimates](docs/TRANSIT_BENCHMARKS.md).'
        if line.startswith('- **Standard BLS is 257-354x'):
            lines[i]='- **A practical BLS upgrade for TESS and ZTF workloads.** Fused phase searches, reusable batch memory and faster Keplerian grid construction reduce search time. The [current benchmark](docs/TRANSIT_BENCHMARKS.md) reports gains with independent recovery and false-positive checks.'
        if line.startswith('- **All four major surveys for ~$33'):
            lines[i]='- **Search-cost estimates tied to measured throughput.** The [current transit benchmark](docs/TRANSIT_BENCHMARKS.md) gives A40 rental equivalents and CPU break-even prices, with preprocessing and full-pipeline costs outside its scope.'
        if line.startswith('Full tables, per-survey costs, and methodology:'):
            lines[i]='Current transit timings, recovery and cost: [one benchmark figure](docs/TRANSIT_BENCHMARKS.md). Earlier measurements for the other algorithms remain in [the benchmark archive](docs/BENCHMARK_RESULTS.md).'
        if line.startswith('v1.0 is a major modernization'):
            start=line.index('with large architectural speedups');end=line.index(', the new survey-scale TLS engine',start)
            lines[i]=line[:start]+'with faster transit searches and Keplerian grid construction ([measured results](docs/TRANSIT_BENCHMARKS.md))'+line[end:]
    p.write_text('\n'.join(lines)+'\n')
    comparison=['# cuvarbase TLS and GTLS: speed, recovery and implementation','',
        'The [current transit benchmark](TRANSIT_BENCHMARKS.md) compares exclusive single-source and batch timings on one A40, together with independent recovery and null tests on observed ZTF and TESS cadences. Its figure and qualifications replace the earlier equal-SDE headline.','',
        '| Stage | cuvarbase v1 TLS | Pinned public GTLS |','|---|---|---|',
        '| Coarse search | Fold into weighted phase bins; reuse the bins across template trials | Sort individual observations by phase; template widths use observation counts |',
        '| Depth / objective | Analytic weighted template-depth fit, with unit baseline | Unweighted window-mean depth estimate with template overshoot, followed by weighted residuals |',
        '| Candidate precision | Exact observation-level refinement of selected top candidates | Different epoch/duration sampling and refinement policy; fast mode returns an SDE spectrum |',
        '| Significance | Native SDE calibrated on independent nulls | Its own native SDE calibrated on the same independent null inputs |','',
        'These are related transit-template algorithms with different numerical searches. A common trial-period array and limb-darkening coefficients do not make them identical. Similar scalar SDE values, including values recomputed with one formula, do not establish equivalent recovery or false-alarm behavior.','',
        'The speed difference combines cuvarbase’s phase-bin architecture with GTLS host orchestration overhead. Measured diagnostic changes batch GTLS’s per-period flux-prefix-sum loop and repeated duration-mask union operations. Full output comparisons and synchronized component timings are in the [current experiment](../analysis/transit-recovery-20260908/README.md) and the [earlier TLS component audit](../analysis/tls-profile-20260908/README.md). These diagnostic patches are separate from the released competitor. Warm CUDA module compilation/lookup was negligible in the earlier profiles.','',
        'The fast cuvarbase engine predates phase 5; the entire advantage is not a phase-5 gain. The current benchmark also tunes documented GTLS fast mode and density constraints, and measures concurrent throughput with separately validated recovery because available GPU memory can change GTLS chunking and its spectrum.','',
        'Earlier CPU TLS failures were zero-sample template/model edge cases. Some happened before the search; the ZTF/Rubin cases completed the period search and failed during output-model construction. Failed API times are excluded from speedup claims.','',
        'The original July comparison and its arithmetic remain in the [preserved document](../analysis/transit-recovery-20260908/sources/claims-before/docs/GTLS_COMPARISON.md) and [provenance audit](../analysis/benchmark-audit-20260906/README.md). In particular, the former 30–171× “equal sensitivity” claim and claims about the GTLS paper’s exact hidden settings are not supported by that evidence.']
    Path('docs/GTLS_COMPARISON.md').write_text('\n'.join(comparison).replace('released competitor','public upstream competitor')+'\n')
    cost=['# Transit-search rental cost','',
        'The [current benchmark figure](TRANSIT_BENCHMARKS.md) pairs measured execution time with independent recovery. Cost savings have the same recovery qualifications as speedups. The A40 bundle used here costs $0.49/hour, including its CPU allocation.','',
        '| Observing pattern | v1 BLS / million | PyPI BLS / million | v1 TLS / million | GTLS / million | CPU BLS hourly break-even |','|---|---:|---:|---:|---:|---:|']
    for p in profiles:
        values=[t[p,m,'batch16']['projected_gpu_usd_per_million'] for m in ['bls_v1','bls_pypi','tls_v1','gtls']]
        speed=t[p,'bls_cpu','batch16']['seconds_per_source']/t[p,'bls_v1','batch16']['seconds_per_source']
        cost.append('| '+names[p]+' | '+' | '.join(f'${v:.2f}' for v in values)+f' | ${.49/speed:.4f}/h |')
    cost+=['',
        'These are linear projections of the median 16-source search throughput, not measured million-source jobs. The boundary includes transfers, periodograms and candidate ranking from prepared arrays; preprocessing, imports, grid construction, I/O, idle time and vetting are excluded. Fresh-grid timings are reported separately. A complete QLP or survey bill cannot be inferred from these values.','',
        'CPU-only break-even price = $0.49 / (CPU time ÷ v1 GPU time), for a CPU service delivering the measured throughput. No standalone CPU rental was benchmarked. The measurement used a 7.65-CPU-equivalent quota on the same Xeon Gold 6342 host; 96 host logical CPUs were not the allocation.','',
        'The [full report](../analysis/transit-recovery-20260908/README.md) contains recovery qualifications, repetitions, hardware, pinned versions and the experiment rental ledger. The old claims of universally cheapest TLS and thousands-fold CPU savings are replaced by these measured, workload-specific projections. [Preserved historical cost document](../analysis/transit-recovery-20260908/sources/claims-before/docs/TLS_COST_ANALYSIS.md).']
    Path('docs/TLS_COST_ANALYSIS.md').write_text('\n'.join(cost)+'\n')
    p=Path('docs/RELEASE_NOTES_v1.0.0.md');old=p.read_text()
    old=re.sub(r'^- \*\*New: survey-scale GPU Transit Least Squares.*$', '- **New GPU Transit Least Squares:** a phase-binned batch engine with exact candidate refinement. The [current ZTF/TESS benchmark](TRANSIT_BENCHMARKS.md) reports its timing advantage over public GTLS together with independent recovery and false-positive qualifications.',old,flags=re.M)
    old=re.sub(r'^- \*\*Standard BLS runs 257.*$', '- **Faster BLS searches and grid construction:** compare actual PyPI 0.2.5, v1 and tested CPU/GPU alternatives in the [current benchmark](TRANSIT_BENCHMARKS.md). The earlier 257–354× Astropy headline used unequal duration searches and is withdrawn as a fair-comparison claim.',old,flags=re.M)
    old=re.sub(r'^- \*\*Versus the previous cuvarbase:.*$', '- **Versus actual PyPI 0.2.5:** fused phase searches, conflict-scatter staging, reusable batch memory, vectorized host scans and grid construction, plus support for the current NumPy/PyCUDA stack. Both releases receive warmed kernels and reusable PyPI memory in the new comparison; its warm speedup is not attributed entirely to compilation caching.',old,flags=re.M)
    start=old.index('## Performance\n');end=old.index('## New features\n',start)
    old=old[:start]+('## Performance\n\nThe [current transit benchmark](TRANSIT_BENCHMARKS.md) is the source for BLS/TLS release claims: one figure, single-source and batch timing, independent recovery, null false positives, and search-cost projections. Equal scalar SDE is not an equal-sensitivity guarantee.\n\nThe former transit headline table and 0.2.6 comparison are retained in the [archived release notes](../analysis/transit-recovery-20260908/sources/claims-before/docs/RELEASE_NOTES_v1.0.0.md). The latest published upgrade baseline is 0.2.5; the 0.2.6 tag was not published to PyPI. Earlier measurements for other algorithms remain in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).\n\n')+old[end:]
    old=re.sub(r'^- \*\*Statistics discipline\*\*:.*$', '- **Statistics:** SDE uses the coarse spectrum while refinement sharpens candidate parameters. Null calibration and independent recovery are required to compare detection performance; a scalar SDE difference or successful golden tests do not establish population sensitivity. An opt-in null bootstrap is available on `tls_search_batch(fap_null_draws=...)`.',old,flags=re.M)
    old=re.sub(r'^- \*\*Survey-speed kernels \(July 2026\)\*\*:.*$', '- **BLS throughput features (July 2026):** fused phase histograms, observation-scatter staging, frequency chunking, and host overhead fixes. The [current benchmark](TRANSIT_BENCHMARKS.md) measures their practical upgrade effect and diagnostic ablations; scattering does not demonstrate a benefit on its three selected cases. Earlier speed ratios are preserved in the archived release notes above.',old,flags=re.M)
    p.write_text(old)
    p=Path('docs/BENCHMARK_RESULTS.md');old=p.read_text()
    start=old.index('## 3. BLS:');end=old.index('## 5. Keplerian',start)
    old=old[:start]+('## 3. BLS: current comparisons\n\nUse the [current transit benchmark](TRANSIT_BENCHMARKS.md) for actual PyPI 0.2.5, v1, Astropy and periodfind comparisons on ZTF/TESS cadences. periodfind provides both CPU and GPU BLS; cuvarbase is not the only GPU BLS implementation. fBLS was screened, with failed/time-limited pilots retained and excluded from speed denominators. The former 257–354× Astropy headline used unequal duration searches.\n\n## 4. TLS: current comparisons\n\nUse the [current transit benchmark](TRANSIT_BENCHMARKS.md) and [implementation comparison](GTLS_COMPARISON.md). Equal SDE did not establish equal sensitivity in the July measurements, and warm GTLS module compilation was not the dominant measured bottleneck. The original BLS/TLS tables remain in the [preserved benchmark document](../analysis/transit-recovery-20260908/sources/claims-before/docs/BENCHMARK_RESULTS.md) and the [provenance audit](../analysis/benchmark-audit-20260906/README.md).\n\n')+old[end:]
    start=old.index('## 6. Combined');end=old.index('## Reproducibility',start)
    old=old[:start]+('## 6. Search-cost projections\n\nCurrent [transit cost estimates](TLS_COST_ANALYSIS.md) derive from measured A40 batch search throughput. They exclude full-pipeline work. Earlier whole-survey dollar totals are preserved in the historical document linked above and should not be advertised as measured complete survey costs.\n\n')+old[end:]
    old=re.sub(r'^- \*\*BLS\*\*:.*$', '- **BLS/TLS:** current speed, independent recovery and cost measurements are in [one transit benchmark figure](TRANSIT_BENCHMARKS.md).',old,flags=re.M)
    old=old.replace('search 4-37x fewer frequencies with no loss in transit detection sensitivity','search 4-37x fewer frequencies in the historical grid examples below; these frequency counts alone do not establish unchanged detection sensitivity')
    p.write_text(old)
    for name in ['docs/BENCHMARK_RESULTS.md','docs/RELEASE_NOTES_v1.0.0.md']:
        p=Path(name);old=p.read_text();first,rest=old.split('\n',1)
        banner=('\n> **Benchmark correction, September 2026.** The transit timing/sensitivity and cost claims below describe historical protocols. Use the [new transit benchmark](TRANSIT_BENCHMARKS.md) for current release claims. Equal scalar SDE did not establish equal sensitivity; some old BLS comparisons used different duration searches; warm GTLS compilation was not the dominant measured bottleneck. Historical values are retained for provenance, not as qualified performance promises.\n')
        if '> **Benchmark correction, September 2026.**' not in old:p.write_text(first+'\n'+banner+rest)
    p=Path('CHANGELOG.rst');old=p.read_text()
    old=re.sub(r'^    \* Measured head-to-head against the previous cuvarbase.*$', '    * The historical July head-to-head used the unpublished 0.2.6 tag and a particular call-per-lightcurve harness. Its 34x loop ratio is not an actual PyPI upgrade comparison or a measurement of warm CUDA compilation cost. The September comparison in ``docs/TRANSIT_BENCHMARKS.md`` uses actual PyPI 0.2.5 with warmed kernels and reusable memory. BLS also fixes the old float32-fold failure on absolute BJD-scale timestamps.',old,flags=re.M)
    marker='Measured end-to-end on an RTX A5000 (``scripts/benchmark_tls_survey.py``'
    if marker in old:
        start=old.index(marker);end=old.index('Batch API validation:',start)
        old=old[:start]+'Timing, recovery and cost claims are superseded by the September 2026 independent-injection benchmark in ``docs/TRANSIT_BENCHMARKS.md``. Equal scalar SDE is not a sensitivity guarantee, and the old thousands-fold CPU and 30-171x GTLS claims must not be read as equivalent-recovery results. '+old[end:]
    old=old.replace('Performance claims re-grounded in measured data (257-354x vs astropy BoxLeastSquares across 7 GPU architectures for standard BLS; honest small-problem caveats for LS)', 'Transit benchmark claims corrected in September 2026: independent recovery/null calibration, actual PyPI baseline, tested CPU/GPU BLS alternatives, and source-verified GTLS comparisons; see ``docs/TRANSIT_BENCHMARKS.md``')
    note=('.. note::\n\n    September 2026 benchmark correction: historical transit speed ratios and\n    equal-SDE statements below are not equivalent-sensitivity guarantees.\n    Current measured comparisons, recovery qualifications and search-cost\n    projections are in ``docs/TRANSIT_BENCHMARKS.md``.\n\n')
    if not old.startswith('.. note::\n\n    September 2026 benchmark correction:'):p.write_text(note+old)
    outputs={str(p.relative_to(stage)):digest(p) for p in stage.rglob('*') if p.is_file()}
    os.chdir(repo)
    allowed={v['path']:v['sha256'] for v in originals};allowed.update(previous)
    for name,new_digest in outputs.items():
        target=repo/name
        if target.exists():assert digest(target) in [allowed.get(name),new_digest], 'Intervening output edit: '+name
    for name in outputs:
        target=repo/name;target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.benchmark-update')
        shutil.copyfile(stage/name,temporary);temporary.replace(target)
    receipt.write_text(json.dumps(dict(outputs=outputs,script_sha256=digest(Path(__file__).resolve()),published=False),indent=2)+'\n')
    staging.cleanup()
    print('Prepared release benchmark page, figure exports, README wording and historical benchmark corrections. No publication performed.')


if __name__=='__main__':main()
