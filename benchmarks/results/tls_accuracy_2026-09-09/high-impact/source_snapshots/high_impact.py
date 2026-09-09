#!/usr/bin/env python3
"""Frozen, focused high-impact recovery pilot; synthetic flux on TESS cadence.

Stages: freeze, generate, run (one method/split), calibrate, analyze.
Keep generated arrays outside the release repository. This pilot measures
recovery and false positives; it does not establish tight equivalence, isolate
binning alone, or provide a new headline timing benchmark.
"""
import argparse
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
import warnings

import numpy as np


COUNTS = dict(calibration=256, injections=256, nulls=256)
METHODS = ('v1_defaults', 'v1_wide', 'v1_fine', 'gtls')
BASELINE = '11317fb0ff1b68af05ae3f67de5f298c9a90e46b'
GTLS_COMMIT = '74e449c325792a763dde4fbffab98039c5e8c111'
SEED = 2026090943


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(value.dtype.str.encode())
    h.update(json.dumps(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def write(path, value, frozen=False):
    path = Path(path)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if frozen and path.exists():
        if path.read_text() != text:
            raise ValueError('Refusing to replace frozen file: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def configs():
    return dict(
        v1_defaults=dict(backend='cuvarbase', t0_oversample=3., n_durations=15,
                         qmin_fac=.5, qmax_fac=2., nbins=None),
        v1_wide=dict(backend='cuvarbase', t0_oversample=3., n_durations=25,
                     qmin_fac=.5 * 4.**(-10./14.), qmax_fac=2., nbins=None),
        v1_fine=dict(backend='cuvarbase', t0_oversample=16., n_durations=32,
                     qmin_fac=.5, qmax_fac=2., nbins=8192),
        gtls=dict(backend='gputls', fast=True, workers=1, T0_fit_margin=.125,
                  duration_grid_step=1.1, R_star_min=.5, R_star_max=2.,
                  M_star_min=1., M_star_max=1.))


def freeze(args):
    root = args.root.resolve()
    if (root / 'design.json').exists():
        raise ValueError('Design already exists; use the existing frozen design')
    repo = Path(__file__).resolve().parents[2]
    paths = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', BASELINE, 'cuvarbase'],
        cwd=repo, text=True).splitlines()
    sources = {}
    for name in paths:
        p = Path(name)
        if p.suffix in ('.py', '.cu', '.cuh') and 'tests' not in p.parts:
            data = subprocess.check_output(['git', 'show', BASELINE + ':' + name], cwd=repo)
            sources[str(p.relative_to('cuvarbase'))] = hashlib.sha256(data).hexdigest()
    archive = repo / 'benchmarks/results/tls_profile_2026-09-08/sources/gtls-head.tar'
    gtls_sources = {}
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            p = Path(member.name)
            if member.isfile() and member.name.startswith('src/gputls/') and p.suffix in ('.py', '.cu', '.cuh'):
                gtls_sources[str(p.relative_to('src/gputls'))] = hashlib.sha256(source.extractfile(member).read()).hexdigest()
    with np.load(args.cadence, allow_pickle=False) as data:
        grid = data['tls_periods']
        if len(grid) != 3084 or len(data['t']) != 9736:
            raise ValueError('Pilot requires the frozen dense TESS 200-second cadence')
        cadence = dict(file_sha256=sha(args.cadence), ndata=len(data['t']),
                       nperiods=len(grid), minimum_period=float(grid.min()),
                       maximum_period=float(grid.max()),
                       arrays={k: array_hash(data[k]) for k in data.files})
    design = dict(
        frozen_at_utc=utc(), schema=1, seed=SEED, counts=COUNTS,
        runner_sha256=sha(__file__), cadence=cadence, configs=configs(),
        population=dict(R_star=1., M_star=1., rp=.0092, impact_uniform=[.94, .96],
                        period_loguniform_days=[2., 6.], epoch_uniform_one_period=True,
                        eccentricity=0., u=[.4804, .1867],
                        oracle_white_snr=[8., 10.], cases_per_snr=128),
        generation=dict(exposure_subsamples=7, random_drop_uniform=[0., .03],
                        minimum_in_transit_points=5, minimum_events=2,
                        relative_errors='Frozen cadence relative_error array',
                        noise='Independent heteroskedastic Gaussian plus OU',
                        ou_amplitude_median_sigma=.25, ou_tau_days=.15,
                        scale='Weighted-centered injected signal has oracle white SNR8 or10; nulls use the same latent-signal noise-scale recipe',
                        flux_baseline=1.),
        search=dict(periods='Full frozen3084period grid, not injection-limited',
                    v1_R_planet_earth_radii=1., v1_refine_top_k=50,
                    v1_refine_oversample=33., v1_batch_size=16,
                    cuvarbase_commit=BASELINE, gtls_commit=GTLS_COMMIT,
                    gtls_source_archive_sha256=sha(archive)),
        statistics=dict(calibration='Strictly above higher95th percentile of256calibration scores, separately for each method',
                        heldout_gate='Threshold file must exist before either held-out run starts',
                        recovery='Primary period drift abs(P_found/P_true-1)*observed_baseline <=0.5*true_duration',
                        failures='Retained; invalid injection is a miss and invalid null score is minus infinity',
                        intervals='Marginal95%Wilson for rates; conservative paired discordant-cell Clopper-Pearson difference intervals',
                        inference='Focused descriptive pilot; no tight equivalence claim or post-result tuning'),
        interpretation=['Widened v1 changes its duration window and automatically increases bin resolution; it is not a pure prior ablation.',
                        'Fine v1 retains the default duration window and changes bins, epoch sampling and duration sampling.',
                        'GTLS differs in template, sample-index duration/epoch search, depth estimation and native score.',
                        'Timing is operational accounting, not a new headline speed benchmark.'],
        expected_sources=dict(cuvarbase=sources, gputls=gtls_sources))
    for method, config in design['configs'].items():
        write(root / 'configs' / (method + '.json'), config, frozen=True)
    write(root / 'design.json', design, frozen=True)
    print(json.dumps(dict(design=str(root / 'design.json'), sha256=sha(root / 'design.json'))))


def load_design(root):
    design = read(root / 'design.json')
    if design['runner_sha256'] != sha(__file__):
        raise ValueError('Runner bytes differ from the frozen protocol')
    if design['counts'] != COUNTS or design['configs'] != configs():
        raise ValueError('Protocol configuration mismatch')
    for method in METHODS:
        if read(root / 'configs' / (method + '.json')) != design['configs'][method]:
            raise ValueError('Frozen method config mismatch: ' + method)
    return design


def make_case(cadence, split, index):
    import batman
    rng = np.random.default_rng(np.random.SeedSequence([SEED, list(COUNTS).index(split), index]))
    raw = [cadence[k] for k in ('t', 'relative_error', 'band', 'exposure_days')]
    keep = rng.random(len(raw[0])) > rng.uniform(0., .03)
    t, relative, band, exposure = [v[keep] for v in raw]
    target = (8., 10.)[index % 2]
    for attempt in range(1, 1001):
        period = float(np.exp(rng.uniform(np.log(2.), np.log(6.))))
        epoch = float(rng.uniform(0., period))
        impact = float(rng.uniform(.94, .96))
        a = (6.6743e-11 * 1.9884e30 * (period * 86400.)**2 / (4 * np.pi**2))**(1./3.) / 6.957e8
        pm = batman.TransitParams()
        pm.t0, pm.per, pm.rp, pm.a = epoch, period, .0092, a
        pm.inc, pm.ecc, pm.w = float(np.degrees(np.arccos(impact / a))), 0., 90.
        pm.u, pm.limb_dark = [.4804, .1867], 'quadratic'
        model = np.ones(len(t))
        for exp in np.unique(exposure):
            take = exposure == exp
            model[take] = batman.TransitModel(pm, t[take], supersample_factor=7,
                                             exp_time=float(exp)).light_curve(pm)
        inside = model < 1 - 1e-9
        events = np.unique(np.rint((t[inside] - epoch) / period).astype(int))
        if inside.sum() >= 5 and len(events) >= 2:
            break
    else:
        raise RuntimeError('Observable injection sampling exhausted')
    w = relative**-2
    signal = model - np.dot(w, model) / w.sum()
    white_scale = np.sqrt(np.dot(w, signal * signal)) / target
    dy = white_scale * relative
    z = rng.normal(size=len(t))
    red = np.empty(len(t))
    red[0] = z[0]
    decay = np.exp(-np.diff(t) / .15)
    innovation = np.sqrt(1 - decay * decay)
    for j in range(1, len(t)):
        red[j] = decay[j-1] * red[j-1] + innovation[j-1] * z[j]
    injected = split == 'injections'
    y = (model if injected else np.ones(len(t))) + rng.normal(size=len(t)) * dy + .25 * np.median(dy) * red
    duration = period / np.pi * np.arcsin(np.sqrt((1 + .0092)**2 - impact**2) / np.sqrt(a*a - impact**2))
    arrays = dict(t=t, y=y, dy=dy, band=band)
    truth = dict(index=index, split=split, injected=injected, seed=[SEED, list(COUNTS).index(split), index],
                 period=period, epoch=epoch, rp=.0092, impact=impact, duration=float(duration),
                 baseline=float(np.ptp(t)), ndata=len(t), n_in_transit=int(inside.sum()),
                 observed_events=len(events), ephemeris_draws=attempt,
                 target_white_oracle_snr=target,
                 latent_white_oracle_snr=float(np.sqrt(np.sum((signal / dy)**2))),
                 median_sigma=float(np.median(dy)),
                 array_sha256={k: array_hash(v) for k, v in arrays.items()})
    return arrays, truth


def generate(args):
    root = args.root.resolve()
    design = load_design(root)
    if sha(args.cadence) != design['cadence']['file_sha256']:
        raise ValueError('Cadence differs from the frozen input')
    manifest_path = root / 'inputs' / 'manifest.json'
    if manifest_path.exists():
        raise ValueError('Input manifest already exists; use the frozen arrays')
    manifest = dict(design_sha256=sha(root / 'design.json'), runner_sha256=sha(__file__),
                    generated_at_utc=utc(), splits={})
    with np.load(args.cadence, allow_pickle=False) as cadence:
        for split, count in COUNTS.items():
            path = root / 'inputs' / (split + '.npz')
            if path.exists():
                raise ValueError('Refusing to overwrite input arrays: ' + str(path))
            arrays = dict(tls_periods=cadence['tls_periods'])
            truths = []
            for index in range(count):
                data, truth = make_case(cadence, split, index)
                truths.append(truth)
                arrays.update({f'{k}_{index}': value for k, value in data.items()})
            metadata = dict(split=split, count=count, cases=truths,
                            design_sha256=sha(root / 'design.json'),
                            period_array_sha256=array_hash(cadence['tls_periods']))
            arrays['metadata'] = np.array(json.dumps(metadata, sort_keys=True, allow_nan=False))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            manifest['splits'][split] = dict(file_sha256=sha(path), metadata=metadata)
            print(json.dumps(dict(generated=split, count=count)), flush=True)
    write(manifest_path, manifest, frozen=True)


def verify_sources(package, expected):
    path = Path(package.__file__).resolve().parent
    actual = {}
    for name, wanted in expected.items():
        source = path / name
        if not source.is_file() or sha(source) != wanted:
            raise ValueError('Installed numerical source differs from pin: ' + str(source))
        actual[name] = wanted
    return dict(path=str(path), files=actual, source_map_sha256=digest(actual))


def gtls_cache(model, truth):
    overview = model.lc_cache_overview
    widths = np.unique(overview['width_in_samples']).astype(int)
    period = truth['period']
    seconds = period * 86400.
    # Literal constants in the pinned GPUFun.getGPUCode; these differ from
    # both Python module constants and the accepted stellar-bound arguments.
    qlo = min(.15, 695508000 * .05 * (4 * seconds / (20848 * 1e15))**(1./3.) / seconds)
    qhi = min(.15, (695508000 * 4 + 2 * 69911000) * (4 * seconds / (416970 * 1e15))**(1./3.) / seconds)
    size = len(model.t)
    correction = 1 + period / float(np.ptp(model.t))
    lo, hi = int(np.floor(qlo * size)), int(np.ceil(qhi * size * correction))
    eligible = widths[(widths >= lo) & (widths <= hi)]
    wanted = truth['duration'] / period * size
    nearest = int(eligible[np.argmin(np.abs(eligible - wanted))]) if len(eligible) else None
    return dict(cache_sha256=array_hash(overview), width_in_samples=widths.tolist(),
                fractional_duration=[float(v) for v in overview['duration']],
                template={k: getattr(model, k) for k in ('per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark')},
                nominal_gpu_width_bounds_at_truth_period=[lo, hi],
                nominal_gpu_eligible_width_count=len(eligible),
                true_geometric_width_in_samples=float(wanted), nearest_eligible_width=nearest,
                nearest_relative_width_error=abs(nearest / wanted - 1) if nearest is not None else None,
                caveat='Nominal per-period CUDA envelope; publicGTLS may evaluate the union of duration choices across a memory-dependent period chunk.')


def run(args):
    root = args.root.resolve()
    design = load_design(root)
    threshold_sha = None
    if args.split != 'calibration':
        thresholds = read(root / 'thresholds.json')  # Before reading held-out input arrays.
        if thresholds['design_sha256'] != sha(root / 'design.json'):
            raise ValueError('Threshold protocol mismatch')
        threshold_sha = sha(root / 'thresholds.json')
    manifest = read(root / 'inputs/manifest.json')
    input_path = root / 'inputs' / (args.split + '.npz')
    if manifest['design_sha256'] != sha(root / 'design.json') or sha(input_path) != manifest['splits'][args.split]['file_sha256']:
        raise ValueError('Input provenance mismatch')
    config = design['configs'][args.method]
    data = np.load(input_path, allow_pickle=False)
    metadata = json.loads(str(data['metadata']))
    if metadata != manifest['splits'][args.split]['metadata']:
        raise ValueError('Input metadata mismatch')
    periods = data['tls_periods']
    if array_hash(periods) != design['cadence']['arrays']['tls_periods']:
        raise ValueError('Period grid changed')
    expected_package = config['backend']
    if expected_package == 'cuvarbase':
        if args.source_root is None:
            raise ValueError('--source-root must identify the baseline11317fb checkout')
        sys.path.insert(0, str(args.source_root.resolve()))
    package = importlib.import_module(expected_package)
    sources = verify_sources(package, design['expected_sources'][expected_package])
    if expected_package == 'cuvarbase':
        from cuvarbase.tls import tls_search_batch
        from cuvarbase.base import ensure_context
        from cuvarbase import tls_models
        import pycuda.driver as driver
        if not tls_models.BATMAN_AVAILABLE:
            raise RuntimeError('A batman template is required; fallback is not permitted')
        ensure_context()
        synchronize = driver.Context.synchronize
        batch_size = 16
    else:
        import cupy as cp
        from gputls import gtls
        from gputls import constants as gtls_constants
        synchronize = cp.cuda.runtime.deviceSynchronize
        batch_size = 1
    output = root / 'results' / args.split / (args.method + '.json')
    identity = dict(method=args.method, split=args.split, count=COUNTS[args.split],
                    design_sha256=sha(root / 'design.json'), runner_sha256=sha(__file__),
                    config=config, config_sha256=sha(root / 'configs' / (args.method + '.json')),
                    input_sha256=sha(input_path), thresholds_sha256=threshold_sha,
                    installed_sources=sources)
    if output.exists():
        record = read(output)
        if any(record[k] != value for k, value in identity.items()):
            raise ValueError('Existing checkpoint has different provenance')
        if record['status'] == 'complete':
            print(json.dumps(dict(already_complete=str(output))))
            return
    else:
        record = dict(**identity, status='running', cases=[], segments=[], packages={})
        record['cpu_quota'] = {name: Path(name).read_text().strip() for name in
            ('/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
             '/sys/fs/cgroup/cpu/cpu.cfs_period_us') if Path(name).exists()}
        record['threads'] = {name: os.getenv(name) for name in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')}
        try:
            record['gpu'] = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,uuid,memory.total,memory.free,driver_version',
                 '--format=csv,noheader'], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            record['gpu'] = None
        for name in ('numpy', 'scipy', 'batman-package', 'pycuda', 'cupy-cuda12x', 'gputls', 'cuvarbase'):
            try:
                record['packages'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        if expected_package == 'gputls':
            record['gtls_host_duration_constants'] = {k: getattr(gtls_constants, k) for k in ('R_STAR_MIN', 'R_STAR_MAX', 'M_STAR_MIN', 'M_STAR_MAX', 'FRACTIONAL_TRANSIT_DURATION_MAX')}
    record['segments'].append(dict(start_index=len(record['cases']), started_at_utc=utc()))
    write(output, record)
    for start in range(len(record['cases']), metadata['count'], batch_size):
        stop = min(start + batch_size, metadata['count'])
        truths = metadata['cases'][start:stop]
        lightcurves = []
        for truth in truths:
            i = truth['index']
            for key, wanted in truth['array_sha256'].items():
                if array_hash(data[f'{key}_{i}']) != wanted:
                    raise ValueError('Prepared input array changed')
            lightcurves.append(tuple(data[f'{key}_{i}'] for key in ('t', 'y', 'dy')))
        synchronize()
        begin = time.perf_counter()
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if expected_package == 'cuvarbase':
                    kwargs = {k: config[k] for k in ('t0_oversample', 'n_durations', 'qmin_fac', 'qmax_fac', 'nbins')}
                    results = tls_search_batch(lightcurves, periods=periods,
                        R_star=1., M_star=1., R_planet=1., u=[.4804, .1867],
                        refine_top_k=50, refine_oversample=33., return_arrays=True, **kwargs)
                    if len(results) != len(lightcurves):
                        raise RuntimeError('Wrong number of public-API results')
                else:
                    model = gtls(*lightcurves[0], verbose=False)
                    kwargs = {k: value for k, value in config.items() if k not in ('backend', 'workers')}
                    raw_periods, raw_power = model.power(periods=periods, R_star=1., M_star=1.,
                        oversampling_factor=3, verbose=False, show_progress_bar=False,
                        transit_template='default', **kwargs)
                    p = np.asarray(np.ma.filled(raw_periods, np.nan), dtype=float)
                    s = np.asarray(np.ma.filled(raw_power, np.nan), dtype=float)
                    good = np.isfinite(p) & np.isfinite(s)
                    j = int(np.nanargmax(np.where(good, s, np.nan))) if good.any() else None
                    results = [dict(period=finite(p[j]) if j is not None else None,
                                    SDE=finite(s[j]) if j is not None else None,
                                    periods=p, chi2=s)]
                synchronize()
            elapsed = time.perf_counter() - begin
            messages = sorted(set(str(w.message) for w in caught))
            if any('falling back to a trapezoid' in message for message in messages):
                raise RuntimeError('Template fallback occurred: ' + '; '.join(messages))
            outputs = []
            for result, truth in zip(results, truths):
                p = np.asarray(result['periods'])
                power = np.asarray(result['chi2'])
                found = finite(result['period']) if result['period'] is not None else None
                score = finite(result['SDE']) if result['SDE'] is not None else None
                valid = found is not None and score is not None and bool(np.isfinite(power).any()) and not result.get('error')
                drift = abs(found / truth['period'] - 1) * truth['baseline'] if valid else None
                row = dict(index=truth['index'], valid=bool(valid), period_found=found, score=score,
                           recovered=bool(valid and drift <= .5 * truth['duration']) if truth['injected'] else None,
                           drift_days=drift, finite_periods=int(np.isfinite(power).sum()),
                           spectrum_sha256=dict(periods=array_hash(p), power=array_hash(power)),
                           api_s=elapsed / len(truths), batch_size=len(truths), warnings=messages,
                           error=result.get('error'))
                if expected_package == 'gputls':
                    row['actual_gtls_cache'] = gtls_cache(model, truth)
                outputs.append(row)
        except Exception:
            elapsed = time.perf_counter() - begin
            error = traceback.format_exc()
            outputs = [dict(index=truth['index'], valid=False, period_found=None, score=None,
                            recovered=False if truth['injected'] else None, error=error,
                            api_s=elapsed / len(truths), batch_size=len(truths),
                            warnings=sorted(set(str(w.message) for w in caught))) for truth in truths]
        record['cases'].extend(outputs)
        write(output, record)
        print(json.dumps(dict(method=args.method, split=args.split, completed=stop,
                              count=metadata['count'], batch_api_s=elapsed)), flush=True)
    record.update(status='complete', finished_at_utc=utc())
    write(output, record)


def scores(record):
    return np.array([row['score'] if row['valid'] and row['score'] is not None else -np.inf for row in record['cases']])


def verified_record(root, method, split, design, manifest):
    path = root / 'results' / split / (method + '.json')
    record = read(path)
    expected = dict(method=method, split=split, count=COUNTS[split], status='complete',
                    design_sha256=sha(root / 'design.json'), runner_sha256=sha(__file__),
                    config=design['configs'][method],
                    config_sha256=sha(root / 'configs' / (method + '.json')),
                    input_sha256=manifest['splits'][split]['file_sha256'])
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('Incomplete or inconsistent result: ' + str(path))
    if [row['index'] for row in record['cases']] != list(range(COUNTS[split])):
        raise ValueError('Missing or duplicated cases: ' + str(path))
    package = design['configs'][method]['backend']
    if record['installed_sources']['files'] != design['expected_sources'][package]:
        raise ValueError('Wrong numerical source: ' + str(path))
    return record


def calibrate(args):
    root = args.root.resolve()
    design = load_design(root)
    manifest = read(root / 'inputs/manifest.json')
    if (root / 'thresholds.json').exists():
        raise ValueError('Calibration already frozen; use existing thresholds')
    # Enforce the planned chronology even if somebody bypassed the run gate.
    for split in ('injections', 'nulls'):
        if (root / 'results' / split).exists() and any((root / 'results' / split).iterdir()):
            raise ValueError('Held-out execution already exists before calibration freeze')
    frozen = dict(frozen_at_utc=utc(), design_sha256=sha(root / 'design.json'),
                  runner_sha256=sha(__file__), input_manifest_sha256=sha(root / 'inputs/manifest.json'),
                  threshold_rule='strict score>higher95thpercentile', methods={})
    for method in METHODS:
        record = verified_record(root, method, 'calibration', design, manifest)
        values = scores(record)
        threshold = float(np.sort(values)[int(np.ceil(.95 * (len(values) - 1)))])
        if not np.isfinite(threshold):
            raise ValueError('No finite calibration threshold for ' + method)
        frozen['methods'][method] = dict(threshold=threshold, n=len(values),
            failed=sum(not row['valid'] for row in record['cases']),
            calibration_result_sha256=sha(root / 'results/calibration' / (method + '.json')))
    write(root / 'thresholds.json', frozen, frozen=True)
    print(json.dumps(dict(thresholds_sha256=sha(root / 'thresholds.json'), methods=list(METHODS))))


def wilson(values):
    n = len(values)
    k = int(np.sum(values))
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return dict(n=n, count=k, rate=p, wilson95=[max(0., center-half), min(1., center+half)])


def paired(left, right):
    from scipy.stats import beta
    left, right = np.asarray(left, bool), np.asarray(right, bool)
    n = len(left)
    wins, losses = int(np.sum(left & ~right)), int(np.sum(~left & right))
    # Four one-sided Clopper-Pearson cell bounds; Bonferroni gives at least
    #95% coverage for this complete two-sided paired-difference interval.
    tail = .05 / 4
    def low(k):
        return float(beta.ppf(tail, k, n-k+1)) if k else 0.
    def high(k):
        return float(beta.ppf(1-tail, k+1, n-k)) if k < n else 1.
    return dict(n=n, left_only=wins, right_only=losses,
                both=int(np.sum(left & right)), neither=int(np.sum(~left & ~right)),
                difference=float(left.mean()-right.mean()),
                conservative_paired95=[low(wins)-high(losses), high(wins)-low(losses)])


def analyze(args):
    root = args.root.resolve()
    design = load_design(root)
    manifest = read(root / 'inputs/manifest.json')
    frozen = read(root / 'thresholds.json')
    if frozen['design_sha256'] != sha(root / 'design.json') or frozen['input_manifest_sha256'] != sha(root / 'inputs/manifest.json'):
        raise ValueError('Frozen calibration provenance mismatch')
    report = dict(design_sha256=sha(root / 'design.json'), runner_sha256=sha(__file__),
                  input_manifest_sha256=sha(root / 'inputs/manifest.json'),
                  thresholds_sha256=sha(root / 'thresholds.json'),
                  scope='Focused high-impact recovery pilot; marginal/descriptive intervals, no tight equivalence claim.',
                  methods={}, comparisons={}, record_sha256={})
    vectors = {}
    truth = manifest['splits']['injections']['metadata']['cases']
    levels = np.array([row['target_white_oracle_snr'] for row in truth])
    for method in METHODS:
        cal = root / 'results/calibration' / (method + '.json')
        if sha(cal) != frozen['methods'][method]['calibration_result_sha256']:
            raise ValueError('Calibration changed after threshold freeze')
        inputs = {split: verified_record(root, method, split, design, manifest) for split in COUNTS}
        for split in ('injections', 'nulls'):
            if inputs[split]['thresholds_sha256'] != sha(root / 'thresholds.json'):
                raise ValueError('Held-out result was not run after this calibration freeze')
        threshold = frozen['methods'][method]['threshold']
        recovered = np.array([row['valid'] and row['recovered'] for row in inputs['injections']['cases']], bool)
        detection = recovered & (scores(inputs['injections']) > threshold)
        false_positive = scores(inputs['nulls']) > threshold
        vectors[method] = dict(detection=detection, false_positive=false_positive)
        summary = dict(threshold=threshold, recovery=wilson(detection), false_positive=wilson(false_positive),
                       period_recovery=wilson(recovered),
                       per_snr={str(level): wilson(detection[levels == level]) for level in (8., 10.)},
                       failed={split: sum(not row['valid'] for row in record['cases']) for split, record in inputs.items()},
                       operational_api_seconds={split: float(sum(row['api_s'] for row in record['cases'])) for split, record in inputs.items()})
        if method == 'gtls':
            audit = [row['actual_gtls_cache'] for row in inputs['injections']['cases'] if 'actual_gtls_cache' in row]
            errors = [row['nearest_relative_width_error'] for row in audit if row['nearest_relative_width_error'] is not None]
            summary['duration_cache_check'] = dict(cases_audited=len(audit),
                no_nominally_eligible_cache_width=sum(row['nearest_eligible_width'] is None for row in audit),
                nearest_relative_width_error_max=max(errors) if errors else None,
                nearest_relative_width_error_median=float(np.median(errors)) if errors else None,
                caveat='This checks nominal duration coverage, not equality of template or numerical search.')
        report['methods'][method] = summary
        for split in COUNTS:
            report['record_sha256'][split + '/' + method] = sha(root / 'results' / split / (method + '.json'))
    for left, right in [('gtls', 'v1_defaults'), ('v1_wide', 'v1_defaults'),
                        ('v1_fine', 'v1_defaults'), ('v1_wide', 'gtls'), ('v1_fine', 'gtls')]:
        report['comparisons'][left + '_minus_' + right] = dict(
            recovery=paired(vectors[left]['detection'], vectors[right]['detection']),
            false_positive=paired(vectors[left]['false_positive'], vectors[right]['false_positive']),
            per_snr={str(level): paired(vectors[left]['detection'][levels == level], vectors[right]['detection'][levels == level]) for level in (8., 10.)})
    write(root / 'analysis.json', report, frozen=True)
    print(json.dumps(report, sort_keys=True, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('freeze', 'generate', 'run', 'calibrate', 'analyze'):
        p = sub.add_parser(name)
        p.add_argument('--root', type=Path, required=True)
        if name in ('freeze', 'generate'):
            p.add_argument('--cadence', type=Path, required=True)
        if name == 'run':
            p.add_argument('--method', choices=METHODS, required=True)
            p.add_argument('--split', choices=tuple(COUNTS), required=True)
            p.add_argument('--source-root', type=Path)
        p.set_defaults(function=globals()[name])
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
