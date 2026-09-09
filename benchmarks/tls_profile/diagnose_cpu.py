#!/usr/bin/env python3
"""Reproduce CPU TLS errors; retain explicitly selected traceback locals.

The only numerical adapter is the existing shared period-grid factory. No
template guard or search repair is applied. Recovering already-computed arrays
from a failed call is labeled separately from a successful public API result.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time
import traceback

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--offset', type=int, default=4)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    d = np.load(args.input)
    meta = json.loads(str(d['metadata']))
    i = args.offset
    order = np.argsort(d[f't_{i}'])
    lc = (d[f't_{i}'][order], (d[f'y_{i}']/d[f'scale_{i}'])[order],
          (d[f'dy_{i}']/d[f'scale_{i}'])[order])
    record = dict(status='running', args=vars(args),
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  ndata=len(lc[0]), nperiods=len(d['periods']), truth=meta['cases'][i],
                  scope='CPU failure reproduction; first call including JIT; no failure repair')

    def save():
        args.out.write_text(json.dumps(record, indent=2, default=str) + '\n')

    save()
    module = importlib.import_module('transitleastsquares.main')
    record['main_source_sha256'] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    module.period_grid = lambda **kwargs: d['periods'].copy()
    from transitleastsquares import transitleastsquares
    events = []
    started = time.perf_counter()

    def instrument(name):
        original = getattr(module, name)

        def call(*a, **kw):
            events.append(dict(function=name, event='enter', seconds=time.perf_counter()-started))
            try:
                return original(*a, **kw)
            finally:
                events.append(dict(function=name, event='exit', seconds=time.perf_counter()-started))
        setattr(module, name, call)

    for name in ['get_cache', 'spectra', 'final_T0_fit', 'fractional_transit']:
        instrument(name)
    try:
        result = transitleastsquares(*lc, verbose=False).power(
            use_threads=args.threads, show_progress_bar=False, verbose=False,
            R_star=1, M_star=1, R_star_min=.05, R_star_max=4,
            M_star_min=.05, M_star_max=1, oversampling_factor=3,
            T0_fit_margin=.125, duration_grid_step=1.1)
        record.update(status='ok', period=float(result.period), SDE=float(result.SDE))
    except Exception as error:
        record.update(status='error', error=traceback.format_exc())
        frames, arrays = [], {}
        tb = error.__traceback__
        while tb:
            frame = tb.tb_frame
            if 'transitleastsquares' in frame.f_code.co_filename:
                row = dict(file=frame.f_code.co_filename, function=frame.f_code.co_name,
                           line=tb.tb_lineno, scalars={})
                for name in ['samples', 'internal_samples', 'used_samples', 'maxwidth_in_samples',
                             'duration', 'period', 'T0', 'SDE', 'depth', 'best_row']:
                    value = frame.f_locals.get(name)
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        row['scalars'][name] = float(value)
                for name in ['full_values', 'scaled_transit', 'downsampled_intransit_flux', 'transit_times']:
                    if name in frame.f_locals:
                        value = np.asarray(frame.f_locals[name])
                        row[name] = dict(shape=list(value.shape), size=int(value.size),
                                         minimum=float(np.min(value)) if value.size else None,
                                         maximum=float(np.max(value)) if value.size else None)
                if frame.f_code.co_name == 'power':
                    for name in ['chi2', 'test_statistic_periods', 'test_statistic_depths', 'power']:
                        if name in frame.f_locals:
                            value = np.asarray(frame.f_locals[name])
                            if value.ndim == 1 and len(value) == len(d['periods']):
                                arrays[name] = value
                    record['completed_search_scalars_before_exception'] = row['scalars']
                frames.append(row)
            tb = tb.tb_next
        record['failure_frames'] = frames
        if arrays:
            np.savez_compressed(args.out.with_suffix('.npz'), **arrays)
            record['salvaged_arrays'] = {k: list(v.shape) for k, v in arrays.items()}
            record['salvaged_arrays_sha256'] = hashlib.sha256(args.out.with_suffix('.npz').read_bytes()).hexdigest()
            record['salvage_warning'] = 'These arrays existed before a public API failure; not a successful API return.'
    record.update(total_first_call_s=time.perf_counter()-started, function_events=events)
    save()
    print(json.dumps({k: record[k] for k in ['status', 'total_first_call_s',
                      'completed_search_scalars_before_exception'] if k in record}), flush=True)


if __name__ == '__main__':
    main()
