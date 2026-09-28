#!/usr/bin/env python3
"""Reproduce isolated exact host optimizations against a recorded git source.

This is a CPU diagnostic, not a GPU or public-search speed denominator.
Run with numerical-library threads limited to one, for example:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m \
    benchmarks.tls_reference.profile_host_optimizations --output profile.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import tracemalloc
import types

import numpy as np

from cuvarbase import tls_reference_math as candidate


BASELINE_REVISION = '6ced75d6d75bfaafa39b78c557fcba86f4651d92'


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def profile_pair(old, new, args, repetitions):
    expected, actual = old(*args), new(*args)
    np.testing.assert_array_equal(actual, expected)
    elapsed = {'baseline': [], 'candidate': []}
    for repetition in range(repetitions):
        pairs = [('baseline', old), ('candidate', new)]
        if repetition % 2:
            pairs.reverse()
        for label, function in pairs:
            begin = time.perf_counter()
            output = function(*args)
            elapsed[label].append(time.perf_counter() - begin)
            np.testing.assert_array_equal(output, expected)
    memory = {}
    for label, function in [('baseline', old), ('candidate', new)]:
        tracemalloc.start()
        output = function(*args)
        memory[label] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        np.testing.assert_array_equal(output, expected)
    medians = {key: statistics.median(value) for key, value in elapsed.items()}
    return dict(exact=True, output_sha256=digest(expected),
                elapsed_seconds=elapsed, median_seconds=medians,
                median_speedup=medians['baseline'] / medians['candidate'],
                peak_tracemalloc_bytes=memory,
                memory_scope='Separate untimed call; Python/NumPy tracked temporary allocations, excluding inputs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-revision', default=BASELINE_REVISION)
    parser.add_argument('--repetitions', type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('--repetitions must be positive')
    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ['git', 'show', args.baseline_revision + ':cuvarbase/tls_reference_math.py'], cwd=root)
    baseline = types.ModuleType('tls_reference_math_baseline')
    exec(compile(source, '<recorded-baseline>', 'exec'), baseline.__dict__)
    record = dict(scope='Isolated CPU diagnostic; no end-to-end GPU speed claim',
                  baseline_revision=args.baseline_revision,
                  baseline_source_sha256=hashlib.sha256(source).hexdigest(),
                  candidate_source_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   system=platform.platform(), machine=platform.machine(),
                                   thread_settings={key: os.environ.get(key) for key in
                                                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}),
                  seed=912, repetitions=args.repetitions, candidate_ranking=[], duration_unions=[])
    for size in (2325, 74616, 235266, 1093617):
        rng = np.random.default_rng(record['seed'] + size)
        periods = np.linspace(.5, 50., size)
        power = np.ma.array(rng.normal(size=size).astype(np.float32),
                            mask=rng.random(size) < .025)
        result = profile_pair(baseline.refinement_candidate_indices,
                              candidate.refinement_candidate_indices,
                              (periods, power), args.repetitions)
        record['candidate_ranking'].append(dict(nperiods=size, **result))
        widths = np.arange(2, 142, 2)
        minima = rng.integers(0, 14, size)
        maxima = rng.integers(14, 144, size)
        result = profile_pair(baseline.chunk_width_masks, candidate.chunk_width_masks,
                              (widths, minima, maxima, max(1, size // 30)), args.repetitions)
        record['duration_unions'].append(dict(nperiods=size, nwidths=len(widths), **result))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(nperiods=size,
                              candidate_ranking=record['candidate_ranking'][-1]['median_speedup'],
                              duration_unions=result['median_speedup'])), flush=True)


if __name__ == '__main__':
    main()
