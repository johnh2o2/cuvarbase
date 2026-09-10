"""Auditable correction of pinned GTLS's full-mode masked-candidate defect.

The installed reference is never modified. A temporary host function differs
only by filtering finite unmasked candidates before ranking.
Templates, CUDA source, duration unions, scoring and final fitting are native.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path

import numpy as np


CORRECTION_ID = 'finite_candidates_before_ranking_v1'
REFERENCE_COMMIT = '74e449c325792a763dde4fbffab98039c5e8c111'


def corrected_source(source):
    old = '    combined = list(enumerate(zip(periods, -power)))'
    new = old + '\n    combined = [item for item in combined if not np.ma.is_masked(item[1][0]) and not np.ma.is_masked(item[1][1]) and np.isfinite(item[1][0]) and np.isfinite(item[1][1])]'
    if source.count(old) != 1:
        raise ValueError('Pinned native host function differs from the audited correction sites')
    return source.replace(old, new)


def identity(core):
    source = inspect.getsource(core.search_multi_periods)
    changed = corrected_source(source)
    return dict(correction=CORRECTION_ID, reference_commit=REFERENCE_COMMIT,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
        corrected_function_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        changes=['Exclude masked/nonfinite periods and scores before the native stable sort'],
        installed_source_unchanged=True, cuda_source_unchanged=True)


@contextmanager
def apply(core):
    original = core.search_multi_periods
    provenance = identity(core)
    namespace = {}
    replacement = corrected_source(inspect.getsource(original))
    # Retain the actual native filename so read-only return profiling also
    # captures the corrected host's unchanged refinement/kernel outputs.
    exec(compile(replacement, core.__file__, 'exec'), core.__dict__, namespace)
    core.search_multi_periods = namespace['search_multi_periods']
    try:
        yield provenance
    finally:
        core.search_multi_periods = original


def finite_candidate_indices(periods, power):
    """Independent literal host selection, used only to prove no-op reuse."""
    combined = list(enumerate(zip(np.ma.asarray(periods), -np.ma.asarray(power))))
    combined = [item for item in combined if not np.ma.is_masked(item[1][0]) and
                not np.ma.is_masked(item[1][1]) and np.isfinite(item[1][0]) and np.isfinite(item[1][1])]
    ranked = sorted(combined, key=lambda item: item[1][1])
    first = [item[0] for item in ranked[:100]]
    remaining = [item for item in ranked if item[0] not in first and item[1][0] > 1]
    second = [item[0] for item in sorted(remaining, key=lambda item: item[1][1])[:100]]
    return np.asarray(first+second, dtype=np.int64)


def no_op_receipt(arrays, result):
    """Sufficient trace evidence that the correction cannot change this run.

    Source/inputs must also be the same. API errors never qualify for reuse.
    The candidate ranking and all actually executed period inputs must remain
    physical, and both chosen-period reads must already be unmasked/finite.
    """
    periods = np.ma.array(arrays['periods'], mask=arrays['stage0_chi2_mask'])
    power = np.ma.array(arrays['stage0_power'], mask=arrays['stage0_power_mask'])
    expected = finite_candidate_indices(periods, power)
    actual = arrays['refinement_indices']
    checks = dict(candidate_indices_equal=np.array_equal(expected, actual),
        no_nonfinite_first_refinement=bool(np.all(np.isfinite(arrays['refinement0_periods']))),
        no_nonfinite_harmonic_refinement=bool(np.all(np.isfinite(arrays['refinement1_periods']))),
        first_selected_period_unmasked=result['stages'][2].get('preceding_selected_period_masked') is False,
        first_selected_period_finite=result['stages'][2].get('preceding_selected_period_finite') is True,
        final_selected_period_finite=result.get('period') is not None and np.isfinite(result['period']))
    if checks['candidate_indices_equal']:
        checks['first_refinement_inputs_equal'] = np.array_equal(
            np.asarray(periods)[expected], arrays['refinement0_periods'])
    else:
        checks['first_refinement_inputs_equal'] = False
    return dict(correction=CORRECTION_ID, proved_no_op=all(checks.values()), checks=checks,
        expected_candidates=len(expected), actual_candidates=len(actual),
        scope='Exact source/inputs plus complete selection trace; no statistical outcome-based reuse')
