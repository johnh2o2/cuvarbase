"""New-allocation figures cannot silently compare different resources or inputs."""
import copy

import pytest

from benchmarks.tls_survey.report_followup import validate_comparison


@pytest.mark.parametrize('changed', [None, 'gpu', 'input', 'science', 'threads', 'population'])
def test_comparison_requires_identical_allocation_and_input_bytes(changed):
    reference = dict(environment=dict(nvidia_smi='gpu-1', cpu_quota_cores=8,
                                     host_memory_limit_bytes=1024,
                                     cpu_math_thread_environment={key: '1' for key in (
                                         'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                         'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')}),
                     science_seal_sha256='frozen-science',
                     cohort=[dict(name=f'a-{index}.npz', regime='tess_solar', nobs=6,
                                  nperiods=10, input_sha256='original-bytes') for index in range(16)])
    candidate = copy.deepcopy(reference)
    if changed == 'gpu':
        candidate['environment']['nvidia_smi'] = 'gpu-2'
    elif changed == 'input':
        candidate['cohort'][0]['input_sha256'] = 'different-bytes'
    elif changed == 'science':
        candidate['science_seal_sha256'] = 'different-science'
    elif changed == 'threads':
        candidate['environment']['cpu_math_thread_environment']['NUMEXPR_NUM_THREADS'] = '4'
    elif changed == 'population':
        candidate['cohort'][0]['regime'] = 'ztf_solar'
    data = {('baseline', 'tess_solar'): reference, ('bls', 'tess_solar'): candidate}
    if changed:
        with pytest.raises(ValueError):
            validate_comparison(data, ('gpu-1', 8, 1024), 'frozen-science')
    else:
        validate_comparison(data, ('gpu-1', 8, 1024), 'frozen-science')
