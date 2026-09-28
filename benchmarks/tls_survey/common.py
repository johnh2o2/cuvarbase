"""Shared immutable identities and predeclared survey populations."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CADENCES = ROOT / 'benchmarks/results/tls_sensitivity_2026-09-09/cadences'
SNRS = (6., 8., 10., 12.)
REGIMES = {
    'tess_solar': dict(cadence='tess_200s', impact=[.2,.7]),
    'tess_highimpact': dict(cadence='tess_200s', impact=[.94,.96], grid_oversampling=9),
    'tess_eccentric': dict(cadence='tess_200s', impact=[.2,.7], period=[6.,12.], eccentricity=[.7,.8], grid_oversampling=9),
    'tess_mdwarf': dict(cadence='tess_200s', impact=[.2,.7], radius=.1, mass=.1),
    'ztf_solar': dict(cadence='ztf', impact=[.2,.7]),
    'ztf_highimpact': dict(cadence='ztf', impact=[.94,.96], grid_oversampling=9),
    'ztf_mdwarf': dict(cadence='ztf', impact=[.2,.7], radius=.1, mass=.1),
    'tess_gap_long': dict(cadence='tess_gap', impact=[.2,.7], period=[15.,25.]),
    'tess_grazing_smeared': dict(cadence='tess_200s', impact=[.999,1.003], exposure_seconds=1800., stride=9, grid_oversampling=24),
    'hatpi_short': dict(cadence='hatpi', impact=[.2,.96], period=[.65,2.], grid_oversampling=9),
}
BLS_CONFIGS = {
    'bls_medium': dict(noverlap=4, dlogq=.1, qmin_factor=1.),
    'bls_fine': dict(noverlap=8, dlogq=.05, qmin_factor=.5),
    'bls_finest': dict(noverlap=16, dlogq=.025, qmin_factor=.25),
    'bls_strong': dict(noverlap=32, dlogq=.0125, qmin_factor=.125),
}


def method_applicable(method,regime):
    return not (method=='bls_strong' and regime=='tess_gap_long')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    v = np.ascontiguousarray(value)
    h = hashlib.sha256(str(v.dtype).encode()+str(v.shape).encode()+v.tobytes())
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_identity():
    relatives = ['benchmarks/tls_accuracy/diagnose.py', 'cuvarbase/tls_reference_math.py',
                 'benchmarks/transit/worker.py','benchmarks/tls_reference/cases.py',
                 'benchmarks/tls_reference/validate.py']
    relatives += ['benchmarks/tls_survey/'+name for name in
                  ('common.py','generate.py','development.py','run.py','analyze.py',
                   'bls_response.py','boundaries.py')]
    relatives += [str(p.relative_to(ROOT)) for p in sorted(CADENCES.glob('*.npz'))]
    return {p: sha(ROOT/p) for p in relatives}


def load_case(folder, entry):
    path = Path(folder)/entry['file']
    if sha(path) != entry['sha256']:
        raise ValueError('Case bytes changed: '+str(path))
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        arrays = {k:data[k] for k in data.files if k != 'metadata'}
    if metadata != entry['metadata']:
        raise ValueError('Case metadata differs from manifest')
    return arrays, metadata


def recovered(found, metadata, aliases=False):
    if found is None or not np.isfinite(found):
        return False
    factors = (.5,1.,2.,1/3,3.) if aliases else (1.,)
    return any(abs(found/(metadata['truth_period']*factor)-1)*metadata['baseline_days']
               <= .5*metadata['duration_days'] for factor in factors)
