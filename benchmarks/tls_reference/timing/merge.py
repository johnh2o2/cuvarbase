#!/usr/bin/env python3
"""Prepare portable timing inputs from accepted original or reproduced studies."""
import argparse
import json
import os
from pathlib import Path
import shutil

if __package__:
    from .cohort import _accepted_origin, accepted_study
    from .common import selected_names, sha
else:
    from cohort import _accepted_origin, accepted_study
    from common import selected_names, sha


REGIMES = ('tess_solar', 'tess_gap', 'ztf_solar')


def merge(studies, output):
    """Keep each source's seal and result directory; never reseal its outcomes."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new timing-input directory')
    origins, cases, identity = {}, [], None
    names = set()
    for label, manifest_path, results_root in studies:
        if label in origins:
            raise ValueError('Study labels must be distinct')
        manifest_path, results_root = Path(manifest_path).resolve(), Path(results_root).resolve()
        manifest = json.loads(manifest_path.read_text())
        acceptance_path = results_root/'acceptance.json'
        _accepted_origin(manifest_path, acceptance_path)
        if identity is None:
            identity = manifest['source_identity']
        elif manifest['source_identity']['production_sources'] != identity['production_sources']:
            raise ValueError('Timing studies used different production sources')
        origins[label] = dict(manifest_path=os.path.relpath(manifest_path, output),
            acceptance_path=os.path.relpath(acceptance_path, output),
            results_root=os.path.relpath(results_root, output),
            manifest_sha256=sha(manifest_path), seal_sha256=manifest['seal_sha256'])
        for case in manifest['cases']:
            if case['metadata'].get('null') is not True or case['metadata']['regime'] not in REGIMES:
                continue
            name = case['file']
            if name in names or Path(name).name != name:
                raise ValueError('Timing input filenames must be distinct plain filenames')
            names.add(name)
            source = manifest_path.parent/name
            if sha(source) != case['sha256']:
                raise ValueError('Timing input bytes differ from their accepted manifest')
            cases.append((dict(case, study_id=label,
                result_root=os.path.relpath(results_root/Path(name).stem, output)), source))
    if not origins:
        raise ValueError('Provide at least one accepted study')
    expected = {name for regime in REGIMES for name in selected_names(regime)}
    if not expected.issubset(names):
        raise ValueError('The fixed16-source null cohort is incomplete; reproduce both main and supplemental studies')
    output.mkdir(parents=True)
    for case, source in cases:
        shutil.copyfile(source, output/case['file'])
        if sha(output/case['file']) != case['sha256']:
            raise ValueError('Copied timing input failed its hash check')
    manifest = dict(schema_version=1, suite='validated_timing_merge',
        source_identity=identity, studies=origins, cases=[case for case, _ in cases],
        scope='Timing inputs retain each original or reproduced study identity; merging does not create independent evidence')
    path = output/'manifest.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    validated = accepted_study(path, output)
    reproduced = validated['evidence_kind'] == 'reproduction'
    acceptance = dict(schema_version=1,
        inputs_manifest_sha256=sha(path), classification=validated['evidence_kind'],
        accepted_studies=validated['accepted_studies'], counts=dict(timing_inputs=len(cases)),
        limits=['This merge creates no new independent evidence',
                'Every timing input retains its separate source manifest, seal and numerical-validation receipt'])
    if reproduced:
        acceptance.update(reproduction_gate={'pass': True}, original_source_identity=identity,
            reproduction_sources=dict(production=identity['production_sources'],
                studies={name: value['reproduction_sources'] for name, value in
                         validated['reproduced_studies'].items()}))
    else:
        acceptance.update(publication_gate={'pass': True}, source_identity=identity)
    (output/'acceptance.json').write_text(json.dumps(acceptance, indent=2)+'\n')
    return dict(manifest=str(path), acceptance=str(output/'acceptance.json'),
                input_count=len(cases), evidence_kind=validated['evidence_kind'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', nargs=3, action='append', required=True,
                        metavar=('NAME', 'MANIFEST', 'RESULTS'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.study, args.out), indent=2))


if __name__ == '__main__':
    main()
