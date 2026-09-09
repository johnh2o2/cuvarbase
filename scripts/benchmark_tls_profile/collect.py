#!/usr/bin/env python3
"""Archive completed profiling evidence and installed-source hashes."""
import hashlib
import json
from pathlib import Path
import sysconfig
import tarfile


def main():
    root = Path('/tmp/cuvarbase-tls-profile')
    assert 'PROFILE_CAMPAIGN_COMPLETE' in (root/'campaign.log').read_text()
    installed = Path(sysconfig.get_paths()['purelib'])
    paths = {'v1': installed/'cuvarbase', 'gtls_pypi': installed/'gputls',
             'gtls_head': root/'gtls-head-install/gputls',
             'cpu_tls': installed/'transitleastsquares'}
    hashes = {}
    for name, folder in paths.items():
        assert folder.is_dir()
        hashes[name] = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(folder.rglob('*')) if p.suffix in ['.py', '.cu', '.cuh']}
    (root/'results/installed-source-hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    files = [p for p in (root/'results').rglob('*') if p.is_file()]
    files += [root/name for name in ['campaign.log', 'setup.log', 'setup.sh',
                                     'profile_tls.py', 'diagnose_cpu.py', 'run_campaign.py', 'collect.py']]
    files += list((root/'inputs').glob('*.npz'))
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (root/'transfer-sha256.json').write_text(json.dumps(manifest, indent=2)+'\n')
    with tarfile.open(root/'evidence.tar', 'w') as archive:
        for p in sorted(files):
            archive.add(p, arcname=str(p.relative_to(root)))
        archive.add(root/'transfer-sha256.json', arcname='transfer-sha256.json')
    print(json.dumps({'files':len(files), 'archive_bytes':(root/'evidence.tar').stat().st_size,
                      'archive_sha256':hashlib.sha256((root/'evidence.tar').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
