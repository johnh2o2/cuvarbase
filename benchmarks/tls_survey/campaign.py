#!/usr/bin/env python3
"""Run a reviewed frozen accuracy campaign; no cloud lifecycle operations.

This operational controller is excluded from the scientific source seal. Every
scientific child checks the sealed sources, and this controller checks them again
at stage boundaries. Restart the same command to resume completed search rows.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'

from common import ROOT, source_identity, sha, write
sys.path.insert(0, str(ROOT))
from run import production_identity


def utc():
    return datetime.now(timezone.utc).isoformat()


def check_manifest(path, split, seal, seal_sha):
    value = json.loads(path.read_text())
    if (value['status'] != 'complete' or value['split'] != split or
            value['seal_sha256'] != seal_sha or value['source_identity'] != seal['source_identity'] or
            value['regimes'] != seal['regimes'] or value['count_per_regime'] != seal['counts'][split]):
        raise ValueError('Existing manifest differs from reviewed design: ' + str(path))
    names = [row['metadata']['name'] for row in value['cases']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate generated inputs')
    for regime in seal['regimes']:
        rows = [row for row in value['cases'] if row['metadata']['regime'] == regime]
        if len(rows) != seal['counts'][split]:
            raise ValueError('Incomplete generated regime: ' + regime)
    return value


def check_result(path, manifest_path, manifest, seal, shard):
    value = json.loads(path.read_text())
    entries = manifest['cases'][shard::seal['execution_shards']]
    expected = {(entry['metadata']['name'], method): entry['sha256'] for entry in entries
                for method in ('tls', seal['bls_selected'][entry['metadata']['regime']]['method'])}
    actual = {(row['name'], row['method']): row['input_sha256'] for row in value['cases']}
    if (value['status'] != 'complete' or value['split'] != manifest['split'] or
            value['manifest_sha256'] != sha(manifest_path) or
            value['production_sources'] != seal['production_sources'] or
            value['runner_sha256'] != sha(ROOT / 'benchmarks/tls_survey/run.py') or
            value['shard_index'] != shard or value['shard_count'] != seal['execution_shards'] or
            len(actual) != len(value['cases']) or actual != expected):
        raise ValueError('Incomplete or mismatched search receipt: ' + str(path))
    return value


class Campaign:
    def __init__(self, args):
        self.args = args
        self.folder = args.work.resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.seal = json.loads(args.seal.read_text())
        if sha(args.seal) != args.seal_sha256:
            raise ValueError('Seal SHA differs from explicitly reviewed SHA')
        self.seal_sha = args.seal_sha256
        self.env = dict(os.environ, LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONUTF8='1',
                        PYTHONPATH=str(ROOT), PATH='/usr/local/cuda/bin:' + os.environ['PATH'])
        self.state_path = self.folder / 'campaign.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else dict(
            created_utc=utc(), seal_sha256=self.seal_sha, stages=[], attempts=[])
        if self.state['seal_sha256'] != self.seal_sha:
            raise ValueError('Cannot reuse work directory for another seal')
        self.children = []

    def save(self):
        write(self.state_path, self.state)

    def guard(self):
        if (sha(self.args.seal) != self.seal_sha or source_identity() != self.seal['source_identity'] or
                production_identity() != self.seal['production_sources']):
            raise ValueError('Scientific or production sources changed after review')

    def commands(self, label, commands):
        self.guard()
        stage = dict(name=label, started_utc=utc(), status='running', workers=[])
        self.state['stages'].append(stage)
        logs = []
        try:
            for index, command in enumerate(commands):
                log_path = self.folder / (label + '-' + str(index) + '.log')
                log = log_path.open('a'); logs.append(log)
                process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                self.children.append(process)
                stage['workers'].append(dict(pid=process.pid, command=command, log=str(log_path)))
            self.save()
            while any(process.poll() is None for process in self.children):
                self.state['heartbeat_utc'] = utc()
                self.save()
                time.sleep(20)
            for process, worker in zip(self.children, stage['workers']):
                worker['exit_code'] = process.wait()
            if any(worker['exit_code'] != 0 for worker in stage['workers']):
                raise RuntimeError('Child failed; preserved all receipts/logs: ' + label)
            stage['status'] = 'complete'
        except BaseException:
            for process in self.children:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process in self.children:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            stage['status'] = 'failed'
            raise
        finally:
            for log in logs:
                log.close()
            self.children = []
            stage['completed_utc'] = utc()
            self.save()

    def python(self, file, *args):
        return [sys.executable, str(ROOT / file), *map(str, args)]

    def generate(self, split):
        self.guard()
        folder = self.folder / ('inputs-' + split)
        manifest_path = folder / 'manifest.json'
        if folder.exists():
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            if manifest.get('status') != 'complete':
                # The generator refuses overwrites. Preserve interruption evidence
                # and repeat its identical predeclared deterministic streams.
                preserved = folder.with_name(folder.name + '.interrupted-' + str(time.time_ns()))
                folder.rename(preserved)
                self.state.setdefault('preserved_interrupted_inputs', []).append(str(preserved))
                self.save()
        if not folder.exists():
            self.commands('generate-' + split, [self.python('benchmarks/tls_survey/generate.py',
                '--split', split, '--count', self.seal['counts'][split], '--regimes',
                ','.join(self.seal['regimes']), '--exposure-nodes', self.seal['exposure_nodes'],
                '--seal', self.args.seal, '--out', folder)])
        check_manifest(manifest_path, split, self.seal, self.seal_sha)
        return manifest_path

    def search(self, split, manifest_path):
        self.guard()
        manifest = check_manifest(manifest_path, split, self.seal, self.seal_sha)
        methods = sorted({'tls'} | {v['method'] for v in self.seal['bls_selected'].values()})
        outputs = [self.folder / (split + '-search-' + str(i) + '.json')
                   for i in range(self.seal['execution_shards'])]
        commands = []
        for shard, output in enumerate(outputs):
            if output.exists() and json.loads(output.read_text()).get('status') == 'complete':
                check_result(output, manifest_path, manifest, self.seal, shard)
                continue
            commands.append(self.python('benchmarks/tls_survey/run.py', '--manifest', manifest_path,
                '--methods', *methods, '--seal', self.args.seal, '--shard-index', shard,
                '--shard-count', self.seal['execution_shards'], '--out', output))
        if commands:
            self.commands('search-' + split, commands)
        for shard, output in enumerate(outputs):
            check_result(output, manifest_path, manifest, self.seal, shard)
        return outputs

    def execute(self):
        self.guard()
        self.state.update(status='running', pid=os.getpid())
        self.state['attempts'].append(dict(started_utc=utc(), controller_sha256=sha(__file__)))
        self.save()
        calibration = self.generate('calibration')
        calibration_results = self.search('calibration', calibration)
        thresholds = self.folder / 'thresholds.json'
        if thresholds.exists():
            value = json.loads(thresholds.read_text())
            if value['seal_sha256'] != self.seal_sha or {
                    row['sha256'] for row in value['receipts']} != {sha(p) for p in calibration_results}:
                raise ValueError('Existing thresholds differ from completed calibration receipts')
        else:
            self.commands('calibrate', [self.python('benchmarks/tls_survey/analyze.py', 'calibrate',
                '--seal', self.args.seal, '--results', *calibration_results, '--out', thresholds)])
        threshold_sha = sha(thresholds)
        if self.state.get('thresholds_sha256', threshold_sha) != threshold_sha:
            raise ValueError('Independently frozen thresholds changed during interruption')
        self.state['thresholds_sha256'] = threshold_sha
        self.save()
        # No test input is generated until independent thresholds have been fixed.
        injections = self.generate('injections')
        nulls = self.generate('nulls')
        injection_results = self.search('injections', injections)
        null_results = self.search('nulls', nulls)
        result = self.folder / 'detection-results.json'
        self.commands('analyze', [self.python('benchmarks/tls_survey/analyze.py', 'analyze',
            '--seal', self.args.seal, '--thresholds', thresholds, '--injections', *injection_results,
            '--nulls', *null_results, '--out', result)])
        if self.args.export_bank:
            bank = self.folder / 'input-bank'
            if not bank.exists():
                self.commands('export-bank', [self.python('benchmarks/tls_reference/inputs.py', 'export',
                    '--study', 'calibration', calibration, '--study', 'injections', injections,
                    '--study', 'nulls', nulls, '--out', bank)])
            self.commands('verify-bank', [self.python('benchmarks/tls_reference/inputs.py', 'verify', '--bank', bank)])
        self.state.update(status='complete', completed_utc=utc(), result_sha256=sha(result))
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument('--seal-sha256', required=True, help='Explicit SHA reviewed before held-out execution')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--export-bank', action='store_true')
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    with (args.work / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = Campaign(args)
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Controller received signal ' + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        try:
            campaign.execute()
        except BaseException:
            campaign.state.update(status='failed', completed_utc=utc(), error=traceback.format_exc())
            campaign.save()
            raise


if __name__ == '__main__':
    main()
