#!/usr/bin/env python3
"""Launch a separately recorded follow-up using the preserved survey sources.

Set numerical-library limits before Python imports NumPy or starts any worker.
The historical experiment, source trees and output directories remain immutable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

THREAD_VARIABLES = (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS',
)


def execution_environment(parent=None):
    environment = dict(os.environ if parent is None else parent)
    environment.update({name: '1' for name in THREAD_VARIABLES})
    environment.update(PYTHONNOUSERSITE='1', PYTHONUTF8='1', LC_ALL='C.UTF-8')
    # Both PyCUDA and the short-prefix compiler invoke nvcc by name.
    environment['PATH'] = '/usr/local/cuda/bin:' + environment.get('PATH', '')
    return environment


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_plan(path, expected):
    path = Path(path).resolve()
    if sha(path) != expected:
        raise ValueError('Follow-up plan changed')
    plan = json.loads(path.read_text())
    root = path.parent
    for relative, digest in plan['files'].items():
        source = root / relative
        if source.is_symlink() or not source.is_file() or sha(source) != digest:
            raise ValueError('Follow-up source/input changed: ' + relative)
    return plan


def launch(plan_path, plan_sha, command, output):
    """Record one attempt, including a failed subprocess; never replace evidence."""
    plan = verify_plan(plan_path, plan_sha)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    environment = execution_environment()
    receipt = dict(plan_sha256=plan_sha, command=command, started_epoch=time.time(),
                   launcher_sha256=sha(__file__),
                   cpu_math_thread_environment={key: environment[key] for key in THREAD_VARIABLES},
                   purpose=plan['purpose'], status='running')
    record = output / 'launch.json'
    record.write_text(json.dumps(receipt, indent=2) + '\n')
    process = None
    try:
        with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
            process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            result = process.wait(timeout=plan['attempt_timeout_seconds'])
        receipt.update(exit_code=result, status='complete' if result == 0 else 'failed')
    except BaseException as error:
        receipt.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        if process is not None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    continue
                # The leader may exit before its worker descendants.
                if sig == signal.SIGKILL:
                    break
        raise
    finally:
        receipt['ended_epoch'] = time.time()
        receipt['elapsed_seconds'] = receipt['ended_epoch'] - receipt['started_epoch']
        record.write_text(json.dumps(receipt, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A child command is required')
    return launch(args.plan, args.plan_sha256, command, args.output)


if __name__ == '__main__':
    sys.exit(main())
