"""
Error-handling hygiene: input validation must not be assert-based
(asserts vanish under ``python -O``) and user-facing errors must be
typed (ValueError/RuntimeError/NotImplementedError), not bare
Exception.
"""
import os
import re
import subprocess
import sys

import numpy as np
import pytest


_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _runtime_sources():
    for root, dirs, files in os.walk(_PKG_DIR):
        if 'tests' in root:
            continue
        for f in files:
            if f.endswith('.py'):
                yield os.path.join(root, f)


def test_no_assert_based_validation_in_runtime_modules():
    offenders = []
    for path in _runtime_sources():
        for i, line in enumerate(open(path), 1):
            if re.match(r"^\s*assert[ (]", line):
                offenders.append("%s:%d" % (os.path.relpath(path), i))
    assert not offenders, offenders


def test_no_bare_exception_raises():
    offenders = []
    for path in _runtime_sources():
        for i, line in enumerate(open(path), 1):
            if 'raise Exception' in line:
                offenders.append("%s:%d" % (os.path.relpath(path), i))
    assert not offenders, offenders


def test_check_k0_raises_value_error():
    from ..lombscargle import check_k0
    # freqs[0] far from any integer multiple of df
    bad = 0.05 + 0.1 * np.arange(10) + 0.033
    with pytest.raises(ValueError, match="k0"):
        check_k0(bad)


def test_check_k0_survives_python_O():
    # Under -O an assert-based check silently disappears; the
    # validation must still raise.
    repo_root = os.path.dirname(_PKG_DIR)
    script = (
        "import numpy as np\n"
        "import conftest  # install GPU stubs\n"
        "from cuvarbase.lombscargle import check_k0\n"
        "bad = 0.05 + 0.1 * np.arange(10) + 0.033\n"
        "try:\n"
        "    check_k0(bad)\n"
        "except ValueError:\n"
        "    print('OK')\n"
        "else:\n"
        "    raise SystemExit('check_k0 validated nothing under -O')\n"
    )
    result = subprocess.run([sys.executable, '-O', '-c', script],
                            cwd=repo_root, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout


def test_eebls_transit_qvals_without_freqs_value_error():
    from ..bls import eebls_transit, eebls_transit_gpu
    t = np.linspace(0, 10, 50)
    y = np.ones(50)
    dy = np.ones(50)
    for func in (eebls_transit, eebls_transit_gpu):
        with pytest.raises(ValueError, match="qvals"):
            func(t, y, dy, qvals=np.array([0.01]))


def test_ce_memory_unsupported_combos_value_error():
    from ..memory.ce_memory import ConditionalEntropyMemory
    with pytest.raises(ValueError, match="balanced_magbins"):
        ConditionalEntropyMemory(weighted=True, balanced_magbins=True)
    with pytest.raises(ValueError, match="compute_log_prob"):
        ConditionalEntropyMemory(weighted=True, compute_log_prob=True)
