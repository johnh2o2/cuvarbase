"""
Error-handling hygiene: input validation must not be assert-based
(asserts vanish under ``python -O``) and user-facing errors must be
typed (ValueError/RuntimeError/NotImplementedError), not bare
Exception.
"""
import ast
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


def _docstring_nodes(tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, 'body', None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            yield node, first.value


def test_docstrings_with_backslashes_are_raw():
    # A non-raw docstring eats its LaTeX: "\right)" becomes a carriage
    # return, "\times" a tab, and the rendered __doc__ / Sphinx math
    # block is corrupt. Any docstring carrying an un-doubled backslash
    # must be a raw string.
    offenders = []
    for path in _runtime_sources():
        with open(path, encoding='utf-8') as f:
            src = f.read()
        tree = ast.parse(src)
        for node, const in _docstring_nodes(tree):
            seg = ast.get_source_segment(src, const)
            if seg is None or '\\' not in seg:
                continue
            m = re.match(r"^(?P<prefix>[rRbBuUfF]*)"
                         r"(?P<q>\"\"\"|'''|\"|')", seg)
            if m is None or 'r' in m.group('prefix').lower():
                continue
            quote = m.group('q')
            literal = seg[len(m.group(0)):-len(quote)]
            # "\\\\" (an escaped backslash) and a trailing "\\" line
            # continuation are deliberate; anything else is an escape
            # sequence eating the text.
            bare = re.sub(r'\\[\\\n]', '', literal)
            if '\\' in bare:
                offenders.append(
                    "%s:%d (%s)" % (os.path.relpath(path, _PKG_DIR),
                                    const.lineno,
                                    getattr(node, 'name', '<module>')))
    assert not offenders, offenders


def test_nfft_memory_math_block_is_intact():
    # defect 12's documentation half: the NFFTMemory epoch/phase
    # convention is a ".. math::" block, so the docstring must be raw.
    from ..memory.nfft_memory import NFFTMemory
    doc = NFFTMemory.__doc__
    assert '\r' not in doc
    assert r'\exp\left(2\pi i f_k (t_j - \mathrm{epoch})\right)' in doc


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


class TestApiStubsImplemented(object):
    """Public API stubs that raised NotImplementedError are now
    implemented (or behave usefully)."""

    def test_ce_memory_requirement_returns_bytes(self):
        from ..ce import ConditionalEntropyAsyncProcess
        proc = ConditionalEntropyAsyncProcess.__new__(
            ConditionalEntropyAsyncProcess)
        proc.phase_bins, proc.mag_bins = 10, 5
        proc.weighted = False
        proc.real_type = np.float32
        small = proc.memory_requirement(100, 1000)
        large = proc.memory_requirement(100, 100000)
        assert small > 0
        assert large > small
        # histogram-dominated: 100k freqs * 50 bins * 4 bytes = 20 MB
        assert large > 100000 * 50 * 4

    def test_ls_memory_is_ready_raises_runtime_error(self):
        from ..memory.lombscargle_memory import LombScargleMemory
        mem = LombScargleMemory(2, None, 8, use_fft=False)
        with pytest.raises(RuntimeError, match="nf is not set"):
            mem.is_ready()


class TestBatchApiHonesty(object):

    def test_batched_run_const_nfreq_default_batch_size_is_1(self):
        import inspect
        from ..lombscargle import LombScargleAsyncProcess
        sig = inspect.signature(
            LombScargleAsyncProcess.batched_run_const_nfreq)
        assert sig.parameters['batch_size'].default == 1

    def test_batch_kernels_are_cached(self):
        # E1: per-call compilation (~0.6-0.9 s vs 2-10 ms of kernel
        # work) was the whole "batch is ~12x slower at TESS scale"
        # regression; the batch kernel must go through the same LRU
        # cache as the single-LC paths (and the old inefficiency
        # warning is retired).
        from .. import bls
        assert not hasattr(bls, '_warn_if_batch_inefficient')
        fns1 = bls._get_cached_batch_kernels(256)
        fns2 = bls._get_cached_batch_kernels(256)
        assert fns1 is fns2
        assert (256, 'batch') in bls._kernel_cache

    def test_bls_memory_host_array_naming(self):
        from ..bls import BLSMemory
        assert hasattr(BLSMemory, 'allocate_host_arrays')
        # deprecated alias retained for compatibility
        assert hasattr(BLSMemory, 'allocate_pinned_arrays')
        # B3: host arrays are now page-locked (pinned) by default, with a
        # graceful fallback to page-aligned memory if pinning fails. The
        # docstring must reflect that (and no longer claim "NOT page-locked").
        doc = BLSMemory.allocate_host_arrays.__doc__
        assert 'page-locked' in doc
        assert 'NOT page-locked' not in doc
        assert 'fall back' in doc
