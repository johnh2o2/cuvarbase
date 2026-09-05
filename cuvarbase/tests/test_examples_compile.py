"""Compile-check the shipped examples and notebook code cells.

``examples/*.py`` and the code cells of ``notebooks/*.ipynb`` are never
executed by the suite (they need a device and, for the notebooks, a
kernel); this is the cheap half of release finding 78: every example
and every notebook cell must at least ``compile()`` cleanly with
warnings turned into errors (a SyntaxWarning such as an invalid escape
sequence fails the test). IPython line/cell magics and shell escapes
(``%matplotlib inline``, ``!pip install``) are stripped first.

Skips when the ``examples/`` or ``notebooks/`` directory is absent
(installed wheel, ``pytest --pyargs cuvarbase``).
"""
import glob
import json
import os
import warnings

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_EXAMPLES = os.path.join(_REPO_ROOT, 'examples')
_NOTEBOOKS = os.path.join(_REPO_ROOT, 'notebooks')


def _compile_strict(src, filename):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        compile(src, filename, 'exec')


def _strip_magics(src):
    """Drop IPython magics / shell escapes; keep line numbers stable."""
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('%') or stripped.startswith('!'):
            out.append('')
        else:
            out.append(line)
    return '\n'.join(out) + '\n'


def _example_files():
    if not os.path.isdir(_EXAMPLES):
        return []
    return sorted(glob.glob(os.path.join(_EXAMPLES, '*.py')))


def _notebook_files():
    if not os.path.isdir(_NOTEBOOKS):
        return []
    return sorted(glob.glob(os.path.join(_NOTEBOOKS, '*.ipynb')))


def _ids(paths):
    return [os.path.basename(p) for p in paths]


_EXAMPLE_FILES = _example_files()
_NOTEBOOK_FILES = _notebook_files()

# Notebooks with a KNOWN compile warning, pending a fix outside the test
# suite. Phase 3 (Sep 2026) found two non-raw matplotlib label strings
# in the PDM notebook's first two code cells -- '$1-\Theta(f)$' and a
# '\c...' TeX macro -- i.e. "invalid escape sequence \T / \c"
# (a DeprecationWarning on 3.9, a SyntaxWarning on 3.12+, a SyntaxError
# in a future Python). The fix is to make those strings raw
# (r'$1-\Theta(f)$'). The entry is strict: once the notebook is fixed
# this xfail turns into a failure and must be deleted.
KNOWN_ESCAPE_OFFENDERS = {
    'Phase Dispersion Minimization.ipynb',
}
_NOTEBOOK_PARAMS = [
    pytest.param(p, marks=pytest.mark.xfail(
        strict=True, raises=SyntaxError,
        reason="known non-raw TeX label strings (see "
               "KNOWN_ESCAPE_OFFENDERS)"))
    if os.path.basename(p) in KNOWN_ESCAPE_OFFENDERS else p
    for p in _NOTEBOOK_FILES]


def test_examples_directory_present_or_installed():
    # In the source tree both directories exist and are non-empty; from
    # an installed wheel neither does and the parametrized tests below
    # are skipped (they parametrize over an empty list).
    if not os.path.isdir(_EXAMPLES) and not os.path.isdir(_NOTEBOOKS):
        pytest.skip("examples/ and notebooks/ not found (running outside "
                    "the source tree)")
    assert _EXAMPLE_FILES or _NOTEBOOK_FILES


@pytest.mark.parametrize('path', _EXAMPLE_FILES, ids=_ids(_EXAMPLE_FILES))
def test_example_compiles_without_warnings(path):
    with open(path, encoding='utf-8') as f:
        src = f.read()
    _compile_strict(src, path)


@pytest.mark.parametrize('path', _NOTEBOOK_PARAMS,
                         ids=_ids(_NOTEBOOK_FILES))
def test_notebook_code_cells_compile_without_warnings(path):
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']
    if not cells:
        pytest.skip("%s has no code cells" % os.path.basename(path))
    n_checked = 0
    for i, cell in enumerate(cells):
        src = cell.get('source', '')
        if isinstance(src, list):
            src = ''.join(src)
        src = _strip_magics(src)
        if not src.strip():
            continue
        _compile_strict(src, '%s[cell %d]' % (os.path.basename(path), i))
        n_checked += 1
    assert n_checked > 0
