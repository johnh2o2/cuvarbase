"""
Orphan-kernel guard: the packaged CUDA sources and the Python loaders
must agree.

``cuvarbase/kernels/wavelet.cu`` shipped in every 0.2.x wheel although
nothing loaded it (Sep 2026 audit, findings 19/61/88). This test keeps
that from happening again, in both directions:

- every ``kernels/*.cu`` stem must appear as a quoted string literal in
  some ``cuvarbase/*.py`` module, i.e. something hands it to
  :func:`cuvarbase.utils.find_kernel` (``bls.py``/``tls.py`` pick the
  stem into a local first, so the literal is what is checked, not the
  ``find_kernel('...')`` call form);
- every ``find_kernel('<stem>')`` literal must resolve to an existing
  ``kernels/<stem>.cu`` file;
- every ``//{INCLUDE <file>}`` directive in a kernel must resolve, and
  every packaged ``*.cuh`` must be included by at least one kernel.

The checks read the installed package (``cuvarbase.__file__``), so they
run unchanged under ``pytest --pyargs cuvarbase`` against a wheel and
double as a package-data check there. Pure CPU; no pycuda needed.
"""
import glob
import os
import re

import cuvarbase
import cuvarbase.utils as utils

PKG_DIR = os.path.dirname(os.path.abspath(cuvarbase.__file__))
KERNEL_DIR = os.path.join(PKG_DIR, 'kernels')

# ``find_kernel('stem')`` / ``find_kernel("stem")`` with a literal argument.
_FIND_KERNEL_LITERAL = re.compile(r"""find_kernel\(\s*['"]([A-Za-z0-9_]+)['"]\s*\)""")

# Stems that are loaded via ``find_kernel``, as of 1.0.0. Kept explicit so
# a renamed kernel file shows up as a failure with a clear message rather
# than as a silent change in the inventory.
EXPECTED_STEMS = {
    'bls', 'bls_optimized', 'bls_batch', 'sparse_bls',
    'ce', 'cunfft', 'lomb', 'nufft_lrt', 'pdm',
    'tls', 'tls_fast',
}


def _package_sources():
    """(path, text) for every .py module in the package (tests excluded)."""
    out = []
    for path in sorted(glob.glob(os.path.join(PKG_DIR, '*.py'))
                       + glob.glob(os.path.join(PKG_DIR, '*', '*.py'))):
        if os.sep + 'tests' + os.sep in path:
            continue
        with open(path, 'r') as f:
            out.append((path, f.read()))
    return out


def _kernel_stems():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(KERNEL_DIR, '*.cu')))


def _header_names():
    return sorted(os.path.basename(p)
                  for p in glob.glob(os.path.join(KERNEL_DIR, '*.cuh')))


def test_kernels_directory_is_packaged():
    assert os.path.isdir(KERNEL_DIR), KERNEL_DIR
    assert _kernel_stems(), "no *.cu files packaged in %s" % KERNEL_DIR


def test_kernel_inventory_matches_expected():
    assert set(_kernel_stems()) == EXPECTED_STEMS, (
        "kernels/*.cu inventory changed; update EXPECTED_STEMS (and the "
        "loader) deliberately. packaged=%r" % _kernel_stems())


def test_every_kernel_file_is_referenced_by_a_loader():
    """No orphan kernels: each *.cu stem is a quoted literal in cuvarbase/*.py."""
    sources = _package_sources()
    orphans = []
    for stem in _kernel_stems():
        literal = re.compile(r"""['"]%s['"]""" % re.escape(stem))
        if not any(literal.search(text) for _, text in sources):
            orphans.append(stem)
    assert not orphans, (
        "kernel file(s) shipped but never loaded (no quoted %r literal in "
        "any cuvarbase/*.py): %r" % ('<stem>', orphans))


def test_every_find_kernel_literal_has_a_file():
    """Every find_kernel('<stem>') literal resolves to a packaged file."""
    literals = set()
    for path, text in _package_sources():
        literals.update(_FIND_KERNEL_LITERAL.findall(text))
    assert literals, "no find_kernel('...') literals found in the package"
    missing = [stem for stem in sorted(literals)
               if not os.path.isfile(utils.find_kernel(stem))]
    assert not missing, "find_kernel literal(s) without a .cu file: %r" % missing
    assert literals <= EXPECTED_STEMS, (
        "find_kernel literal(s) not in EXPECTED_STEMS: %r"
        % sorted(literals - EXPECTED_STEMS))


def test_find_kernel_resolves_every_expected_stem():
    for stem in sorted(EXPECTED_STEMS):
        path = utils.find_kernel(stem)
        assert os.path.isfile(path), path
        assert os.path.getsize(path) > 0, path


def test_include_directives_resolve_and_headers_are_used():
    """//{INCLUDE x} targets exist; every packaged .cuh is included somewhere."""
    included = set()
    for path in glob.glob(os.path.join(KERNEL_DIR, '*.cu')):
        with open(path, 'r') as f:
            text = f.read()
        for target in utils._INCLUDE_RE.findall(text):
            included.add(target)
            assert os.path.isfile(os.path.join(KERNEL_DIR, target)), (
                "%s includes missing file %s" % (os.path.basename(path), target))
    unused = sorted(set(_header_names()) - included)
    assert not unused, "packaged .cuh never //{INCLUDE}d by any kernel: %r" % unused
