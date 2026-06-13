"""
Single-source guard for the BLS kernel files.

``bls.cu`` and ``bls_optimized.cu`` share most of their device/global
functions. The duplication once shipped a silent-wrong-results bug (the
``reduction_max`` s>32 candidate drop was originally fixed in only one
copy -- commit 77b4333), so the shared functions now live in a single
file, ``bls_common.cuh``, which both kernels inline via the
``//{INCLUDE bls_common.cuh}`` directive (expanded by
``utils._module_reader`` at load time).

These tests assert the include mechanism instead of comparing two copies:
- both kernels carry the include directive,
- the shared functions are defined once (in bls_common.cuh) and never
  redefined in either .cu file, so drift is structurally impossible,
- the directive really expands (so both kernels see the shared bodies),
- the intentionally-divergent functions still live in each .cu file.
"""
import os
import re

from cuvarbase.utils import find_kernel, _module_reader

# Functions that are *supposed* to differ between the two files and so
# stay out of the shared header.
INTENTIONALLY_DIVERGENT = {
    # full tree reduction (bls.cu) vs tree-to-warp + shuffle
    # (bls_optimized.cu, including the s >= 32 fix from 72ae029)
    'reduction_max',
    # interleaved [yw, w] shared layout (bls.cu) vs separate arrays
    # (bls_optimized.cu)
    'full_bls_no_sol',
    'full_bls_no_sol_optimized',
}

INCLUDE_DIRECTIVE = '//{INCLUDE bls_common.cuh}'

# An INCLUDE directive standing on its own line (the form _module_reader
# expands). Anchored so it ignores prose that merely mentions the
# directive inside a comment.
_DIRECTIVE_LINE = re.compile(r"^[ \t]*//\{INCLUDE\s", re.M)


def _common_path():
    return os.path.join(os.path.dirname(find_kernel('bls')),
                        'bls_common.cuh')


def _func_names(src):
    """Names of every __device__/__global__ function defined in ``src``."""
    return set(re.findall(
        r"^__(?:device|global)__[^\n]*?(\w+)\s*\(", src, re.M))


def test_both_kernels_inline_the_shared_header():
    for name in ('bls', 'bls_optimized'):
        raw = open(find_kernel(name)).read()
        assert _DIRECTIVE_LINE.search(raw), (
            "%s.cu must inline the shared functions via %r on its own line"
            % (name, INCLUDE_DIRECTIVE))


def test_shared_functions_defined_once_in_common_header():
    common = open(_common_path()).read()
    shared = _func_names(common)
    # the shared surface should not silently shrink
    assert len(shared) >= 12, sorted(shared)

    # none of the shared functions may be redefined in either .cu file --
    # that is the only way they could drift again.
    for name in ('bls', 'bls_optimized'):
        local = _func_names(open(find_kernel(name)).read())
        clash = shared & local
        assert not clash, (
            "%s.cu redefines shared function(s) %s already provided by "
            "bls_common.cuh -- delete the local copy so they cannot drift"
            % (name, sorted(clash)))


def test_include_directive_expands_shared_bodies():
    # _module_reader must inline the header so nvcc sees the shared
    # bodies; check a representative shared function appears in both
    # assembled sources.
    shared = _func_names(open(_common_path()).read())
    for name in ('bls', 'bls_optimized'):
        assembled = _module_reader(find_kernel(name),
                                   cpp_defs=dict(BLOCK_SIZE=256))
        assert not _DIRECTIVE_LINE.search(assembled), (
            "%s: include directive line was not expanded" % name)
        missing = shared - _func_names(assembled)
        assert not missing, (
            "%s: shared function(s) %s missing after include expansion"
            % (name, sorted(missing)))


def test_intentionally_divergent_functions_live_in_the_cu_files():
    std = _func_names(open(find_kernel('bls')).read())
    opt = _func_names(open(find_kernel('bls_optimized')).read())
    assert 'reduction_max' in std and 'reduction_max' in opt
    assert 'full_bls_no_sol' in std
    assert 'full_bls_no_sol_optimized' in opt
