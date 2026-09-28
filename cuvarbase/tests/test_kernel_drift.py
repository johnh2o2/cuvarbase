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
- the intentionally-divergent functions still live in each .cu file,
- and any function name defined in BOTH .cu files (a helper duplicated
  instead of moved to the header) must have an identical normalized
  body -- the cross-file comparison the Jul-2026 audit found missing:
  without it, a same-name helper added to both files could drift again
  exactly like the original reduction_max bug. Only ``reduction_max``
  itself is exempt (divergent by design).

The last two checks are generalized to EVERY kernel file
(``kernels/*.cu`` and ``*.cuh``): a ``#define NAME value`` that appears
in more than one file must carry the same value(s) everywhere, and a
``__device__``/``__global__`` function defined in more than one file
must have one body, with the intentionally divergent copies listed
explicitly. The Sep 2026 audit (defect 20) found
``sparse_bls_simple.cu`` still carrying ``MAX_W_COMPLEMENT 1E-9`` after
PR #65 had set 1E-4 in ``sparse_bls.cu`` (powers up to 4.6 in pure
noise on the opt-in kernel); the define check would have caught it.

The function extractor accepts a definition wherever its qualifier
appears on the line -- indented, or behind ``extern "C"``, ``static``,
``inline``, ``__forceinline__`` or a one-line ``template<...>`` -- so
the ``extern "C" __global__`` TLS kernels are covered too (the
column-0 anchor the guard first shipped with skipped them); a name
defined more than once in one file keeps every body. Both guards are
proved to bite on a mutated scratch copy of the kernel directory
(:func:`test_guards_bite_on_a_mutated_copy`).
"""
import glob
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

# Cross-file duplicated functions whose divergence is intentional: the
# named FILES hold a sanctioned variant and are excluded from the
# body comparison for that name; every other copy must still be
# identical. Keep this list short and justified.
INTENTIONALLY_DIVERGENT_COPIES = {
    # full tree reduction (bls.cu) vs tree-to-warp + shuffle
    # (bls_optimized.cu); see INTENTIONALLY_DIVERGENT above
    'reduction_max': {'bls.cu', 'bls_optimized.cu'},
    # cunfft.cu const-qualifies the parameters (CONSTANT int); the
    # arithmetic is the same as bls_common.cuh's mod()
    'mod': {'cunfft.cu'},
    # ce.cu is the FLT (float-or-double) variant using floor();
    # tls.cu declares it inline with a different parameter name. The
    # float copies in bls_common.cuh and sparse_bls.cu must agree.
    'mod1': {'ce.cu', 'tls.cu'},
}

# #define names whose values legitimately differ between files (none
# today: MIN_W was a dead define in bls.cu/bls_optimized.cu -- the
# shared bls_value uses literals -- and was deleted rather than
# whitelisted). Map name -> set of files allowed to disagree.
INTENTIONALLY_DIVERGENT_DEFINES = {}

# An INCLUDE directive standing on its own line (the form _module_reader
# expands). Anchored so it ignores prose that merely mentions the
# directive inside a comment.
_DIRECTIVE_LINE = re.compile(r"^[ \t]*//\{INCLUDE\s", re.M)


def _common_path():
    return os.path.join(os.path.dirname(find_kernel('bls')),
                        'bls_common.cuh')


# A __device__/__global__ definition (or prototype) header. The
# qualifier may be indented and may follow ``extern "C"``, ``static``,
# ``inline``, ``__forceinline__``/``__noinline__`` or a one-line
# ``template<...>``, and several CUDA qualifiers may be chained
# (``__host__ __device__``). ``[^\n{;]*?`` keeps the match on one
# line and stops at a body or a prototype's ``;``.
_FUNC_DEF = re.compile(
    r"^[ \t]*(?:(?:extern\s+\"C\"|static|inline|__forceinline__|"
    r"__noinline__|template\s*<[^>\n]*>)\s+)*"
    r"__(?:device|global|host)__"
    r"(?:\s+__(?:device|global|host|forceinline__|noinline__)__)*"
    r"[^\n{;]*?(\w+)\s*\(", re.M)


def _func_names(src):
    """Names of every __device__/__global__ function defined in ``src``."""
    return set(_FUNC_DEF.findall(src))


def _strip_comments(src):
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    return src


def _func_bodies(src):
    """Map name -> frozenset of normalized sources (signature +
    brace-matched body) for every __device__/__global__ function
    defined in ``src``. A name defined more than once in the file
    (e.g. in both branches of an ``#ifdef``) keeps every body;
    prototypes (``;`` before any ``{``) are skipped."""
    src = _strip_comments(src)
    bodies = {}
    for m in _FUNC_DEF.finditer(src):
        name = m.group(1)
        open_brace = src.find('{', m.end())
        semicolon = src.find(';', m.end())
        if open_brace < 0 or 0 <= semicolon < open_brace:
            continue
        depth, i = 1, open_brace + 1
        while i < len(src) and depth:
            if src[i] == '{':
                depth += 1
            elif src[i] == '}':
                depth -= 1
            i += 1
        # normalize whitespace so formatting-only differences don't count
        text = ' '.join(src[m.start():i].split())
        bodies[name] = bodies.get(name, frozenset()) | {text}
    return bodies


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


def test_no_cross_file_drift_of_duplicated_functions():
    # A helper defined in BOTH .cu files (rather than moved into
    # bls_common.cuh) is a fresh drift hazard the header mechanism
    # cannot see. Any such duplicate must be byte-identical after
    # comment stripping + whitespace normalization. reduction_max is
    # the one sanctioned divergence (tree reduction vs warp shuffle).
    std = _func_bodies(open(find_kernel('bls')).read())
    opt = _func_bodies(open(find_kernel('bls_optimized')).read())

    duplicated = (set(std) & set(opt)) - {'reduction_max'}
    drifted = sorted(name for name in duplicated
                     if std[name] != opt[name])
    assert not drifted, (
        "function(s) %s are defined in BOTH bls.cu and bls_optimized.cu "
        "with differing bodies -- move the shared implementation into "
        "bls_common.cuh (or, if the divergence is intentional, rename "
        "or whitelist it here) so the copies cannot silently drift"
        % drifted)

    # the guard itself must stay exercised: reduction_max is the known
    # duplicated-and-divergent pair, so the extractor must see it in
    # both files (guards against the regex/brace-matcher going stale)
    assert 'reduction_max' in std and 'reduction_max' in opt
    assert std['reduction_max'] != opt['reduction_max']


# --------------------------------------------------------------------
# all kernel files
# --------------------------------------------------------------------

def _all_kernel_files():
    kdir = os.path.dirname(find_kernel('bls'))
    files = sorted(glob.glob(os.path.join(kdir, '*.cu'))
                   + glob.glob(os.path.join(kdir, '*.cuh')))
    assert len(files) >= 10, files
    return files


_DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+(?:\([^)]*\))?)"
                     r"(?:[ \t]+(.*?))?[ \t]*$", re.M)


def _defines(src):
    """name -> set of values defined for it in ``src`` (a name defined
    in both branches of an #ifdef, e.g. FLT double/float, yields both
    values; the SET must then agree across files)."""
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    out = {}
    for m in _DEFINE.finditer(src):
        value = re.sub(r"//.*$", "", m.group(2) or "").strip()
        out.setdefault(m.group(1), set()).add(' '.join(value.split()))
    return out


def _define_drift(paths):
    """(drifted, per_name) for the #defines of ``paths``: ``drifted``
    lists ``(name, {file: sorted values})`` for every name whose value
    set differs between two non-whitelisted files."""
    per_name = {}
    for path in paths:
        for name, values in _defines(open(path).read()).items():
            per_name.setdefault(name, {})[os.path.basename(path)] = values

    drifted = []
    for name, per_file in sorted(per_name.items()):
        files = {f: v for f, v in per_file.items()
                 if f not in INTENTIONALLY_DIVERGENT_DEFINES.get(name, ())}
        if len(files) < 2:
            continue
        if len(set(frozenset(v) for v in files.values())) > 1:
            drifted.append((name, {f: sorted(v) for f, v in files.items()}))
    return drifted, per_name


def _function_drift(paths):
    """(drifted, per_name) for the __device__/__global__ functions of
    ``paths``: ``drifted`` lists ``(name, sorted files)`` for every name
    whose bodies differ between two non-whitelisted files."""
    per_name = {}
    for path in paths:
        for name, body in _func_bodies(open(path).read()).items():
            per_name.setdefault(name, {})[os.path.basename(path)] = body

    drifted = []
    for name, per_file in sorted(per_name.items()):
        copies = {f: b for f, b in per_file.items()
                  if f not in INTENTIONALLY_DIVERGENT_COPIES.get(name, ())}
        if len(copies) >= 2 and len(set(copies.values())) > 1:
            drifted.append((name, sorted(copies)))
    return drifted, per_name


def test_same_named_defines_agree_across_all_kernel_files():
    drifted, per_name = _define_drift(_all_kernel_files())
    assert not drifted, (
        "#define(s) with different values in different kernel files "
        "(the MAX_W_COMPLEMENT 1E-9 vs 1E-4 drift of sparse_bls_simple.cu "
        "was exactly this): %s -- use one value, or move the constant "
        "into a shared header" % drifted)

    # the guard itself must see the shared constants it protects
    assert 'MAX_W_COMPLEMENT' in per_name and 'RESTRICT' in per_name
    assert len(per_name['RESTRICT']) >= 5


def test_no_cross_file_drift_of_duplicated_functions_in_any_kernel():
    drifted, per_name = _function_drift(_all_kernel_files())
    assert not drifted, (
        "function(s) defined in several kernel files with differing "
        "bodies: %s -- share one implementation (bls_common.cuh-style "
        "include) or list the sanctioned variant in "
        "INTENTIONALLY_DIVERGENT_COPIES with a reason" % drifted)

    # every whitelisted entry must still correspond to a real duplicate
    # (a stale whitelist would hide a future rename)
    for name, files in INTENTIONALLY_DIVERGENT_COPIES.items():
        assert name in per_name and len(per_name[name]) >= 2, name
        assert files <= set(per_name[name]), (name, files,
                                              sorted(per_name[name]))
    # the extractor sees the known duplicates
    assert {'get_id', 'mod1', 'atomicAddDouble'} <= set(
        n for n, d in per_name.items() if len(d) >= 2)
    # ... and the ``extern "C" __global__`` kernels the column-0 anchor
    # of the first version of this guard could not see (review finding
    # on 398cd60): a copy of one of these drifting in another file must
    # be caught like any other.
    assert set(per_name['tls_search_kernel']) == {'tls.cu'}
    assert set(per_name['tls_search_kernel_keplerian']) == {'tls.cu'}
    assert set(per_name['tls_fast_search_kernel']) == {'tls_fast.cu'}
    assert set(per_name['tls_refine_kernel']) == {'tls_fast.cu'}
    # every name is a definition, never a prototype: each body is braced
    for name, per_file in per_name.items():
        for file, bodies in per_file.items():
            assert all(b.endswith('}') for b in bodies), (name, file)


def test_extractor_accepts_prefixed_and_indented_qualifiers():
    src = """
extern "C" __global__ void k_extern(int a) { return; }
    __device__ int k_indented(int a) { return a; }
static __device__ __forceinline__ float k_static(float x) { return x; }
inline __device__ float k_inline(float x) { return x; }
template <typename T> __device__ T k_template(T x) { return x; }
__host__ __device__ int k_host_device(int a) { return a; }
__device__ int k_prototype(int a);
"""
    names = _func_names(src)
    assert names == {'k_extern', 'k_indented', 'k_static', 'k_inline',
                     'k_template', 'k_host_device', 'k_prototype'}
    bodies = _func_bodies(src)
    assert set(bodies) == names - {'k_prototype'}   # prototype skipped
    assert bodies['k_extern'] == {
        'extern "C" __global__ void k_extern(int a) { return; }'}


def test_guards_bite_on_a_mutated_copy(tmp_path):
    """Proof that both all-kernel guards detect real drift: on a scratch
    copy of the kernel directory, re-create defect 20 (a second sparse
    kernel file whose ``MAX_W_COMPLEMENT`` disagrees) and add a
    divergent copy of an ``extern "C" __global__`` kernel (indented,
    to exercise both blind spots of the original extractor) and check
    that exactly those two names are reported."""
    import shutil
    live = _all_kernel_files()
    copies = []
    for path in live:
        dst = tmp_path / os.path.basename(path)
        shutil.copy(path, dst)
        copies.append(str(dst))
    assert not _define_drift(copies)[0]
    assert not _function_drift(copies)[0]

    # 1. the sparse_bls_simple.cu drift of defect 20, re-created: a
    #    second file with the same functions but the stale define
    src = (tmp_path / 'sparse_bls.cu').read_text()
    assert re.search(r'^#define MAX_W_COMPLEMENT 1E-4$', src, re.M)
    simple = tmp_path / 'sparse_bls_simple.cu'
    simple.write_text(re.sub(r'^(#define MAX_W_COMPLEMENT )\S+$',
                             r'\g<1>1E-9', src, count=1, flags=re.M))
    copies.append(str(simple))
    drifted, _ = _define_drift(copies)
    assert [name for name, _ in drifted] == ['MAX_W_COMPLEMENT'], drifted
    # identical function copies are not drift
    assert not _function_drift(copies)[0]

    # 2. a divergent copy of an extern "C" kernel in another file
    (body,) = _func_bodies((tmp_path / 'tls.cu').read_text())[
        'tls_search_kernel']
    assert body.startswith('extern "C" __global__ void tls_search_kernel(')
    mutant = body[:-1] + ' int drift_mutant = 1; }'
    with open(tmp_path / 'tls_fast.cu', 'a') as f:
        f.write('\n    ' + mutant + '\n')
    drifted, per_name = _function_drift(copies)
    assert [name for name, _ in drifted] == ['tls_search_kernel'], drifted
    assert set(per_name['tls_search_kernel']) == {'tls.cu', 'tls_fast.cu'}
