"""
Kernel-drift guard for the duplicated BLS kernel files.

``bls.cu`` and ``bls_optimized.cu`` share most of their device/global
functions. The duplication already shipped one silent-wrong-results
bug (the ``reduction_max`` s>32 candidate drop was originally fixed in
only one copy — commit 77b4333), so this test fails whenever a
shared-name function is edited in one file but not the other.

Intentional differences are normalized away (comments, whitespace,
float-literal suffixes, the ``mod1`` vs ``mod1_fast`` name) or
whitelisted (``reduction_max``: the standard file uses a full tree
reduction while the optimized file reduces to warp level and finishes
with shuffles — different strategies, both correct).
"""
import re

from cuvarbase.utils import find_kernel

# Functions that are *supposed* to differ between the two files.
INTENTIONALLY_DIVERGENT = {
    # full tree reduction (bls.cu) vs tree-to-warp + shuffle
    # (bls_optimized.cu, including the s >= 32 fix from 72ae029)
    'reduction_max',
}


def _extract_functions(path):
    src = open(path).read()
    funcs = {}
    for m in re.finditer(
            r"^__(?:device|global)__[^\n]*?(\w+)\s*\(", src, re.M):
        name = m.group(1)
        i = src.index('{', m.start())
        depth, j = 1, i + 1
        while depth and j < len(src):
            if src[j] == '{':
                depth += 1
            elif src[j] == '}':
                depth -= 1
            j += 1
        funcs[name] = src[m.start():j]
    return funcs


def _normalize(body):
    # comments
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    # the optimized file uses mod1_fast where bls.cu uses mod1
    body = body.replace("mod1_fast", "mod1")
    # float-literal cosmetics: 1e-10f == 1e-10, 0.f == 0. == 0
    body = re.sub(r"(?<=[\d.])f\b", "", body)
    body = re.sub(r"(\d+)\.(?=\s|\)|,|;|/| )", r"\1", body)
    return re.sub(r"\s+", " ", body).strip()


def test_shared_bls_kernel_functions_do_not_drift():
    std = _extract_functions(find_kernel('bls'))
    opt = _extract_functions(find_kernel('bls_optimized'))

    shared = sorted((set(std) & set(opt)) - INTENTIONALLY_DIVERGENT)
    # the shared surface itself should not silently shrink
    assert len(shared) >= 12, shared

    drifted = [name for name in shared
               if _normalize(std[name]) != _normalize(opt[name])]
    assert not drifted, (
        "shared kernel function(s) %s differ between bls.cu and "
        "bls_optimized.cu beyond the normalized cosmetics — apply "
        "the change to both copies (precedent: the reduction_max "
        "s>32 bug was fixed in only one copy)" % drifted)


def test_intentionally_divergent_functions_exist_in_both():
    std = _extract_functions(find_kernel('bls'))
    opt = _extract_functions(find_kernel('bls_optimized'))
    for name in INTENTIONALLY_DIVERGENT:
        assert name in std and name in opt
