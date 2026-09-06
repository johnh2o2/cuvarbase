"""CI packaging smoke test for the *installed* wheel (or sdist).

Run from a clean environment where cuvarbase was installed from the built
artifact (pip install --no-deps dist/*.whl), so pycuda is genuinely absent.
Three things are checked:

1. ``import cuvarbase`` requires neither pycuda nor a CUDA context (the
   primary context is created lazily on first GPU use, not at import).
2. With pycuda stubbed, every submodule in ``cuvarbase._SUBMODULES`` and
   the shipped ``cuvarbase.tests`` package import -- catching
   missing-subpackage bugs that source-tree testing hides (e.g. the v1.0
   wheel that omitted cuvarbase.base/cuvarbase.memory entirely).
3. Every kernel stem the package hands to ``find_kernel('...')`` resolves
   to a packaged ``kernels/<stem>.cu`` file, every ``//{INCLUDE x}``
   directive resolves, and the packaged inventory matches the loaders
   (a shared .cuh left out of package-data, or an orphan kernel, fails
   here).
"""
import glob
import importlib
import os
import re
import sys
import types

# Make sure we import the installed package, not the source tree.
sys.path = [p for p in sys.path if os.path.abspath(p) != os.getcwd()]

# --- Part 1: GPU-less, pycuda-less import ---------------------------------
# pycuda is not installed in this venv, so a successful import proves the
# package top-level does not import it (no eager CUDA context).
import cuvarbase  # noqa: E402
assert 'pycuda' not in sys.modules, \
    "import cuvarbase pulled in pycuda -- the CUDA context is no longer " \
    "supposed to be created at import time"
pkg_dir = os.path.dirname(os.path.abspath(cuvarbase.__file__))
# The package must come from the venv's site-packages, not from the
# source checkout this script lives in (a plain substring test against
# the cwd misfires whenever the venv happens to sit below the cwd).
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assert not pkg_dir.startswith(_repo_root + os.sep), \
    "cuvarbase imported from the source checkout %s, not the installed " \
    "package" % _repo_root
print('GPU-less import OK:', cuvarbase.__version__, 'from', pkg_dir)

# --- Part 2: stubbed-pycuda deep import of every submodule -----------------
for name in ['pycuda', 'pycuda.autoprimaryctx', 'pycuda.autoinit',
             'pycuda.driver', 'pycuda.gpuarray', 'pycuda.compiler',
             'pycuda.tools']:
    sys.modules[name] = types.ModuleType(name)
sys.modules['pycuda.compiler'].SourceModule = object
sys.modules['pycuda.tools'].context_dependent_memoize = lambda f: f
sys.modules['pycuda.tools'].mark_cuda_test = lambda f: f

submodules = sorted(cuvarbase._SUBMODULES)
assert submodules, "cuvarbase._SUBMODULES is empty"
for name in submodules:
    importlib.import_module('cuvarbase.' + name)
print('submodules import OK (%d): %s' % (len(submodules), ', '.join(submodules)))

import cuvarbase.tests  # noqa: E402
tests_dir = os.path.dirname(os.path.abspath(cuvarbase.tests.__file__))
n_tests = len(glob.glob(os.path.join(tests_dir, 'test_*.py')))
assert n_tests > 0, "cuvarbase.tests ships no test_*.py modules"
print('cuvarbase.tests OK (%d test modules)' % n_tests)

from cuvarbase.base import GPUAsyncProcess, ensure_context  # noqa: E402, F401
from cuvarbase.memory import BLSBatchMemory  # noqa: E402, F401
import cuvarbase.utils  # noqa: E402

# --- Part 3: packaged kernels match the loaders ---------------------------
find_kernel_literal = re.compile(
    r"""find_kernel\(\s*['"]([A-Za-z0-9_]+)['"]\s*\)""")
sources = [p for p in glob.glob(os.path.join(pkg_dir, '*.py'))
           + glob.glob(os.path.join(pkg_dir, '*', '*.py'))
           if os.sep + 'tests' + os.sep not in p]
literal_stems = set()
source_text = {}
for path in sources:
    with open(path, 'r') as f:
        source_text[path] = f.read()
    literal_stems.update(find_kernel_literal.findall(source_text[path]))
assert literal_stems, "no find_kernel('...') literals in the installed package"

kernel_dir = os.path.join(pkg_dir, 'kernels')
packaged = sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(kernel_dir, '*.cu')))
assert packaged, "no kernels/*.cu packaged under %s" % kernel_dir

missing = [s for s in sorted(literal_stems)
           if not os.path.isfile(cuvarbase.utils.find_kernel(s))]
assert not missing, "kernel file(s) missing from the package: %r" % missing

# Stems that bls.py/tls.py bind to a local before calling find_kernel are
# not find_kernel('...') literals; they must still be quoted somewhere.
orphans = [s for s in packaged
           if not any(re.search(r"""['"]%s['"]""" % re.escape(s), txt)
                      for txt in source_text.values())]
assert not orphans, "packaged kernel(s) no loader references: %r" % orphans

for path in glob.glob(os.path.join(kernel_dir, '*.cu')):
    with open(path, 'r') as f:
        for target in cuvarbase.utils._INCLUDE_RE.findall(f.read()):
            assert os.path.isfile(os.path.join(kernel_dir, target)), \
                "%s includes %s, missing from the package" % (
                    os.path.basename(path), target)
headers = sorted(os.path.basename(p)
                 for p in glob.glob(os.path.join(kernel_dir, '*.cuh')))
assert 'bls_common.cuh' in headers, "bls_common.cuh missing from the package"

print('kernels OK: %d .cu (%s), %d .cuh (%s)' % (
    len(packaged), ', '.join(packaged), len(headers), ', '.join(headers)))
print('wheel import OK:', cuvarbase.__version__)
