"""pycuda compiler-cache behaviour + why CE recompiles every call."""
import time, os, numpy as np, warnings
warnings.simplefilter('ignore')
import pycuda.compiler as pc, pycuda.driver as cuda
from cuvarbase.base import ensure_context; ensure_context()
from cuvarbase.utils import _module_reader, find_kernel
from cuvarbase.ce import ConditionalEntropyAsyncProcess
print("PYCUDA_CACHE_DIR=%r PYCUDA_DISABLE_CACHE=%r" % (os.environ.get('PYCUDA_CACHE_DIR'), os.environ.get('PYCUDA_DISABLE_CACHE')))
cd = os.path.expanduser('~/.cache/pycuda')
print("cache dir exists:", os.path.isdir(cd), "entries:", sum(len(f) for _, _, f in os.walk(cd)) if os.path.isdir(cd) else 0)
src = _module_reader(find_kernel('ce'), cpp_defs=dict(NPHASE=10, NMAG=5, PHASE_OVERLAP=0, MAG_OVERLAP=0))
for i in range(3):
    t0 = time.perf_counter(); m = pc.SourceModule(src, options=['--use_fast_math']); dt = time.perf_counter() - t0
    print("SourceModule(ce.cu) call %d: %.0f ms" % (i, 1e3*dt))
src2 = _module_reader(find_kernel('bls'), cpp_defs=dict(BLOCK_SIZE=256))
for i in range(2):
    t0 = time.perf_counter(); m = pc.SourceModule(src2, options=['--use_fast_math']); dt = time.perf_counter() - t0
    print("SourceModule(bls.cu) call %d: %.0f ms" % (i, 1e3*dt))
# nvcc-free module load for comparison
t0 = time.perf_counter(); cubin = pc.compile(src, options=['--use_fast_math']); print("pc.compile (cached cubin bytes) %.0f ms" % (1e3*(time.perf_counter()-t0)))
t0 = time.perf_counter(); cuda.module_from_buffer(cubin); print("module_from_buffer %.0f ms" % (1e3*(time.perf_counter()-t0)))
ce = ConditionalEntropyAsyncProcess()
rng = np.random.RandomState(0); t = np.sort(rng.rand(500)); y = rng.randn(500); dy = np.ones(500)
ce.run([(t, y, dy)], freqs=np.linspace(1, 5, 100)); ce.finish()
print("CE prepared_functions keys:", sorted(ce.prepared_functions.keys()))
print("run() gate checks for 'ce_wt' in keys ->", 'ce_wt' in ce.prepared_functions, "=> recompiles every call")
