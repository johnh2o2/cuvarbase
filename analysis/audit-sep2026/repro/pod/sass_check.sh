#!/bin/bash
# Does float atomicAdd on shared memory compile to a native ATOMS.ADD or to a CAS spin loop on sm_89?
set -e
cd /workspace/scratch
python - <<'PY'
from cuvarbase.utils import find_kernel, _module_reader
for k in ('bls', 'bls_batch', 'sparse_bls'):
    open(f'/workspace/scratch/{k}_expanded.cu','w').write(_module_reader(find_kernel(k), cpp_defs=dict(BLOCK_SIZE=256)))
PY
for k in bls bls_batch; do
  nvcc -arch=sm_89 --use_fast_math -cubin -o ${k}.cubin ${k}_expanded.cu 2>&1 | grep -v "^$" | head -5
  echo "--- $k: shared atomics in SASS (full_bls_no_sol_fused / full_bls_batch_fused):"
  cuobjdump -sass ${k}.cubin | grep -E "ATOMS|ATOM\.|RED\." | sed 's/^ *//' | awk '{print $2}' | sort | uniq -c
  echo "--- $k: register/shared usage:"
  cuobjdump -res-usage ${k}.cubin 2>/dev/null | grep -A1 "full_bls" | grep -E "Function|REG" | head -12
done
nvcc -arch=sm_89 --use_fast_math -cubin -o sparse.cubin sparse_bls_expanded.cu 2>&1 | head -3
cuobjdump -res-usage sparse.cubin | grep -A1 sparse | head -4
