#!/bin/bash
# run the edge harness one entry point per process so a sticky CUDA error cannot poison the others
cd /workspace/cuvarbase
export PYTHONDONTWRITEBYTECODE=1
for n in $(python -c "import sys; sys.path.insert(0,'/workspace/scratch/xcut'); import entry_points as E; print(' '.join(E.ALL))" 2>/dev/null); do
  timeout 600 python /workspace/scratch/xcut/h2_edge.py $n 2>&1 | grep -v "warnings.warn\|UserWarning\|^/workspace\|PyCUDA WARNING\|cuMemFree"
done
