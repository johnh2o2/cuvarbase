#!/bin/bash
set -euo pipefail
unset PYTHONPATH
mkdir -p /tmp/cuvarbase-phase5/venvs /workspace/artifact-run
for artifact in wheel sdist; do
    if [[ $artifact == wheel ]]; then package=(/workspace/cuvarbase/dist/cuvarbase-1.0.0-*.whl); else package=(/workspace/cuvarbase/dist/cuvarbase-1.0.0.tar.gz); fi
    test "${#package[@]}" = 1
    bare=/tmp/cuvarbase-phase5/venvs/$artifact-smoke
    gpu=/tmp/cuvarbase-phase5/venvs/$artifact-test
    python -m venv "$bare"
    "$bare/bin/pip" install 'numpy>=1.22' 'scipy>=1.8'
    "$bare/bin/pip" install --no-deps "${package[0]}"
    "$bare/bin/python" -c 'import importlib.util; assert importlib.util.find_spec("pycuda") is None'
    python -m venv "$gpu"
    "$gpu/bin/pip" install "${package[0]}[test]" cufinufft
    "$gpu/bin/pip" freeze
 done
