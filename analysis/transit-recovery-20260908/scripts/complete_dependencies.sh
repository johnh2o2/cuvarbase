#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd /tmp/cuvarbase-tls-profile
# Initial pilot revealed missing optional build/import dependencies, not algorithm failures.
modern/bin/python -m pip install --report recovery/results/build-dependency-install.json cython matplotlib
modern/bin/python -m pip install --force-reinstall --no-deps --no-build-isolation ./periodfind-source
modern/bin/python -c 'from periodfind.gpu import BoxLeastSquares; print(BoxLeastSquares())'
modern/bin/python -m pip freeze > recovery/results/modern-freeze.txt
echo DEPENDENCIES_COMPLETE
