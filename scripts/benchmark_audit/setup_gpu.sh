#!/usr/bin/env bash
# Disposable GPU host only. Source is supplied by git archive of 1032caf.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
ROOT=/tmp/cuvarbase-benchmark-audit
cd "$ROOT"
tar -xf source-v1.tar -C source-v1
python3 -m venv modern
modern/bin/python -m pip install --upgrade pip wheel setuptools
modern/bin/python -m pip install 'numpy==2.2.6' 'scipy==1.15.3' 'pycuda==2025.1.2' 'cupy-cuda12x==13.6.0' 'astropy==8.0.1' 'nifty-ls==1.1.0' 'cufinufft==2.5.1' 'gputls==0.4.4' 'transitleastsquares==1.32' batman-package threadpoolctl
modern/bin/python -m pip install --no-deps ./source-v1
modern/bin/python -m pip freeze > results/modern-freeze.txt
python3 -m venv legacy
legacy/bin/python -m pip install --upgrade pip wheel 'setuptools<81'
legacy/bin/python -m pip install 'numpy==1.23.5' 'scipy==1.10.1' 'pycuda==2022.2.2' scikit-cuda future threadpoolctl
legacy/bin/python -m pip install --no-deps 'cuvarbase==0.2.5'
legacy/bin/python -m pip freeze > results/legacy-freeze.txt
nvidia-smi > results/nvidia-smi.txt
nvcc --version > results/nvcc.txt
lscpu > results/lscpu.txt
cat /sys/fs/cgroup/cpu.max > results/cpu-max.txt
echo SETUP_COMPLETE
