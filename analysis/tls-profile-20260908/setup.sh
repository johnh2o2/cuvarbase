#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUTF8=1
cd /tmp/cuvarbase-tls-profile
mkdir -p results sources source-v1 gtls-source
tar -xf payload.tar
tar -xf source-v1.tar -C source-v1
tar -xf gtls-head.tar -C gtls-source
python3 -m venv modern
modern/bin/python -m pip install --quiet --upgrade pip wheel 'setuptools<76'
modern/bin/python -m pip install --quiet --report results/install.json \
  numpy==2.2.6 scipy==1.15.3 pycuda==2025.1.2 cupy-cuda12x==13.6.0 \
  gputls==0.4.4 transitleastsquares==1.32 batman-package==2.5.3 numba==0.67.0
modern/bin/python -m pip install --quiet --no-deps ./source-v1
modern/bin/python -m pip install --quiet --no-deps --target gtls-head-install ./gtls-source
modern/bin/python -m pip freeze > results/pip-freeze.txt
nvidia-smi > results/nvidia-smi.txt
nvcc --version > results/nvcc.txt
lscpu > results/lscpu.txt
if [[ -f /sys/fs/cgroup/cpu.max ]]; then
  cat /sys/fs/cgroup/cpu.max > results/cpu-max.txt
else
  cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us > results/cpu-quota-period.txt
fi
echo SETUP_COMPLETE
