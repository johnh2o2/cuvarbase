#!/usr/bin/env bash
set -euo pipefail
cd /tmp/cuvarbase-tls-profile
mkdir -p recovery/results recovery/sources gtls-head-source
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=7
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
python3 -m venv modern
modern/bin/python -m pip install --report recovery/sources/validation-install.json numpy==2.2.6 scipy==1.15.3 cupy-cuda12x==13.6.0 numba==0.67.0 batman-package==2.5.3 pynvml==13.0.1 tqdm==4.70.0
tar -xf gtls-head.tar -C gtls-head-source
modern/bin/python -m pip install --no-deps --target gtls-head-install ./gtls-head-source
modern/bin/python -m pip freeze > recovery/sources/validation-freeze.txt
nvidia-smi -q -x > recovery/sources/gpu.xml
lscpu > recovery/sources/cpu.txt
cat /sys/fs/cgroup/cpu.max > recovery/sources/cpu-quota.txt
printf 'SETUP_COMPLETE\n'
