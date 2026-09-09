#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
cd /tmp/cuvarbase-tls-profile
mkdir -p recovery/results recovery/inputs recovery/scripts periodfind-source fBLS-source
tar -xf periodfind-source.tar -C periodfind-source
tar -xf fBLS-source.tar -C fBLS-source
modern/bin/python -m pip install --report recovery/results/extra-modern-install.json astropy==8.0.1 cython matplotlib
modern/bin/python - <<'PY'
from pathlib import Path
import subprocess,re,json,hashlib
p=Path('periodfind-source/setup.py');before=p.read_text()
cap=subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],text=True).strip().replace('.','')
after,n=re.subn(r'compute_capabilities = \[[^\]]+\]','compute_capabilities = ['+cap+']',before)
assert n==1;p.write_text(after)
Path('recovery/results/periodfind-build-adjustment.json').write_text(json.dumps(dict(
    file='setup.py',change='Compile only the measured GPU architecture; numerical source unchanged',
    before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(after.encode()).hexdigest())))
PY
modern/bin/python -m pip install --no-deps --no-build-isolation ./periodfind-source
python3 -m venv legacy
legacy/bin/python -m pip install --upgrade pip wheel 'setuptools<81'
legacy/bin/python -m pip install --report recovery/results/legacy-install.json numpy==1.23.5 scipy==1.10.1 pycuda==2022.2.2 scikit-cuda==0.5.3 future threadpoolctl
legacy/bin/python -m pip install --no-deps ./cuvarbase-0.2.5-py2.py3-none-any.whl
curl --proto '=https' --tlsv1.2 --fail --location https://sh.rustup.rs -o recovery/results/rustup-init.sh
sha256sum recovery/results/rustup-init.sh > recovery/results/rustup-init.sha256
sh recovery/results/rustup-init.sh -y --profile minimal
source /root/.cargo/env
modern/bin/python -m pip install --report recovery/results/periodfind-cpu-install.json ./periodfind-source/rust
modern/bin/python -m pip freeze > recovery/results/modern-freeze.txt
legacy/bin/python -m pip freeze > recovery/results/legacy-freeze.txt
echo RECOVERY_SETUP_COMPLETE
