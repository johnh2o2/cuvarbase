#!/bin/bash
set -Eeuo pipefail
export CUDA_DIR=$(ls -d /usr/local/cuda-* | sort -V | tail -1)
export PATH="$CUDA_DIR/bin:$PATH" CUDA_HOME="$CUDA_DIR" LD_LIBRARY_PATH="$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
unset PYTHONPATH
LOGDIR=/workspace/logs
mkdir -p "$LOGDIR"
trap 'rc=$?; echo "$rc" > "$LOGDIR/gate.exit"; date -u +%Y-%m-%dT%H:%M:%SZ; exit "$rc"' EXIT
run_step() {
    name=$1; shift
    date -u +%Y-%m-%dT%H:%M:%SZ
    echo "START $name: $*"
    if "$@" > "$LOGDIR/$name.log" 2>&1; then
        echo "PASS $name"
    else
        rc=$?; echo "FAIL $name ($rc)"; tail -60 "$LOGDIR/$name.log"; return "$rc"
    fi
}
T=1032caf029570dc4841db1c594a2cbb1654e8fd8
run_step apt bash -c 'apt-get update -qq && apt-get install -y -qq rsync'
run_step clone git clone https://github.com/johnh2o2/cuvarbase.git /workspace/cuvarbase
cd /workspace/cuvarbase
run_step checkout git checkout --detach "$T"
test "$(git rev-parse HEAD)" = "$T"
test "$(git rev-parse 'HEAD^{tree}')" = b023c3e8d163010dbae2fc0b7cd5204ca04384d1
git diff --exit-code
run_step install pip install --break-system-packages -e '.[test]' cufinufft -r docs/requirements.txt build twine
run_step preflight python -c 'import pycuda.driver as d, batman, transitleastsquares, nfft, astropy, cufinufft, cuvarbase; d.init(); print("GPU:",d.Device(0).name()); print(cuvarbase.__version__, cuvarbase.__file__); assert cuvarbase.__file__.startswith("/workspace/cuvarbase/")'
{
    echo "commit $T"; echo "tree $(git rev-parse 'HEAD^{tree}')"; date -u
    nvidia-smi; nvcc --version; python --version; pip freeze; git status --porcelain
} > "$LOGDIR/env_record.txt" 2>&1
run_step suite_full python -m pytest -p no:cacheprovider -v -rs
run_step release_gate python scripts/check_release_gate.py
# Command-line assignment is needed because docs/Makefile overrides environment SPHINXOPTS.
run_step docs_build env SPHINXOPTS='-E -a -W --keep-going' make -C docs html SPHINXOPTS='-E -a -W --keep-going'
run_step docs_figures python /workspace/verify_docs.py
run_step build python -m build
run_step twine_check python -m twine check --strict dist/*
run_step pkg_info python /workspace/check_pkg_info.py
run_step artifact_setup bash /workspace/artifact_setup.sh
cd /workspace/artifact-run
run_step wheel_smoke /workspace/venvs/wheel-smoke/bin/python /workspace/cuvarbase/scripts/ci_wheel_smoke.py
run_step sdist_smoke /workspace/venvs/sdist-smoke/bin/python /workspace/cuvarbase/scripts/ci_wheel_smoke.py
run_step artifact_paths python /workspace/verify_artifact_paths.py
run_step wheel_pyargs /workspace/venvs/wheel-test/bin/python -m pytest --pyargs cuvarbase -p no:cacheprovider -v -rs
run_step sdist_pyargs /workspace/venvs/sdist-test/bin/python -m pytest --pyargs cuvarbase -p no:cacheprovider -v -rs
run_step wheel_run /workspace/venvs/wheel-test/bin/python /workspace/wheel_run.py
cd /workspace/cuvarbase
run_step source_clean git status --porcelain
run_step site_archive bash -c 'rm -rf docs/build/html/.doctrees docs/build/html/.buildinfo; touch docs/build/html/.nojekyll; tar -C docs/build/html -czf /workspace/site-1032caf029570dc4841db1c594a2cbb1654e8fd8.tgz .; sha256sum dist/* /workspace/site-1032caf029570dc4841db1c594a2cbb1654e8fd8.tgz'
echo 'ALL GATE COMMANDS PASSED'
