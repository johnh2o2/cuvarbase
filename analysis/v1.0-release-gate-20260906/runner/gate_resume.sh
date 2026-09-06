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
cd /workspace/cuvarbase
run_step artifact_setup bash /workspace/artifact_setup.sh
cd /workspace/artifact-run
run_step wheel_smoke /tmp/cuvarbase-phase5/venvs/wheel-smoke/bin/python /workspace/cuvarbase/scripts/ci_wheel_smoke.py
run_step sdist_smoke /tmp/cuvarbase-phase5/venvs/sdist-smoke/bin/python /workspace/cuvarbase/scripts/ci_wheel_smoke.py
run_step artifact_paths python /workspace/verify_artifact_paths.py
run_step wheel_pyargs /tmp/cuvarbase-phase5/venvs/wheel-test/bin/python -m pytest --pyargs cuvarbase -p no:cacheprovider -v -rs
run_step sdist_pyargs /tmp/cuvarbase-phase5/venvs/sdist-test/bin/python -m pytest --pyargs cuvarbase -p no:cacheprovider -v -rs
run_step wheel_run /tmp/cuvarbase-phase5/venvs/wheel-test/bin/python /workspace/wheel_run.py
cd /workspace/cuvarbase
run_step source_clean git status --porcelain
run_step site_archive bash -c 'rm -rf docs/build/html/.doctrees docs/build/html/.buildinfo; touch docs/build/html/.nojekyll; tar -C docs/build/html -czf /workspace/site-1032caf029570dc4841db1c594a2cbb1654e8fd8.tgz .; sha256sum dist/* /workspace/site-1032caf029570dc4841db1c594a2cbb1654e8fd8.tgz'
echo 'ALL GATE COMMANDS PASSED'
