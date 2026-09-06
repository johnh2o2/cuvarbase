#!/bin/bash
# Phase 4 campaign: 6 processes, balanced by single-process compute
# (the A40 gives ~2x throughput for >= 4 concurrent processes; more
# processes do not help). Usage on the pod:
#   OUT=/workspace/p4run bash launch_campaign.sh [extra harness args, e.g. --quick]
set -e
CUDA_DIR=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
export PATH=${CUDA_DIR}/bin:$PATH CUDA_HOME=${CUDA_DIR} LD_LIBRARY_PATH=${CUDA_DIR}/lib64:$LD_LIBRARY_PATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /workspace/cuvarbase
OUT=${OUT:-/workspace/p4run}
mkdir -p "$OUT"
EXTRA="$@"
run() {   # name, args...
    local name=$1; shift
    nohup python scripts/nufft_lrt_validation.py "$@" $EXTRA --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1 &
    echo "started $name (pid $!): $*"
}
# A: white (all arms) + calibration
run A_white        --configs white
# B: white_bjd (all arms)
run B_whitebjd     --configs white_bjd --skip-calibration
# C: red_1x (all arms incl. lrt_flat)
run C_red1x        --configs red_1x
# D: red_3x (all arms incl. lrt_flat)
run D_red3x        --configs red_3x
# E: red_sys basis-free arms, then red_sys_nzm sequential
run E_redsys_main  --configs red_sys --arms lrt,lrt_auto,bls,tls
run E2_nzm_seq     --configs red_sys_nzm --arms lrt_seq
# F: red_sys detectors, then red_sys_nzm Detector A
run F_redsys_det   --configs red_sys --arms lrt_marg,lrt_seq
run F2_nzm_marg    --configs red_sys_nzm --arms lrt_marg
date > "$OUT/started_at"
echo "all started; logs in $OUT"
