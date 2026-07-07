#!/bin/bash
# TRUE cold single-shot: clear the on-disk kernel caches before EACH run so the
# JIT compile happens from scratch every time (first-run / fresh-container case).
# Each (method, baseline) is a fresh process. Reports full_wall (python import +
# CUDA context init + compile + search) and search_compile (compile + search).
export PATH=/usr/local/cuda/bin:$PATH
cd /root
printf "%-17s %6s %11s %13s %8s %6s\n" method baseline full_wall_s search_compile nper SDE
for base in 200 500 1000 1500; do
  for m in cuv_tls_default cuv_tls_matched gtls_skip8; do
    rm -rf ~/.cache/pycuda /root/.cache/pycuda ~/.cupy ~/.nv/ComputeCache 2>/dev/null
    t0=$(date +%s.%N)
    OUT=$(python3 cold_shot.py --method "$m" --baseline "$base" 2>/dev/null | grep RESULT)
    t1=$(date +%s.%N)
    wall=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
    sc=$(echo "$OUT"   | grep -oE 'search_compile_s=[0-9.]+' | cut -d= -f2)
    nper=$(echo "$OUT" | grep -oE 'nper=[0-9]+' | cut -d= -f2)
    sde=$(echo "$OUT"  | grep -oE 'SDE=[0-9.]+' | cut -d= -f2)
    printf "%-17s %6d %11s %13s %8s %6s\n" "$m" "$base" "$wall" "${sc:-ERR}" "${nper:-?}" "${sde:-?}"
  done
done
echo "DONE_COLD"
