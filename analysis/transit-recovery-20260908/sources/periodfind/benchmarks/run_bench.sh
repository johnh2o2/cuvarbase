#!/bin/bash
#SBATCH --job-name=periodfind-bench
#SBATCH --partition=skylake-gpu
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks/bench_%j.log

set -euo pipefail

module load gcc/13.2.0 python/3.11.5
source /home/mcoughli/periodfind/.venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda

cd /home/mcoughli/periodfind

echo "=== Node: $(hostname), GPUs: ==="
nvidia-smi --list-gpus

BENCH_DIR=benchmarks

# Run 1: CPU + single GPU
echo ""
echo "########## 1x P100 + CPU ##########"
CUDA_VISIBLE_DEVICES=0 python "$BENCH_DIR/throughput_bench.py" \
    --gpu-label "1x P100" \
    -o "$BENCH_DIR/throughput_results_1gpu.csv"

# Run 2: 2x GPU only (skip CPU — already measured above)
echo ""
echo "########## 2x P100 (GPU only) ##########"
CUDA_VISIBLE_DEVICES=0,1 python "$BENCH_DIR/throughput_bench.py" \
    --gpu-only --gpu-label "2x P100" \
    -o "$BENCH_DIR/throughput_results_2gpu.csv"

# Merge CSVs into the main results file
python -c "
import csv, sys

rows = []
for path in ['$BENCH_DIR/throughput_results_1gpu.csv',
             '$BENCH_DIR/throughput_results_2gpu.csv']:
    with open(path) as f:
        rows.extend(list(csv.DictReader(f)))

out = '$BENCH_DIR/throughput_results.csv'
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f'Merged {len(rows)} rows -> {out}')
"

# Generate plots
python "$BENCH_DIR/plot_throughput.py"

echo ""
echo "=== Done ==="
