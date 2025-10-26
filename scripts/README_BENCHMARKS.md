# Running Benchmarks on RunPod

## Quick Start

```bash
# 1. Sync code to RunPod
./scripts/sync-to-runpod.sh

# 2. SSH to RunPod and estimate runtime
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519
cd /workspace/cuvarbase
python3 scripts/estimate_benchmark_time.py

# 3. Start benchmark in persistent session
./scripts/run_benchmark_remote.sh

# 4. Detach from session (benchmark continues)
# Press: Ctrl+B, then D

# 5. Later: Reattach to check progress
tmux attach -t cuvarbase_benchmark

# 6. Or: Monitor log in real-time
tail -f benchmark_results_*/benchmark.log
```

## Expected Runtime

For `sparse_bls` algorithm with default settings:
- **Total time**: ~2-3 minutes on RTX A5000
- **CPU measurements**: ~2 minutes (8 experiments)
- **GPU measurements**: ~25 seconds (11 experiments)
- **Extrapolated**: 5 experiments (instant)

Breakdown by configuration:
```
ndata=10:   All measured (very fast, <1s each)
ndata=100:  Most measured, large batches extrapolated
ndata=1000: Only small batches measured, rest extrapolated
```

## Session Management

### Check if benchmark is running
```bash
tmux ls
```

### Attach to running benchmark
```bash
tmux attach -t cuvarbase_benchmark
```

### Detach without stopping
```
Press: Ctrl+B, then D
```

### Kill benchmark session
```bash
tmux kill-session -t cuvarbase_benchmark
```

### View live progress
```bash
# Find the latest results directory
ls -dt benchmark_results_* | head -1

# Tail the log
tail -f benchmark_results_*/benchmark.log
```

## Output Files

Results are saved to `benchmark_results_YYYYMMDD_HHMMSS/`:
```
benchmark_results_20250125_143022/
├── benchmark.log              # Full log with timestamps
├── results.json              # Raw benchmark data
├── report.md                 # Markdown summary
├── benchmark_sparse_bls_scaling.png  # Scaling plots
└── ...
```

## Downloading Results

### From RunPod to local machine:
```bash
# On local machine
scp -P <PORT> -i ~/.ssh/id_ed25519 \
    root@<HOST>:/workspace/cuvarbase/benchmark_results_*/* \
    ./local_results/
```

### Or use rsync for efficiency:
```bash
rsync -avz -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" \
    root@<HOST>:/workspace/cuvarbase/benchmark_results_*/ \
    ./local_results/
```

## Customization

### Adjust timeouts
Edit `scripts/run_benchmark_remote.sh`:
```bash
--max-cpu-time 600    # 10 minutes instead of 5
--max-gpu-time 240    # 4 minutes instead of 2
```

### Add more algorithms
Edit `scripts/run_benchmark_remote.sh`:
```bash
--algorithms sparse_bls bls_gpu_fast lombscargle
```

### Change grid
Edit `scripts/benchmark_algorithms.py`:
```python
ndata_values = [50, 200, 500]    # Different sizes
nbatch_values = [1, 5, 20, 50]   # Different batches
```

## Troubleshooting

### Benchmark hangs
```bash
# Check GPU status
nvidia-smi

# Check if process is running
tmux attach -t cuvarbase_benchmark
# Look for active Python process

# If truly hung, kill and restart
tmux kill-session -t cuvarbase_benchmark
./scripts/run_benchmark_remote.sh
```

### Out of memory
Reduce batch sizes in the grid:
```python
nbatch_values = [1, 10, 100]  # Skip 1000
```

### Session lost
Tmux persists! Just reattach:
```bash
tmux attach -t cuvarbase_benchmark
```

### Can't find results
```bash
# List all benchmark result directories
ls -ltr benchmark_results_*/

# Check if benchmark completed
grep -r "Benchmark Completed" benchmark_results_*/
```

## Performance Tips

1. **First run**: CUDA compilation adds ~30s overhead
2. **Subsequent runs**: Much faster, kernels are cached
3. **GPU memory**: ~2GB VRAM used for largest configs
4. **CPU usage**: Minimal, mostly GPU-bound
5. **Disk I/O**: Negligible, results are small (~1MB)

## Interpreting Results

### Good speedup patterns:
- Small problems (ndata<100): 1-10x speedup
- Medium problems (ndata~100): 10-50x speedup
- Large problems (ndata>500): 50-200x speedup

### Red flags:
- GPU slower than CPU: Problem too small, kernel overhead dominates
- No improvement with batch: Memory bottleneck or CPU preprocessing
- Declining speedup: Memory bandwidth saturation

See `BENCHMARKING.md` for detailed interpretation guide.
