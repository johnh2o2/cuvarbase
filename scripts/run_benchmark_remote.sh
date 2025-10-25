#!/bin/bash
#
# Run benchmarks on RunPod with persistence
#
# This script runs benchmarks inside tmux so they continue even if SSH disconnects.
# Results are saved to timestamped files.

set -e

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="benchmark_results_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/benchmark.log"
RESULTS_FILE="${OUTPUT_DIR}/results.json"
SESSION_NAME="cuvarbase_benchmark"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Starting benchmark at $(date)" | tee "${LOG_FILE}"
echo "Output directory: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "Session name: ${SESSION_NAME}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Check if tmux session already exists
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Benchmark session '${SESSION_NAME}' already exists!" | tee -a "${LOG_FILE}"
    echo "Options:" | tee -a "${LOG_FILE}"
    echo "  1. Attach to existing session: tmux attach -t ${SESSION_NAME}" | tee -a "${LOG_FILE}"
    echo "  2. Kill existing session: tmux kill-session -t ${SESSION_NAME}" | tee -a "${LOG_FILE}"
    exit 1
fi

# Create tmux session and run benchmark
echo "Creating tmux session '${SESSION_NAME}'..." | tee -a "${LOG_FILE}"
echo "Benchmark will continue running even if you disconnect." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Create detached tmux session with benchmark command
tmux new-session -d -s "${SESSION_NAME}" bash -c "
    set -e
    cd $(pwd)

    echo '========================================' | tee -a '${LOG_FILE}'
    echo 'Benchmark Starting' | tee -a '${LOG_FILE}'
    echo 'Started at: \$(date)' | tee -a '${LOG_FILE}'
    echo '========================================' | tee -a '${LOG_FILE}'
    echo '' | tee -a '${LOG_FILE}'

    # Set CUDA environment
    export PATH=/usr/local/cuda-12.8/bin:\$PATH
    export CUDA_HOME=/usr/local/cuda-12.8
    export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\$LD_LIBRARY_PATH

    echo 'GPU Information:' | tee -a '${LOG_FILE}'
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee -a '${LOG_FILE}'
    echo '' | tee -a '${LOG_FILE}'

    echo 'Python version:' | tee -a '${LOG_FILE}'
    python3 --version | tee -a '${LOG_FILE}'
    echo '' | tee -a '${LOG_FILE}'

    echo 'Starting benchmarks...' | tee -a '${LOG_FILE}'
    echo '' | tee -a '${LOG_FILE}'

    # Run benchmark with moderate timeouts
    # CPU timeout: 5 minutes (300s)
    # GPU timeout: 2 minutes (120s)
    python3 scripts/benchmark_algorithms.py \
        --algorithms sparse_bls \
        --max-cpu-time 300 \
        --max-gpu-time 120 \
        --output '${RESULTS_FILE}' \
        2>&1 | tee -a '${LOG_FILE}'

    BENCHMARK_EXIT_CODE=\$?

    echo '' | tee -a '${LOG_FILE}'
    echo '========================================' | tee -a '${LOG_FILE}'
    echo 'Benchmark Completed' | tee -a '${LOG_FILE}'
    echo 'Finished at: \$(date)' | tee -a '${LOG_FILE}'
    echo 'Exit code: \$BENCHMARK_EXIT_CODE' | tee -a '${LOG_FILE}'
    echo '========================================' | tee -a '${LOG_FILE}'

    if [ \$BENCHMARK_EXIT_CODE -eq 0 ]; then
        echo '' | tee -a '${LOG_FILE}'
        echo 'Generating visualizations...' | tee -a '${LOG_FILE}'

        python3 scripts/visualize_benchmarks.py \
            '${RESULTS_FILE}' \
            --output-prefix '${OUTPUT_DIR}/benchmark' \
            --report '${OUTPUT_DIR}/report.md' \
            2>&1 | tee -a '${LOG_FILE}'

        echo '' | tee -a '${LOG_FILE}'
        echo 'Results saved to: ${OUTPUT_DIR}' | tee -a '${LOG_FILE}'
        echo '' | tee -a '${LOG_FILE}'
        echo 'Files created:' | tee -a '${LOG_FILE}'
        ls -lh '${OUTPUT_DIR}'/ | tee -a '${LOG_FILE}'
    else
        echo '' | tee -a '${LOG_FILE}'
        echo 'Benchmark failed with exit code \$BENCHMARK_EXIT_CODE' | tee -a '${LOG_FILE}'
    fi

    echo '' | tee -a '${LOG_FILE}'
    echo 'Session will remain open. Press Ctrl+C to exit or detach with Ctrl+B then D' | tee -a '${LOG_FILE}'

    # Keep session alive
    exec bash
"

echo "" | tee -a "${LOG_FILE}"
echo "Benchmark started in background tmux session!" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Commands:" | tee -a "${LOG_FILE}"
echo "  - View progress:  tmux attach -t ${SESSION_NAME}" | tee -a "${LOG_FILE}"
echo "  - Detach:         Press Ctrl+B, then D" | tee -a "${LOG_FILE}"
echo "  - Check status:   tmux ls" | tee -a "${LOG_FILE}"
echo "  - View log:       tail -f ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Results will be saved to: ${OUTPUT_DIR}/" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Show initial log output
sleep 2
echo "Initial output:" | tee -a "${LOG_FILE}"
echo "---" | tee -a "${LOG_FILE}"
tail -20 "${LOG_FILE}"
