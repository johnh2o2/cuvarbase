#!/bin/bash
# Run cuvarbase benchmarks across multiple GPU types on RunPod.
#
# Creates a pod for each GPU, runs benchmarks, downloads results, terminates.
# Requires RUNPOD_API_KEY in .runpod.env
#
# Usage:
#   ./scripts/benchmark_all_gpus.sh
#   ./scripts/benchmark_all_gpus.sh "NVIDIA H100 80GB HBM3" "NVIDIA H200"

set -eE

# Cleanup function to terminate pod on failure
cleanup_pod() {
    if [ -n "${CURRENT_POD_ID}" ]; then
        echo "Cleaning up: terminating pod ${CURRENT_POD_ID}..."
        curl -s --request POST \
            --header 'content-type: application/json' \
            --url "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
            --data "{\"query\": \"mutation { podTerminate(input: {podId: \\\"${CURRENT_POD_ID}\\\"}) }\"}" > /dev/null 2>&1 || true
        CURRENT_POD_ID=""
    fi
}
trap cleanup_pod ERR

CURRENT_POD_ID=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# Load config
if [ ! -f .runpod.env ]; then
    echo "Error: .runpod.env not found"
    exit 1
fi
source .runpod.env

if [ -z "${RUNPOD_API_KEY}" ]; then
    echo "Error: RUNPOD_API_KEY not set"
    exit 1
fi

API_URL="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
RESULTS_DIR="${PROJECT_DIR}/benchmarks/results/by_gpu"
mkdir -p "${RESULTS_DIR}"

# SSH key option
SSH_KEY_OPT=""
if [ -f ~/.ssh/id_ed25519 ]; then
    SSH_KEY_OPT="-i ~/.ssh/id_ed25519"
fi

# GPU types to benchmark (RunPod type ID -> our short name -> benchmark --gpu-model)
# Format: "RUNPOD_TYPE_ID|SHORT_NAME|BENCHMARK_GPU_MODEL"
if [ $# -gt 0 ]; then
    # User specified GPU types on command line — use them as RunPod type IDs
    GPU_LIST=()
    for gpu in "$@"; do
        case "$gpu" in
            *V100*)    GPU_LIST+=("${gpu}|V100|V100") ;;
            *4000*Ada*) GPU_LIST+=("${gpu}|RTX_4000_Ada|RTX_4000_Ada") ;;
            *4090*)    GPU_LIST+=("${gpu}|RTX_4090|RTX_4090") ;;
            *L40)      GPU_LIST+=("${gpu}|L40|L40") ;;
            *A100*SXM*) GPU_LIST+=("${gpu}|A100_SXM|A100_SXM") ;;
            *H100*HBM*|*H100*SXM*) GPU_LIST+=("${gpu}|H100_SXM|H100_SXM") ;;
            *H200*)    GPU_LIST+=("${gpu}|H200_SXM|H200_SXM") ;;
            *)         GPU_LIST+=("${gpu}|unknown|H100_SXM") ;;
        esac
    done
else
    GPU_LIST=(
        "Tesla V100-SXM2-16GB|V100|V100"
        "NVIDIA RTX 4000 Ada Generation|RTX_4000_Ada|RTX_4000_Ada"
        "NVIDIA GeForce RTX 4090|RTX_4090|RTX_4090"
        "NVIDIA L40|L40|L40"
        "NVIDIA A100-SXM4-80GB|A100_SXM|A100_SXM"
        "NVIDIA H100 80GB HBM3|H100_SXM|H100_SXM"
        "NVIDIA H200|H200_SXM|H200_SXM"
    )
fi

# Benchmark parameters
NDATA=10000
NBATCH=10
NFREQ=5000
BASELINE=3652.5
ALGORITHMS="bls_standard ls"

echo "=============================================="
echo "  cuvarbase Multi-GPU Benchmark Suite"
echo "=============================================="
echo "GPUs to benchmark: ${#GPU_LIST[@]}"
echo "Parameters: ndata=${NDATA}, nbatch=${NBATCH}, nfreq=${NFREQ}"
echo "Results directory: ${RESULTS_DIR}"
echo ""

TOTAL_GPUS=${#GPU_LIST[@]}
CURRENT=0
FAILED_GPUS=()

for gpu_entry in "${GPU_LIST[@]}"; do
    IFS='|' read -r GPU_TYPE SHORT_NAME GPU_MODEL <<< "$gpu_entry"
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "=============================================="
    echo "  [${CURRENT}/${TOTAL_GPUS}] ${SHORT_NAME} (${GPU_TYPE})"
    echo "=============================================="

    RESULT_FILE="${RESULTS_DIR}/benchmark_${SHORT_NAME}.json"
    POD_ID=""

    # --- Skip if results already exist ---
    if [ -f "${RESULT_FILE}" ]; then
        echo "Results already exist at ${RESULT_FILE}, skipping."
        continue
    fi

    # --- Create pod ---
    echo "Creating pod..."
    RESPONSE=$(curl -s --request POST \
        --header 'content-type: application/json' \
        --url "${API_URL}" \
        --data "{\"query\": \"mutation { podFindAndDeployOnDemand(input: { cloudType: ALL, gpuCount: 1, volumeInGb: 50, containerDiskInGb: 40, minVcpuCount: 2, minMemoryInGb: 15, gpuTypeId: \\\"${GPU_TYPE}\\\", name: \\\"cuvarbase-bench-${SHORT_NAME}\\\", imageName: \\\"${IMAGE}\\\", ports: \\\"22/tcp\\\", volumeMountPath: \\\"/workspace\\\" }) { id costPerHr } }\"}")

    POD_ID=$(echo "${RESPONSE}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'errors' in data:
    print('ERROR:' + data['errors'][0]['message'], file=sys.stderr)
    sys.exit(1)
print(data['data']['podFindAndDeployOnDemand']['id'])
" 2>&1)

    if [[ "${POD_ID}" == ERROR:* ]] || [ -z "${POD_ID}" ]; then
        echo "FAILED to create pod: ${POD_ID}"
        echo "Response: ${RESPONSE}"
        FAILED_GPUS+=("${SHORT_NAME}: pod creation failed")
        continue
    fi

    COST=$(echo "${RESPONSE}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data['data']['podFindAndDeployOnDemand']['costPerHr'])
")
    CURRENT_POD_ID="${POD_ID}"
    echo "Pod ${POD_ID} created (\$${COST}/hr)"

    # --- Wait for SSH ---
    echo "Waiting for pod to start..."
    MAX_WAIT=300
    WAITED=0
    SSH_IP=""
    SSH_PORT=""

    while [ ${WAITED} -lt ${MAX_WAIT} ]; do
        sleep 10
        WAITED=$((WAITED + 10))

        STATUS_RESPONSE=$(curl -s --request POST \
            --header 'content-type: application/json' \
            --url "${API_URL}" \
            --data "{\"query\": \"query { pod(input: {podId: \\\"${POD_ID}\\\"}) { id desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }\"}")

        eval "$(echo "${STATUS_RESPONSE}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
pod = data['data']['pod']
status = pod.get('desiredStatus', 'UNKNOWN')
print(f'POD_STATUS={status}')
runtime = pod.get('runtime')
if runtime and runtime.get('ports'):
    for port in runtime['ports']:
        if port['privatePort'] == 22 and port['isIpPublic']:
            print(f\"SSH_IP={port['ip']}\")
            print(f\"SSH_PORT={port['publicPort']}\")
" 2>/dev/null)" 2>/dev/null || true

        printf "\r  Status: %-10s Waited: %ds" "${POD_STATUS}" "${WAITED}"

        if [ -n "${SSH_IP}" ] && [ -n "${SSH_PORT}" ]; then
            echo ""
            break
        fi
    done

    if [ -z "${SSH_IP}" ] || [ -z "${SSH_PORT}" ]; then
        echo ""
        echo "Pod did not become SSH-ready within ${MAX_WAIT}s, terminating..."
        curl -s --request POST \
            --header 'content-type: application/json' \
            --url "${API_URL}" \
            --data "{\"query\": \"mutation { podTerminate(input: {podId: \\\"${POD_ID}\\\"}) }\"}" > /dev/null
        FAILED_GPUS+=("${SHORT_NAME}: SSH timeout")
        continue
    fi

    echo "SSH available at ${SSH_IP}:${SSH_PORT}"

    # --- Setup SSH via proxy ---
    echo "Setting up SSH..."
    POD_HOST_ID=$(curl -s --request POST \
        --header "content-type: application/json" \
        --url "${API_URL}" \
        --data "{\"query\": \"query { pod(input: {podId: \\\"${POD_ID}\\\"}) { machine { podHostId } } }\"}" \
        | python3 -c "import sys, json; print(json.load(sys.stdin)['data']['pod']['machine']['podHostId'])" 2>/dev/null) || true

    PROXY_SSH="ssh -tt -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${SSH_KEY_OPT} ${POD_HOST_ID}@ssh.runpod.io"

    # Start SSHD and add key
    echo 'ssh-keygen -A 2>/dev/null; service ssh start; mkdir -p /root/.ssh; chmod 700 /root/.ssh; echo "SSHD_SETUP_DONE"; exit' \
        | ${PROXY_SSH} 2>&1 | grep -q "SSHD_SETUP_DONE" || true

    if [ -f ~/.ssh/id_ed25519.pub ]; then
        LOCAL_PUBKEY=$(cat ~/.ssh/id_ed25519.pub)
        echo "mkdir -p /root/.ssh && echo \"${LOCAL_PUBKEY}\" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && echo AUTH_OK; exit" \
            | ${PROXY_SSH} 2>&1 | grep -q "AUTH_OK" || true
    fi

    # Wait for direct SSH
    SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ${SSH_KEY_OPT} -p ${SSH_PORT}"
    SSH_TARGET="root@${SSH_IP}"
    SSH_READY=false
    SSH_WAIT=0

    while [ ${SSH_WAIT} -lt 60 ]; do
        if ssh ${SSH_OPTS} ${SSH_TARGET} "echo ok" >/dev/null 2>&1; then
            SSH_READY=true
            break
        fi
        sleep 5
        SSH_WAIT=$((SSH_WAIT + 5))
    done

    if [ "${SSH_READY}" != true ]; then
        echo "Direct SSH failed, terminating pod..."
        curl -s --request POST \
            --header 'content-type: application/json' \
            --url "${API_URL}" \
            --data "{\"query\": \"mutation { podTerminate(input: {podId: \\\"${POD_ID}\\\"}) }\"}" > /dev/null
        FAILED_GPUS+=("${SHORT_NAME}: SSH connection failed")
        continue
    fi

    echo "SSH connected."

    # --- Sync code (tarball + scp, more reliable than piped tar) ---
    echo "Syncing code..."
    LOCAL_TAR="/tmp/cuvarbase_sync.tar.gz"
    # Use COPYFILE_DISABLE to prevent macOS resource fork/xattr inclusion
    COPYFILE_DISABLE=1 tar czf "${LOCAL_TAR}" \
        --no-mac-metadata --no-xattrs 2>/dev/null \
        --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='.pytest_cache' --exclude='build' --exclude='dist' \
        --exclude='*.egg-info' --exclude='.runpod.env' --exclude='work' \
        --exclude='testing' --exclude='*.png' --exclude='*.gif' \
        --exclude='benchmarks/results/by_gpu' --exclude='.claude' \
        --exclude='._*' --exclude='.DS_Store' \
        -C "${PROJECT_DIR}" . 2>/dev/null || \
    COPYFILE_DISABLE=1 tar czf "${LOCAL_TAR}" \
        --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='.pytest_cache' --exclude='build' --exclude='dist' \
        --exclude='*.egg-info' --exclude='.runpod.env' --exclude='work' \
        --exclude='testing' --exclude='*.png' --exclude='*.gif' \
        --exclude='benchmarks/results/by_gpu' --exclude='.claude' \
        --exclude='._*' --exclude='.DS_Store' \
        -C "${PROJECT_DIR}" . 2>/dev/null

    SCP_OPTS="-P ${SSH_PORT} -o ConnectTimeout=30 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=10 ${SSH_KEY_OPT}"
    SSH_XFER_OPTS="-o ConnectTimeout=30 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=10 ${SSH_KEY_OPT} -p ${SSH_PORT}"

    SYNC_OK=false
    set +eE  # Disable error trapping during sync attempts
    for SYNC_TRY in 1 2 3; do
        echo "  Sync attempt ${SYNC_TRY}: uploading tarball via ssh..."
        # Use ssh stdin pipe (works even when scp is blocked)
        UPLOAD_OUT=$(cat "${LOCAL_TAR}" | ssh ${SSH_XFER_OPTS} ${SSH_TARGET} "cat > /tmp/cuvarbase_sync.tar.gz && echo UPLOAD_OK" 2>&1) || true
        if ! echo "${UPLOAD_OUT}" | grep -q "UPLOAD_OK"; then
            echo "  Upload failed: ${UPLOAD_OUT}"
            sleep 10
            continue
        fi
        echo "  Sync attempt ${SYNC_TRY}: extracting on remote..."
        EXTRACT_OUT=$(ssh ${SSH_XFER_OPTS} ${SSH_TARGET} "mkdir -p /workspace/cuvarbase && tar xzf /tmp/cuvarbase_sync.tar.gz --no-same-owner -C /workspace/cuvarbase 2>/dev/null; ls /workspace/cuvarbase/pyproject.toml && echo SYNC_OK" 2>&1) || true
        echo "  Remote output: ${EXTRACT_OUT}"
        if echo "${EXTRACT_OUT}" | grep -q "SYNC_OK"; then
            SYNC_OK=true
            break
        fi
        echo "  Extract failed"
        sleep 10
    done
    set -eE  # Re-enable error trapping
    rm -f "${LOCAL_TAR}"

    if [ "${SYNC_OK}" != true ]; then
        echo "Code sync failed after 3 attempts, terminating pod..."
        curl -s --request POST \
            --header 'content-type: application/json' \
            --url "${API_URL}" \
            --data "{\"query\": \"mutation { podTerminate(input: {podId: \\\"${POD_ID}\\\"}) }\"}" > /dev/null
        CURRENT_POD_ID=""
        FAILED_GPUS+=("${SHORT_NAME}: code sync failed")
        continue
    fi

    # --- Install dependencies and run benchmarks ---
    echo "Installing and running benchmarks..."
    ssh ${SSH_OPTS} ${SSH_TARGET} bash << ENDSSH
set -e

cd /workspace/cuvarbase

# CUDA env
export PATH=/usr/local/cuda/bin:\$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH

# Show GPU info
echo "GPU INFO:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Install cuvarbase
echo ""
echo "Installing cuvarbase..."
pip install --break-system-packages -q -e .[test] 2>&1 | tail -3

# Patch scikit-cuda for numpy 2.x
python3 << 'ENDPYTHON'
import re, os, glob
for filepath in glob.glob('/usr/local/lib/python*/dist-packages/skcuda/*.py'):
    with open(filepath, 'r') as f:
        content = f.read()
    original = content
    content = re.sub(
        r'num_types\s*=\s*\[np\.(?:type|sctype)Dict\[t\]\s+for\s+t\s+in\s*\\\\?\s*\n\s*np\.typecodes\[.AllInteger.\]\+np\.typecodes\[.AllFloat.\]\]',
        'num_types = [np.int8, np.int16, np.int32, np.int64,\n'
        '             np.uint8, np.uint16, np.uint32, np.uint64,\n'
        '             np.float16, np.float32, np.float64]',
        content
    )
    content = re.sub(r'np\.sctypes\[(["\047])float\1\]', '[np.float16, np.float32, np.float64]', content)
    content = re.sub(r'np\.sctypes\[(["\047])int\1\]', '[np.int8, np.int16, np.int32, np.int64]', content)
    content = re.sub(r'np\.sctypes\[(["\047])uint\1\]', '[np.uint8, np.uint16, np.uint32, np.uint64]', content)
    content = re.sub(r'np\.sctypes\[(["\047])complex\1\]', '[np.complex64, np.complex128]', content)
    # Fix np.float, np.int, np.complex removed in numpy 2.x
    # Only replace standalone np.float( calls, not np.float32/64 etc.
    content = re.sub(r'\bnp\.float\b(?!16|32|64|128|_)', 'float', content)
    content = re.sub(r'\bnp\.int\b(?!8|16|32|64|_)', 'int', content)
    content = re.sub(r'\bnp\.complex\b(?!64|128|_)', 'complex', content)
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  Patched {os.path.basename(filepath)}")
ENDPYTHON

# Install CPU baselines
echo ""
echo "Installing CPU baselines..."
pip install --break-system-packages -q astropy nifty-ls transitleastsquares PyAstronomy 2>&1 | tail -3

# Verify
echo ""
python3 -c "import cuvarbase; print(f'cuvarbase OK')"
python3 -c "import pycuda.driver as cuda; cuda.init(); d=cuda.Device(0); print(f'GPU: {d.name()} ({d.total_memory()//1024**2} MB)')"

# Run benchmarks
echo ""
echo "=========================================="
echo "  RUNNING BENCHMARKS"
echo "=========================================="
python3 scripts/benchmark_algorithms.py \
    --algorithms ${ALGORITHMS} \
    --ndata ${NDATA} \
    --nbatch ${NBATCH} \
    --nfreq ${NFREQ} \
    --baseline ${BASELINE} \
    --gpu-model ${GPU_MODEL} \
    --output /workspace/benchmark_${SHORT_NAME}.json

echo ""
echo "BENCHMARK COMPLETE"
ENDSSH

    BENCH_EXIT=$?

    if [ ${BENCH_EXIT} -ne 0 ]; then
        echo "Benchmark failed with exit code ${BENCH_EXIT}"
        FAILED_GPUS+=("${SHORT_NAME}: benchmark failed (exit ${BENCH_EXIT})")
    fi

    # --- Download results ---
    echo "Downloading results..."
    SCP_OPTS="-P ${SSH_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ${SSH_KEY_OPT}"
    scp ${SCP_OPTS} ${SSH_TARGET}:/workspace/benchmark_${SHORT_NAME}.json \
        "${RESULT_FILE}" 2>/dev/null || {
        echo "Failed to download via scp, trying ssh cat..."
        ssh ${SSH_OPTS} ${SSH_TARGET} "cat /workspace/benchmark_${SHORT_NAME}.json" > "${RESULT_FILE}" 2>/dev/null || {
            echo "Failed to download results"
            FAILED_GPUS+=("${SHORT_NAME}: download failed")
        }
    }

    if [ -f "${RESULT_FILE}" ]; then
        echo "Results saved: ${RESULT_FILE}"
    fi

    # --- Terminate pod ---
    echo "Terminating pod ${POD_ID}..."
    curl -s --request POST \
        --header 'content-type: application/json' \
        --url "${API_URL}" \
        --data "{\"query\": \"mutation { podTerminate(input: {podId: \\\"${POD_ID}\\\"}) }\"}" > /dev/null
    CURRENT_POD_ID=""
    echo "Pod terminated."

done

# --- Final summary ---
echo ""
echo "=============================================="
echo "  BENCHMARK RUN COMPLETE"
echo "=============================================="
echo ""

RESULT_FILES=$(ls "${RESULTS_DIR}"/benchmark_*.json 2>/dev/null)
if [ -n "${RESULT_FILES}" ]; then
    echo "Results collected:"
    for f in ${RESULT_FILES}; do
        echo "  $(basename ${f})"
    done
else
    echo "No results collected!"
fi

if [ ${#FAILED_GPUS[@]} -gt 0 ]; then
    echo ""
    echo "FAILURES:"
    for f in "${FAILED_GPUS[@]}"; do
        echo "  - ${f}"
    done
fi

echo ""
echo "To combine results:"
echo "  python3 scripts/combine_gpu_benchmarks.py ${RESULTS_DIR}/"
