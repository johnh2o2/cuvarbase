#!/bin/bash
# Run arbitrary command on RunPod instance

set -e

# Load RunPod configuration
if [ ! -f .runpod.env ]; then
    echo "Error: .runpod.env not found!"
    echo "Copy .runpod.env.template to .runpod.env and fill in your RunPod details"
    exit 1
fi

source .runpod.env

# Build SSH connection string
SSH_OPTS="-p ${RUNPOD_SSH_PORT}"
if [ -n "${RUNPOD_SSH_KEY}" ]; then
    SSH_OPTS="${SSH_OPTS} -i ${RUNPOD_SSH_KEY}"
fi

SSH_HOST="${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}"

# Parse command
COMMAND="${@}"

echo "=========================================="
echo "Running command on RunPod"
echo "=========================================="
echo "Command: ${COMMAND}"
echo ""

# First sync the code
echo "Step 1: Syncing code..."
./scripts/sync-to-runpod.sh

echo ""
echo "Step 2: Running command on RunPod..."
echo "=========================================="

# Run command remotely with auto-detected CUDA path
ssh ${SSH_OPTS} ${SSH_HOST} "CUDA_DIR=\$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1) && export PATH=\${CUDA_DIR}/bin:\$PATH && export CUDA_HOME=\${CUDA_DIR} && export LD_LIBRARY_PATH=\${CUDA_DIR}/lib64:\$LD_LIBRARY_PATH && cd ${RUNPOD_REMOTE_DIR} && ${COMMAND}"

echo ""
echo "=========================================="
echo "Command complete!"
echo "=========================================="
