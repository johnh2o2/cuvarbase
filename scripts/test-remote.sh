#!/bin/bash
# Run tests on RunPod instance

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

# Parse arguments
TEST_PATH="${1:-cuvarbase/tests/}"
PYTEST_ARGS="${@:2}"

echo "=========================================="
echo "Running tests on RunPod"
echo "=========================================="
echo "Test path: ${TEST_PATH}"
echo "Additional pytest args: ${PYTEST_ARGS}"
echo ""

# First sync the code
echo "Step 1: Syncing code..."
./scripts/sync-to-runpod.sh

echo ""
echo "Step 2: Running tests on RunPod..."
echo "=========================================="

# Run tests remotely and stream output
ssh ${SSH_OPTS} ${SSH_HOST} "export PATH=/usr/local/cuda-12.8/bin:\$PATH && export CUDA_HOME=/usr/local/cuda-12.8 && export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\$LD_LIBRARY_PATH && cd ${RUNPOD_REMOTE_DIR} && pytest ${TEST_PATH} ${PYTEST_ARGS} -v"

echo ""
echo "=========================================="
echo "Tests complete!"
echo "=========================================="
