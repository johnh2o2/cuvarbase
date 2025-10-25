#!/bin/bash
# Sync local cuvarbase code to RunPod instance

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
if [ ! -z "${RUNPOD_SSH_KEY}" ]; then
    SSH_OPTS="${SSH_OPTS} -i ${RUNPOD_SSH_KEY}"
fi

SSH_HOST="${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}"

echo "Syncing cuvarbase to RunPod..."
echo "Target: ${SSH_HOST}:${RUNPOD_REMOTE_DIR}"

# Create remote directory if it doesn't exist
ssh ${SSH_OPTS} ${SSH_HOST} "mkdir -p ${RUNPOD_REMOTE_DIR}"

# Sync code using rsync (excludes git, pycache, etc.)
rsync -avz --progress \
    --no-perms --no-owner --no-group \
    -e "ssh ${SSH_OPTS}" \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache/' \
    --exclude 'build/' \
    --exclude 'dist/' \
    --exclude '*.egg-info/' \
    --exclude '.runpod.env' \
    --exclude 'work/' \
    --exclude 'testing/' \
    --exclude '*.png' \
    --exclude '*.gif' \
    ./ ${SSH_HOST}:${RUNPOD_REMOTE_DIR}/

echo "Sync complete!"
