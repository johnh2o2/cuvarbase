#!/bin/bash
# Initial setup of cuvarbase development environment on RunPod

set -e

# Load RunPod configuration
if [ ! -f .runpod.env ]; then
    echo "Error: .runpod.env not found!"
    echo "Copy .runpod.env.template to .runpod.env and fill in your RunPod details"
    exit 1
fi

source .runpod.env

# Build SSH connection string
SSH_OPTS="-p ${RUNPOD_SSH_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
if [ -n "${RUNPOD_SSH_KEY}" ]; then
    SSH_OPTS="${SSH_OPTS} -i ${RUNPOD_SSH_KEY}"
fi

SSH_HOST="${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}"

echo "=========================================="
echo "Setting up cuvarbase on RunPod"
echo "=========================================="

# Sync code first
echo "Step 1: Syncing code..."
./scripts/sync-to-runpod.sh

echo ""
echo "Step 2: Installing cuvarbase in development mode..."
ssh ${SSH_OPTS} ${SSH_HOST} REMOTE_DIR="${RUNPOD_REMOTE_DIR:-/workspace/cuvarbase}" bash << 'ENDSSH'
set -e

cd "${REMOTE_DIR}"

# Set up CUDA environment (auto-detect version)
if [ -d /usr/local/cuda ]; then
    export PATH=/usr/local/cuda/bin:$PATH
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
elif [ -d /usr/local/cuda-12.4 ]; then
    export PATH=/usr/local/cuda-12.4/bin:$PATH
    export CUDA_HOME=/usr/local/cuda-12.4
    export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
fi

# Check if CUDA is available
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "Warning: nvidia-smi not found. Make sure CUDA is installed."
fi

# Install cuvarbase in development mode with test dependencies
echo ""
echo "Installing cuvarbase and dependencies..."
pip install --break-system-packages -e .[test]

# Patch scikit-cuda for numpy 2.x compatibility
echo ""
echo "Patching scikit-cuda for numpy 2.x compatibility..."
python << 'ENDPYTHON'
import re
import os
import glob

skcuda_files = glob.glob('/usr/local/lib/python*/dist-packages/skcuda/*.py')
if not skcuda_files:
    print("Warning: skcuda not found, skipping patch")
    exit(0)

for filepath in skcuda_files:
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # Replace num_types list comprehension using typeDict or sctypeDict
    # This handles both np.typeDict and np.sctypeDict variants
    content = re.sub(
        r'num_types\s*=\s*\[np\.(?:type|sctype)Dict\[t\]\s+for\s+t\s+in\s*\\?\s*\n\s*np\.typecodes\[.AllInteger.\]\+np\.typecodes\[.AllFloat.\]\]',
        'num_types = [np.int8, np.int16, np.int32, np.int64,\n'
        '             np.uint8, np.uint16, np.uint32, np.uint64,\n'
        '             np.float16, np.float32, np.float64]',
        content
    )

    # Replace np.sctypes with explicit types
    content = re.sub(r'np\.sctypes\[(["\'])float\1\]', '[np.float16, np.float32, np.float64]', content)
    content = re.sub(r'np\.sctypes\[(["\'])int\1\]', '[np.int8, np.int16, np.int32, np.int64]', content)
    content = re.sub(r'np\.sctypes\[(["\'])uint\1\]', '[np.uint8, np.uint16, np.uint32, np.uint64]', content)
    content = re.sub(r'np\.sctypes\[(["\'])complex\1\]', '[np.complex64, np.complex128]', content)

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  Patched {os.path.basename(filepath)}")

print("All scikit-cuda files patched for numpy 2.x compatibility")
ENDPYTHON

echo ""
echo "Verifying installation..."
python -c "import cuvarbase; print(f'✓ cuvarbase version: {cuvarbase.__version__}')"
python -c "import pycuda.driver as cuda; cuda.init(); dev = cuda.Device(0); print(f'✓ CUDA available: {cuda.Device.count()} device(s)'); print(f'✓ GPU: {dev.name()} ({dev.total_memory()//1024**2} MB)')"

echo ""
echo "✓ Setup complete!"
ENDSSH

echo ""
echo "=========================================="
echo "RunPod environment ready!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  - Run tests: ./scripts/test-remote.sh"
echo "  - Sync code: ./scripts/sync-to-runpod.sh"
echo "  - SSH in: ssh ${SSH_OPTS} ${SSH_HOST}"
