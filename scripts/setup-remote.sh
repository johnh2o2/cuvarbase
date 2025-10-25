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
SSH_OPTS="-p ${RUNPOD_SSH_PORT}"
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
ssh ${SSH_OPTS} ${SSH_HOST} bash << 'ENDSSH'
set -e

cd /workspace/cuvarbase

# Set up CUDA environment
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

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

# Find skcuda installation (could be in different python versions)
skcuda_paths = glob.glob('/usr/local/lib/python*/dist-packages/skcuda/misc.py')
if not skcuda_paths:
    print("Warning: skcuda/misc.py not found, skipping patch")
    exit(0)

misc_path = skcuda_paths[0]
print(f"Patching {misc_path}...")

# Read the file
with open(misc_path, 'r') as f:
    content = f.read()

# Replace the problematic lines around line 637
old_code = """# List of available numerical types provided by numpy:
num_types = [np.sctypeDict[t] for t in \\
             np.typecodes['AllInteger']+np.typecodes['AllFloat']]"""

new_code = """# List of available numerical types provided by numpy:
# Fixed for numpy 2.x compatibility
try:
    num_types = [np.sctypeDict[t] for t in \\
                 np.typecodes['AllInteger']+np.typecodes['AllFloat']]
except KeyError:
    # numpy 2.x: build list manually
    num_types = [np.int8, np.int16, np.int32, np.int64,
                 np.uint8, np.uint16, np.uint32, np.uint64,
                 np.float16, np.float32, np.float64]"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(misc_path, 'w') as f:
        f.write(content)
    print(f"✓ Patched {misc_path}")
else:
    print(f"Note: Already patched or code structure changed")

# Patch np.sctypes usage across all scikit-cuda files
print("")
print("Patching np.sctypes usage in scikit-cuda...")
skcuda_files = glob.glob('/usr/local/lib/python*/dist-packages/skcuda/*.py')

for filepath in skcuda_files:
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # Replace np.sctypes with explicit types
    content = re.sub(
        r'np\.sctypes\[(["\'])float\1\]',
        '[np.float16, np.float32, np.float64]',
        content
    )
    content = re.sub(
        r'np\.sctypes\[(["\'])int\1\]',
        '[np.int8, np.int16, np.int32, np.int64]',
        content
    )
    content = re.sub(
        r'np\.sctypes\[(["\'])uint\1\]',
        '[np.uint8, np.uint16, np.uint32, np.uint64]',
        content
    )
    content = re.sub(
        r'np\.sctypes\[(["\'])complex\1\]',
        '[np.complex64, np.complex128]',
        content
    )

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Patched {os.path.basename(filepath)}")

print("✓ All scikit-cuda files patched for numpy 2.x compatibility")
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
