# RunPod Development Workflow

This guide explains how to develop cuvarbase locally while testing on RunPod GPU instances.

## Overview

Since cuvarbase requires CUDA-enabled GPUs, this workflow allows you to:
- Develop and edit code locally (with Claude Code or your preferred tools)
- Automatically sync code to RunPod
- Run GPU-dependent tests on RunPod
- Stream test results back to your local terminal

## Initial Setup

### 1. Configure RunPod Connection

Copy the template configuration file:

```bash
cp .runpod.env.template .runpod.env
```

Edit `.runpod.env` with your RunPod instance details:

```bash
# Get these from your RunPod pod's "Connect" button -> SSH
RUNPOD_SSH_HOST=ssh.runpod.io
RUNPOD_SSH_PORT=12345                    # Your pod's SSH port
RUNPOD_SSH_USER=root

# Optional: Path to SSH key (if using key-based auth)
# RUNPOD_SSH_KEY=~/.ssh/runpod_rsa

# Remote directory where code will be synced
RUNPOD_REMOTE_DIR=/workspace/cuvarbase
```

### 2. Initial RunPod Environment Setup

Run the setup script once to install cuvarbase on your RunPod instance:

```bash
./scripts/setup-remote.sh
```

This will:
- Sync your code to RunPod
- Install cuvarbase in development mode (`pip install -e .[test]`)
- Verify CUDA is available
- Confirm installation

## Daily Development Workflow

### Sync Code to RunPod

After making local changes, sync to RunPod:

```bash
./scripts/sync-to-runpod.sh
```

This uses `rsync` to efficiently transfer only changed files.

### Run Tests on RunPod

Execute tests remotely and see results in your local terminal:

```bash
# Run all tests
./scripts/test-remote.sh

# Run specific test file
./scripts/test-remote.sh cuvarbase/tests/test_lombscargle.py

# Run with pytest options
./scripts/test-remote.sh cuvarbase/tests/test_bls.py -k test_specific_function -v
```

The script will:
1. Sync your latest code
2. Run pytest on RunPod
3. Stream output back to your terminal

### Direct SSH Access

If you need to manually interact with the RunPod instance:

```bash
# Using the configured values from .runpod.env
source .runpod.env
ssh -p ${RUNPOD_SSH_PORT} ${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}
```

## Example Development Session

```bash
# 1. Make changes locally (edit code with Claude Code, VS Code, etc.)
vim cuvarbase/lombscargle.py

# 2. Run tests on RunPod to verify
./scripts/test-remote.sh cuvarbase/tests/test_lombscargle.py

# 3. If tests pass, commit your changes
git add cuvarbase/lombscargle.py
git commit -m "Improve lombscargle performance"
```

## Tips

### Working with Claude Code

You can develop entirely in your local terminal with Claude Code:
- Claude Code helps you write/edit code locally
- Run `./scripts/test-remote.sh` to test on GPU
- Claude Code sees the test output and helps debug

### Faster Iteration

For rapid testing of a single test:

```bash
./scripts/test-remote.sh cuvarbase/tests/test_ce.py::test_single_function -v
```

### Checking GPU Status

SSH into RunPod and run:

```bash
nvidia-smi
```

### Re-installing Dependencies

If you update `requirements.txt` or `pyproject.toml`:

```bash
./scripts/setup-remote.sh
```

This re-runs the installation process.

## Troubleshooting

### SSH Connection Issues

Test your SSH connection manually:

```bash
source .runpod.env
ssh -p ${RUNPOD_SSH_PORT} ${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}
```

If this fails, check:
- RunPod instance is running
- SSH port is correct (check RunPod dashboard)
- SSH key permissions: `chmod 600 ~/.ssh/runpod_rsa`

### Import Errors on RunPod

If you get import errors, ensure cuvarbase is installed in editable mode:

```bash
ssh -p ${RUNPOD_SSH_PORT} ${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}
cd /workspace/cuvarbase
pip install -e .[test]
```

### CUDA Not Found

Verify CUDA toolkit is installed on RunPod:

```bash
ssh -p ${RUNPOD_SSH_PORT} ${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}
nvidia-smi
nvcc --version
```

Most RunPod templates include CUDA by default.

## Security Notes

- `.runpod.env` is gitignored to protect your credentials
- Never commit `.runpod.env` to version control
- Keep `.runpod.env.template` updated with the latest configuration structure

## Advanced Usage

### Custom Remote Directory

Change `RUNPOD_REMOTE_DIR` in `.runpod.env`:

```bash
RUNPOD_REMOTE_DIR=/root/projects/cuvarbase
```

Then re-run setup:

```bash
./scripts/setup-remote.sh
```

### Running Jupyter Notebooks

SSH into RunPod and start Jupyter:

```bash
ssh -p ${RUNPOD_SSH_PORT} -L 8888:localhost:8888 ${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}
cd /workspace/cuvarbase
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

Open http://localhost:8888 in your local browser.

### Persistent Storage

RunPod's `/workspace` directory is persistent. Large datasets or results can be stored there and will survive pod restarts.

## Scripts Reference

- `scripts/sync-to-runpod.sh` - Sync local code to RunPod
- `scripts/test-remote.sh` - Run tests on RunPod and show results
- `scripts/setup-remote.sh` - Initial environment setup
- `.runpod.env` - Your RunPod configuration (not in git)
- `.runpod.env.template` - Template for configuration
