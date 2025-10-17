# Migration Guide: Upgrading to cuvarbase 0.4.0

This guide helps users upgrade from earlier versions (especially Python 2.7) to cuvarbase 0.4.0.

## What's Changed

### Breaking Changes

**Python Version Requirement**
- **OLD:** Python 2.7, 3.4, 3.5, 3.6
- **NEW:** Python 3.7, 3.8, 3.9, 3.10, 3.11 or later
- **Action:** Upgrade your Python installation if needed

**Dependencies**
- **Removed:** `future` package (no longer needed)
- **Updated:** `numpy>=1.17` (was `>=1.6`)
- **Updated:** `scipy>=1.3` (was unspecified)
- **Action:** Dependencies will be updated automatically during installation

### Non-Breaking Changes

**API Compatibility**
- ✅ All public APIs remain unchanged
- ✅ Function signatures are the same
- ✅ Return values are the same
- ✅ No code changes needed if you're on Python 3.7+

## Step-by-Step Upgrade

### For Python 3.7+ Users (Easy)

If you're already using Python 3.7 or later, upgrading is simple:

```bash
# Upgrade cuvarbase
pip install --upgrade cuvarbase

# That's it! Your existing code should work without changes
```

### For Python 2.7 Users (Requires Python Upgrade)

If you're still on Python 2.7, you need to upgrade Python first:

**Option 1: Use Conda (Recommended)**
```bash
# Create a new environment with Python 3.11
conda create -n cuvarbase-py311 python=3.11
conda activate cuvarbase-py311

# Install cuvarbase
pip install cuvarbase
```

**Option 2: System Python Upgrade**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.11 python3.11-pip

# macOS with Homebrew
brew install python@3.11

# Install cuvarbase with the new Python
python3.11 -m pip install cuvarbase
```

**Option 3: Use Docker (Easiest)**
```bash
# Use the provided Docker image
docker pull nvidia/cuda:11.8.0-devel-ubuntu22.04
docker run -it --gpus all nvidia/cuda:11.8.0-devel-ubuntu22.04

# Inside the container:
pip3 install cuvarbase
```

### Updating Your Code

**If you're migrating from Python 2.7, update your scripts:**

**Before (Python 2.7):**
```python
from __future__ import print_function, division
from builtins import range

import cuvarbase.bls as bls

# Your code here
```

**After (Python 3.7+):**
```python
# No __future__ or builtins imports needed!
import cuvarbase.bls as bls

# Your code here - everything else stays the same!
```

## Common Issues and Solutions

### Issue 1: ImportError for 'future' package

**Error:**
```
ImportError: No module named 'future'
```

**Solution:**
This is expected! The `future` package is no longer needed. Simply upgrade cuvarbase:
```bash
pip install --upgrade cuvarbase
```

### Issue 2: Python version too old

**Error:**
```
ERROR: Package 'cuvarbase' requires a different Python: 3.6.x not in '>=3.7'
```

**Solution:**
Upgrade to Python 3.7 or later (see upgrade steps above).

### Issue 3: PyCUDA installation problems

**Error:**
```
ERROR: Failed building wheel for pycuda
```

**Solution:**
This is a known issue with PyCUDA. Try:
```bash
# Install CUDA toolkit first (if not installed)
# Then install numpy before pycuda
pip install numpy>=1.17
pip install pycuda

# Finally install cuvarbase
pip install cuvarbase
```

Or use Docker (recommended):
```bash
docker run -it --gpus all nvidia/cuda:11.8.0-devel-ubuntu22.04
pip3 install cuvarbase
```

### Issue 4: Existing code breaks with syntax errors

**Error:**
```python
print "Hello"  # SyntaxError in Python 3
```

**Solution:**
Update Python 2 syntax to Python 3:
```python
print("Hello")  # Python 3 syntax
```

Use the `2to3` tool to automatically convert:
```bash
2to3 -w yourscript.py
```

## Testing Your Migration

After upgrading, test your installation:

```python
# Test basic import
import cuvarbase
print(f"cuvarbase version: {cuvarbase.__version__}")

# Test core functionality
from cuvarbase import bls
print("BLS module loaded successfully")

# Your existing tests should pass
```

## Docker Quick Start

The easiest way to get started with cuvarbase 0.4.0:

```bash
# Build the Docker image
cd cuvarbase/
docker build -t cuvarbase:0.4.0 .

# Run with GPU support
docker run -it --gpus all cuvarbase:0.4.0

# Inside the container, install cuvarbase
pip3 install cuvarbase

# Start using it!
python3
>>> import cuvarbase
>>> # Your code here
```

## Rollback (If Needed)

If you need to rollback to the previous version:

```bash
# Install the last Python 2.7-compatible version
pip install cuvarbase==0.2.5

# Note: You'll need Python 2.7 or 3.4-3.6 for this version
```

## Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/johnh2o2/cuvarbase/issues)
2. Review the [Installation Guide](INSTALL.rst)
3. Read the [Implementation Notes](IMPLEMENTATION_NOTES.md)
4. Open a new issue with:
   - Your Python version: `python --version`
   - Your cuvarbase version: `pip show cuvarbase`
   - The full error message
   - Your operating system

## What's Next?

Future improvements planned (see MODERNIZATION_ROADMAP.md):
- Phase 3: Enhanced documentation
- Phase 4: Expanded test coverage
- Phase 5: Optional CPU fallback with Numba
- Phase 6: Performance optimizations
- Phase 7: API improvements

## Summary

**For most users:**
- If on Python 3.7+: Just `pip install --upgrade cuvarbase`
- If on Python 2.7: Upgrade Python first, then install cuvarbase
- No code changes needed (if already using Python 3)

**Key Benefits of 0.4.0:**
- Cleaner, more maintainable code
- Modern Python packaging
- Better compatibility with current Python ecosystem
- CI/CD for quality assurance
- Docker support for easy deployment

---

**Questions?** Open an issue on GitHub or refer to the documentation.

**Date:** 2025-10-14  
**Version:** 0.4.0  
**Python Required:** 3.7+
