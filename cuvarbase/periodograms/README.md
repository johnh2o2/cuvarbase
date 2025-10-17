# Periodograms Module

This module will contain structured implementations of various periodogram and 
period-finding algorithms.

## Planned Structure

The periodograms module is designed to organize related algorithms together:

```
periodograms/
├── __init__.py           # Main exports
├── bls/                  # Box Least Squares
│   ├── __init__.py
│   ├── core.py          # Main BLS implementation
│   └── variants.py      # BLS variants
├── ce/                   # Conditional Entropy
│   ├── __init__.py
│   └── core.py
├── lombscargle/          # Lomb-Scargle
│   ├── __init__.py
│   └── core.py
├── nfft/                 # Non-equispaced FFT
│   ├── __init__.py
│   └── core.py
└── pdm/                  # Phase Dispersion Minimization
    ├── __init__.py
    └── core.py
```

## Current Status

Currently, this module provides imports for backward compatibility. The actual
implementations remain in the root `cuvarbase/` directory to minimize disruption.

Future work could move implementations here for better organization.

## Usage

```python
# Current usage (backward compatible)
from cuvarbase import LombScargleAsyncProcess, ConditionalEntropyAsyncProcess

# Future usage (when migration is complete)
from cuvarbase.periodograms import LombScargleAsyncProcess
from cuvarbase.periodograms import ConditionalEntropyAsyncProcess
```

## Design Goals

1. **Clear organization**: Group related algorithms together
2. **Discoverability**: Easy to find and understand available methods
3. **Extensibility**: Simple to add new periodogram variants
4. **Backward compatibility**: Existing code continues to work
