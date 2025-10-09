# Memory Module

This module contains classes for managing GPU memory allocation and data transfer
for various periodogram computations.

## Contents

### `NFFTMemory`
Memory management for Non-equispaced Fast Fourier Transform operations.

**Used by:** `NFFTAsyncProcess`, `LombScargleAsyncProcess`

### `ConditionalEntropyMemory`
Memory management for Conditional Entropy period-finding operations.

**Used by:** `ConditionalEntropyAsyncProcess`

### `LombScargleMemory`
Memory management for Lomb-Scargle periodogram computations.

**Used by:** `LombScargleAsyncProcess`

## Design Philosophy

Memory management classes are separated from computation logic to:

1. **Improve modularity**: Memory allocation code is isolated and reusable
2. **Enable testing**: Memory classes can be tested independently
3. **Support flexibility**: Different memory strategies can be swapped easily
4. **Enhance clarity**: Clear separation between data management and computation

## Common Patterns

All memory classes follow similar patterns:

```python
# Create memory container
memory = SomeMemory(stream=stream, **kwargs)

# Set data
memory.fromdata(t, y, dy, allocate=True)

# Transfer to GPU
memory.transfer_data_to_gpu()

# Compute (in parent process class)
# ...

# Transfer results back
memory.transfer_results_to_cpu()
```

## Usage

```python
from cuvarbase.memory import NFFTMemory, ConditionalEntropyMemory, LombScargleMemory

# Or for backward compatibility:
from cuvarbase.cunfft import NFFTMemory
from cuvarbase.ce import ConditionalEntropyMemory
from cuvarbase.lombscargle import LombScargleMemory
```

Note: The old import paths still work for backward compatibility.
