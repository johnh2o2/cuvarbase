# Cuvarbase Architecture

This document describes the organization and architecture of the cuvarbase codebase.

## Overview

Cuvarbase provides GPU-accelerated implementations of various period-finding and
variability analysis algorithms for astronomical time series data.

## Directory Structure

```
cuvarbase/
├── __init__.py              # Main package exports
├── base/                    # Core abstractions and base classes
│   ├── __init__.py
│   ├── async_process.py    # GPUAsyncProcess base class
│   └── README.md
├── memory/                  # GPU memory management
│   ├── __init__.py
│   ├── nfft_memory.py      # NFFT memory management
│   ├── ce_memory.py        # Conditional Entropy memory
│   ├── lombscargle_memory.py  # Lomb-Scargle memory
│   └── README.md
├── periodograms/            # Periodogram implementations (future)
│   ├── __init__.py
│   └── README.md
├── kernels/                 # CUDA kernel source files
│   ├── bls.cu
│   ├── ce.cu
│   ├── cunfft.cu
│   ├── lomb.cu
│   └── pdm.cu
├── tests/                   # Unit tests
│   └── ...
├── bls.py                   # Box Least Squares implementation
├── ce.py                    # Conditional Entropy implementation
├── lombscargle.py           # Lomb-Scargle implementation
├── cunfft.py                # NFFT implementation
├── pdm.py                   # Phase Dispersion Minimization
├── core.py                  # Backward compatibility wrapper
└── utils.py                 # Utility functions
```

## Module Organization

### Base Module (`cuvarbase.base`)

Contains fundamental abstractions used across all periodogram implementations:

- **`GPUAsyncProcess`**: Base class for GPU-accelerated computations
  - Manages CUDA streams for asynchronous operations
  - Provides template methods for compilation and execution
  - Implements batched processing for large datasets

### Memory Module (`cuvarbase.memory`)

Encapsulates GPU memory management for different algorithms:

- **`NFFTMemory`**: Memory management for NFFT operations
- **`ConditionalEntropyMemory`**: Memory for conditional entropy
- **`LombScargleMemory`**: Memory for Lomb-Scargle computations

**Benefits:**
- Separation of concerns: memory allocation separate from computation
- Reusability: memory patterns can be shared
- Testability: memory management can be tested independently
- Clarity: clear API for data transfer between CPU and GPU

### Periodograms Module (`cuvarbase.periodograms`)

Placeholder for future organization of periodogram implementations.
Currently provides backward-compatible imports.

### Implementation Files

Core algorithm implementations (currently at package root):

- **`bls.py`**: Box Least Squares periodogram for transit detection
- **`ce.py`**: Conditional Entropy period finder
- **`lombscargle.py`**: Generalized Lomb-Scargle periodogram
- **`cunfft.py`**: Non-equispaced Fast Fourier Transform
- **`pdm.py`**: Phase Dispersion Minimization

### CUDA Kernels (`cuvarbase/kernels`)

GPU kernel implementations in CUDA C:
- Compiled at runtime using PyCUDA
- Optimized for specific periodogram computations

## Design Principles

### 1. Abstraction Through Inheritance

All periodogram implementations inherit from `GPUAsyncProcess`:

```python
class SomeAsyncProcess(GPUAsyncProcess):
    def _compile_and_prepare_functions(self):
        # Compile CUDA kernels
        pass
    
    def run(self, data, **kwargs):
        # Execute computation
        pass
```

### 2. Memory Management Separation

Memory management is separated from computation logic:

```python
# Memory class handles allocation/transfer
memory = SomeMemory(stream=stream)
memory.fromdata(t, y, allocate=True)

# Process class handles computation
process = SomeAsyncProcess()
result = process.run(data, memory=memory)
```

### 3. Asynchronous GPU Operations

All operations use CUDA streams for asynchronous execution:
- Enables overlapping of computation and data transfer
- Supports concurrent processing of multiple datasets
- Improves GPU utilization

### 4. Backward Compatibility

The restructuring maintains complete backward compatibility:

```python
# Old imports still work
from cuvarbase import GPUAsyncProcess
from cuvarbase.cunfft import NFFTMemory

# New imports are also available
from cuvarbase.base import GPUAsyncProcess  
from cuvarbase.memory import NFFTMemory
```

## Common Patterns

### Creating a Periodogram Process

```python
import pycuda.autoprimaryctx
from cuvarbase import LombScargleAsyncProcess

# Create process
proc = LombScargleAsyncProcess(nstreams=2)

# Prepare data
data = [(t1, y1, dy1), (t2, y2, dy2)]

# Run computation
results = proc.run(data)

# Wait for completion
proc.finish()

# Extract results
freqs, powers = results[0]
```

### Batched Processing

```python
# Process large datasets in batches
results = proc.batched_run(large_data, batch_size=10)
```

### Memory Reuse

```python
# Allocate memory once
memory = proc.allocate(data)

# Reuse for multiple runs
results1 = proc.run(data1, memory=memory)
results2 = proc.run(data2, memory=memory)
```

## Extension Points

### Adding a New Periodogram

1. Create a new memory class in `cuvarbase/memory/`
2. Inherit from `GPUAsyncProcess`
3. Implement required methods:
   - `_compile_and_prepare_functions()`
   - `run()`
   - `allocate()` (optional)
4. Add CUDA kernel to `cuvarbase/kernels/`
5. Add tests to `cuvarbase/tests/`

### Example

```python
from cuvarbase.base import GPUAsyncProcess
from cuvarbase.memory import BaseMemory

class NewPeriodogramMemory(BaseMemory):
    # Memory management implementation
    pass

class NewPeriodogramProcess(GPUAsyncProcess):
    def _compile_and_prepare_functions(self):
        # Load and compile CUDA kernel
        pass
    
    def run(self, data, **kwargs):
        # Execute computation
        pass
```

## Testing

Tests are organized in `cuvarbase/tests/`:
- Each implementation has corresponding test file
- Tests verify both correctness and performance
- Comparison with CPU reference implementations

## Future Improvements

1. **Complete periodograms module migration**: Move implementations to subpackages
2. **Unified memory interface**: Create common base class for memory managers
3. **Plugin architecture**: Enable easy addition of new algorithms
4. **Documentation generation**: Auto-generate API docs from docstrings
5. **Performance profiling**: Built-in profiling utilities

## Dependencies

- **PyCUDA**: Python interface to CUDA
- **scikit-cuda**: Additional CUDA functionality (FFT)
- **NumPy**: Array operations
- **SciPy**: Scientific computing utilities

## References

For more details on specific modules:
- [Base Module](base/README.md)
- [Memory Module](memory/README.md)
- [Periodograms Module](periodograms/README.md)
