# Base Module

This module contains the core base classes and abstractions used throughout cuvarbase.

## Contents

### `GPUAsyncProcess`

The base class for all GPU-accelerated periodogram computations. It provides:

- Stream management for asynchronous GPU operations
- Abstract methods for compilation and execution
- Batched processing capabilities
- Common patterns for GPU workflow

## Usage

This module is primarily used internally. For user-facing functionality, see the main
periodogram implementations in `cuvarbase.ce`, `cuvarbase.lombscargle`, etc.

```python
from cuvarbase.base import GPUAsyncProcess

# Or for backward compatibility:
from cuvarbase import GPUAsyncProcess
```

## Design

The `GPUAsyncProcess` class follows a template pattern where subclasses implement:
- `_compile_and_prepare_functions()`: Compile CUDA kernels
- `run()`: Execute the computation

This provides a consistent interface across different periodogram methods.
