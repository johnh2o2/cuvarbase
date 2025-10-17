# Contributing to cuvarbase

Thank you for your interest in contributing to cuvarbase! This document provides guidelines and standards for maintaining code quality and consistency.

## Code of Conduct

Please be respectful and constructive in all interactions with the project community.

## Development Setup

### Prerequisites

- Python 3.7 or later
- CUDA-capable GPU (NVIDIA)
- CUDA Toolkit (11.x or 12.x recommended)
- PyCUDA >= 2017.1.1 (avoid 2024.1.2)
- scikit-cuda

### Installation for Development

```bash
git clone https://github.com/johnh2o2/cuvarbase.git
cd cuvarbase
pip install -e .[test]
```

### Running Tests

```bash
pytest cuvarbase/tests/
```

## Code Standards

### Python Version Support

- **Minimum Python version**: 3.7
- **Tested versions**: 3.7, 3.8, 3.9, 3.10, 3.11, 3.12
- Do not use Python 2.7 compatibility code

### Naming Conventions

Follow PEP 8 naming conventions:

- **Classes**: `PascalCase` (e.g., `GPUAsyncProcess`, `NFFTMemory`)
- **Functions**: `snake_case` (e.g., `conditional_entropy`, `lomb_scargle_async`)
- **Variables**: `snake_case` (e.g., `block_size`, `max_frequency`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_BLOCK_SIZE`)
- **Private members**: prefix with `_` (e.g., `_compile_and_prepare_functions`)

#### CUDA/GPU Specific Naming

For clarity in GPU code, we use suffixes to indicate memory location:
- `_g`: GPU memory (e.g., `t_g`, `freqs_g`)
- `_c`: CPU/host memory (e.g., `ce_c`, `results_c`)
- `_d`: Device functions (in CUDA kernels)

### Code Style

#### Imports

Group imports in the following order, separated by blank lines:
1. Standard library imports
2. Third-party imports (numpy, scipy, pycuda, etc.)
3. Local application imports

```python
import sys
import resource

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from .core import GPUAsyncProcess
from .utils import find_kernel
```

#### Type Hints

While not required for all code, type hints are encouraged for public APIs:

```python
def autofrequency(
    t: np.ndarray,
    nyquist_factor: float = 5,
    samples_per_peak: float = 5,
    minimum_frequency: float = None,
    maximum_frequency: float = None
) -> np.ndarray:
    """Generate frequency grid for periodogram."""
    ...
```

#### Docstrings

Use NumPy-style docstrings for all public functions and classes:

```python
def function_name(param1, param2, param3=None):
    """
    Brief description of function.

    Longer description if needed, explaining the purpose and behavior
    in more detail.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2
    param3 : type, optional (default: None)
        Description of param3

    Returns
    -------
    return_type
        Description of return value

    Raises
    ------
    ExceptionType
        When this exception is raised

    Examples
    --------
    >>> result = function_name(1, 2)
    >>> print(result)
    3

    See Also
    --------
    related_function : Related functionality

    Notes
    -----
    Additional information about implementation details or caveats.
    """
    ...
```

#### Comments

- Use inline comments sparingly and only when the code is not self-explanatory
- Prefer descriptive variable names over comments
- Document complex algorithms with block comments or docstrings

### CUDA Kernel Conventions

For CUDA kernels (`.cu` files):

- Use `__global__` for GPU kernel functions
- Use `__device__` for device-only functions
- Document kernel parameters and thread/block organization
- Use descriptive names: `kernel_name` or `operation_type`

Example:
```cuda
__global__ void compute_periodogram(
    FLT *t,           // observation times
    FLT *y,           // observation values
    FLT *freqs,       // frequency grid
    FLT *output,      // output periodogram
    unsigned int n,   // number of observations
    unsigned int nf   // number of frequencies
) {
    // Kernel implementation
}
```

### Memory Management

- Always check for GPU memory allocation failures
- Use CUDA streams for asynchronous operations
- Clean up GPU resources in class destructors or context managers
- Document memory ownership and transfer patterns

### Testing

- Write unit tests for new functionality
- Tests should be in `cuvarbase/tests/`
- Use `pytest` for test framework
- Mock GPU operations when appropriate to allow CPU-only testing
- Test edge cases and error conditions

Example test structure:
```python
def test_function_name():
    """Test brief description."""
    # Setup
    data = np.array([...])
    
    # Execute
    result = function_name(data)
    
    # Assert
    assert result.shape == expected_shape
    np.testing.assert_allclose(result, expected, rtol=1e-5)
```

### Documentation

- Update documentation when changing public APIs
- Include examples in docstrings
- Add entries to CHANGELOG.rst for significant changes
- Update README.rst if changing installation or usage

## Pull Request Process

1. **Fork and branch**: Create a feature branch from `main`
2. **Make changes**: Follow the code standards above
3. **Test**: Ensure all tests pass
4. **Document**: Update docstrings and documentation
5. **Commit**: Use clear, descriptive commit messages
6. **Pull Request**: Submit PR with description of changes

### Commit Messages

Use clear, descriptive commit messages:
- Start with a verb in imperative mood (e.g., "Add", "Fix", "Update")
- Keep first line under 72 characters
- Add detailed description if needed

Examples:
```
Add support for weighted conditional entropy

Fix memory leak in BLS computation

Update documentation for NUFFT LRT method
- Add examples
- Clarify parameter descriptions
- Fix typos
```

## Performance Considerations

When contributing GPU code:
- Profile before optimizing
- Document any performance-critical sections
- Consider memory bandwidth vs. computation tradeoffs
- Test with various GPU architectures when possible

## Questions?

If you have questions about contributing, please:
- Check existing documentation
- Look at similar code in the repository
- Open an issue for discussion

Thank you for contributing to cuvarbase!
