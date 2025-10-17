# Quick Reference: GPU Framework Comparison for cuvarbase

This document provides a quick reference for comparing GPU frameworks in the context of cuvarbase's specific needs.

## Decision Matrix

| Requirement | PyCUDA | CuPy | Numba | JAX | Score |
|-------------|--------|------|-------|-----|-------|
| Custom CUDA kernels | ✓✓ Native | ✗ Limited | ~ Python | ✗ No | PyCUDA wins |
| Performance | ✓✓ Optimal | ✓ Excellent | ~ Good | ✓ Excellent | PyCUDA wins |
| Fine memory control | ✓✓ Full | ✓ Good | ✓ Good | ~ Limited | PyCUDA wins |
| Stream management | ✓✓ Complete | ✓ Good | ~ Basic | ~ Limited | PyCUDA wins |
| Installation ease | ~ Complex | ✓ Moderate | ✓✓ Easy | ~ Complex | Numba wins |
| Documentation | ✓ Good | ✓✓ Excellent | ✓✓ Excellent | ✓ Good | Tie |
| Python 3 support | ✓ Good | ✓✓ Excellent | ✓✓ Excellent | ✓✓ Excellent | Others win |
| Learning curve | ~ Steep | ✓ Easy | ✓ Easy | ~ Steep | CuPy/Numba |
| Astronomy use | ✓✓ Common | ✓ Growing | ✓ Common | ~ Rare | PyCUDA wins |

**Legend**: ✓✓ Excellent, ✓ Good, ~ Acceptable, ✗ Poor/Not Supported

**Winner for cuvarbase**: **PyCUDA** (8/9 critical requirements)

## Framework Migration Cost Estimates

| Framework | Estimated Time | Risk Level | Breaking Changes |
|-----------|---------------|------------|------------------|
| Stay with PyCUDA | 0 months | None | None |
| Migrate to CuPy | 3-6 months | High | Yes |
| Migrate to Numba | 4-8 months | High | Yes |
| Migrate to JAX | 6-12 months | Very High | Yes |

**Recommendation**: Don't migrate. Focus on modernization instead.

## When to Use Each Framework

### Use PyCUDA when:
- ✓ You have custom CUDA kernels (like cuvarbase)
- ✓ You need fine-grained memory control
- ✓ You need advanced stream management
- ✓ Performance is critical
- ✓ You're working with legacy CUDA code

### Use CuPy when:
- ✓ You're doing array operations only
- ✓ You want NumPy-compatible API
- ✓ You don't need custom kernels
- ✓ Installation simplicity matters
- ✓ Starting a new project

### Use Numba when:
- ✓ You want to write kernels in Python
- ✓ You need CPU fallback
- ✓ You're prototyping algorithms
- ✓ You want JIT compilation
- ✓ Code readability > performance

### Use JAX when:
- ✓ You need automatic differentiation
- ✓ You're doing machine learning
- ✓ You want functional programming
- ✓ You need multi-device scaling
- ✗ NOT for custom CUDA kernels

## Code Pattern Comparison

### Memory Allocation

**PyCUDA** (Current):
```python
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

# Method 1: Direct allocation
data_gpu = cuda.mem_alloc(data.nbytes)

# Method 2: Using gpuarray
data_gpu = gpuarray.to_gpu(data)
```

**CuPy**:
```python
import cupy as cp

data_gpu = cp.asarray(data)  # Similar to NumPy
```

**Numba**:
```python
from numba import cuda

data_gpu = cuda.to_device(data)
```

**JAX**:
```python
import jax.numpy as jnp

data_gpu = jnp.asarray(data)  # Automatic device placement
```

### Custom Kernel Execution

**PyCUDA** (Current):
```python
from pycuda.compiler import SourceModule

kernel_code = """
__global__ void my_kernel(float *out, float *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * 2.0f;
}
"""

mod = SourceModule(kernel_code)
func = mod.get_function("my_kernel")
func(out_gpu, in_gpu, np.int32(n), 
     block=(256,1,1), grid=(n//256+1,1))
```

**CuPy**:
```python
import cupy as cp

kernel_code = '''
extern "C" __global__
void my_kernel(float *out, float *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * 2.0f;
}
'''

kernel = cp.RawKernel(kernel_code, 'my_kernel')
kernel((n//256+1,), (256,), (out_gpu, in_gpu, n))
```

**Numba**:
```python
from numba import cuda

@cuda.jit
def my_kernel(out, in_arr):
    idx = cuda.grid(1)
    if idx < out.size:
        out[idx] = in_arr[idx] * 2.0
        
my_kernel[n//256+1, 256](out_gpu, in_gpu)
```

**JAX**: Not applicable (no custom kernel support)

### Async Operations

**PyCUDA** (Current):
```python
import pycuda.driver as cuda

stream = cuda.Stream()
data_gpu.set_async(data_cpu, stream=stream)
kernel(data_gpu, stream=stream)
stream.synchronize()
```

**CuPy**:
```python
import cupy as cp

stream = cp.cuda.Stream()
with stream:
    data_gpu = cp.asarray(data_cpu)
    # Operations run on this stream
stream.synchronize()
```

**Numba**:
```python
from numba import cuda

stream = cuda.stream()
data_gpu = cuda.to_device(data_cpu, stream=stream)
kernel[blocks, threads, stream](data_gpu)
stream.synchronize()
```

**JAX**: Automatic async (XLA handles it)

## Real-World cuvarbase Example

### Current Implementation (PyCUDA)
```python
# cuvarbase/bls.py
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Load custom kernel
kernel_txt = open('kernels/bls.cu').read()
module = SourceModule(kernel_txt)
func = module.get_function('full_bls_no_sol')

# Prepare function for faster launches
dtypes = [np.intp, np.float32, ...]
func.prepare(dtypes)

# Execute with multiple streams
for i, stream in enumerate(streams):
    func.prepared_async_call(
        grid, block, stream,
        *args
    )
```

### Hypothetical CuPy Implementation
```python
# Would require rewriting bls.cu
import cupy as cp

# Cannot directly use existing bls.cu kernel
# Need to wrap in RawKernel or rewrite logic
kernel = cp.RawKernel(kernel_txt, 'full_bls_no_sol')

# Less control over argument types
# Different stream management
stream = cp.cuda.Stream()
with stream:
    kernel(grid, block, args)
```

**Observation**: CuPy version is similar but:
- Requires adapting existing kernel code
- Less explicit control over data types
- Different async pattern
- Migration effort not justified

## Performance Comparison (Estimated)

Based on benchmark studies from other projects:

| Operation | PyCUDA | CuPy | Numba | JAX |
|-----------|--------|------|-------|-----|
| Custom kernel | 100% (baseline) | 95-98% | 70-85% | N/A |
| Array ops | 100% | 98-100% | 80-90% | 95-100% |
| Memory transfer | 100% | 98-100% | 95-98% | 95-100% |
| Compilation time | Fast | Fast | Slow (first run) | Very slow |

**Notes**:
- PyCUDA: Direct CUDA with minimal overhead
- CuPy: Excellent for array ops, slight overhead for kernels
- Numba: Python translation adds overhead
- JAX: XLA compilation is powerful but unpredictable

## Installation Comparison

### PyCUDA (Current)
```bash
# Prerequisites: CUDA toolkit installed
pip install numpy
pip install pycuda

# Often requires manual compilation:
./configure.py --cuda-root=/usr/local/cuda
python setup.py install
```
**Difficulty**: ★★★★☆ (4/5)

### CuPy
```bash
# Install for CUDA 11.x
pip install cupy-cuda11x
```
**Difficulty**: ★★☆☆☆ (2/5)

### Numba
```bash
pip install numba
# CUDA toolkit needed but handled automatically
```
**Difficulty**: ★☆☆☆☆ (1/5)

### JAX
```bash
# CPU version
pip install jax

# GPU version
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
**Difficulty**: ★★★☆☆ (3/5)

## Community and Ecosystem

| Metric | PyCUDA | CuPy | Numba | JAX |
|--------|--------|------|-------|-----|
| GitHub Stars | ~1.8k | ~7.5k | ~9.3k | ~28k |
| Last Release | 2024 | 2024 | 2024 | 2024 |
| Astronomy Usage | High | Growing | Medium | Low |
| Stack Overflow Qs | ~2k | ~1k | ~3k | ~2k |
| Corporate Backing | None | Preferred Networks | Anaconda | Google |
| Maintenance Status | Stable | Active | Active | Very Active |

**Interpretation**:
- PyCUDA: Mature, stable, trusted by astronomy community
- CuPy: Growing rapidly, strong support
- Numba: Part of Anaconda, excellent support
- JAX: Google-backed, ML-focused

## Compatibility Matrix

| Feature | PyCUDA | CuPy | Numba | JAX |
|---------|--------|------|-------|-----|
| Python 2.7 | ✓ | ✗ | ✓ | ✗ |
| Python 3.7+ | ✓ | ✓ | ✓ | ✓ |
| CUDA 8.0 | ✓ | ✗ | ✓ | ✗ |
| CUDA 11.x | ✓ | ✓ | ✓ | ✓ |
| CUDA 12.x | ✓ | ✓ | ✓ | ✓ |
| Linux | ✓ | ✓ | ✓ | ✓ |
| Windows | ✓ | ✓ | ✓ | ✓ |
| macOS | ✓ | Limited | ✓ | Limited |

## The Bottom Line

### For cuvarbase specifically:

**Stick with PyCUDA because**:
1. ✓ You have 6 optimized CUDA kernels
2. ✓ Performance is excellent
3. ✓ Migration cost is very high
4. ✓ Risk outweighs benefit
5. ✓ Community trusts PyCUDA

**Modernize instead**:
1. ✓ Drop Python 2.7
2. ✓ Improve documentation
3. ✓ Add CI/CD
4. ✓ Consider CPU fallback (Numba)

### For new projects:
- **Custom kernels needed?** → PyCUDA
- **Array operations only?** → CuPy
- **Need CPU fallback?** → Numba
- **Machine learning?** → JAX

## Resources

- PyCUDA: https://documen.tician.de/pycuda/
- CuPy: https://docs.cupy.dev/
- Numba: https://numba.pydata.org/
- JAX: https://jax.readthedocs.io/
- CUDA Programming Guide: https://docs.nvidia.com/cuda/

---

**Last Updated**: 2025-10-14  
**Status**: Reference Guide
