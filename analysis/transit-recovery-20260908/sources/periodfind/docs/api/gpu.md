# GPU Backend

The GPU backend uses CUDA via Cython extensions. Requires the CUDA toolkit.

```bash
pip install cython numpy
pip install -e .
```

## Classes

The GPU module re-exports four classes compiled from Cython/CUDA:

- `periodfind.gpu.ConditionalEntropy`
- `periodfind.gpu.AOV`
- `periodfind.gpu.LombScargle`
- `periodfind.gpu.FPW`

These have the **same interface** as the [CPU backend classes](cpu.md).
Constructors accept the same parameters, and the `.calc()` method has an
identical signature and return type.

!!! note
    Because the GPU classes are compiled Cython extension types, their
    documentation cannot be auto-generated without a CUDA build environment.
    Refer to the [CPU Backend](cpu.md) docs for the full API — the interface
    is identical.

## Usage

```python
# Direct import
from periodfind.gpu import ConditionalEntropy, AOV, LombScargle, FPW

ce = ConditionalEntropy(n_phase=10, n_mag=10)
results = ce.calc(times, mags, periods, period_dts)

# Or use the device-agnostic factory
import periodfind
periodfind.set_device('gpu')
ce = periodfind.ConditionalEntropy(n_phase=10, n_mag=10)
```
