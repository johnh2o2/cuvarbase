# import pycuda.autoinit causes problems when running e.g. FFT
import pycuda.autoprimaryctx

# Version
__version__ = "0.3.0"

# For backward compatibility, import all main classes
from .base import GPUAsyncProcess
from .memory import (
    NFFTMemory, 
    ConditionalEntropyMemory, 
    LombScargleMemory
)

# Import periodogram implementations
from .cunfft import NFFTAsyncProcess, nfft_adjoint_async
from .ce import ConditionalEntropyAsyncProcess, conditional_entropy, conditional_entropy_fast
from .lombscargle import LombScargleAsyncProcess, lomb_scargle_async
from .pdm import PDMAsyncProcess
from .bls import *
from .nufft_lrt import NUFFTLRTAsyncProcess, NUFFTLRTMemory

__all__ = [
    'GPUAsyncProcess',
    'NFFTMemory',
    'ConditionalEntropyMemory',
    'LombScargleMemory',
    'NFFTAsyncProcess',
    'ConditionalEntropyAsyncProcess',
    'LombScargleAsyncProcess',
    'PDMAsyncProcess',
    'NUFFTLRTAsyncProcess',
    'NUFFTLRTMemory',
]
