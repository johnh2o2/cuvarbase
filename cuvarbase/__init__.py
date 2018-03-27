#import pycuda.autoinit 
__all__ = ['bls', 'ce', 'core', 'cunfft', 'lombscargle', 'pdm', 'utils']
__version__ = "0.2.0"

import atexit
import pycuda.tools
import pycuda.driver as cuda

def _cleanup_cuda():
	#print("CLEANING CONTEXTS")
	while cuda.Context.get_current() is not None:
		#print(" --> FOUND ONE <--")
		cuda.Context.pop()
		#print("     [cleaned]")

	pycuda.tools.clear_context_caches()

atexit.register(_cleanup_cuda)
