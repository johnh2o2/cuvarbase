
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class GPUAsyncProcess(object):
	def __init__(self, device=0, reader=None, nstreams=None, function_kwargs=None):
		self.reader = reader
		self.function_kwargs = {} if function_kwargs is None else function_kwargs
		self.streams = []
		self.gpu_data = []
		self.results = []
		self._adjust_nstreams = nstreams is None
		if not nstreams is None:
			self._create_streams(nstreams)
				
		self.prepared_functions = {}	

	def _create_streams(self, n):
		for i in range(n):
			self.streams.append(cuda.Stream())

	def _compile_and_prepare_functions(self):
		raise NotImplementedError()

	def run(self, *args, **kwargs):
		raise NotImplementedError()

	def finish(self):
		for i, stream in enumerate(self.streams):
			stream.synchronize()
