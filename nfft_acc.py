import numpy as np 
import sys
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from scipy import fftpack
import skcuda.fft as cufft
from nfft import nfft_adjoint as nfft_adjoint_cpu
from nfft.utils import nfft_matrix
from nfft.kernels import KERNELS
import matplotlib.pyplot as plt
from time import time


gpu_nfft_utils = SourceModule(open('nfft_cuda.cu', 'r').read())

gpu_precompute_psi = gpu_nfft_utils.get_function("precompute_psi")
gpu_fast_gaussian_grid = gpu_nfft_utils.get_function("fast_gaussian_grid")
gpu_slow_gaussian_grid = gpu_nfft_utils.get_function("slow_gaussian_grid")
gpu_divide_phi_hat = gpu_nfft_utils.get_function("divide_phi_hat")
gpu_center_fft     = gpu_nfft_utils.get_function('center_fft')

BLOCK_SIZE=256
def nfft_adjoint_accelerated(x, y, N, m=8, fast=True, sigma=2, stream=None, 
							just_return_gridded_data=False):	
	n0 = np.int32(len(x))
	n = np.int32(sigma * N)
	m = np.int32(m)

	y_batch = np.ravel(y)
	nbatch = np.int32( len(y) if hasattr(y[0], '__iter__') else 1)

	b =  np.float32(float(2 * sigma * m) / ((2 * sigma - 1) * np.pi))
	
	g    = gpuarray.to_gpu(np.zeros(nbatch * n, dtype=np.float32))
	ghat = gpuarray.to_gpu(np.zeros(nbatch * n, dtype=np.complex64))

	y_g = gpuarray.to_gpu(np.array(y_batch, dtype=np.float32))
	x_g = gpuarray.to_gpu(np.array(x, dtype=np.float32))

	if fast:

		q1 = gpuarray.to_gpu(np.zeros(n0, dtype=np.float32))
		q2 = gpuarray.to_gpu(np.zeros(n0, dtype=np.float32))
		q3 = gpuarray.to_gpu(np.zeros(2 * m + 1, dtype=np.float32))
		GRID_SIZE = int(np.ceil(float(n0 + 2 * m + 1) / BLOCK_SIZE))
		kernel_kwargs = dict( block=(BLOCK_SIZE, 1, 1), grid=(GRID_SIZE, 1), stream=stream)

		gpu_precompute_psi(x_g, q1, q2, q3, n0, n,  m, b, **kernel_kwargs)

		GRID_SIZE = int(np.ceil(float(n0 * nbatch) / BLOCK_SIZE))
		kernel_kwargs = dict( block=(BLOCK_SIZE, 1, 1), grid=(GRID_SIZE, 1), stream=stream)
		gpu_fast_gaussian_grid(x_g, y_g, g, q1, q2, q3, n0, n, nbatch, m, **kernel_kwargs)

	GRID_SIZE = int(np.ceil(float(n * nbatch) / BLOCK_SIZE))
	kernel_kwargs = dict( block=(BLOCK_SIZE, 1, 1), grid=(GRID_SIZE, 1), stream=stream)

	if not fast:
		gpu_slow_gaussian_grid( x_g, y_g, g, n0, n, nbatch,  m, b, **kernel_kwargs)

	if just_return_gridded_data:
		return g
	
	gpu_center_fft(g, ghat, n, nbatch, **kernel_kwargs)

	plan = cufft.Plan(n, np.complex64, np.complex64, batch=nbatch, 
						stream=stream, istride=1, ostride=1, idist=n, odist=n)

	cufft.ifft(ghat, ghat, plan)

	gpu_divide_phi_hat(ghat, n, nbatch, b, **kernel_kwargs)

	gh = ghat.get()
	#print(gh)

	results = []
	for i in range(nbatch):
		
		inds = i * n + n // 2 - N // 2 + np.arange(N)

		if fast:
			results.append(gh[inds] / n)	
		else:
			results.append(gh[inds] / n) 

	#del g, ghat, y_g, x_g, q1, q2, q3
	return results


def test_ffts():
	ndata = 1000
	rand = np.random.RandomState(100)
	y = rand.randn(ndata)

	yhat = np.empty(len(y))

	yg = gpuarray.to_gpu(y.astype(np.complex64))
	yghat = gpuarray.to_gpu(yhat.astype(np.complex64))

	plan = Plan(len(y), np.complex64, np.complex64)
	ifft(yg, yghat, plan)

	yhat = fftpack.ifft(y) * len(y)

	print((yhat - yghat.get() )/ np.absolute(yhat))

def test_batched_gridding():
	ndata = 10000
	sigma = 2
	nbatch = 10
	n = int(1E6)
	fast = False
	rand = np.random.RandomState(100)

	get_y = lambda N : rand.rand(N)
	get_x = lambda N : shifted(np.sort(rand.rand(N)) - 0.5)
	
	x = get_x(ndata)

	yarrs = [ get_y(len(x)) for i in range(nbatch) ]

	batched_grid = lambda *args, **kwargs : nfft_adjoint_accelerated(*args, 
											  just_return_gridded_data=True, fast=fast, 
											  sigma=sigma, **kwargs)

	gb = batched_grid(x, yarrs, n)
	grids_batched = []
	for i in range(nbatch):
		grids_batched.append(gb[n*i:n*(i+1)])

	grids_nonbatched = []
	for y in yarrs:
		grids_nonbatched.append(batched_grid(x, y, n))

	for i, (gbatch, gnonbatch) in enumerate(zip(grids_batched, grids_nonbatched)):
		for j, (G, GN) in enumerate(zip(gbatch, gnonbatch)):
			if 2 * abs(G - GN) / abs(G + GN) > 1E-5:
				print(i, j, G, GN)
				

def shifted(x):
    """Shift x values to the range [-0.5, 0.5)"""
    return -0.5 + (x + 0.5) % 1

if __name__ == '__main__':
	#test_batched_gridding()
	ndata = 10000
	sigma = 2
	m=8
	n = 2 ** int(np.ceil(np.log2(1E6)))
	rand = np.random.RandomState(100)
	signal_freqs = np.linspace(0.1, 0.4, 10)

	random_times = lambda N : shifted(np.sort(rand.rand(N) - 0.5))
	noise = lambda : 0.1 * rand.randn(len(x))
	omega = lambda freq : 2 * np.pi * freq * len(x) 
	phase = lambda : 2 * np.pi * rand.rand()

	
	random_signal = lambda X, frq : np.cos(omega(frq) * X - phase()) + noise()

	x = random_times(ndata)
	y = [ random_signal(x, freq) for freq in signal_freqs ]

	t0 = time()
	fhats = nfft_adjoint_accelerated(x, y, n, fast=True, sigma=sigma, m=m)
	dt_fast = time() - t0
	ncpu = len(signal_freqs)
	t0 = time()
	fhat_cpus = [ nfft_adjoint_cpu(x, Y, n, 
						sigma=sigma, m=m, use_fft=True, truncated=True) for i, Y in enumerate(y) if i < ncpu ]
	
	dt_cpu = time() - t0

	print(dt_fast / len(signal_freqs), dt_cpu / ncpu)
	
	#sys.exit()
	#fhat_cpus = nfft_adjoint_accelerated(x, y, n, m, fast=False)
	import matplotlib.pyplot as plt

	
	for i, (fhat, fhat_cpu) in enumerate(zip(fhats, fhat_cpus)):
		freqs = np.arange(len(fhat)) - len(fhat) / 2
		f, ax = plt.subplots()
		X = np.absolute(fhat_cpu)
		Y = np.absolute(fhat)
		#ax.scatter(freqs, 2 * (Y - X) / np.median(Y + X), marker='.', s=1, alpha=0.5)
		
		ax.scatter(X, Y, s=1, alpha=0.05)
		#ax.plot(X, color='k')
		#ax.plot(Y, color='r', alpha=0.5)
		#ax.set_xscale('log')
		#ax.set_yscale('log')
		#ax.set_xlim(1E-1, 1.1 * max([ max(X), max(Y) ]))
		#ax.set_ylim(1E-1, 1.1 * max([ max(X), max(Y) ]))
		#ax.plot(freqs, np.absolute(fhat_cpu), color='b', alpha=0.6 / (i + 1))
		#ax.plot(freqs, np.absolute(fhat) , color='r', alpha=0.6 / (i + 1))
		#ax.axvline( freq * ndata)
		
		#xmin, xmax = ax.get_xlim()
		#xline = np.logspace(np.log10(xmin), np.log10(xmax), 1000)
		#ax.plot(xline, xline, ls=':', color='k')
		plt.show()
		plt.close(f)