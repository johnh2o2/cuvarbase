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


gpu_nfft_utils = SourceModule(open('nfft_cuda.cu', 'r').read())

gpu_precompute_psi = gpu_nfft_utils.get_function("precompute_psi")
gpu_fast_gaussian_grid = gpu_nfft_utils.get_function("fast_gaussian_grid")
gpu_slow_gaussian_grid = gpu_nfft_utils.get_function("slow_gaussian_grid")
gpu_divide_phi_hat = gpu_nfft_utils.get_function("divide_phi_hat")
gpu_resize_for_batch_fft = gpu_nfft_utils.get_function('resize_for_batch_fft')

BLOCK_SIZE=256
def nfft_adjoint_accelerated(x, f, n, m, fast=True):
	x_batch, f_batch, n_batch, n0_batch = [], [], [], []
	nbatch = 1
	if hasattr(n, '__iter__'):
	
		for X, Y, NG in zip(x, f, n):

			x_batch.extend(list(X))
			f_batch.extend(list(Y))
			n_batch.append(NG)
			n0_batch.append(len(X))
		nbatch = len(n)
	else:
		x_batch = list(x)
		f_batch = list(f)
		n_batch = [ n ]
		n0_batch = [ len(x) ]

	n_total = sum(n_batch)
	n0_total = sum(n0_batch)

	sigma = float(n_batch[0]) / float(n0_batch[0])

	b =  (2 * sigma * m) / ((2 * sigma - 1) * np.pi)

	fftsize = max(n_batch)
	
	g = gpuarray.zeros(n_total, dtype=np.float32)
	ghat = gpuarray.zeros(nbatch * fftsize, dtype=np.complex64)

	

	n_g = gpuarray.to_gpu(np.array(n_batch, dtype=np.int32))
	n0_g = gpuarray.to_gpu(np.array(n0_batch, dtype=np.int32))

	f_g = gpuarray.to_gpu(np.array(f_batch, dtype=np.float32))
	x_g = gpuarray.to_gpu(np.array(x_batch, dtype=np.float32))

	if fast:

		q1 = gpuarray.zeros(n0_total, dtype=np.float32)
		q2 = gpuarray.zeros(n0_total, dtype=np.float32)
		q3 = gpuarray.zeros(2 * m + 1, dtype=np.float32)

		GRID_SIZE = int(np.ceil(float(n0_total + 2 * m + 1) / BLOCK_SIZE))# = int(2**(np.ceil(np.log2(n))))
	
		gpu_precompute_psi(x_g, q1, q2, q3, n0_g, n_g, np.int32(nbatch),  np.int32(m), np.float32(b), block=(BLOCK_SIZE, 1, 1), grid=(GRID_SIZE, 1))
	
		gpu_fast_gaussian_grid(g, f_g, x_g, n_g, n0_g, np.int32(nbatch),  np.int32(m), q1, q2, q3, block=(BLOCK_SIZE, 1, 1), grid=(GRID_SIZE, 1))

	GRID_SIZE = int(np.ceil(float(n_total) / BLOCK_SIZE))
	

	if not fast:
		gpu_slow_gaussian_grid(g, f_g, x_g, n_g, n0_g, np.int32(nbatch),  np.int32(m), np.float32(b), block=(BLOCK_SIZE, 1, 1), grid=(GRID_SIZE, 1))
	
	gpu_resize_for_batch_fft(g, ghat, n_g, np.int32(nbatch), np.int32(fftsize), block=(BLOCK_SIZE, 1, 1), grid=(GRID_SIZE, 1));
	
	#sys.exit()
	plan = cufft.Plan(fftsize, np.complex64, np.complex64, batch=nbatch, idist=fftsize, odist=fftsize, istride=1, ostride=1)
	#plan = cufft.Plan(fftsize, np.complex64, np.complex64)
	cufft.ifft(ghat, ghat, plan)

	GRID_SIZE = int(np.ceil(nbatch * float(fftsize) / BLOCK_SIZE))
	gpu_divide_phi_hat(ghat, n_g, np.int32(m), np.int32(nbatch), np.int32(fftsize), np.float32(b), block=(BLOCK_SIZE, 1, 1), grid=(GRID_SIZE, 1))


	gh = ghat.get()
	#print(gh)

	results = []
	for i in range(nbatch):
		N = n0_batch[i]
		#low  = i * fftsize + np.arange(N//2, N)
		#high = i * fftsize + np.arange(N//2)
		inds = i * fftsize + fftsize // 2 - N // 2 + np.arange(N)
		#results.append(np.concatenate([ gh[klow], gh[khigh] ]))
		#results.append(gh[-(N//2) + np.arange(N)])
		if fast:
			results.append(gh[inds])	
		else:
			results.append(gh[inds] / n_batch[i]) 
	#if nbatch == 1:
	#	return results[0]
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

	sys.exit()


def batched_grid(x, y, n, m, fast=True):
	x_batch, y_batch, n_batch, n0_batch = [], [], [], []
	nbatch = 1
	if hasattr(n, '__iter__'):
	
		for X, Y, NG in zip(x, y, n):

			x_batch.extend(list(X))
			y_batch.extend(list(Y))
			n_batch.append(NG)
			n0_batch.append(len(X))
		nbatch = len(n)
	else:
		x_batch = list(x)
		y_batch = list(y)
		n_batch = [ n ]
		n0_batch = [ len(x) ]

	n_total = sum(n_batch)
	n0_total = sum(n0_batch)

	sigma = float(n_batch[0]) / float(n0_batch[0])

	b =  (2 * sigma * m) / ((2 * sigma - 1) * np.pi)

	g = gpuarray.zeros(n_total, dtype=np.float32)

	n_g = gpuarray.to_gpu(np.array(n_batch, dtype=np.int32))
	n0_g = gpuarray.to_gpu(np.array(n0_batch, dtype=np.int32))

	y_g = gpuarray.to_gpu(np.array(y_batch, dtype=np.float32))
	x_g = gpuarray.to_gpu(np.array(x_batch, dtype=np.float32))

	if fast:

		q1 = gpuarray.zeros(n0_total, dtype=np.float32)
		q2 = gpuarray.zeros(n0_total, dtype=np.float32)
		q3 = gpuarray.zeros(2 * m + 1, dtype=np.float32)

		GRID_SIZE = int(np.ceil(float(n0_total + 2 * m + 1) / BLOCK_SIZE))# = int(2**(np.ceil(np.log2(n))))
	
		gpu_precompute_psi(x_g, q1, q2, q3, n0_g, n_g, np.int32(nbatch),  np.int32(m), np.float32(b), block=(BLOCK_SIZE, 1, 1), grid=(GRID_SIZE, 1))
	
		gpu_fast_gaussian_grid(g, y_g, x_g, n_g, n0_g, np.int32(nbatch),  np.int32(m), q1, q2, q3, block=(BLOCK_SIZE, 1, 1), grid=(GRID_SIZE, 1))

	GRID_SIZE = int(np.ceil(float(n_total) / BLOCK_SIZE))
	

	if not fast:
		gpu_slow_gaussian_grid(g, y_g, x_g, n_g, n0_g, np.int32(nbatch),  np.int32(m), np.float32(b), block=(BLOCK_SIZE, 1, 1), grid=(GRID_SIZE, 1))
	
	
	return g.get()


def test_batched_gridding():
	ndata = 10
	sigma = 3
	nbatch = 10
	m = 8
	s = 0.1
	fast = False
	rand = np.random.RandomState(100)

	get_y = lambda N : rand.rand(N)
	get_x = lambda N : shifted(np.sort(rand.rand(N)) - 0.5)
	
	xarrs = [ get_x(ndata - 2 * int(0.5 * s * ndata * rand.randn())) for i in range(nbatch) ]
	yarrs = [ get_y(len(X)) for X in xarrs ]

	n = [ sigma * len(X) for X in xarrs]

	gb = batched_grid(xarrs, yarrs, n, m, fast=fast)

	grids_batched = []
	offset = 0
	for ngrid in n:
		grids_batched.append(gb[offset:ngrid + offset])
		offset += ngrid

	grids_nonbatched = []
	for X, Y, N in zip(xarrs, yarrs, n):
		grids_nonbatched.append(batched_grid(X, Y, N, m, fast=fast ))

	for i, (gbatch, gnonbatch) in enumerate(zip(grids_batched, grids_nonbatched)):

		for j, (G, GN) in enumerate(zip(gbatch, gnonbatch)):
			if 2 * abs(G - GN) / abs(G + GN) > 1E-5:
				print(i, j, G, GN)
				

def shifted(x):
    """Shift x values to the range [-0.5, 0.5)"""
    return -0.5 + (x + 0.5) % 1

if __name__ == '__main__':
	#test_ffts()

	#test_batched_gridding()
	#sys.exit()
	ndata = 30000
	rand = np.random.RandomState(100)
	signal_freqs = np.linspace(0.1, 0.4, 1000)
	x, y, ndatas = [], [], []

	for freq in [ 0.1 ]:
		#dsize = ndata - 2 * int(0.05 * ndata * rand.randn())
		dsize = ndata
		X = shifted(np.sort(rand.rand(dsize) - 0.5))
		Y = np.zeros_like(X)
		for i, frq in enumerate(signal_freqs):
			Y += np.cos(2 * np.pi * frq * X * len(X) - 2* np.pi * rand.rand()) 
		Y += 0.1 * rand.randn(len(X))
		ndatas.append(len(X))
		x.append(X)
		y.append(Y)

	sigma = 3
	m = 8

	n = [ nd * sigma for nd in ndatas ]

	fhats = nfft_adjoint_accelerated(x, y, n, m, fast=False)
	fhat_cpus = [ nfft_adjoint_cpu(X, Y, len(X), sigma=sigma, m=m, use_fft=True, truncated=True) for X, Y in zip(x, y) ]
	#fhat_cpus = nfft_adjoint_accelerated(x, y, n, m, fast=False)
	import matplotlib.pyplot as plt

	f, ax = plt.subplots()
	for i, (fhat, fhat_cpu) in enumerate(zip(fhats, fhat_cpus)):
		freqs = np.arange(len(fhat)) - len(fhat) / 2

		X = np.absolute(fhat_cpu)
		Y = np.absolute(fhat)
		ax.scatter(X, 2 * (Y - X) / np.mean(X + Y), marker='.', s=1, alpha=0.5)
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