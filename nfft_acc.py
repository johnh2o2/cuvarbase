#!/usr/bin/env python

import numpy as np 
import sys
import pycuda.driver as cuda
#import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from scipy import fftpack
#import skcuda.fft as cufft
from nfft import nfft_adjoint as nfft_adjoint_cpu
from nfft.utils import nfft_matrix
from nfft.kernels import KERNELS
import matplotlib.pyplot as plt
from time import time
import resource
import threading

#cuda.init()





def pdm_accelerated(x, y, err, freqs, batch_size=1, dphi=0.05, block_size=128):

        nlcs = np.int32( len(y) if hasattr(y[0], '__iter__') else 1)
	
	weights = [ np.power(s, -2).astype(np.float32) for s in err ]
	weights = [ wgt / np.sum(wgt) for wgt in weights ]

	streams = [ cuda.Stream() for i in range(nlcs) ]

	w_gs = [ gpuarray.zeros(len(x), np.float32) for i in range(nlcs) ]
	y_gs = [ gpuarray.zeros(len(x), np.float32) for i in range(nlcs) ]
	power_gs = [ gpuarray.zeros( len(freqs), np.float32) for i in range(nlcs) ]

	x_g = gpuarray.to_gpu(x.astype(np.float32))
	freqs_g = gpuarray.to_gpu(freqs.astype(np.float32))
	
	power_cpu = [ np.zeros(len(freqs), dtype=np.float32) for i in range(nlcs) ]

	ndata = np.int32(len(x))
	nfreqs = np.int32(len(freqs))
	dphi  = np.float32(dphi)	
	
	block = (block_size, 1, 1)
	grid_size = int(np.ceil(float(nfreqs) / block_size))
	grid = (grid_size, 1)
	for i in range(nlcs):
		w_gs[i].set_async(weights[i], stream=streams[i])
		y_gs[i].set_async(y[i].astype(np.float32), stream=streams[i])
	
	for i in range(nlcs):
		gpu_pdm.prepared_async_call(grid, block, streams[i],
				x_g.ptr, y_gs[i].ptr, w_gs[i].ptr, freqs_g.ptr, power_gs[i].ptr, 
				ndata, nfreqs, dphi)
	for i in range(nlcs):
	#for i in range(nlcs):
		#power_gs[i].get_async(stream=streams[i], ary=power_cpu[i])
		power_gs[i].get(ary=power_cpu[i])

	return power_cpu
	
def nfft_adjoint_accelerated(x, y, N, m=8, fast=True, sigma=2, batch_size=5, 
							just_return_gridded_data=False, block_size=160, prepared=True,
							run_async=True):	
	gpu_nfft_utils = SourceModule(open('nfft_cuda.cu', 'r').read(), options=[ '--use_fast_math'])
	gpu_precompute_psi = gpu_nfft_utils.get_function("precompute_psi").prepare([ np.intp, np.intp, np.intp, np.intp, np.int32, np.int32, np.int32, np.float32 ])
	gpu_fast_gaussian_grid = gpu_nfft_utils.get_function("fast_gaussian_grid").prepare([ np.intp, np.intp, np.intp, np.intp, np.intp, np.intp, np.int32, np.int32, np.int32, np.int32])
	gpu_slow_gaussian_grid = gpu_nfft_utils.get_function("slow_gaussian_grid").prepare([ np.intp, np.intp, np.intp, np.int32, np.int32, np.int32, np.int32, np.float32 ])
	gpu_divide_phi_hat = gpu_nfft_utils.get_function("divide_phi_hat").prepare([ np.intp, np.int32, np.int32, np.float32 ])
	gpu_center_fft     = gpu_nfft_utils.get_function('center_fft').prepare([ np.intp, np.intp, np.int32, np.int32 ])
	n0 = np.int32(len(x))
	n = np.int32(sigma * N)
	m = np.int32(m)
	b =  np.float32(float(2 * sigma * m) / ((2 * sigma - 1) * np.pi))

	kernel_kwargs = lambda nthreads, stream : dict(block=(block_size, 1, 1), 
									        grid=(int(np.ceil(float(nthreads) / block_size)), 1), 
									        stream=stream)
	y_all = np.ravel(y).astype(np.float32)

	ndata = np.int32( len(y) if hasattr(y[0], '__iter__') else 1)
	batch_size = min([ batch_size, ndata ])
	nbatches = ndata // batch_size
	assert(nbatches * batch_size == ndata)

	streams, events = [], []
	marker_names = [ 'htod_start', 'htod_stop', 'precompute_psi_start', 'precompute_psi_stop', 
					'gridding_start', 'gridding_stop', 'center_fft_start', 'center_fft_stop', 
					'divide_phi_start', 'divide_phi_stop', 'dtoh_start', 'dtoh_stop',
					'fft_start', 'fft_stop']
	
	streams = [ None for batch in range(nbatches) ]
	if run_async:
		streams = [ cuda.Stream() for batch in range(nbatches) ]
	events =  [ dict([ (name, cuda.Event()) for name in marker_names ]) for batch in range(nbatches) ]
	
	#g, ghat, x_g, y_g, q1, q2, q3 = [],[],[],[],[],[],[]
	x_g = gpuarray.to_gpu(np.array(x, dtype=np.float32))
	q1, q2, q3 = None, None, None
	if fast:
		q1 = gpuarray.zeros(n0, np.float32)
		q2 = gpuarray.zeros(n0, dtype=np.float32)
		q3 = gpuarray.zeros(2 * m + 1, dtype=np.float32)

		kw = kernel_kwargs(n0 + 2 * m + 1, None)
		#event['precompute_psi_start'].record()
		gpu_precompute_psi.prepared_call(kw['grid'], kw['block'],
						x_g.ptr, q1.ptr, q2.ptr, q3.ptr, n0, n, m, b)

		#event['precompute_psi_stop'].record(stream)
	plot_filter = False
	if plot_filter:
		Q1 = q1.get()
		Q2 = q2.get()
		Q3 = q3.get()
		f = [ [ Q1[i] * pow(Q2[i], l) * Q3[l] for l in range(len(Q3)) ] for i in range(len(Q1)) ]
		L = np.arange(2 * m + 1) - m
		for F in f:
			plt.plot(L, F, alpha=0.02, color='k')
		plt.show()
		sys.exit()
	g_batch    = [ gpuarray.zeros(n  * batch_size, np.float32)   for i in range(nbatches) ]
	ghat_batch = [ gpuarray.zeros(n  * batch_size, np.complex64) for i in range(nbatches) ]
	y_g_batch  = [ gpuarray.zeros(n0 * batch_size, np.float32)   for i in range(nbatches) ]
	batch_size = np.int32(batch_size)

	ghat_cpu = np.zeros(n * ndata, dtype=np.complex64)
	
	for i, (stream, event) in enumerate(zip(streams, events)):
		print(i)
		event['htod_start'].record(stream)
		y_batch = y_all[i * batch_size * n0 : (i+1) * batch_size * n0]

		y_g_batch[i].set_async(y_batch, stream=stream)
		event['htod_stop'].record(stream)

		if not run_async and not stream is None:
			stream.synchronize()

	#for i, (stream, event, (bsize, y_batch)) in enumerate(zip(streams, events, batches)):
		if fast:
			kw = kernel_kwargs(n0 * batch_size, stream)
			event['gridding_start'].record(stream)

			if prepared:
				gpu_fast_gaussian_grid.prepared_async_call(kw['grid'], kw['block'], kw['stream'], 
					x_g.ptr, y_g_batch[i].ptr, g_batch[i].ptr, q1.ptr, q2.ptr, q3.ptr, n0, n, batch_size, m)
			else:
				gpu_fast_gaussian_grid(x_g, y_g_batch[i], g_batch[i], q1, q2, q3, n0, n, batch_size, m, **kw)

		else:
			kw = kernel_kwargs(n * batch_size, stream)
			event['gridding_start'].record(stream)
			if prepared:
				gpu_slow_gaussian_grid.prepared_async_call(kw['grid'], kw['block'], kw['stream'], 
						x_g.ptr, y_g_batch[i].ptr, g_batch[i].ptr, n0, n, batch_size,  m, b)
			else:
				gpu_slow_gaussian_grid(x_g, y_g_batch[i], g_batch[i], n0, n, batch_size,  m, b, **kw)

		event['gridding_stop'].record(stream)	
		if not run_async and not stream is None:
			stream.synchronize()
		#gb = g_batch[i].get()[:n]
		#print(min(gb), max(gb))
		#plt.scatter(x, y_g_batch[i].get()[:n0], s=2, c='k')# y_g_batch[i].get(), alpha=0.2)
		#plt.plot(np.linspace(-0.5, 0.5, n), g_batch[i].get()[:n], alpha=0.2)
		#plt.show()
		#sys.exit()
		
		kw = kernel_kwargs(n * batch_size, stream)
		if prepared:
			event['center_fft_start'].record(stream)
			gpu_center_fft.prepared_async_call(kw['grid'], kw['block'], kw['stream'],
							g_batch[i].ptr, ghat_batch[i].ptr, n, batch_size)
		else:
			event['center_fft_start'].record(stream)
			gpu_center_fft(g_batch[i], ghat_batch[i], n, batch_size, **kw)

		if not run_async and not stream is None:
			stream.synchronize()
		
		event['center_fft_stop'].record(stream)
		event['fft_start'].record(stream)
		plan = cufft.Plan(n, np.complex64, np.complex64, batch=batch_size, 
				   stream=stream, istride=1, ostride=1, idist=n, odist=n)	
		cufft.ifft(ghat_batch[i], ghat_batch[i], plan)
		event['fft_stop'].record(stream)
		if not run_async and not stream is None:
			stream.synchronize()

		
		event['divide_phi_start'].record(stream)
		if prepared:
			gpu_divide_phi_hat.prepared_async_call(kw['grid'], kw['block'], kw['stream'],
							ghat_batch[i].ptr, n, batch_size, b)
		else:
			gpu_divide_phi_hat(ghat_batch[i], n, batch_size, b, **kw)
		if not run_async and not stream is None:
			stream.synchronize()
	
		event['divide_phi_stop'].record(stream)

	for i, (stream, event) in enumerate(zip(streams, events)):
		ghat_batch[i].get_async(stream=stream, ary=ghat_cpu[i * batch_size * n:(i+1)*batch_size * n])	
	#for i, (stream, event, (bsize, y_batch)) in enumerate(zip(streams, events, batches)):
		
	#	event['dtoh_start'].record(stream)
	#	event['dtoh_stop'].record(stream)

	#for i in range(len(streams)):
	#	gh.extend(gh_temps[i].tolist())
	#gh = np.array(gh)
	results = []
	for i in range(ndata):
		
		inds = i * n + n // 2 - N // 2 + np.arange(N)

		if fast:
			results.append(ghat_cpu[inds] / n)	
		else:
			results.append(ghat_cpu[inds] / n) 
	return results
	#return []

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
	ndata = 1000
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


	from multiprocessing import Pool
	import matplotlib.pyplot as plt
	
	ndata = 500
	p_min = 0.1 * 24 * 60. # minimum period (minutes)
	T = 10.    # baseline (years)
	H = 1 	   # number of harmonics
	oversampling = 5 # df = 1 / (o * T)
	fast = True
	batch_size = 10
	nlcs = 4 * batch_size
	block_size = 160

	# nominal number of frequencies needed
	Nf = int(oversampling * H * T * 60 * 24 * 365. / p_min)
	#print(Nf)
	#Nf = 10
	sigma = 2
	noise_sigma = 0.1
	m=8

	# nearest power of 2
	n = 2 ** int(np.ceil(np.log2(Nf)))
	print(n, Nf, float(n) / float(Nf))

	rand = np.random.RandomState(100)
	signal_freqs = np.linspace(0.1, 0.4, nlcs)


	random_times = lambda N : shifted(np.sort(rand.rand(N) - 0.5))
	noise = lambda : noise_sigma * rand.randn(len(x))
	omega = lambda freq : 2 * np.pi * freq * len(x) 
	phase = lambda : 2 * np.pi * rand.rand()

	
	random_signal = lambda X, frq : np.cos(omega(frq) * X - phase()) + noise()

	x = random_times(ndata)
	x_pdm = (x + 0.5) * T * 365
	y = [ random_signal(x, freq) for freq in signal_freqs ]
	err = [ noise_sigma * np.ones_like(Y) for Y in y ]
	df = 1./(T * oversampling)
	
	freqs_pdm = df * (0.5 +  np.arange(Nf))
#	cuda.start_profiler()
#	ref = cuda.Event()
#	ref.record()
	
	nbatches = int(np.ceil(float(len(y)) / batch_size))

	def gthread(args):
		n, batches = args
		print(n)
		proc = GPUBatchProcess(pdm_batch_function, device=n, function_kwargs=dict(freqs=freqs_pdm))
		print(proc)
		results = []
		for batch in batches:
			results.append(proc.run(batch))
		
		proc.finish()
		return results
	#ngpus = cuda.Device.count()
	#dev = cuda.Device(0)
	#ctx = dev.make_context()
	ngpus=1
	#cuda.start_profiler()
	pool = Pool(processes=ngpus)
	nbatches_per_gpu = int(np.ceil(float(nbatches) / ngpus))
	batches = [[]]
	for i, (ydat, errdat) in enumerate(zip(y, err)):
		data = [ (T, Y, E) for T, Y, E in zip(x_pdm, ydat, errdat) ]
		if len(batches[-1]) >= batch_size:
			batches.append([ data ])
		else:
			batches[-1].append(data)
	
	gpu_batches = [ (i, [ batches[i * nbatches_per_gpu + j] for j in range(nbatches_per_gpu) ]) for i in range(ngpus) ]

	
	results = pool.map(gthread, gpu_batches)
	

#	f, ax = plt.subplots()
#	ax.plot(frqs, pows[0])
#	plt.show()
#	sys.exit()
	
	#del ctx
	#cuda.stop_profiler()
	sys.exit()
	
	t0 = time()
	pows_pdm = pdm_accelerated(x_pdm, y, err, freqs_pdm, dphi=0.1, block_size=128)
	dt_async = time() - t0
	
	t0 = time()
	for Y, ERR in zip(y, err):
		pows_pdm = pdm_accelerated(x_pdm, Y, ERR, freqs_pdm, dphi=0.1, block_size=128)
	dt_serial = time() - t0
	print(dt_batched / len(y), dt_serial / len(y))
	#f, ax = plt.subplots()
	
	#j = 0
	#ax.plot(freqs_pdm, pows_pdm[j])
	#ax.axvline(signal_freqs[j] * len(x), ls=':', color='k')
	#plt.show()
	#t0 = time()
	#fhats = nfft_adjoint_accelerated(x, y, n, fast=fast, sigma=sigma, batch_size=batch_size,
	#								m=m, block_size=block_size)
	
	#dt_batch = time() - t0

	#fhats_nb = []
	#t0 = time()
	#for Y in y:
	#	fhats_nb.extend(nfft_adjoint_accelerated(x, Y, n, fast=fast, sigma=sigma, 
			#				m=m, block_size=block_size))
	#dt_nonbatch = time() - t0


	#warp_size = 32
	#timing_info = []
	#for warp_multiple in 1 + np.arange(32):
	#	block_size = warp_multiple * warp_size
	#	t0 = time()
	#	fhats = nfft_adjoint_accelerated(x, y, n, fast=True, sigma=sigma, 
	#											m=m, block_size=block_size)
	#	dt_fast = time() - t0
	#	timing_info.append((block_size, dt_fast))

	#for b, dt in timing_info:
	#	print(b, dt)

	ncpu = len(signal_freqs)
	t0 = time()
	fhat_cpus = [ nfft_adjoint_cpu(x, Y, n, 
									sigma=sigma, m=m, 
									use_fft=True, 
									truncated=True) \
					for i, Y in enumerate(y) if i < ncpu ]
	
	dt_cpu = time() - t0

	print(dt_batch / len(signal_freqs), dt_nonbatch / len(signal_freqs), dt_cpu / ncpu)
	
	#sys.exit()
	#fhat_cpus = nfft_adjoint_accelerated(x, y, n, m, fast=False)

	
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
