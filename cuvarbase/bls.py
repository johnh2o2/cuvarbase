

import sys
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel

from pycuda.compiler import SourceModule
from .core import GPUAsyncProcess
from .utils import find_kernel

import resource
import numpy as np


def bin_yw(phi, y, w, nbins):
	
	bins = np.floor(phi * nbins).astype(np.int32)
	y_bins = np.zeros(nbins, np.float32)
	w_bins = np.zeros(nbins, np.float32)
	for b, Y, W in zip(bins, y, w):
		w_bins[b] += W
		y_bins[b] += Y * W

	return y_bins, w_bins

def reduction_max(max_func, arr, arr_args, nfreq, nbins, 
					stream, final_arr, final_argmax_arr, 
					final_index, block_size):
	# assert power of 2
	assert(block_size - 2 * (block_size / 2) == 0)
	
	block = (block_size, 1, 1)
	grid_size = int(np.ceil(float(nbins) / block_size)) * nfreq
	grid = (grid_size, 1)
	nbins0 = nbins

	init = np.int32(1)
	while (grid_size > nfreq):


		max_func.prepared_async_call(grid, block, stream, 
			                        arr.ptr, arr_args.ptr,
			                        np.int32(nfreq), np.int32(nbins0), 
			                        np.int32(nbins), 
			                        arr.ptr, arr_args.ptr, np.int32(0), init)
		init = np.int32(0)

		nbins0 = grid_size / nfreq
		grid_size = int(np.ceil(float(nbins0) / block_size)) * nfreq
		grid = (grid_size, 1)

	max_func.prepared_async_call(grid, block, stream,
									arr.ptr,  arr_args.ptr,
									np.int32(nfreq), np.int32(nbins0), np.int32(nbins),
									final_arr.ptr, final_argmax_arr.ptr,
									np.int32(final_index), init)
	



def bls_test(t, y, dy, freqs, max_mem_gb=2, qmin=0.02, qmax=0.5, nstreams=10, 
				bin_width=0.2, block_size=256, cpu_bin=False):
	module = SourceModule(open(find_kernel('bls'), 'r').read().replace('#define BLOCK_SIZE', 
											'#define BLOCK_SIZE %d'%(block_size)), 
												options=['--use_fast_math'])

	gpu_bls = module.get_function('binned_bls').prepare([ np.intp, np.intp,
								 np.intp, np.int32, np.float32, np.float32 ]) 

	gpu_max = module.get_function('reduction_max').prepare([ np.intp, np.int32, np.int32,
															np.int32, np.intp, np.int32 ])
	

	gpu_bin = module.get_function('bin_and_phase_fold').prepare([np.intp, np.intp, np.intp,
																np.intp, np.intp, 
																np.float32, np.int32, np.int32])
	#gpu_max = ReductionKernel(np.float32, neutral='0', reduce_expr='max(a,b)', 
	#							map_expr='x[i]', arguments='float *x')
	qmin = np.float32(qmin)
	qmax = np.float32(qmax)

	nbins = np.int32(1./(bin_width * qmin))
	ndata = np.int32(len(t))
	grid_size = int(np.ceil(float(nbins) / block_size))
	
	w = np.power(dy, -2)
	w /= sum(w)

	t_g = gpuarray.to_gpu(np.array(t).astype(np.float32))
	yw_g = gpuarray.to_gpu(np.array(y * w).astype(np.float32))
	w_g = gpuarray.to_gpu(np.array(w).astype(np.float32))

	yw_g_bins, w_g_bins, bls_tmp_gs, streams = [],[],[],[]
	for i in range(nstreams):
		streams.append(cuda.Stream())
		yw_g_bins.append(gpuarray.zeros(nbins, dtype=np.float32))
		w_g_bins.append(gpuarray.zeros(nbins, dtype=np.float32))
		bls_tmp_gs.append(gpuarray.zeros(nbins, dtype=np.float32))


	bls_g = gpuarray.zeros(len(freqs), dtype=np.float32)

	block=(block_size, 1, 1)
	
	grid = (grid_size, 1)
	

	for i, freq in enumerate(freqs):
		freq = freqs[i]

		j = i % nstreams
		yw_g_bin = yw_g_bins[j]
		w_g_bin = w_g_bins[j]
		bls_tmp_g = bls_tmp_gs[j]
		stream = streams[j]

		yw_g_bin.fill(np.float32(0), stream=stream)
		w_g_bin.fill(np.float32(0), stream=stream)

		bin_grid = (int(np.ceil(float(ndata) / block_size)), 1)

		gpu_bin.prepared_async_call(bin_grid, block, stream,
									t_g.ptr, yw_g.ptr, w_g.ptr, 
									yw_g_bin.ptr, w_g_bin.ptr,
									np.float32(freq), ndata, nbins)
		#stream.synchronize()

		gpu_bls.prepared_async_call(grid, block, stream, 
						yw_g_bin.ptr, w_g_bin.ptr, bls_tmp_g.ptr, nbins, qmin, qmax)
		

		#bls_tmp = bls_tmp_g.get()

		#stream.synchronize()
		reduction_max(gpu_max, bls_tmp_g, 1, nbins, stream, bls_g, i,  block_size)

		#stream.synchronize()

		#import matplotlib.pyplot as plt
		#plt.plot(bls_tmp)
		#plt.axhline(bls_g.get()[i])
		#plt.show()
	return bls_g.get()

def get_qphi(nbins0, nbinsf, alpha, noverlap):
	q_phi = []

	x = np.float32(1.)
	dphi = np.float32(np.float32(1.)/noverlap)
	while(np.int32(x * nbins0) <= nbinsf):

		nb = np.int32(x * nbins0)
		q = np.float32(1.)/nb

		qp = []
		for s in range(noverlap):
			for i in range(nb):
				phi = (i * q + np.float32(s) * dphi)%1.0
				qp.append((q, phi))

		q_phi.extend(qp)
		x *= np.float32(alpha)

	return q_phi

def bls_test_bst(t, y, dy, freqs, max_mem_gb=2, qmin=0.02, qmax=0.5, nstreams=10, 
				noverlap=10, alpha=1.5, block_size=256, cpu_bin=False, 
				batch_size=1, plot_status=False):
	module = SourceModule(open(find_kernel('bls'), 'r').read().replace('#define BLOCK_SIZE', 
											'#define BLOCK_SIZE %d'%(block_size)), 
												options=['--use_fast_math'])

	gpu_bls = module.get_function('binned_bls_bst').prepare([ np.intp, np.intp,
								 np.intp, np.int32 ]) 

	gpu_max = module.get_function('reduction_max').prepare([ np.intp, np.intp,
															 np.int32, np.int32, np.int32, 
															 np.intp, np.intp, np.int32, np.int32 ])
	

	gpu_bin = module.get_function('bin_and_phase_fold_bst_multifreq').prepare([np.intp, np.intp, np.intp,
																np.intp, np.intp, np.intp,  
																np.int32, np.int32, np.int32, 
																np.int32, np.int32, np.int32,
																np.float32, np.int32])
	nbins0 = 1
	nbinsf = 1
	while (int(1./(qmax)) > nbins0):
		nbins0 = np.ceil(alpha * nbins0)

	while (int(1./(qmin)) > nbinsf):
		nbinsf = np.ceil(alpha * nbinsf)

	nbins0 = np.int32(nbins0)
	nbinsf = np.int32(nbinsf)
	ndata = np.int32(len(t))


	q_phi = get_qphi(nbins0, nbinsf, alpha, noverlap)
	qvals, phivals = zip(*q_phi)

	nbins_tot = len(q_phi) / noverlap

	gs = batch_size * len(q_phi)

	grid_size = int(np.ceil(float(gs) / block_size))
	
	w = np.power(dy, -2)
	w /= sum(w)
	ybar = np.dot(w, y)
	YY = np.dot(w, np.power(y - ybar, 2))

	t_g = gpuarray.to_gpu(np.array(t).astype(np.float32))
	yw_g = gpuarray.to_gpu(np.array((y - ybar) * w).astype(np.float32))
	w_g = gpuarray.to_gpu(np.array(w).astype(np.float32))
	freqs_g = gpuarray.to_gpu(np.array(freqs).astype(np.float32))

	yw_g_bins, w_g_bins, bls_tmp_gs, bls_tmp_sol_gs, streams = [],[],[],[],[]
	for i in range(nstreams):
		streams.append(cuda.Stream())
		yw_g_bins.append(gpuarray.zeros(gs, dtype=np.float32))
		w_g_bins.append(gpuarray.zeros(gs, dtype=np.float32))
		bls_tmp_gs.append(gpuarray.zeros(gs, dtype=np.float32))
		bls_tmp_sol_gs.append(gpuarray.zeros(gs, dtype=np.int32))


	bls_g = gpuarray.zeros(len(freqs), dtype=np.float32)
	bls_sol_g = gpuarray.zeros(len(freqs), dtype=np.int32)

	block=(block_size, 1, 1)
	
	grid = (grid_size, 1)

	nbatches = int(np.ceil(float(len(freqs)) / batch_size))
	for batch in range(nbatches):

		nf = batch_size if batch < nbatches - 1 else \
		                   len(freqs) - batch * batch_size

		nf = np.int32(nf)
		j = batch % nstreams
		yw_g_bin = yw_g_bins[j]
		w_g_bin = w_g_bins[j]
		bls_tmp_g = bls_tmp_gs[j]
		bls_tmp_sol_g = bls_tmp_sol_gs[j]

		stream = streams[j]

		yw_g_bin.fill(np.float32(0), stream=stream)
		w_g_bin.fill(np.float32(0), stream=stream)

		bin_grid = (int(np.ceil(float(ndata * nf) / block_size)), 1)

		gpu_bin.prepared_async_call(bin_grid, block, stream,
									t_g.ptr, yw_g.ptr, w_g.ptr, 
									yw_g_bin.ptr, w_g_bin.ptr, freqs_g.ptr,
									ndata, nf, nbins0, nbinsf, 
									np.int32(batch_size * batch), noverlap,
									alpha, nbins_tot)
		#stream.synchronize()

		gpu_bls.prepared_async_call(grid, block, stream, 
						yw_g_bin.ptr, w_g_bin.ptr, 
						bls_tmp_g.ptr,  nf * nbins_tot * noverlap)
		
		bls_tmp = None if not plot_status else bls_tmp_g.get()

		#stream.synchronize()
		reduction_max(gpu_max, bls_tmp_g, bls_tmp_sol_g, 
						nf, nbins_tot * noverlap, stream, bls_g, bls_sol_g, 
						batch * batch_size,  block_size)
		if plot_status:
			import matplotlib.pyplot as plt
			for i in range(nf):
				plt.title("frequency %d, (%.3e)"%(i, freqs[i]))
				for j in range(noverlap):
					offset = i * nbins_tot * noverlap + j * nbins_tot
					inds = slice(offset, offset + nbins_tot)
					plt.plot(bls_tmp[inds])
				plt.axhline(bls_g.get()[i + batch * batch_size], color='r')
				#plt.axhline(bls_tmp2[i], ls=':', color='k')
				plt.show()
			#import sys
			#sys.exit()
	bls_sols = bls_sol_g.get()
	assert(not any(bls_sols < 0))
	#print(np.unique(bls_sols))
	qphi_sols = [ (qvals[b], phivals[b]) for b in bls_sols ]

	return bls_g.get()/YY, qphi_sols