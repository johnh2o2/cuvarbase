#include <stdio.h>
#define RESTRICT __restrict__
#define CONSTANT const
//{CPP_DEFS}

// Device/global functions shared with bls_optimized.cu live in a single
// source file to prevent the two kernels from drifting apart (see
// bls_common.cuh and test_kernel_drift.py). Only the functions that
// differ on purpose stay below: full_bls_no_sol (this file uses the
// interleaved [yw, w] shared layout and a full tree reduction) versus
// full_bls_no_sol_optimized in bls_optimized.cu, and reduction_max (full
// tree reduction here vs warp-shuffle finish there).
//{INCLUDE bls_common.cuh}

__global__ void full_bls_no_sol(
	                    const float* __restrict__ t,
	                    const float* __restrict__ yw,
	                    const float* __restrict__ w,
						float* __restrict__ bls,
						const float* __restrict__ freqs,
						const unsigned int * __restrict__ nbins0,
						const unsigned int * __restrict__ nbinsf,
						unsigned int ndata,
						unsigned int nfreq,
						unsigned int freq_offset,
						unsigned int hist_size,
						unsigned int noverlap,
						float dlogq,
						float dphi,
                        unsigned int ignore_negative_delta_sols){
	unsigned int i = get_id();

	extern __shared__ float sh[];

	float *block_bins = sh;
	float *best_bls = (float *)&sh[2 * hist_size];

	__shared__ float f0;
	__shared__ int nb0, nbf, max_bin_width;

#ifdef USE_LOG_BIN_SPACING
	__shared__ int tot_nbins;
#endif

	unsigned int s;
	int b;
	float phi, bls1, bls2, thread_max_bls, thread_yw, thread_w;

	// this will be inefficient for block sizes >> number of bins per frequency
	unsigned int i_freq = blockIdx.x;
	while (i_freq < nfreq){

		thread_max_bls = 0.f;

		if (threadIdx.x == 0){
			// read frequency from global memory
			f0 = freqs[i_freq + freq_offset];

			// read nbins from global memory
			nb0 = nbins0[i_freq + freq_offset];
			nbf = nbinsf[i_freq + freq_offset];

			max_bin_width = divrndup(nbf, nb0);

#ifdef USE_LOG_BIN_SPACING
			tot_nbins = count_tot_nbins(nb0, nbf, dlogq);
#endif
		}

		// wait for broadcasting to finish
		__syncthreads();

		// intialize bins to 0 (synchronization is necessary here...)
		for(unsigned int k = threadIdx.x; k < nbf; k += blockDim.x){
			block_bins[2 * k] = 0.f;
			block_bins[2 * k + 1] = 0.f;
		}

		// wait for initialization to finish
		__syncthreads();

		// histogram the data
		for (unsigned int k = threadIdx.x; k < ndata; k += blockDim.x){
			phi = mod1(t[k] * f0);

			b = mod((int) floorf(((float) nbf) * phi - dphi), (int) nbf);

			// shared memory atomics should (hopefully) be faster.
			atomicAdd(&(block_bins[2 * b]), yw[k]);
			atomicAdd(&(block_bins[2 * b + 1]), w[k]);
		}

		// wait for everyone to finish adding data to the histogram
		__syncthreads();

		// get max bls for this THREAD
#ifdef USE_LOG_BIN_SPACING
		for (unsigned int n = threadIdx.x; n < tot_nbins; n += blockDim.x){

			unsigned int bin_offset = 0;
			unsigned int nb = nb0;
			while ((bin_offset + nb) * noverlap < n){
				bin_offset += nb;
				nb += dnbins(nb, dlogq);
			}

			b = (((int) n) - ((int) (bin_offset * noverlap))) % nb;
			s = (((int) n) - ((int) (bin_offset * noverlap))) / nb;

			thread_yw = 0.f;
			thread_w = 0.f;
			unsigned int m0 = 0;

			for (unsigned int m = b; m < b + nb; m ++){
				thread_yw += block_bins[2 * (m % nbf)];
				thread_w += block_bins[2 * (m % nbf) + 1];
			}

			bls1 = bls_value(thread_yw, thread_w, ignore_negative_delta_sols);
			if (bls1 > thread_max_bls)
				thread_max_bls = bls1;
		}

#else
		for (unsigned int n = threadIdx.x; n < nbf; n += blockDim.x){

			thread_yw = 0.f;
			thread_w = 0.f;
			unsigned int m0 = 0;

			for (unsigned int m = 1; m < max_bin_width; m += dnbins(m, dlogq)){
				for (s = m0; s < m; s++){
					thread_yw += block_bins[2 * ((n + s) % nbf)];
					thread_w += block_bins[2 * ((n + s) % nbf) + 1];
				}
				m0 = m;

				bls1 = bls_value(thread_yw, thread_w, ignore_negative_delta_sols);
				if (bls1 > thread_max_bls)
					thread_max_bls = bls1;
			}
		}
#endif

		best_bls[threadIdx.x] = thread_max_bls;

		// wait for everyone to finish
		__syncthreads();

		// get max bls for this BLOCK
		for(unsigned int k = (blockDim.x / 2); k > 0; k /= 2){
			if(threadIdx.x < k){
				bls1 = best_bls[threadIdx.x];
				bls2 = best_bls[threadIdx.x + k];

				best_bls[threadIdx.x] = (bls1 > bls2) ? bls1 : bls2;
			}
			__syncthreads();
		}

		// store block max to global memory
		if (threadIdx.x == 0)
			bls[i_freq + freq_offset] = best_bls[0];

		// increment frequency
		i_freq += gridDim.x;
	}
}


__global__ void reduction_max(float *arr, unsigned int *arr_args, unsigned int nfreq,
	                          unsigned int nbins, unsigned int stride,
                              float *block_max, unsigned int *block_arg_max,
                              unsigned int offset, unsigned int init){

	__shared__ float partial_max[BLOCK_SIZE];
	__shared__ unsigned int partial_arg_max[BLOCK_SIZE];

	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int nblocks_per_freq = gridDim.x / nfreq;
	unsigned int nthreads_per_freq = blockDim.x * nblocks_per_freq;




	//	freq_no / b
	//			----block 1 -----       ----- block N ------------------------
	//		  0 | 0 1 2 .. B - 1 | ... | (N - 1)B, ... , ndata, ..., N * B - 1|
	//
	//			---block N + 1---       ---- block 2N ------------------------
	//		  1 | 0 1 2 .. B - 1 | ... | (N - 1)B, ... , ndata, ..., N * B - 1|
	//			...
	//
	//			---(nf - 1)N ----       --- nf * N ---
	//   nf - 1 | ..             | ... |             |

	unsigned int fno = id / nthreads_per_freq;
	unsigned int b   = id % nthreads_per_freq;

	// read part of array from global memory into shared memory
	partial_max[threadIdx.x] = (fno < nfreq && b < nbins) ?
	                                 arr[fno * stride + b] : -1.f;

	partial_arg_max[threadIdx.x] = (fno < nfreq && b < nbins) ?
									(
										(init == 1) ?
											b : arr_args[fno * stride + b]
									) : 0;

	__syncthreads();

	float m1, m2;

	// reduce to find max of shared memory array
	for(int s = blockDim.x / 2; s > 0; s /= 2){
		if(threadIdx.x < s){
			m1 = partial_max[threadIdx.x];
			m2 = partial_max[threadIdx.x + s];

			partial_max[threadIdx.x] = (m1 > m2) ? m1 : m2;

			partial_arg_max[threadIdx.x] = (m1 > m2) ?
			 						partial_arg_max[threadIdx.x] :
			 						partial_arg_max[threadIdx.x + s];
		}

		__syncthreads();
	}

	// store partial max back into global memory
	if (threadIdx.x == 0 && fno < nfreq){
		unsigned int i = (gridDim.x == nfreq) ? 0 :
			                 fno * stride - fno * nblocks_per_freq;

		i += blockIdx.x + offset;

		block_max[i] = partial_max[0];
		block_arg_max[i] = partial_arg_max[0];
	}
}
