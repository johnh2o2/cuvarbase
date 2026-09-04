#include <stdio.h>
#define RESTRICT __restrict__
#define CONSTANT const
//{CPP_DEFS}

// Optimized version of BLS kernel with following improvements:
// 1. Fixed bank conflicts (separate yw/w arrays)
// 2. Explicit use of fast math intrinsics
// 3. Better memory access patterns
// 4. Warp-level reduction in final stages
//
// Device/global functions shared with bls.cu live in bls_common.cuh
// (inlined below) so the two kernels cannot drift apart. Only the
// functions that differ on purpose stay in this file: the bank-conflict
// -free full_bls_no_sol_optimized and the warp-shuffle reduction_max.
//{INCLUDE bls_common.cuh}

// OPTIMIZED VERSION of full_bls_no_sol
// Key improvements:
// 1. Separate yw/w arrays to avoid bank conflicts
// 2. Explicit fast math intrinsics
// 3. Warp-level reduction for final max finding
__global__ void full_bls_no_sol_optimized(
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

	// OPTIMIZATION: Separate yw/w arrays to avoid bank conflicts
	// Old layout: [yw0, w0, yw1, w1, ...]
	// New layout: [yw0, yw1, ..., ywN, w0, w1, ..., wN]
	float *block_bins_yw = sh;
	float *block_bins_w = (float *)&sh[hist_size];
	float *best_bls = (float *)&sh[2 * hist_size];

	__shared__ float f0;
	__shared__ int nb0, nbf, max_bin_width;

#ifdef USE_LOG_BIN_SPACING
	__shared__ int tot_nbins;
#endif

	unsigned int s;
	int b;
	float phi, bls1, bls2, thread_max_bls, thread_yw, thread_w;

	unsigned int i_freq = blockIdx.x;
	while (i_freq < nfreq){

		thread_max_bls = 0.f;

		if (threadIdx.x == 0){
			f0 = freqs[i_freq + freq_offset];
			nb0 = nbins0[i_freq + freq_offset];
			nbf = nbinsf[i_freq + freq_offset];
			// Widest box: floor(nbf / nb0), i.e. the largest m whose
			// q = m/nbf still satisfies q <= 1/nb0 (= the discretized
			// qmax).  This used to be divrndup(nbf, nb0) with a strict
			// `m < max_bin_width` loop, which is the same bound whenever
			// nb0 does not divide nbf but drops the qmax box itself when
			// it does (Sep 2026 audit, id 64: qmin=0.025/qmax=0.1 tested
			// only q <= 0.075).
			max_bin_width = nbf / nb0;

#ifdef USE_LOG_BIN_SPACING
			tot_nbins = count_tot_nbins(nb0, nbf, dlogq);
#endif
		}

		__syncthreads();

		// Initialize bins to 0 - now separate arrays
		for(unsigned int k = threadIdx.x; k < nbf; k += blockDim.x){
			block_bins_yw[k] = 0.f;
			block_bins_w[k] = 0.f;
		}

		__syncthreads();

		// Histogram the data - OPTIMIZATION: use fast math
		for (unsigned int k = threadIdx.x; k < ndata; k += blockDim.x){
			phi = mod1(t[k] * f0);

			b = mod((int) floorf(((float) nbf) * phi - dphi), (int) nbf);

			// OPTIMIZATION: Atomic adds on separate arrays (no bank conflicts)
			atomicAdd(&(block_bins_yw[b]), yw[k]);
			atomicAdd(&(block_bins_w[b]), w[k]);
		}

		__syncthreads();

		// Get max bls for this thread
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

			for (unsigned int m = b; m < b + nb; m ++){
				thread_yw += block_bins_yw[m % nbf];
				thread_w += block_bins_w[m % nbf];
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

			for (unsigned int m = 1; m <= max_bin_width; m += dnbins(m, dlogq)){
				for (s = m0; s < m; s++){
					thread_yw += block_bins_yw[(n + s) % nbf];
					thread_w += block_bins_w[(n + s) % nbf];
				}
				m0 = m;

				bls1 = bls_value(thread_yw, thread_w, ignore_negative_delta_sols);
				if (bls1 > thread_max_bls)
					thread_max_bls = bls1;
			}
		}
#endif

		best_bls[threadIdx.x] = thread_max_bls;

		__syncthreads();

		// Standard tree reduction down to single warp (32 threads)
		for(unsigned int k = (blockDim.x / 2); k >= 32; k /= 2){
			if(threadIdx.x < k){
				bls1 = best_bls[threadIdx.x];
				bls2 = best_bls[threadIdx.x + k];

				best_bls[threadIdx.x] = (bls1 > bls2) ? bls1 : bls2;
			}
			__syncthreads();
		}

		// Final warp reduction using shuffle (no sync needed)
		// After the loop above, best_bls[0...31] contains the values to reduce
		if (threadIdx.x < 32){
			float val = best_bls[threadIdx.x];

			// Warp shuffle reduction (no __syncthreads needed within a warp)
			for(int offset = 16; offset > 0; offset /= 2){
				float other = __shfl_down_sync(0xffffffff, val, offset);
				val = (val > other) ? val : other;
			}

			if (threadIdx.x == 0)
				best_bls[0] = val;
		}

		// Store result
		if (threadIdx.x == 0)
			bls[i_freq + freq_offset] = best_bls[0];

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

	unsigned int fno = id / nthreads_per_freq;
	unsigned int b   = id % nthreads_per_freq;

	partial_max[threadIdx.x] = (fno < nfreq && b < nbins) ?
	                                 arr[fno * stride + b] : -1.f;

	partial_arg_max[threadIdx.x] = (fno < nfreq && b < nbins) ?
									(
										(init == 1) ?
											b : arr_args[fno * stride + b]
									) : 0;

	__syncthreads();

	float m1, m2;

	// Reduce to find max - standard reduction down to warp level.
	// NOTE: must be s >= 32 (not s > 32) so the s=32 fold runs and only
	// 32 candidates survive for the warp-shuffle stage below; with s > 32
	// elements 32..63 were silently dropped (same bug fixed in
	// full_bls_no_sol_optimized by commit 72ae029).
	for(int s = blockDim.x / 2; s >= 32; s /= 2){
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

	// OPTIMIZATION: Final warp reduction with shuffle
	if (threadIdx.x < 32){
		float val = partial_max[threadIdx.x];
		unsigned int arg = partial_arg_max[threadIdx.x];

		for(int offset = 16; offset > 0; offset /= 2){
			float other_val = __shfl_down_sync(0xffffffff, val, offset);
			unsigned int other_arg = __shfl_down_sync(0xffffffff, arg, offset);

			if (other_val > val){
				val = other_val;
				arg = other_arg;
			}
		}

		if (threadIdx.x == 0){
			partial_max[0] = val;
			partial_arg_max[0] = arg;
		}
	}

	__syncthreads();

	// Store result
	if (threadIdx.x == 0 && fno < nfreq){
		unsigned int i = (gridDim.x == nfreq) ? 0 :
			                 fno * stride - fno * nblocks_per_freq;

		i += blockIdx.x + offset;

		block_max[i] = partial_max[0];
		block_arg_max[i] = partial_arg_max[0];
	}
}
