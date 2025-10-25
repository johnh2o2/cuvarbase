#include <stdio.h>
#define RESTRICT __restrict__
#define CONSTANT const
#define MIN_W 1E-3
//{CPP_DEFS}

// Optimized version of BLS kernel with following improvements:
// 1. Fixed bank conflicts (separate yw/w arrays)
// 2. Explicit use of fast math intrinsics
// 3. Better memory access patterns
// 4. Warp-level reduction in final stages

__device__ unsigned int get_id(){
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int mod(int a, int b){
	int r = a % b;
	return (r < 0) ? r + b : r;
}

__device__ float mod1_fast(float a){
    // Use fast intrinsic instead of floorf
	return a - __float2int_rd(a);
}

__device__ float bls_value(float ybar, float w, unsigned int ignore_negative_delta_sols){
	float bls = (w > 1e-10f && w < 1.f - 1e-10f) ? ybar * ybar / (w * (1.f - w)) : 0.f;
    return ((ignore_negative_delta_sols == 1) & (ybar > 0.f)) ? 0.f : bls;
}

__global__ void binned_bls_bst(float *yw, float *w, float *bls, unsigned int n, unsigned int ignore_negative_delta_sols){
	unsigned int i = get_id();

	if (i < n){
		bls[i] = bls_value(yw[i], w[i], ignore_negative_delta_sols);
	}
}


__device__ unsigned int dnbins(unsigned int nbins, float dlogq){
	if (dlogq < 0.f)
		return 1;

	unsigned int n = (unsigned int) __float2int_rd(dlogq * nbins);

	return (n == 0) ? 1 : n;
}

__device__ unsigned int nbins_iter(unsigned int i, unsigned int nb0, float dlogq){
	if (i == 0)
		return nb0;

	unsigned int nb = nb0;
	for(int j = 0; j < i; j++)
		nb += dnbins(nb, dlogq);

	return nb;
}

__device__ unsigned int count_tot_nbins(unsigned int nbins0, unsigned int nbinsf, float dlogq){
	unsigned int ntot = 0;

	for(int i = 0; nbins_iter(i, nbins0, dlogq) <= nbinsf; i++)
		ntot += nbins_iter(i, nbins0, dlogq);
	return ntot;
}

__global__ void store_best_sols_custom(unsigned int *argmaxes, float *best_phi,
	                            float *best_q, float *q_values,
	                            float *phi_values, unsigned int nq, unsigned int nphi,
	                            unsigned int nfreq, unsigned int freq_offset){

	unsigned int i = get_id();

	if (i < nfreq){
		unsigned int imax = argmaxes[i + freq_offset];

		best_phi[i + freq_offset] = phi_values[imax / nq];
		best_q[i + freq_offset] = q_values[imax % nq];
	}
}


__device__ int divrndup(int a, int b){
	return (a % b > 0) ? a/b + 1 : a/b;
}

__global__ void store_best_sols(unsigned int *argmaxes, float *best_phi,
	                            float *best_q,
	                            unsigned int nbins0, unsigned int nbinsf,
	                            unsigned int noverlap,
	                            float dlogq, unsigned int nfreq, unsigned int freq_offset){

	unsigned int i = get_id();

	if (i < nfreq){
		unsigned int imax = argmaxes[i + freq_offset];
		float dphi = 1.f / noverlap;

		unsigned int nb = nbins0;
		unsigned int bin_offset = 0;
		unsigned int i_iter = 0;
		while ((bin_offset + nb) * noverlap <= imax){
			bin_offset += nb;
			nb = nbins_iter(++i_iter, nbins0, dlogq);
		}

		float q = 1.f / nb;
		int s = (((int) imax) - ((int) (bin_offset * noverlap))) / nb;
		int jphi = (((int) imax) - ((int) (bin_offset * noverlap))) % nb;

		float phi = mod1_fast((float) (((double) q) * (((double) jphi) + ((double) s) * ((double) dphi))));

		best_phi[i + freq_offset] = phi;
		best_q[i + freq_offset] = q;
	}
}

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
			max_bin_width = divrndup(nbf, nb0);

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
			phi = mod1_fast(t[k] * f0);

			b = mod((int) __float2int_rd(((float) nbf) * phi - dphi), (int) nbf);

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

			for (unsigned int m = 1; m < max_bin_width; m += dnbins(m, dlogq)){
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

		// OPTIMIZATION: Use warp shuffle for final warp reduction
		// Standard tree reduction down to warp size
		for(unsigned int k = (blockDim.x / 2); k > 32; k /= 2){
			if(threadIdx.x < k){
				bls1 = best_bls[threadIdx.x];
				bls2 = best_bls[threadIdx.x + k];

				best_bls[threadIdx.x] = (bls1 > bls2) ? bls1 : bls2;
			}
			__syncthreads();
		}

		// Final warp reduction using shuffle (no sync needed)
		if (threadIdx.x < 32){
			float val = best_bls[threadIdx.x];

			// Warp shuffle reduction (no __syncthreads needed)
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


__global__ void bin_and_phase_fold_bst_multifreq(
	                    float *t, float *yw, float *w,
						float *yw_bin, float *w_bin, float *freqs,
						unsigned int ndata, unsigned int nfreq, unsigned int nbins0, unsigned int nbinsf,
						unsigned int freq_offset, unsigned int noverlap, float dlogq,
						unsigned int nbins_tot){
	unsigned int i = get_id();

	if (i < ndata * nfreq){
		unsigned int i_data = i % ndata;
		unsigned int i_freq = i / ndata;

		unsigned int offset = i_freq * nbins_tot * noverlap;

		float W = w[i_data];
		float YW = yw[i_data];

		float phi = mod1_fast(t[i_data] * freqs[i_freq + freq_offset]);

		float dphi = 1.f / noverlap;
		unsigned int nbtot = 0;
		unsigned int nb, b;

		for(int j = 0; nbins_iter(j, nbins0, dlogq) <= nbinsf; j++){
			nb = nbins_iter(j, nbins0, dlogq);

			for (int s = 0; s < noverlap; s++){
				b = (unsigned int) mod((int) __float2int_rd(nb * phi - s * dphi), nb);
				b += offset + s * nb + noverlap * nbtot;

				atomicAdd(&(yw_bin[b]), YW);
				atomicAdd(&(w_bin[b]), W);
			}
			nbtot += nb;
		}
	}
}


__global__ void bin_and_phase_fold_custom(
	                    float *t, float *yw, float *w,
						float *yw_bin, float *w_bin, float *freqs,
						float *q_values, float *phi_values,
						unsigned int nq, unsigned int nphi, unsigned int ndata,
						unsigned int nfreq, unsigned int freq_offset){
	unsigned int i = get_id();

	if (i < ndata * nfreq){
		unsigned int i_data = i % ndata;
		unsigned int i_freq = i / ndata;

		unsigned int offset = i_freq * nq * nphi;

		float W = w[i_data];
		float YW = yw[i_data];

		float phi = mod1_fast(t[i_data] * freqs[i_freq + freq_offset]);

		for(int pb = 0; pb < nphi; pb++){
			float dphi = phi - phi_values[pb];
			dphi -= __float2int_rd(dphi);

			for(int qb = 0; qb < nq; qb++){
				if (dphi < q_values[qb]){
					atomicAdd(&(yw_bin[pb * nq + qb + offset]), YW);
					atomicAdd(&(w_bin[pb * nq + qb + offset]), W);
				}
			}
		}
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

	// Reduce to find max - standard reduction down to warp level
	for(int s = blockDim.x / 2; s > 32; s /= 2){
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
