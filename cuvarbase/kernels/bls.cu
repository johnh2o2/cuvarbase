#include <stdio.h>
#define RESTRICT __restrict__
#define CONSTANT const
#define MIN_W 1E-3
//{CPP_DEFS}

__device__ unsigned int get_id(){
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int mod(int a, int b){
	int r = a % b;
	return (r < 0) ? r + b : r;
}

__device__ float mod1(float a){
	return a - floorf(a);
}

__device__ float bls_value(float ybar, float w){
	return (w > 1e-10 && w < 1.f - 1e-10) ? ybar * ybar / (w * (1.f - w)) : 0.f;
}

__global__ void binned_bls_bst(float *yw, float *w, float *bls, int n){
	unsigned int i = get_id();

	if (i < n){
		bls[i] = bls_value(yw[i], w[i]);
	}
}


__global__ void store_best_sols_custom(int *argmaxes, float *best_phi, 
	                            float *best_q, float *q_values,
	                            float *phi_values, int nq, int nphi,
	                            int nfreq, int freq_offset){

	unsigned int i = get_id();

	if (i < nfreq){
		int imax = argmaxes[i + freq_offset];

		best_phi[i + freq_offset] = phi_values[imax / nq];
		best_q[i + freq_offset] = q_values[imax % nq];
	}
}


__device__ int divrndup(int a, int b){
	return (a % b > 0) ? a/b + 1 : a/b;
}

__device__ int nbins_iter(int i, int nb0, float dlogq){
	int nb = nb0;
	for(int j = 0; j < i; j++)
		nb += (int) ceilf(dlogq * nb);

	return nb;
}

__device__ int count_tot_nbins(int nbins0, int nbinsf, float dlogq){
	int ntot = 0;

	for(int i = 0; nbins_iter(i, nbins0, dlogq) <= nbinsf; i++)
		ntot += nbins_iter(i, nbins0, dlogq);
	return ntot;
}



__global__ void store_best_sols(int *argmaxes, float *best_phi, 
	                            float *best_q,
	                            int nbins0, int nbinsf, int noverlap, 
	                            float dlogq, int nfreq, int freq_offset){

	unsigned int i = get_id();

	if (i < nfreq){
		int imax = argmaxes[i + freq_offset];
		float dphi = 1.f / noverlap;

		int nb = 0;
		int bin_offset = 0;
		for (int j = 0; (bin_offset + nbins_iter(j, nbins0, dlogq)) * noverlap < imax; j++){
			nb = nbins_iter(j, nbins0, dlogq);
			bin_offset += nb;
		}

		float q = 1.f / nb;
		int s = (imax - bin_offset * noverlap) / nb;
		int jphi = (imax - bin_offset * noverlap) % nb;
		
		float phi = mod1(q * (jphi + s * dphi));

		best_phi[i + freq_offset] = phi;
		best_q[i + freq_offset] = q;
	}
}

// needs ndata * nfreq threads
// noverlap -- number of overlapped bins (noverlap * (1 / q) total bins)
// Note: this thread heavily utilizes global atomic operations, and could
//       likely be improved by 1-2 orders of magnitude for large Ndata (10^4)
//       if shared memory atomics were utilized.
__global__ void bin_and_phase_fold_bst_multifreq(
	                    float *t, float *yw, float *w,
						float *yw_bin, float *w_bin, float *freqs,
						int ndata, int nfreq, int nbins0, int nbinsf,
						int freq_offset, int noverlap, float dlogq,
						int nbins_tot){
	unsigned int i = get_id();

	if (i < ndata * nfreq){
		int i_data = i % ndata;
		int i_freq = i / ndata;

		int offset = i_freq * nbins_tot * noverlap;

		float W = w[i_data];
		float YW = yw[i_data];

		// get phase [0, 1)
		float phi = mod1(t[i_data] * freqs[i_freq + freq_offset]);

		float dphi = 1.f / noverlap;
		int nbtot = 0;

		// iterate through bins (logarithmically spaced)
		for(int j = 0; nbins_iter(j, nbins0, dlogq) <= nbinsf; j++){
			int nb = nbins_iter(j, nbins0, dlogq);

			// iterate through offsets [ 0, 1./sigma, ..., 
			//                           (sigma - 1) / sigma ]
			for (int s = 0; s < noverlap; s++){
				int b = (int) floorf(nb * phi - s * dphi);
				b = mod(b, nb) + offset + s * nb + noverlap * nbtot;

				atomicAdd(&(yw_bin[b]), YW);
				atomicAdd(&(w_bin[b]), W);
			}
			nbtot += nb;
		}
	}
}

// needs as many threads as we can get. Each block works on a single frequency
// and then moves on to another frequency
//
// NO ATOMICS! The best solution is not kept to save memory. Requires multiple
// passes through the data at each frequency, which may slow things down considerably.
// 
// npasses: (nbins_tot * noverlap) / BLOCK_SIZE ~ 1e5 * 2 / 256 = 781
// bigger block sizes are BETTER!
//
// limited by shared memory = (2 * block_size + 1) * sizeof(float), means
// a maximum of MAX_SHARED_MEMORY * block_size / ((2 * block_size + 1) * sizeof(float))
// ~ (48KB / 4 B) * (1 / (2 + 1/block_size)) ~ 48 / 8 * 1000 = 6000 threads
__global__ void full_bls_no_sol_fast(
	                    float *t, float *yw, float *w,
						float *bls, float *freqs,
						unsigned int *nbins0, unsigned int *nbinsf, 
						unsigned int ndata, unsigned int nfreq,
						unsigned int freq_offset, unsigned int noverlap,
						float dlogq){
	
	unsigned int i = get_id();

	__shared__ float block_yw_bin[BLOCK_SIZE];
	__shared__ float block_w_bin[BLOCK_SIZE];
	__shared__ float block_max_bls;

	unsigned int nb, nb0, nbf, nbins_tot, nrounds, i_bin, j_bin, 
	             bin_offset, s;
	int b;
	float f0, bls1, bls2;

	// this will be inefficient for block sizes >> number of bins per frequency
	unsigned int i_freq = blockIdx.x;

	float dphi = 1.f/noverlap;

	while (i_freq < nfreq){
		if (threadIdx.x == 0){

			// initialize block max
			block_max_bls = 0.f;

			// read frequency from global memory
			f0 = freqs[i_freq + freq_offset];

			// read nbins from global memory
			nb0 = nbins0[i_freq + freq_offset];
			nbf = nbinsf[i_freq + freq_offset];

		}

		__syncthreads();

		// read frequency from global memory
		f0 = freqs[i_freq + freq_offset];

		// read nbins from global memory
		nb0 = nbins0[i_freq + freq_offset];
		nbf = nbinsf[i_freq + freq_offset];

		// total bins for this frequency
		nbins_tot = count_tot_nbins(nb0, nbf, dlogq);

		// number of bins per thread
		nrounds = divrndup(nbins_tot * noverlap, BLOCK_SIZE);

		for (unsigned int j = 0; j < nrounds; j++){
			i_bin = j * blockDim.x + threadIdx.x;

			// intialize bins to 0 (no synchronization necessary)
			block_yw_bin[threadIdx.x] = 0.f;
			block_w_bin[threadIdx.x] = 0.f;

			// get our bearings
			//  - nb: number of bins
			//  - bin_offset: sum(nb, bin < this bin)
			//  - s: overlap number
			//  - j_bin: bin number
			bin_offset = 0;
			nb = nb0;
			for (unsigned int i_iter = 0; (bin_offset + nb) * noverlap < i_bin; i_iter++){
				bin_offset += nb;
				nb = nbins_iter(i_iter + 1, nb0, dlogq);
			}

			s = (i_bin - bin_offset * noverlap) / nb;
			j_bin = (i_bin - bin_offset * noverlap) % nb;

			for (unsigned int k = 0; k < ndata; k++){

				// get bin number for this datapoint
				b = (int) floorf(nb * mod1(t[k] * f0) - s * dphi);
				b += (b < 0) ? nb : 0;

				// write to your bin
				if (b == j_bin){
					block_yw_bin[threadIdx.x] += yw[k];
					block_w_bin[threadIdx.x] += w[k];
				}
			}

			// convert to bls
			block_yw_bin[threadIdx.x] = bls_value(block_yw_bin[threadIdx.x], 
				                                  block_w_bin[threadIdx.x]);

			// wait for everyone in this block to finish binning the data
			__syncthreads();
			
			// find the max_bls value in this block
			for(unsigned int k = (blockDim.x / 2); k > 0; k /= 2){
				if(threadIdx.x < k){
					bls1 = block_yw_bin[threadIdx.x];
					bls2 = block_yw_bin[threadIdx.x + k];
					
					block_yw_bin[threadIdx.x] = (bls1 > bls2) ? bls1 : bls2;
				}
				__syncthreads();
			}

			// store block max if it's greater than the running max value
			if (threadIdx.x == 0 && block_yw_bin[0] > block_max_bls)
				block_max_bls = block_yw_bin[0];

			// wait until we've finished storing the block_max
			__syncthreads();

		}
		
		// write max_bls for frequency to global memory!
		if (threadIdx.x == 0)
			bls[i_freq + freq_offset] = block_max_bls;
		
		i_freq += gridDim.x;
	}
}

__device__ unsigned int dnbins(unsigned int nbins, float dlogq){

	if (dlogq < 0)
		return 1;

	unsigned int n = (unsigned int) floorf(dlogq * nbins);

	return (n == 0) ? 1 : n;
}

// needs as many threads as we can get. Each block works on a single frequency
// and then moves on to another frequency
//
// Uses shared memory atomics to parallelize data reads
// bigger block sizes are BETTER!
//
// limited by shared memory = (2 * block_size + 1) * sizeof(float), means
// a maximum of MAX_SHARED_MEMORY * block_size / ((2 * block_size + 1) * sizeof(float))
// ~ (48KB / 4B) * (1 / (2 + 1/block_size)) ~ 48 / 8 * 1000 = 6000 threads
//
// requires (block_size + 2 * hist_size) extra shared memory
__global__ void full_bls_no_sol_fast_sma_linbins(
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
						float dphi,
						float dlogq){
	
	unsigned int i = get_id();

	extern __shared__ float sh[];

	float *block_bins = sh;
	float *best_bls = (float *)&sh[2 * hist_size];

	__shared__ float f0;
	__shared__ int nb0, nbf, max_bin_width;

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

			b = mod((int) floorf(nbf * phi - dphi), nbf);

			// shared memory atomics should (hopefully) be faster.
			atomicAdd(&(block_bins[2 * b]), yw[k]);
			atomicAdd(&(block_bins[2 * b + 1]), w[k]);
		}

		// wait for everyone to finish adding data to the histogram
		__syncthreads();

		// get max bls for this THREAD
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

				bls1 = bls_value(thread_yw, thread_w);
				if (bls1 > thread_max_bls)
					thread_max_bls = bls1;
			}
		}

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


// needs as many threads as we can get. Each block works on a single frequency
// and then moves on to another frequency
//
// Uses shared memory atomics to parallelize data reads
// bigger block sizes are BETTER!
//
// limited by shared memory = (2 * block_size + 1) * sizeof(float), means
// a maximum of MAX_SHARED_MEMORY * block_size / ((2 * block_size + 1) * sizeof(float))
// ~ (48KB / 4B) * (1 / (2 + 1/block_size)) ~ 48 / 8 * 1000 = 6000 threads
__global__ void full_bls_no_sol_fast_sma(
	                    const float* __restrict__ t, 
	                    const float* __restrict__ yw, 
	                    const float* __restrict__ w,
						float * __restrict__ bls, 
						const float* __restrict__ freqs,
						const unsigned int * __restrict__ nbins0, 
						const unsigned int * __restrict__ nbinsf, 
						unsigned int ndata, 
						unsigned int nfreq,
						unsigned int freq_offset, 
						unsigned int noverlap, 
						float dlogq){
	
	unsigned int i = get_id();

	__shared__ float block_bins[2 * BLOCK_HIST_SIZE];
	__shared__ float block_max_bls, f0;
	__shared__ unsigned int nb0, nbf;

	unsigned int nb, nbins_tot, nrounds, i_bin, 
	             s, nbtot;
	int b;
	float phi, bls1, bls2;

	// this will be inefficient for block sizes >> number of bins per frequency
	unsigned int i_freq = blockIdx.x;
	float dphi = 1.f/noverlap;
	while (i_freq < nfreq){

		
		if (threadIdx.x == 0){
			// make sure block_max_bls is stored as zero
			block_max_bls = 0.f;

			// read frequency from global memory
			f0 = freqs[i_freq + freq_offset];

			// read nbins from global memory
			nb0 = nbins0[i_freq + freq_offset];
			nbf = nbinsf[i_freq + freq_offset];
		}

		__syncthreads();

		// total bins for this frequency
		nbins_tot = count_tot_nbins(nb0, nbf, dlogq);

		// number of bins per thread
		nrounds = divrndup(nbins_tot * noverlap, BLOCK_HIST_SIZE);

		for (unsigned int j = 0; j < nrounds; j++){
			i_bin = j * BLOCK_HIST_SIZE;

			int K = (BLOCK_HIST_SIZE > nbins_tot - i_bin) ? BLOCK_HIST_SIZE : nbins_tot - i_bin;

			// intialize bins to 0 (synchronization is necessary here...)
			for(unsigned int k = threadIdx.x; k < K; k += blockDim.x){
				block_bins[2 * k] = 0.f;
				block_bins[2 * k + 1] = 0.f;
			}

			// wait for initialization to finish
			__syncthreads();

			for (unsigned int k = threadIdx.x; k < ndata; k += blockDim.x){
				phi = mod1(t[k] * f0);
				
				nbtot = 0;

				// iterate through bins (logarithmically spaced)
				for(unsigned int m = 0; nbins_iter(m, nb0, dlogq) <= nbf; m++){
					nb = nbins_iter(m, nb0, dlogq);

					// skip all bins below range
					// max_bin_index = nbtot * noverlap + nb * noverlap - i_bin
					if (nbtot * noverlap + nb * noverlap < i_bin)
						continue;

					// quit once we've reached the maximum number of bins in this round
					// min_bin_index = nbtot * noverlap - i_bin
					if (nbtot * noverlap > i_bin + BLOCK_HIST_SIZE)
						break;

					// iterate through offsets [ 0, 1./sigma, ..., 
					//                           (sigma - 1) / sigma ]
					for (s = 0; s < noverlap; s++){
						// get the histogram index for this datapoint given
						// overlap number and bin size
						b = (int) floorf(nb * phi - s * dphi);
						b = mod(b, nb) + s * nb + noverlap * nbtot;

						if (b - i_bin < BLOCK_HIST_SIZE && b >= i_bin){
							// shared memory atomics should (hopefully) be faster.
							atomicAdd(&(block_bins[2 * (b - i_bin)]), yw[k]);
							atomicAdd(&(block_bins[2 * (b - i_bin) + 1]), w[k]);
						}
					}
					nbtot += nb;
				}
			}

			// wait for everyone to finish adding data to the histogram
			__syncthreads();

			// convert to bls
			for(unsigned int k = threadIdx.x; k < BLOCK_HIST_SIZE; k+=blockDim.x){
				block_bins[2 * k] = bls_value(block_bins[2 * k], 
				                              block_bins[2 * k + 1]);
			}

			// wait for everyone to convert binned data to BLS
			__syncthreads();


			// find the max_bls value in this block
			for(unsigned int k = 0; k < divrndup(BLOCK_HIST_SIZE, blockDim.x); k++){
				for(unsigned int m = (blockDim.x / 2); m > 0; m /= 2){
					if(threadIdx.x < m){
						bls1 = block_bins[2 * (threadIdx.x + k * blockDim.x)];
						bls2 = block_bins[2 * (threadIdx.x + k * blockDim.x + m)];
						
						block_bins[2 * (threadIdx.x + k * blockDim.x)] = (bls1 > bls2) ? bls1 : bls2;
					}
					__syncthreads();
				}
			}

			// store block max if it's greater than the running max value
			if (threadIdx.x == 0 && block_bins[0] > block_max_bls)
				block_max_bls = block_bins[0];

			// wait until we've finished storing max BLS for the block
			__syncthreads();

		}
		
		// write (global) bls max for this frequency to global memory!
		if (threadIdx.x == 0)
			bls[i_freq + freq_offset] = block_max_bls;

		// increment frequency
		i_freq += gridDim.x;
	}
}

// needs ndata * nfreq threads
// noverlap -- number of overlapped bins (noverlap * (1 / q) total bins)
__global__ void bin_and_phase_fold_custom(
	                    float *t, float *yw, float *w,
						float *yw_bin, float *w_bin, float *freqs,
						float *q_values, float *phi_values, 
						int nq, int nphi, int ndata, 
						int nfreq, int freq_offset){
	unsigned int i = get_id();

	if (i < ndata * nfreq){
		int i_data = i % ndata;
		int i_freq = i / ndata;

		int offset = i_freq * nq * nphi;

		float W = w[i_data];
		float YW = yw[i_data];

		// get phase [0, 1)
		float phi = mod1(t[i_data] * freqs[i_freq + freq_offset]);

		for(int pb = 0; pb < nphi; pb++){
			float dphi = phi - phi_values[pb];
			dphi -= floorf(dphi);

			for(int qb = 0; qb < nq; qb++){
				if (dphi < q_values[qb]){
					atomicAdd(&(yw_bin[pb * nq + qb + offset]), YW);
					atomicAdd(&(w_bin[pb * nq + qb + offset]), W);
				}
			}
		}
	}
}




__global__ void reduction_max(float *arr, int *arr_args, int nfreq, 
	                          int nbins, int stride,
                              float *block_max, int *block_arg_max, 
                              int offset, int init){

	__shared__ float partial_max[BLOCK_SIZE];
	__shared__ int partial_arg_max[BLOCK_SIZE];

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int nblocks_per_freq = gridDim.x / nfreq;
	int nthreads_per_freq = blockDim.x * nblocks_per_freq;


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

	int fno = id / nthreads_per_freq;
	int b   = id % nthreads_per_freq;

	// read part of array from global memory into shared memory
	partial_max[threadIdx.x] = (fno < nfreq && b < nbins) ?
	                                 arr[fno * stride + b] : -1.f;

	partial_arg_max[threadIdx.x] = (fno < nfreq && b < nbins) ?
									(
										(init == 1) ?
											b : arr_args[fno * stride + b]
									) : -1;

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
		int i = (gridDim.x == nfreq) ? 0 :
			fno * stride - fno * nblocks_per_freq;

		i += blockIdx.x + offset;

		block_max[i] = partial_max[0];
		block_arg_max[i] = partial_arg_max[0];
	}
}
