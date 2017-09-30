#include <stdio.h>
#define RESTRICT __restrict__
#define CONSTANT const
#define MIN_W 1E-3
//{CPP_DEFS}

__device__ int get_id(){
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
	int i = get_id();

	if (i < n){
		bls[i] = bls_value(yw[i], w[i]);
	}
}

__global__ void store_best_sols(int *argmaxes, float *best_phi, 
	                            float *best_q,
	                            int nbins0, int nbinsf, int noverlap, 
	                            float dlogq, int nfreq, int freq_offset){

	int i = get_id();

	if (i < nfreq){
		int imax = argmaxes[i + freq_offset];
		float dphi = 1.f / noverlap;
		int nb = nbins0;
		float x = 1.f;
		int offset = 0;

		while(offset + noverlap * nb <= imax){	
			offset += noverlap * nb;

			x *= (1 + dlogq);
			nb = (int) (x * nbins0);
		}

		float q = 1.f / nb;
		int s = (imax - offset) / nb;

		int jphi = (imax - offset) % nb;
		
		float phi = mod1(q * (jphi + s * dphi));

		best_phi[i + freq_offset] = phi;
		best_q[i + freq_offset] = q;
	}
}

__global__ void store_best_sols_custom(int *argmaxes, float *best_phi, 
	                            float *best_q, float *q_values,
	                            float *phi_values, int nq, int nphi,
	                            int nfreq, int freq_offset){

	int i = get_id();

	if (i < nfreq){
		int imax = argmaxes[i + freq_offset];

		best_phi[i + freq_offset] = phi_values[imax / nq];
		best_q[i + freq_offset] = q_values[imax % nq];
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
	int i = get_id();

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
		for(float x = 1.f; ((int) (x * nbins0)) <= nbinsf; 
			                              x *= (1 + dlogq)){
			int nb = (int) (x * nbins0);
			float q = 1.f / nb;

			// iterate through offsets [ 0, 1./sigma, ..., 
			//                           (sigma - 1) / sigma ]
			for (int s = 0; s < noverlap; s++){

				int b = (int) floorf(nb * mod1(phi - s * q * dphi));

				b += offset + s * nb + noverlap * nbtot;
				atomicAdd(&(yw_bin[b]), YW);
				atomicAdd(&(w_bin[b]), W);
			}
			nbtot += nb;
		}
	}
}

__device__ int count_tot_nbins(int nbins0, int nbinsf, float dlogq){
	float x = 1.f;
	int ntot = 0;

	while((int) (x * nbins0) <= nbinsf){
		ntot += (int) (x * nbins0);
		x *= (1.f + dlogq);
	}

	return ntot;
}

// needs ndata * nfreq threads
// noverlap -- number of overlapped bins (noverlap * (1 / q) total bins)
// Note: this thread heavily utilizes global atomic operations, and could
//       likely be improved by 1-2 orders of magnitude for large Ndata (10^4)
//       if shared memory atomics were utilized.
__global__ void full_bls_no_sol_fast(
	                    float *t, float *yw, float *w,
						float *bls, float *freqs,
						int ndata, int nfreq, int *nbins0, int *nbinsf,
						int freq_offset, const int noverlap, float dlogq,
						int nbins_tot){
	int i = get_id();

	__shared__ float block_yw_bin[BLOCK_SIZE];
	__shared__ float block_w_bin[BLOCK_SIZE];
	__shared__ float block_max_bls;

	int nb, nb0, nbf, nbins_tot, nrounds, i_bin, j_bin, bin_offset, s;
	float x;

	int i_freq = blockIdx.x;
	while (i_freq + freq_offset < nfreq){
		nb0 = nbins0[i_freq + freq_offset];
		nbf = nbinsf[i_freq + freq_offset];

		// total bins for this frequency
		nbins_tot = count_tot_nbins(nb0, nbf, dlogq);

		// number of bins per thread
		nrounds = (int) ceil((nbins_tot * noverlap) / BLOCK_SIZE);

		// intialize bins to 0 (no synchronization necessary)
		block_yw_bin[threadIdx.x] = 0.f;
		block_w_bin[threadIdx.x] = 0.f;

		for (int j = 0; j < nrounds; j++){
			i_bin = j * blockDim.x + threadIdx.x;

			nb = nb0;
			bin_offset = 0;
			x = 1.f;
			while ((bin_offset + nb) * noverlap <= i_bin){
				bin_offset += nb;
				x *= (1.f + dlogq);
				nb = (int) (x * nb0);
			}

			if (nb > nbf)
				break;

			s = (i_bin - bin_offset * noverlap) / nb;
			j_bin = (i_bin - bin_offset * noverlap) % nb;


			for (int k = 0; k < ndata; k++){
				
			}


		__syncthreads();






		}
		__syncthreads();
	}



		float sums[noverlap];

		int i_freq = i / (nbins_tot);

		int offset = i_freq * nbins_tot * noverlap;

		float W = w[i_data];
		float YW = yw[i_data];

		// get phase [0, 1)
		float phi = mod1(t[i_data] * freqs[i_freq + freq_offset]);

		float dphi = 1.f / noverlap;
		int nbtot = 0;

		// iterate through bins (logarithmically spaced)
		for(float x = 1.f; ((int) (x * nbins0)) <= nbinsf; 
			                              x *= (1 + dlogq)){
			int nb = (int) (x * nbins0);
			float q = 1.f / nb;

			// iterate through offsets [ 0, 1./sigma, ..., 
			//                           (sigma - 1) / sigma ]
			for (int s = 0; s < noverlap; s++){

				int b = (int) floorf(nb * mod1(phi - s * q * dphi));

				b += offset + s * nb + noverlap * nbtot;
				atomicAdd(&(yw_bin[b]), YW);
				atomicAdd(&(w_bin[b]), W);
			}
			nbtot += nb;
		}
	}
}

/*
// needs nbins_tot * nfreq threads
// noverlap -- number of overlapped bins (noverlap * (1 / q) total bins)
// requires (nfreq * nbins_tot) / block_size * (sizeof float + sizeof int) + ndata * (3 * sizeof float) global memory
// and (2 * sizeof(float) + sizeof(int)) * noverlap * block_size * nblocks bytes of shared memory
// nblocks ~ (nfreq * nbins_tot) / block_size so (2 * sizeof(float)) * noverlap * nfreq * nbins_tot
// nfreqs = SHARED_MEMORY_LIMIT / ((2 * sizeof(float) * noverlap * nbins_tot)
// nbins_tot = nbinsf * log(nbinsf) ~ 1/qmin log(1e4) ~ 10/qmin ~ 1e5
// noverlap ~ 2
// m = 12 bytes
// so nfreqs * (12 * 2 * 1e5) ~ 4.8e6 ==> 48 ~ 12 * 2 * nfreqs
__global__ void bin_and_phase_fold_efficiently(
	                    float *t, float *yw, float *w,
						float *binned_bls, float *freqs,
						int ndata, int nfreq, int *nbins0, int *nbinsf,
						int freq_offset, int noverlap, float dlogq,
						int total_bin_count){
	int i = get_id();

	// size: noverlap * BLOCK_SIZE * (2 * sizeof(float) + sizeof(int))
	extern __shared__ float shared_bins[];

	float *shared_yw = shared_bins;
	float *shared_w = (float*)&shared_yw[noverlap * blockDim.x];

	int b, nbtot, nb, i_bin, j, i_freq;
	float f0, dphi, x, phi;

	dphi = 1.f / noverlap;

	// initialize shared arrays
	for (int i = 0; i < noverlap; i++){
		shared_yw[threadIdx.x * noverlap + i] = 0.f;
		shared_w[threadIdx.x * noverlap + i] = 0.f;
	}

	__syncthreads();

	// bin the data
	if (i < total_bin_count{
		// 
		i_freq = 0;
		int total_nbin_offset = 0;
		int nb0 = nbins0[freq_offset];
		int nbf = nbinsf[freq_offset];
		int ntot = count_tot_nbins(nb0, nbf, dlogq);

		while (i_freq + 1 < nfreq && total_nbin_offset + ntot < i){
			i_freq ++;
			total_nbin_offset += ntot;
			nb0 = nbins0[i_freq];
			nbf = nbinsf[i_freq];
			ntot = count_tot_nbins(nb0, nbf, dlogq);
		}
		
		int i_bin = i - total_nbin_offset;
		nb = nb0;
		int nboffset = 0;
		x = 1.f;
		while(nboffset + nb < i_bin){
			nboffset += nb;
			x *= (1.f + dlogq);
			nb = (int) (x * nb0);
		}

		int j_bin = i_bin - nboffset;

		f0 = freqs[i_freq + freq_offset];

		for (int i = 0; i < ndata; i++){

			// get phase * nbins
			phi = nb * mod1(t[i] * f0);

			if (phi < j_bin || phi > j_bin + 2)
				continue;

			// if it's in this bin or the next bin, keep going 
			for (int s = 0; s < noverlap; s++){
				j = (int) floor(phi - s * dphi);

				j += (j < 0) ? nb : 0;

				if (j != b)
					break;

				shared_yw[noverlap * threadIdx.x + s] += yw[i];
				shared_w[noverlap * threadIdx.x + s] += w[i];
			}
		}
	}
	
	// wait till all bins are finished
	__syncthreads();

	if (i < total_bin_count){
		int ind_offset = (total_nbin_offset + nboffset) * noverlap;
		binned_bls[i] = bls_value(shared_yw[, w1);
	}

	float bls_max = 0.f;
	int argmax = 0;

	for(int s = 0; s < noverlap; s++){
		// reduce to find max BLS values of this block
		// and only store those to global memory
		for(int k = (blockDim.x / 2); k > 0; k /= 2){
			if(threadIdx.x < k){
				int i1 = noverlap * threadIdx.x + s;
				int i2 = noverlap * (threadIdx.x + k) + s;

				float ybar1 = shared_yw[i1];
				float w1 = shared_w[i1];

				float ybar2 = shared_yw[i2];
				float w2 = shared_w[i2];

				bls1 = bls_value(ybar1, w1);
				bls2 = bls_value(ybar2, w2);

				shared_yw[i1] = (bls1 > bls2) ? bls1 : bls2;

				shared_argmax[i1] = (bls1 > bls2) ?
				 						shared_argmax[i1] :
				 						shared_argmax[i2];
			}

			__syncthreads();
		}
		if (threadIdx.x == 0 && shared_yw[s] > bls_max){
			bls_max = shared_yw[s];
			argmax = shared_argmax[s];
		}
	}

	// write to global memory
	if (threadIdx.x == 0){
		block_bls[blockIdx.x] = bls_max;
		block_arg_max[blockIdx.x] = argmax;
	}

}

*/

// needs ndata * nfreq threads
// noverlap -- number of overlapped bins (noverlap * (1 / q) total bins)
__global__ void bin_and_phase_fold_custom(
	                    float *t, float *yw, float *w,
						float *yw_bin, float *w_bin, float *freqs,
						float *q_values, float *phi_values, 
						int nq, int nphi, int ndata, 
						int nfreq, int freq_offset){
	int i = get_id();

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




__global__ void inplace_max_reduce(float *arr, int *arr_args, int n){

	__shared__ float partial_max[BLOCK_SIZE];
	__shared__ int partial_arg_max[BLOCK_SIZE];

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int nblocks_per_ = gridDim.x / nfreq;
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
