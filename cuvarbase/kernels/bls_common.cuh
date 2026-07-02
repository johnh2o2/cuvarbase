// Shared device/global functions for the BLS kernels.
//
// bls.cu and bls_optimized.cu both inline this file via the
// //{INCLUDE bls_common.cuh} directive (expanded by utils._module_reader
// at load time). Single-sourcing these functions removes the historical
// drift hazard between the two kernel files: the reduction_max s>32
// candidate-drop bug (commit 77b4333) was originally fixed in only one
// copy because the same function lived in two places. Keep functions that
// differ on purpose -- reduction_max (full tree vs warp shuffle) and
// full_bls_no_sol / full_bls_no_sol_optimized -- in their own files.

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

__device__ float bls_value(float ybar, float w, unsigned int ignore_negative_delta_sols){
	// if ignore negative delta sols is turned on, that means only solutions where
	// the mean amplitude within the transit is _lower_ than the mean amplitude of
	// the source are considered: it will ignore "inverted dips"
	//
	// The upper w bound must be a float32-meaningful complement: the old
	// `w < 1.f - 1e-10f` compiled to `w < 1.f` (1e-10 < ulp(1)/2), so a
	// box capturing ALL the statistical weight passed the guard with
	// (1.f - w) equal to pure atomic-roundoff noise (~1e-5 for n~1e4
	// points) and ybar likewise roundoff around 0 -- a 0/0 that showed
	// up as nondeterministic bogus peaks on single-site data at alias
	// frequencies (PR #65 reproducer, HATPI). 1e-4 exceeds worst-case
	// accumulation error with margin; no legitimate transit solution
	// holds >99.99% of the total weight (there would be no
	// out-of-transit baseline). The lower bound is unchanged: small-w
	// sums of positive weights carry no cancellation.
	float bls = (w > 1e-10f && w < 1.f - 1e-4f) ? ybar * ybar / (w * (1.f - w)) : 0.f;
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

	unsigned int n = (unsigned int) floorf(dlogq * nbins);

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

		float phi = mod1((float) (((double) q) * (((double) jphi) + ((double) s) * ((double) dphi))));

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

		// get phase [0, 1)
		float phi = mod1(t[i_data] * freqs[i_freq + freq_offset]);

		float dphi = 1.f / noverlap;
		unsigned int nbtot = 0;
		unsigned int nb, b;

		// iterate through bins (logarithmically spaced)
		for(int j = 0; nbins_iter(j, nbins0, dlogq) <= nbinsf; j++){
			nb = nbins_iter(j, nbins0, dlogq);

			// iterate through offsets [ 0, 1./sigma, ...,
			//                           (sigma - 1) / sigma ]
			for (int s = 0; s < noverlap; s++){
				b = (unsigned int) mod((int) floorf(nb * phi - s * dphi), nb);
				b += offset + s * nb + noverlap * nbtot;

				atomicAdd(&(yw_bin[b]), YW);
				atomicAdd(&(w_bin[b]), W);
			}
			nbtot += nb;
		}
	}
}

// needs ndata * nfreq threads
// noverlap -- number of overlapped bins (noverlap * (1 / q) total bins)
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
