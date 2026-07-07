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

__device__ double mod1d(double a){
	return a - floor(a);
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
	                            double *phi_values, unsigned int nq, unsigned int nphi,
	                            unsigned int nfreq, unsigned int freq_offset){

	unsigned int i = get_id();

	if (i < nfreq){
		unsigned int imax = argmaxes[i + freq_offset];

		best_phi[i + freq_offset] = (float) phi_values[imax / nq];
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

// Fused-noverlap fast BLS kernel (one block per frequency, grid-stride).
//
// The multi-pass host loop launches the full fold+histogram+scan kernel
// ``noverlap`` times with the phase-bin grid shifted by 1/noverlap of a
// bin between passes and takes the elementwise max. This kernel fuses
// all passes into ONE launch: it histograms the data once at
// ``noverlap``-times finer phase resolution and derives every pass's
// box sums from runs of fine bins.
//
// Derivation. Pass s assigns a point with phase phi to coarse bin
//   b_s = floor(nbf*phi - s/noverlap) mod nbf.
// With u = nbf*phi and fine bin j = floor(noverlap*u) mod (noverlap*nbf):
//   b_s = floor((j - s)/noverlap) mod nbf        (integer identity)
// so the box of pass s starting at coarse bin n with width m covers
// exactly the fine bins [noverlap*n + s, noverlap*(n+m) + s): every
// (n, s) box is a contiguous run of noverlap*m fine bins whose fine
// start jj = noverlap*n + s enumerates [0, noverlap*nbf) bijectively.
//
// Float32 caveat: the host only routes here for power-of-two noverlap
// with base dphi == 0, where fl(noverlap*u) == noverlap*u and
// u - s/noverlap are exact, so bin assignment is bit-identical to the
// multi-pass kernels; other noverlap values fall back to the host
// loop. (Box SUMS still differ from the multi-pass path at float32
// rounding level: fine-bin partials accumulate in a different order,
// on top of the run-to-run atomic nondeterminism both paths share.)
//
// Cost vs the host loop: shared-memory atomics and folds drop by
// noverlap-x (histogram built once), per-frequency fixed costs (bin
// init, syncthreads, block reduction) are paid once instead of
// noverlap times; the box scan reads noverlap-x more (cheap,
// conflict-free) fine-bin partials. Shared memory grows to
// 2 * noverlap * max_nbins + blockDim floats; the host checks the
// limit and falls back to the multi-pass loop when it doesn't fit.
//
// hist_size here is the FINE histogram size: noverlap * max(nbinsf).
__global__ void full_bls_no_sol_fused(
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
	extern __shared__ float sh[];

	// separate yw/w arrays (bank-conflict-free layout)
	float *fine_yw = sh;
	float *fine_w = (float *)&sh[hist_size];
	float *best_bls = (float *)&sh[2 * hist_size];

	__shared__ float f0;
	__shared__ int nb0, nbf, max_bin_width, nfine;

	float phi, bls1, bls2, thread_max_bls, thread_yw, thread_w;

	unsigned int i_freq = blockIdx.x;
	while (i_freq < nfreq){

		thread_max_bls = 0.f;

		if (threadIdx.x == 0){
			f0 = freqs[i_freq + freq_offset];
			nb0 = nbins0[i_freq + freq_offset];
			nbf = nbinsf[i_freq + freq_offset];
			max_bin_width = divrndup(nbf, nb0);
			nfine = nbf * ((int) noverlap);
		}

		__syncthreads();

		for(unsigned int k = threadIdx.x; k < nfine; k += blockDim.x){
			fine_yw[k] = 0.f;
			fine_w[k] = 0.f;
		}

		__syncthreads();

		// fold + fine histogram: ndata (not noverlap*ndata) atomics
		for (unsigned int k = threadIdx.x; k < ndata; k += blockDim.x){
			phi = mod1(t[k] * f0);

			// u reproduces the multi-pass pass-0 expression exactly;
			// dphi is 0 on this path (host guarantees it).
			float u = ((float) nbf) * phi - dphi;
			int j = mod((int) floorf(((float) noverlap) * u), nfine);

			atomicAdd(&(fine_yw[j]), yw[k]);
			atomicAdd(&(fine_w[j]), w[k]);
		}

		__syncthreads();

		// scan: fine start jj <-> (coarse start n = jj/noverlap,
		// pass s = jj%noverlap); box width m coarse = noverlap*m fine
		for (unsigned int jj = threadIdx.x; jj < nfine; jj += blockDim.x){

			thread_yw = 0.f;
			thread_w = 0.f;
			unsigned int f_m0 = 0;

			for (unsigned int m = 1; m < max_bin_width; m += dnbins(m, dlogq)){
				unsigned int f_m = m * noverlap;
				for (unsigned int u = f_m0; u < f_m; u++){
					unsigned int idx = jj + u;
					if (idx >= (unsigned int) nfine)
						idx -= nfine;
					thread_yw += fine_yw[idx];
					thread_w += fine_w[idx];
				}
				f_m0 = f_m;

				bls1 = bls_value(thread_yw, thread_w, ignore_negative_delta_sols);
				if (bls1 > thread_max_bls)
					thread_max_bls = bls1;
			}
		}

		best_bls[threadIdx.x] = thread_max_bls;

		__syncthreads();

		// tree reduction to one warp, then warp shuffle
		for(unsigned int k = (blockDim.x / 2); k >= 32; k /= 2){
			if(threadIdx.x < k){
				bls1 = best_bls[threadIdx.x];
				bls2 = best_bls[threadIdx.x + k];
				best_bls[threadIdx.x] = (bls1 > bls2) ? bls1 : bls2;
			}
			__syncthreads();
		}

		if (threadIdx.x < 32){
			float val = best_bls[threadIdx.x];
			for(int offset = 16; offset > 0; offset /= 2){
				float other = __shfl_down_sync(0xffffffff, val, offset);
				val = (val > other) ? val : other;
			}
			if (threadIdx.x == 0)
				best_bls[0] = val;
		}

		if (threadIdx.x == 0)
			bls[i_freq + freq_offset] = best_bls[0];

		i_freq += gridDim.x;
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
						float *yw_bin, float *w_bin, double *freqs,
						float *q_values, double *phi_values,
						double epoch,
						unsigned int nq, unsigned int nphi, unsigned int ndata,
						unsigned int nfreq, unsigned int freq_offset){
	unsigned int i = get_id();

	if (i < ndata * nfreq){
		unsigned int i_data = i % ndata;
		unsigned int i_freq = i / ndata;

		unsigned int offset = i_freq * nq * nphi;

		float W = w[i_data];
		float YW = yw[i_data];

		// Fold in single precision with the float32-cast frequency,
		// exactly like bin_and_phase_fold_bst_multifreq and the CPU
		// reference single_bls (which folds with float32(t) *
		// float32(freq)). freqs stay double ONLY for the epoch
		// re-referencing below -- folding with the double frequency
		// would shift each phase by up to ~|f64 - f32|* t relative to
		// the reference and flip bin membership of edge points.
		float f0 = (float) freqs[i_freq + freq_offset];

		// get phase [0, 1)
		float phi = mod1(t[i_data] * f0);

		for(int pb = 0; pb < nphi; pb++){
			// Re-reference the trial phase (given in the original input
			// timescale) to the subtracted epoch, in double precision:
			// epoch * freq can be ~1e6 cycles for BJD-scale epochs.
			// phi_values are double so this matches the float64
			// conversion (phi0 - epoch*freq) % 1 in single_bls bit for
			// bit before the float32 cast.
			float phi0 = (float)mod1d(phi_values[pb] - (epoch * freqs[i_freq + freq_offset]));
			float dphi = phi - phi0;
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
