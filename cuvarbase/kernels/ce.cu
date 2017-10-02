#include <stdio.h>

//{CPP_DEFS}

#ifndef MAX_SHARED_MEM_SIZE
	#define MAX_SHARED_MEM_SIZE 48000
#endif

#ifdef DOUBLE_PRECISION
	#define ATOMIC_ADD atomicAddDouble
	#define FLT double
#else
	#define ATOMIC_ADD atomicAdd
	#define FLT float
#endif

__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                       (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__device__ int phase_ind(FLT ft){
	FLT phi = ft - floor(ft);
	int n = (int) (phi * NPHASE);
	return n % NPHASE;
}

__device__ int posmod(int n, int N){
	int nmodN = n % N;
	return (nmodN < 0) ? nmodN + N : nmodN;
}

__device__ FLT mod1(FLT x){
	return x - floor(x);
}

__global__ void histogram_data_weighted(FLT *t, FLT *y, FLT *dy, 
	                                    FLT *bin, FLT *freqs,
	                                    unsigned int nfreq, unsigned int ndata, 
	                                    FLT max_phi){

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int i_freq = i / ndata;
	unsigned int j_data = i % ndata;

	if (i_freq < nfreq){
		FLT Y = y[j_data];
		FLT DY = dy[j_data];
		
		int n0 = phase_ind(freqs[i_freq] * t[j_data]);
		unsigned int offset = i_freq * (NMAG * NPHASE);

		int m0 = (int) (Y * NMAG);

		for(int m = 0; m < NMAG; m++){
			FLT z = (((FLT) m) / NMAG - Y);
			if (abs(z) > max_phi * DY && m != m0)
				continue;
			FLT zmax = z + (1 + MAG_OVERLAP) / ((FLT) NMAG);
			FLT wtot = normcdf(zmax / DY) - normcdf(z / DY);

			for(int n = n0; n >= n0 - PHASE_OVERLAP; n--)
				ATOMIC_ADD(&(bin[offset + posmod(n, NPHASE) * NMAG + m]), wtot);
			
		}
	}

}

__global__ void histogram_data_count(FLT *t, unsigned int *y,
	                                 unsigned int *bin,
	                                 FLT *freqs, unsigned int nfreq, 
	                                 unsigned int ndata){

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int i_freq = i / ndata;
	unsigned int j_data = i % ndata;
	if (i_freq < nfreq){
		unsigned int offset = i_freq * (NMAG * NPHASE);
		unsigned int m0 = y[j_data];
		int n0 = phase_ind(freqs[i_freq] * t[j_data]);

		for (int n = n0; n >= n0 - PHASE_OVERLAP; n--){
			for (int m = m0; m >= 0 && m >= m0 - MAG_OVERLAP; m--) {
				atomicInc(&(bin[offset + posmod(n, NPHASE) * NMAG + m]), 
				      (PHASE_OVERLAP + 1) * (MAG_OVERLAP + 1) * ndata);
			}
		}	
	}
}


__global__ void ce_classical_fast(const FLT * __restrict__ t, 
	                              const unsigned int * __restrict__ y,
	                              const FLT * __restrict__ freqs, 
	                              FLT * __restrict__ ce, 
	                              unsigned int nfreq,
	                              unsigned int freq_offset,
	                              unsigned int ndata,
	                              unsigned int nphase,
	                              unsigned int nmag,
	                              unsigned int phase_overlap,
	                              unsigned int mag_overlap){

	extern __shared__ unsigned int sh[];

	// (unsigned int + FLT) * nmag * nphase + nphase * (unsigned int) 
	//__shared__ float *t_sh = sh;
	//__shared__ unsigned int *y_sh = (unsigned int *)&t_sh[ndata];
	//__shared__ unsigned int *bin = (unsigned int *)&y_sh[ndata];

	unsigned int * block_bin = (unsigned int *)sh;
	unsigned int * block_bin_phi = (unsigned int *)&block_bin[nmag * nphase];
	FLT * Hc = (FLT *)&block_bin_phi[nphase];
	__shared__ FLT f0;

	// each block works on a single frequency.
	unsigned int i_freq = blockIdx.x;

	unsigned int i, m0, n0, N, Nphi;
	int m, n;

	FLT dm0 = (((FLT) mag_overlap) + 1.f) / nmag;
	while (i_freq < nfreq){

		// read frequency from global data
		if (threadIdx.x == 0){
			f0 = freqs[i_freq + freq_offset];
		}

		// initialise blocks to zero
		for(i = threadIdx.x; i < nmag * nphase; i += blockDim.x){
			if (i < nphase)
				block_bin_phi[i] = 0;
			
			block_bin[i] = 0;
		}

		__syncthreads();

		// make 2d histogram
		for(i = threadIdx.x; i < ndata; i += blockDim.x){
			m0 = y[i];
			n0 = (unsigned int) floor(nphase * mod1(t[i] * f0));

			for (n = n0; n >= n0 - phase_overlap; n--){
				for (m = m0; m >= 0 && m >= m0 - mag_overlap; m--)
					atomicInc(&(block_bin[posmod(n, nphase) * nmag + m]), 
					      (phase_overlap + 1) * (mag_overlap + 1) * ndata);
				
			}
		}	

		__syncthreads();

		// Get the total number of data points across phi bins
		for(n=threadIdx.x; n < nmag * nphase; n+=blockDim.x){
			n0 = n % nphase;
			m0 = n / nmag;

			atomicAdd(&(block_bin_phi[n0]), block_bin[n0 * nmag + m0]);
		}

		__syncthreads();

		// Convert to dH
		for(n=threadIdx.x; n < nmag * nphase; n+=blockDim.x){
			n0 = n % nphase;
			m0 = n / nmag;

			// adjust mag bin width for overlapping mag bins (phase bins are periodic)
			FLT dm = (m0 + mag_overlap > nmag) ? (((int) nmag) - ((int) m0)) * dm0 / mag_overlap : dm0;

			N = block_bin[n0 * nmag + m0];
			Nphi = block_bin_phi[n0];

			Hc[n0 * nmag + m0] = N * log((dm * Nphi) / N);
		}

		__syncthreads();

		//add up contributions
		for(n=(nmag * nphase) / 2; n > 0; n/=2){
			for (m = threadIdx.x; m < n; m += blockDim.x)
				Hc[m] += Hc[m + n];
		}

		// add up total bin counts
		for(n = nphase / 2; n > 0; n/=2){
			for (m = threadIdx.x; m < n; m += blockDim.x)
				block_bin_phi[m] += block_bin_phi[m + n];
		}

		__syncthreads();


		// write result to global memory
		if (threadIdx.x == 0)
			ce[i_freq + freq_offset] = Hc[0] / block_bin_phi[0];
		
		i_freq += gridDim.x;
	}
}



__global__ void weighted_ce(FLT *bins, unsigned int nfreq, FLT *ce){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nfreq){
		FLT Hc = 0.f;
		FLT bin_tot = 0.f;
		FLT dm = ((FLT)(MAG_OVERLAP + 1)) / NMAG;
		for(int n=0; n < NPHASE; n++){
			unsigned int offset = i * (NMAG * NPHASE) + n * NMAG;

			FLT p_phi_n = 0.f;
			for (int m=0; m < NMAG; m++)
				p_phi_n += bins[offset + m];

			for (int m=0; m < NMAG; m++){
				FLT pmn = bins[offset + m];
				bin_tot += pmn;

				if (pmn > 0.f && p_phi_n > 1E-10)
					Hc += pmn * log((dm * p_phi_n) / pmn);
			}
		}
		ce[i] = Hc / bin_tot;
	}
}

__global__ void standard_ce(unsigned int *bins, unsigned int nfreq,
                            FLT *ce){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nfreq){
		FLT Hc = 0.f;
		FLT dm = ((FLT)(MAG_OVERLAP + 1)) / NMAG;
		unsigned int bin_tot = 0;
		for(int n=0; n < NPHASE; n++){
			unsigned int offset = i * (NMAG * NPHASE) + n * NMAG;

			unsigned int Nphi = 0;
			for (int m=0; m < NMAG; m++)
				Nphi += bins[offset + m];

			if (Nphi == 0)
				continue;

			for (int m=0; m < NMAG; m++){
				unsigned int N = bins[offset + m];

				if (N == 0)
					continue;

				bin_tot += N;
				Hc += N * log((dm * Nphi) / N);
			}
		}
		
		ce[i] = Hc / bin_tot;
	}
}

__global__ void constdpdm_ce(unsigned int *bins, unsigned int nfreq,
                             FLT *ce, FLT *mag_bwf){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nfreq){
		FLT Hc = 0.f;
		unsigned int bin_tot = 0;
		for(int n=0; n < NPHASE; n++){
			unsigned int offset = i * (NMAG * NPHASE) + n * NMAG;

			unsigned int Nphi = 0;
			for (int m=0; m < NMAG; m++)
				Nphi += bins[offset + m];
			
			if (Nphi == 0)
				continue;

			for (int m=0; m < NMAG; m++){
				unsigned int N = bins[offset + m];

				if (N == 0)
					continue;
				
				bin_tot += N;
				Hc += N * log((mag_bwf[m] * Nphi) / N);
			}
		}
		
		ce[i] = Hc / bin_tot;
	}
}

__global__ void log_prob(unsigned int *bins, unsigned int nfreq,
                         FLT *log_proba, FLT *mag_bin_fracs){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nfreq){
		FLT logP = 0.f;
		for(int n=0; n < NPHASE; n++){
			unsigned int offset = i * (NMAG * NPHASE) + n * NMAG;

			unsigned int Nphi = 0;
			for (int m=0; m < NMAG; m++)
				Nphi += bins[offset + m];
			
			if (Nphi == 0)
				continue;

			for (int m=0; m < NMAG; m++){
				FLT N = (FLT) (bins[offset + m]);

				FLT Nexp = Nphi * mag_bin_fracs[m];

				if (Nexp < 1e-9)
					continue;

				logP += N * log(Nexp) - Nexp - lgamma(N + 1.f);
			}
		}
		
		log_proba[i] = logP / (PHASE_OVERLAP + 1.f);
	}
}

