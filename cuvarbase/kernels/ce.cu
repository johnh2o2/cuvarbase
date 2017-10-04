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


__device__ FLT mod1(FLT x){
	return x - floor(x);
}

__device__ int phase_ind(FLT ft){
	int n = (int) (mod1(ft) * NPHASE);
	return n % NPHASE;
}

__device__ int posmod(int n, int N){
	return (n < 0) ? n + N : n % N;
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

		for (int n = (int) n0; n >= (((int) n0) - PHASE_OVERLAP); n--){
			for (int m = (int) m0; m >= 0 && m >= (((int) m0) - MAG_OVERLAP); m--) {
				atomicInc(&(bin[offset + posmod(n, NPHASE) * NMAG + m]), 
				           (PHASE_OVERLAP + 1) * (MAG_OVERLAP + 1) * ndata);
			}
		}	
	}
}

__device__ unsigned int rnduppow2(unsigned int u){
	unsigned int v = u;
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;

	return v;
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

	// align!
	unsigned int r = ((nmag * nphase + nphase) * sizeof(unsigned int)) % sizeof(FLT);
	FLT * Hc = (FLT *)&block_bin_phi[nphase + r];
	__shared__ FLT f0;

	// each block works on a single frequency.
	unsigned int i_freq = blockIdx.x;

	unsigned int i, N, Nphi;
	unsigned int ntot_2 = rnduppow2(nmag * nphase);
	unsigned int nphase_2 = rnduppow2(nphase);
	int m, n, m0, n0;

	FLT dm0 = ((FLT) (mag_overlap + 1.f)) / nmag;
	FLT dm;
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
			Hc[i] = 0.f;
		}

		__syncthreads();

		// make 2d histogram
		for(i = threadIdx.x; i < ndata; i += blockDim.x){
			m0 = (int) (y[i]);
			n0 = ((int) floor(nphase * mod1(t[i] * f0))) % nphase;

			for (n = n0; n >= (((int) n0) - ((int) phase_overlap)); n--){
				for (m = m0; m >= 0 && m >= (((int) m0) - ((int) mag_overlap)); m--)
					atomicInc(&(block_bin[posmod(n, nphase) * nmag + m]), 
					      (phase_overlap + 1) * (mag_overlap + 1) * ndata);
				
			}
		}	

		__syncthreads();

		// Get the total number of data points across phi bins
		for(n=threadIdx.x; n < nmag * nphase; n+=blockDim.x)
			atomicAdd(&(block_bin_phi[n / nmag]), block_bin[n]);

		__syncthreads();

		// Convert to dH
		for(n=threadIdx.x; n < nmag * nphase; n+=blockDim.x){
			m0 = n % nmag;
			n0 = n / nmag;

			N = block_bin[n];
			Nphi = block_bin_phi[n0];

			if (Nphi * N == 0)
				continue;

			// adjust mag bin width for overlapping mag bins (phase bins are periodic)
			dm = (m0 + mag_overlap + 1 > nmag) ? (((int) nmag) -  m0) * dm0 / (1.f + mag_overlap) : dm0;

			Hc[n] = ((FLT) N) * log((dm * ((FLT) Nphi)) / ((FLT) N));
		}

		__syncthreads();
		
		//add up contributions
		for(n = ntot_2 / 2; n > 0; n/=2){
			for (m = threadIdx.x; m < n && m + n < nmag * nphase; m += blockDim.x)
				Hc[m] += Hc[m + n];
			__syncthreads();
		}

		// add up total bin counts
		for(n = nphase_2 / 2; n > 0; n/=2){
			for (m = threadIdx.x; m < n && m + n < nphase; m += blockDim.x)
				block_bin_phi[m] += block_bin_phi[m + n];
			__syncthreads();
		}

		// write result to global memory
		if (threadIdx.x == 0)
			ce[i_freq + freq_offset] = Hc[0] / block_bin_phi[0];
		
		i_freq += gridDim.x;
	}
}




__global__ void ce_classical_faster(const FLT * __restrict__ t, 
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
	unsigned int * block_bin = (unsigned int *)sh;
	unsigned int * block_bin_phi = (unsigned int *)&block_bin[nmag * nphase];

	// align!
	unsigned int r = ((nmag * nphase + nphase) * sizeof(unsigned int)) % sizeof(FLT);
	FLT * Hc = (FLT *)&block_bin_phi[nphase + r];
	FLT * t_sh = (FLT *)&Hc[nmag * nphase];
	unsigned int * y_sh = (unsigned int *)&t_sh[ndata];
	__shared__ FLT f0;

	unsigned int i, N, Nphi;
	// each block works on a single frequency.
	unsigned int i_freq = blockIdx.x;
	unsigned int ntot_2 = rnduppow2(nmag * nphase);
	unsigned int nphase_2 = rnduppow2(nphase);
	int m, n, m0, n0;

	// load data into shared memory
	for (int i = threadIdx.x; i < ndata; i += blockDim.x){
		t_sh[i] = t[i];
		y_sh[i] = y[i];
	}
	
	__syncthreads();

	FLT dm0 = ((FLT) (mag_overlap + 1.f)) / nmag;
	FLT dm;
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
			Hc[i] = 0.f;
		}

		__syncthreads();

		// make 2d histogram
		for(i = threadIdx.x; i < ndata; i += blockDim.x){
			m0 = (int) (y[i]);
			n0 = ((int) floor(nphase * mod1(t_sh[i] * f0))) % nphase;

			for (n = n0; n >= (((int) n0) - ((int) phase_overlap)); n--){
				for (m = m0; m >= 0 && m >= (((int) m0) - ((int) mag_overlap)); m--)
					atomicInc(&(block_bin[posmod(n, nphase) * nmag + m]), 
					          (phase_overlap + 1) * (mag_overlap + 1) * ndata);
			}
				
		}

		__syncthreads();

		// Get the total number of data points across phi bins
		for(n=threadIdx.x; n < nmag * nphase; n+=blockDim.x)
			atomicAdd(&(block_bin_phi[n / nmag]), block_bin[n]);

		__syncthreads();

		// Convert to dH
		for(n=threadIdx.x; n < nmag * nphase; n+=blockDim.x){
			m0 = n % nmag;
			n0 = n / nmag;

			Nphi = block_bin_phi[n0];
			N = block_bin[n];
			if (Nphi*N == 0)
				continue;

			// adjust mag bin width for overlapping mag bins (phase bins are periodic)
			dm = (m0 + mag_overlap + 1 > ((int) nmag)) ? (((int) nmag) -  m0) * dm0 / (1.f + mag_overlap) : dm0;

			Hc[n] = ((FLT) N) * log((dm * ((FLT) Nphi)) / ((FLT) N));
		}

		__syncthreads();

		//add up contributions
		for(n = ntot_2 / 2; n > 0; n/=2){
			for (m = threadIdx.x; (m < n) && ((m + n) < nmag * nphase); m += blockDim.x)
				Hc[m] += Hc[m + n];
			__syncthreads();
		}

		// add up total bin counts
		for(n = nphase_2 / 2; n > 0; n/=2){
			for (m = threadIdx.x; (m < n) && ((m + n) < nphase); m += blockDim.x)
				block_bin_phi[m] += block_bin_phi[m + n];
			__syncthreads();
		}

		// write result to global memory
		if (threadIdx.x == 0)
			ce[i_freq + freq_offset] = Hc[0] / ((FLT) (block_bin_phi[0]));
		
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
	FLT dm, dm0, Hc;
	unsigned int bin_tot, offset, Nphi, N;

	if (i < nfreq){
		Hc = 0.f;
		dm0 = ((FLT)(MAG_OVERLAP + 1)) / NMAG;
		bin_tot = 0;
		for(int n=0; n < NPHASE; n++){
			offset = i * (NMAG * NPHASE) + n * NMAG;

			Nphi = 0;
			for (int m=0; m < NMAG; m++)
				Nphi += bins[offset + m];

			if (Nphi == 0)
				continue;

			for (int m=0; m < NMAG; m++){
				N = bins[offset + m];

				if (N == 0)
					continue;

				bin_tot += N;

				// adjust mag bin width for overlapping bins
				dm = (m + MAG_OVERLAP + 1 > NMAG) ? (((FLT) NMAG) -  ((FLT) m)) * dm0 / (1.f + MAG_OVERLAP) : dm0;
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

