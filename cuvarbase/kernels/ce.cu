#include <stdio.h>

//{CPP_DEFS}

#ifdef DOUBLE_PRECISION
	#define ATOMIC_ADD atomicAddDouble
	#define FLT double
#else
	#define ATOMIC_ADD atomicAdd
	#define FLT float
#endif


__device__ int phase_ind(FLT ft){
	FLT phi = ft - floor(ft);
	int n = (int) (phi * NPHASE);
	return n % NPHASE;
}

__device__ int posmod(int n, int N){
	int nmodN = n % N;
	return (nmodN < 0) ? nmodN + N : nmodN;
}

__global__ void histogram_data_weighted(FLT *t, FLT *y, FLT *dy, FLT *bin, FLT *freqs,
	                               int nfreq, int ndata, FLT max_phi){

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int i_freq = i / ndata;
	int j_data = i % ndata;

	if (i_freq < nfreq && j_data < ndata){
		FLT Y = y[j_data];
		FLT DY = dy[j_data];
		
		int n0 = phase_ind(freqs[i_freq] * t[j_data]);
		int offset = i_freq * (NMAG * NPHASE);

		int m0 = (int) (Y * NMAG);

		for(int m = 0; m < NMAG; m++){
			FLT z = (((float) m) / NMAG - Y);
			if (abs(z) > max_phi * DY && m != m0)
				continue;
			FLT zmax = z + (1 + MAG_OVERLAP) / ((float) NMAG);
			FLT wtot = normcdf(zmax / DY) - normcdf(z / DY);

			//if (wtot > 1E-2)
			//	printf("%e %e %e %e %e\n", wtot, z, zmax, Y, DY);
			for(int n = n0; n >= n0 - PHASE_OVERLAP; n--)
				ATOMIC_ADD(&(bin[offset + posmod(n, NPHASE) * NMAG + m]), wtot);
			
		}
	}

}

__global__ void histogram_data_count(FLT *t, unsigned int *y, unsigned int *bin, FLT *freqs,
	                                 int nfreq, int ndata){

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int i_freq = i / ndata;
	int j_data = i % ndata;

	if (i_freq < nfreq && j_data < ndata){
		int offset = i_freq * (NMAG * NPHASE);
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


__global__ void weighted_ce(FLT *bins, int nfreq, FLT *ce){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nfreq){
		FLT Hc = 0.f;
		FLT bin_tot = 0.f;
		for(int n=0; n < NPHASE; n++){
			int offset = i * (NMAG * NPHASE) + n * NMAG;

			FLT p_phi_n = 0.f;
			for (int m=0; m < NMAG; m++)
				p_phi_n += bins[offset + m];

			for (int m=0; m < NMAG; m++){
				FLT pmn = bins[offset + m];
				bin_tot += pmn;

				if (pmn > 0.f && p_phi_n > 1E-10)
					Hc += pmn * log(p_phi_n / pmn);
			}
		}
		ce[i] = Hc / bin_tot;
	}
}

__global__ void standard_ce(unsigned int *bins, int nfreq,
                                    FLT *ce){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nfreq){
		FLT Hc = 0.f;
		unsigned int bin_tot = 0;
		for(int n=0; n < NPHASE; n++){
			int offset = i * (NMAG * NPHASE) + n * NMAG;

			unsigned int p_phi_n = 0;
			for (int m=0; m < NMAG; m++)
				p_phi_n += bins[offset + m];

			for (int m=0; m < NMAG; m++){
				FLT pmn = bins[offset + m];
				bin_tot += pmn;

				if (pmn > 0 && p_phi_n > 0)
					Hc += pmn * log(((float) p_phi_n) / ((float) pmn));
			}
		}
		
		ce[i] = Hc / bin_tot;
	}
}