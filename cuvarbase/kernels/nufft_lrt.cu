#include <stdio.h>
#include <pycuda-complex.hpp>

#define RESTRICT __restrict__
#define CONSTANT const
#define PI 3.14159265358979323846264338327950288f
//{CPP_DEFS}

#ifdef DOUBLE_PRECISION
	#define FLT double
#else
	#define FLT float
#endif

#define CMPLX pycuda::complex<FLT>

// Compute matched filter statistic for NUFFT LRT
// Implements: sum(Y * conj(T) / P_s) / sqrt(sum(|T|^2 / P_s))
__global__ void nufft_matched_filter(
	CMPLX *RESTRICT Y,         // NUFFT of lightcurve, length nf
	CMPLX *RESTRICT T,         // NUFFT of template, length nf
	FLT *RESTRICT P_s,         // Power spectrum estimate, length nf
	FLT *RESTRICT weights,     // Frequency weights (for one-sided spectrum), length nf
	FLT *RESTRICT results,     // Output results [numerator, denominator], length 2
	CONSTANT int nf,           // Number of frequency samples
	CONSTANT FLT eps_floor)    // Floor for power spectrum to avoid division by zero
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Shared memory for reduction
	extern __shared__ FLT sdata[];
	FLT *s_num = sdata;
	FLT *s_den = &sdata[blockDim.x];
	
	FLT num_sum = 0.0f;
	FLT den_sum = 0.0f;
	
	// Each thread processes one or more frequency bins
	if (i < nf) {
		FLT P_inv = 1.0f / fmaxf(P_s[i], eps_floor);
		FLT w = weights[i];
		
		// Numerator: real(Y * conj(T) * w / P_s)
		CMPLX YT_conj = Y[i] * conj(T[i]);
		num_sum = YT_conj.real() * w * P_inv;
		
		// Denominator: |T|^2 * w / P_s
		FLT T_mag_sq = (T[i].real() * T[i].real() + T[i].imag() * T[i].imag());
		den_sum = T_mag_sq * w * P_inv;
	}
	
	// Store partial sums in shared memory
	s_num[threadIdx.x] = num_sum;
	s_den[threadIdx.x] = den_sum;
	__syncthreads();
	
	// Reduction in shared memory
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			s_num[threadIdx.x] += s_num[threadIdx.x + s];
			s_den[threadIdx.x] += s_den[threadIdx.x + s];
		}
		__syncthreads();
	}
	
	// Write result for this block to global memory
	if (threadIdx.x == 0) {
		atomicAdd(&results[0], s_num[0]);
		atomicAdd(&results[1], s_den[0]);
	}
}

// Compute power spectrum estimate from NUFFT
// Simple smoothed periodogram approach
__global__ void estimate_power_spectrum(
	CMPLX *RESTRICT Y,         // NUFFT of data, length nf
	FLT *RESTRICT P_s,         // Output power spectrum, length nf
	CONSTANT int nf,           // Number of frequency samples
	CONSTANT int smooth_window,// Smoothing window size
	CONSTANT FLT eps_floor)    // Floor value as fraction of median
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < nf) {
		// Compute periodogram value: |Y[i]|^2
		FLT power = Y[i].real() * Y[i].real() + Y[i].imag() * Y[i].imag();
		
		// Simple boxcar smoothing
		FLT smoothed = 0.0f;
		int count = 0;
		int half_window = smooth_window / 2;
		
		for (int j = -half_window; j <= half_window; j++) {
			int idx = i + j;
			if (idx >= 0 && idx < nf) {
				FLT val = Y[idx].real() * Y[idx].real() + Y[idx].imag() * Y[idx].imag();
				smoothed += val;
				count++;
			}
		}
		
		P_s[i] = smoothed / count;
	}
}

// Apply frequency weights for one-sided spectrum conversion
__global__ void compute_frequency_weights(
	FLT *RESTRICT weights,     // Output weights, length nf
	CONSTANT int nf,           // Number of frequency samples
	CONSTANT int n_data)       // Original data length (for determining Nyquist)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < nf) {
		// Weights for converting two-sided to one-sided spectrum
		if (i == 0) {
			weights[i] = 1.0f;
		} else if (i < nf - 1) {
			weights[i] = 2.0f;
		} else {
			// Last frequency (Nyquist for even n_data)
			weights[i] = (n_data % 2 == 0) ? 1.0f : 2.0f;
		}
	}
}

// Demean data on GPU
__global__ void demean_data(
	FLT *RESTRICT data,        // Data to demean (in-place), length n
	CONSTANT int n,            // Length of data
	CONSTANT FLT mean)         // Mean to subtract
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < n) {
		data[i] -= mean;
	}
}

// Compute mean of data (reduction kernel)
__global__ void compute_mean(
	FLT *RESTRICT data,        // Input data, length n
	FLT *RESTRICT result,      // Output mean
	CONSTANT int n)            // Length of data
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	extern __shared__ FLT sdata[];
	
	FLT sum = 0.0f;
	if (i < n) {
		sum = data[i];
	}
	
	sdata[threadIdx.x] = sum;
	__syncthreads();
	
	// Reduction
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}
		__syncthreads();
	}
	
	if (threadIdx.x == 0) {
		atomicAdd(result, sdata[0] / n);
	}
}

// Generate transit template (simple box model)
__global__ void generate_transit_template(
	FLT *RESTRICT t,           // Time values, length n
	FLT *RESTRICT template_out,// Output template, length n
	CONSTANT int n,            // Length of data
	CONSTANT FLT period,       // Orbital period
	CONSTANT FLT epoch,        // Transit epoch
	CONSTANT FLT duration,     // Transit duration
	CONSTANT FLT depth)        // Transit depth
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < n) {
		// Phase fold
		FLT phase = fmodf(t[i] - epoch, period) / period;
		if (phase < 0) phase += 1.0f;
		
		// Center phase around 0.5
		if (phase > 0.5f) phase -= 1.0f;
		
		// Check if in transit
		FLT phase_width = duration / (2.0f * period);
		if (fabsf(phase) <= phase_width) {
			template_out[i] = -depth;
		} else {
			template_out[i] = 0.0f;
		}
	}
}
