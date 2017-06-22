#include <pycuda-complex.hpp>
#include <stdio.h>
#define PI 3.14159265358979323846264338327950288f
#define RESTRICT __restrict__
#define CONSTANT const
//#define RESTRICT
//#define CONSTANT

__device__ float gauss_filter(CONSTANT float x, CONSTANT int n, CONSTANT float b){
	return expf(-(n*n*x*x) / b) / sqrt(PI * b);
}

__device__ float gauss_filter_i(CONSTANT float x, CONSTANT float b){
	return expf(-(x*x) / b) / sqrt(PI * b);
}
__device__ int mod (CONSTANT int a, CONSTANT int b)
{
   int ret = a % b;
   return (ret < 0) ? ret + b : ret;
}

__device__ float diffmod(CONSTANT float a, CONSTANT float b, CONSTANT float M) {
	float ret = a - b;
	if (fabsf(ret) > M/2){
		if (ret > 0)
			return ret - M;
		return M + ret;
	}
	return ret;
}

__global__ void center_fft(
	float * RESTRICT in, 
	pycuda::complex<float> *out, 
	CONSTANT int n, 
	CONSTANT int nbatch)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	int batch = i / n;

	if (batch < nbatch) {
		int k = mod(i, n);

		int shift = (k % 2 == 0) ? 1 : -1;
		out[i] = pycuda::complex<float>(in[batch * n + k] * shift, 0.f);
	}
}

__global__ void precompute_psi(
	float *RESTRICT x, 
	float *RESTRICT q1, 
	float *RESTRICT q2, 
	float *RESTRICT q3, 
	CONSTANT int n0, 
	CONSTANT int n, 
	CONSTANT int m, 
	CONSTANT float b){
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	float binv = 1.f/b;	
	if (i < n0){
		//int u = (int) floorf(n * x[i] - m);

		float xg = m + (n * x[i] - floorf(n * x[i]));
		
		q1[i] = expf(-xg * xg * binv)/ sqrtf(PI * b);
		q2[i] = expf( 2 * xg * binv);

	} else if (i - n0 < 2 * m + 1) {
		int l = i - n0; 
		q3[l] = expf(-l * l * binv);	
	}

}

//TODO: precompute filter values (NOT fast gaussian gridding), and inverse filter values
//      and use these in place of the gaussian gridding


//__global__ void precompute_filter(float *x, float *inv_filter, float *filter)
//TODO: debug fast gaussian gridding...

__global__ void fast_gaussian_grid( 
	float *RESTRICT x, 
	float *RESTRICT y, 
	float *RESTRICT g, 
	float *RESTRICT q1, 
	float *RESTRICT q2, 
	float *RESTRICT q3, 
	CONSTANT int n0, 
	CONSTANT int n, 
	CONSTANT int nbatch, 
	CONSTANT int m)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	int batch = i / n0;

	if (batch < nbatch){
		int i0 = i % n0;
		int u = (int) floorf(n * (x[i0] + 0.5f) - m);
		
		float Q = q1[i0];
		float Q2 = q2[i0];
		float Y  = y[i];

		for(int k = u; k < u + 2 * m + 1; k++){
			atomicAdd(g + mod(k, n) + batch * n, Q * q3[k - u] * Y);// Q * q3[k - u] * Y);
			Q *= Q2;
		}
	}
}


__global__ void slow_gaussian_grid( 
	float *RESTRICT x, 
	float *RESTRICT y, 
	float *RESTRICT g, 
	CONSTANT int n0, 
	CONSTANT int n, 
	CONSTANT int nbatch, 
	CONSTANT int m, 
	CONSTANT float b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	int batch = i / n;

	if (batch < nbatch){
			
		float dx, dgi;
		int data_index, data_coord;

		int grid_index = i - n * batch;
		int grid_min = mod(grid_index - m, n);
	
		for ( data_index = 0; 
					data_index < n0
			     && (n * (x[data_index] + 0.5f)) 
			               < grid_min; 
			  data_index++);
		
		int d_max = data_index + n0;
		while(data_index < d_max){
			data_coord = mod(data_index, n0);
			
			dgi = n * (x[data_coord] + 0.5f);

			dx = diffmod(dgi, grid_index, n);

			if ( dx > m )
				break;

			g[i] += gauss_filter_i(dx, b) * y[data_coord + n0 * batch];
			data_index ++;
		}
		
	}
}

/*
__global__ void divide_phi_hat(
	pycuda::complex<float> *g, 
	CONSTANT int n, 
	CONSTANT int N,
	CONSTANT int nbatch, 
	CONSTANT float b,
	CONSTANT float phi0)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	int batch = i / N;

	pycuda::complex<float> I = pycuda::complex<float>(0.f, 1.f);
	
	if (batch < nbatch){
		int m = i % N;

		int kprime         = m - N/2;
		float Kprime       = (PI * kprime) / n;

		// Now convert shortened N/2 (I)FFT to the full length-N transform
		int k = m + (n-N)/2;
		int kmod = (k < n/2) ? k : n - k;
		
		float C = cosf(k * PI / n);
		float S = sinf(k * PI / n);
		
		pycuda::complex<float> G1 = g[batch * n + kmod];
		pycuda::complex<float> G2 = g[batch * n + n/2 - kmod];

		float ghat_re = 0.5 * ((G1 + G2).real() - (G1 - G2).real() * S 
							 + (G1 + G2).imag() * C);
		float ghat_im = 0.5 * ((G1 - G2).imag() - (G1 + G2).imag() * S 
			                 - (G1 - G2).real() * C);
		if (k >= n/2)
			ghat_im *= -1;

		pycuda::complex<float> ghat = pycuda::complex<float>(ghat_re, ghat_im);

		// *= exp(i * (2 * pi * phi0) * (k - n / 2)) for t[0] != 0
		float theta_k = 2.f * PI * phi0 * kprime;
		ghat *= cosf(theta_k) + I * sinf(theta_k);

		// normalization factor from gridding kernel (gaussian)
		g[i]          = ghat * expf(b * Kprime * Kprime);
	} 
	
}
*/

__global__ void divide_phi_hat(
	pycuda::complex<float> *g, 
	CONSTANT int n, // sigma * N
	CONSTANT int N, // number of desired frequency samples
	CONSTANT int nbatch, // number of transforms
	CONSTANT float b,    // scale factor
	CONSTANT float phi0) // (unscaled) phase shift resulting from t[0] != 0
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	int batch = i / N;

	pycuda::complex<float> I = pycuda::complex<float>(0.f, 1.f);
	
	if (batch < nbatch){
		int m = i % N;

		int kprime         = m - N/2;
		float Kprime       = (PI * kprime) / n;
		int k = m + (n-N)/2;
		
		pycuda::complex<float> G = g[batch * n + k];

		// *= exp(i * (2 * pi * phi0) * (k - n / 2)) for t[0] != 0
		float theta_k = 2.f * PI * phi0 * kprime;
		G *= cosf(theta_k) + I * sinf(theta_k);

		// Not sure why this is needed but necessary to be consistent
		// with jake vanderplas' NFFT (and I assume any other implementation)
		G *= (m % 2 == 0) ? 1 : -1;

		// normalization factor from gridding kernel (gaussian)
		g[i]          = G * expf(b * Kprime * Kprime);
	} 
	
}


