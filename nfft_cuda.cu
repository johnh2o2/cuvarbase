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

		// this centers the fft ( * e^(-i pi n))
		float shift = (k % 2 == 0) ? 1 : -1;

		out[i] = ( pycuda::complex<float> ) (shift * in[i]);
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

		float xg = m + (n * x[i] - floorf(n * x[i])) - 0.5;
		
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


__global__ void divide_phi_hat(
	pycuda::complex<float> *g, 
	CONSTANT int n, 
	CONSTANT int nbatch, 
	CONSTANT float b)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	int batch = i / n;

	if (batch < nbatch){
		int grid_index = i - batch * n;

		int k          = (int) floorf(grid_index - n / 2);
		float K        = PI * k / n;
		g[i]          *= n * exp(b * K * K);
	} 
	
}

