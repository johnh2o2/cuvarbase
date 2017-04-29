#include <pycuda-complex.hpp>
#include <stdio.h>
#define PI 3.14159265358979323846264338327950288f


__device__ float gauss_filter(float x, int n, float b){
	return exp(-(n*n*x*x) / b) / sqrt(PI * b);
}

__device__ float gauss_filter_i(float x, float b){
	return exp(-(x*x) / b) / sqrt(PI * b);
}
__device__ int mod (int a, int b)
{
   int ret = a % b;
   return (ret < 0) ? ret + b : ret;
}

__device__ float diffmod(float a, float b, float M) {
	float ret = a - b;
	if (fabsf(ret) > M/2){
		if (ret > 0)
			return ret - M;
		return M + ret;
	}
	return ret;
}

__global__ void center_fft(float *in, pycuda::complex<float> *out, int n, int nbatch){
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	int batch = i / n;

	if (batch < nbatch) {
		int k = mod(i, n);

		// this centers the fft ( * e^(-i pi n))
		float shift = (k % 2 == 0) ? 1 : -1;

		out[i] = ( pycuda::complex<float> ) (shift * in[i]);
	}
}

__global__ void precompute_psi(float *x, float *q1, float *q2, float *q3, int n0, int n, int m, float b){
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	if (i < n0){
		
		int u = (int) floor(n * x[i] - m);

		float xg = n * x[i] - ((float) u);

		q1[i] = exp(-xg * xg / b)/ sqrt(PI * b);
		q2[i] = exp(  2 * xg / b);

	} else if (i - n0 < 2 * m + 1) {
		int l = i - n0; 
		q3[l] = exp(-0.5 * l * l);	
	}

}

//TODO: precompute filter values (NOT fast gaussian gridding), and inverse filter values
//      and use these in place of the gaussian gridding

//TODO: debug fast gaussian gridding...

__global__ void fast_gaussian_grid( float *x, float *y, float *g, float *q1, float *q2, float *q3, int n0, int n, int nbatch, int m){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	int batch = i / n0;

	if (batch < nbatch){
		int i0 = mod(i, n0);
		int u = (int) floorf(n * (x[i0] + 0.5) - m);
		
		float Q = q1[i0];
		float Y = y[i];
		int gcoord;

		for(int k = u; k < u + 2 * m + 1; k++){
			gcoord = mod(k, n);
			atomicAdd(g + gcoord + batch * n,  Q * q3[k - u] * Y);
			Q *=  q2[i0];
		}
	}
}


__global__ void slow_gaussian_grid( float *x, float *y, float *g, int n0, int n, int nbatch, int m, float b){
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
			data_coord = mod(data_index, n0) + n0 * batch;

			dgi = n * (x[data_coord - n0*batch] + 0.5f);

			dx = diffmod(dgi, grid_index, n);

			if ( dx > m )
				break;

			g[i] += gauss_filter_i(dx, b) * y[data_coord];

			data_index ++;
		}
	}
}


__global__ void divide_phi_hat(pycuda::complex<float> *g, int n, int nbatch, float b){
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	int batch = i / n;

	if (batch < nbatch){
		int grid_index = i - batch * n;

		int k          = (int) floorf(grid_index - n / 2);
		float K        = PI * k / n;
		g[i]          *= n * exp(b * K * K);
	} 
	
}
