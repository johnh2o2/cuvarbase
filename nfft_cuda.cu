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

__device__ float moddiff(float a, float b, float M) {
	float ret = a - b;
	if (fabsf(ret) > M/2){
		if (ret > 0)
			return ret - M;
		return M + ret;
	}
	return ret;
}

__global__ void resize_for_batch_fft(float *in, pycuda::complex<float> *out, int *n, int nbatch, int fftsize){
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	int batch = 0;
	int grid_offset = 0;
	while (batch < nbatch){
		if(i >= grid_offset + n[batch]){
			
			grid_offset += n[batch];
			batch ++;
		} else { break; }
	}

	if (batch < nbatch) {
		int k = i - grid_offset;
		int fft_index = batch * fftsize + k;

		// this shifts fft 0 point to center
		float shift = (k % 2 == 0) ? 1 : -1;

		out[fft_index] = ( pycuda::complex<float> ) (shift * in[i]);
	}
}

__global__ void precompute_psi(float *x, float *q1, float *q2, float *q3, int *n0, int *n, int nbatch, int m, float b){
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	int batch = 0;
	int data_offset = 0;
	while (batch < nbatch){
		if(i >= data_offset + n0[batch]){
			data_offset += n0[batch];
			batch ++;
		} else { break; }
	}

	if (batch < nbatch){
		
		int u = (int) floor(n[batch] * x[i] - m);

		float y = n[batch] * x[i] - ((float) u);

		q1[i] = exp(-y * y / b)/ sqrt(PI * b);
		q2[i] = exp( 2 * y / b);

	} else if (i - data_offset < 2 * m + 1) {
		int l = i - data_offset; 
		q3[l] = exp(-0.5 * l * l);	
	}

}
__global__ void fast_gaussian_grid(float *g, float *f, float *x, int *n, int *n0, int nbatch, int m, float *q1, float *q2, float *q3){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	int batch = 0;
	int grid_offset = 0;
	int data_offset = 0;
	while (batch < nbatch){
		if(i >= grid_offset + n[batch]){
			grid_offset += n[batch];
			data_offset += n0[batch];
			batch ++;
		} else { break; }
	}

	if (batch < nbatch){
		
		int u = (int) floor(n[batch] * (x[i] + 0.5) - m);
		
		float Q = q1[i];
		float F = f[i];
		int gcoord;

		//int start = (k0 < grid_offset)     ? grid_offset : k0;
		//int end   = (k0 + 2 * m + 1 > grid_offset + n[batch] ) 
		//					? grid_offset + n[batch] 
		//					: k0 + 2 * m + 1;

		for(int k = u; k < u + 2 * m + 1; k++){
			gcoord = mod(k, n[batch]);
			atomicAdd(g + gcoord + grid_offset,  Q * q3[k - u] * F);
			Q *=  q2[i];
		}
	}
}


__global__ void slow_gaussian_grid(float *g, float *f, float *x, int *n, int *n0, int nbatch, int m, float b){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	int batch       = 0;
	int grid_offset = 0;
	int data_offset = 0;
	while (batch < nbatch){
		if(i >= grid_offset + n[batch]){
			
			grid_offset += n[batch];
			data_offset += n0[batch];
			batch ++;
		} else { break; }
	}

	if (batch < nbatch){

		float dx, dgi;
		int data_index, data_coord;

		int grid_index = i - grid_offset;

		int nd = n0[batch];
		int ng = n[batch];

		int grid_min = mod(grid_index - m, ng);
	
		for ( data_index = 0; 
					data_index < nd 
			     && (ng * (x[data_index + data_offset] + 0.5f)) 
			               < grid_min; 
			  data_index++);

		int d_max = data_index + nd;
		while(data_index < d_max){
			data_coord = mod(data_index, nd) + data_offset;

			dgi = ng * (x[data_coord] + 0.5f);

			dx = moddiff(dgi, grid_index, ng);

			if ( dx > m )
				break;

			g[i] += gauss_filter_i(dx, b) * f[data_coord];

			data_index ++;
		}
	}
}


__global__ void divide_phi_hat(pycuda::complex<float> *g, int *n, int m, int nbatch, int fftsize, float b){
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	int batch = i / fftsize;

	if (batch < nbatch){
		int grid_index = i - batch * fftsize;

		int k          = (int) floorf(grid_index - fftsize / 2);
		float K        = PI * k / fftsize;
		g[grid_index] *= n[batch] * exp(b * K * K);
	} 
	
}