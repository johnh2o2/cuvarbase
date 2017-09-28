#include <stdio.h>
#include <pycuda-complex.hpp>

#define RESTRICT __restrict__
#define CONSTANT const
#define PI 3.14159265358979323846264338327950288f
#define FILTER gauss_filter
//{CPP_DEFS}

#ifdef DOUBLE_PRECISION
	#define ATOMIC_ADD atomicAddDouble
	#define FLT double

#else
	#define ATOMIC_ADD atomicAdd
	#define FLT float
#endif

#define CMPLX pycuda::complex<FLT>

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


__device__ FLT gauss_filter(CONSTANT FLT x, CONSTANT FLT b) {
	return exp(-(x*x) / b) / sqrt(PI * b);
}

__device__ int mod(CONSTANT int a, CONSTANT int b) {
   int ret = a % b;
   return (ret < 0) ? ret + b : ret;
}

__device__ float modflt(CONSTANT FLT a, CONSTANT FLT b){
	return a - floor(a / b) * b;
}

__device__ FLT diffmod(CONSTANT FLT a, CONSTANT FLT b, CONSTANT FLT M) {
	FLT ret = a - b;
	if (fabsf(ret) > M/2){
		if (ret > 0)
			return ret - M;
		return M + ret;
	}
	return ret;
}

__global__ void nfft_shift(
	CMPLX *in,
	CMPLX *out,
	CONSTANT int ng,
	CONSTANT int nbatch,
	CONSTANT FLT x0,
	CONSTANT FLT xf,
	CONSTANT FLT spp,
	CONSTANT FLT f0){

	int i = blockIdx.x *blockDim.x + threadIdx.x;

	int batch = i / ng;

	if (batch < nbatch) {
        FLT k0 = f0 * spp * (xf - x0);

		FLT phi = (2.f * PI * (i % ng) * k0) / ng;

        CMPLX shift = CMPLX(cos(phi), sin(phi));

		out[i] = shift * in[i];
	}
}

__global__ void precompute_psi(
	FLT *RESTRICT x, // observation times
	FLT * q1,        // precomputed filter values (length n0)
	FLT * q2,        // precomputed filter values (length n0)
	FLT * q3,        // precomputed filter values (length 2 * m + 1)
	CONSTANT int n0,     // data size
	CONSTANT int ng,      // grid size
	CONSTANT int m,      // max filter radius
	CONSTANT FLT b,      // filter scaling
	CONSTANT FLT x0,     // min(x)
	CONSTANT FLT xf,     // max(x)
	CONSTANT FLT spp)    // samples per peak
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	FLT binv = 1.f/b;
	if (i < n0){
		FLT xg = (x[i] - x0) / (spp * (xf - x0));

		xg = m + modflt(ng * xg, 1.f);

		q1[i] = exp(-xg * (xg * binv)) / sqrt(b * PI);
		q2[i] = exp( 2.f * xg * binv);

	} else if (i - n0 < 2 * m + 1) {
		int l = i - n0;
		q3[l] = exp(-l * l * binv);
	}

}

__global__ void fast_gaussian_grid(
	FLT *RESTRICT x,     // data (observation times), length n0
	FLT *RESTRICT y,     // data (observations), length (nbatch * n0)
	CMPLX * grid,          // grid, length n * nbatch
	FLT *RESTRICT q1,	 // precomputed filter values
	FLT *RESTRICT q2,	 // precomputed filter values
	FLT *RESTRICT q3,	 // precomputed filter values
	CONSTANT int n0,     // data size
	CONSTANT int ng,      // grid size
	CONSTANT int nbatch, // number of grids/datasets
	CONSTANT int m,      // max filter radius
	CONSTANT FLT x0,     // min(x)
	CONSTANT FLT xf,     // max(x)
	CONSTANT FLT spp)    // samples per peak
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int batch = i / n0;

	if (batch < nbatch){

		// datapoint
		int di = i % n0;

		// scale
		FLT xval = (x[di] - x0) / (spp * (xf - x0));

		// observation
		FLT yi = y[i];

		// nearest gridpoint (rounding down)
		int u = (int) floorf(ng * xval - m);

		// precomputed filter values
		FLT Q  = q1[di];
		FLT Q2 = q2[di];

		// add datapoint to grid
		for(int k = 0; k < 2 * m + 1; k++){
            FLT dg = Q * q3[k] * yi;
            if (!(isnan(dg) || isinf(dg)))
			    ATOMIC_ADD(&(grid[mod(k + u, ng) + batch * ng]._M_re),
				          dg);
            else
                break;
            Q *= Q2;
		}
	}
}



__global__ void slow_gaussian_grid(
	FLT *RESTRICT x,     // data (observation times)
	FLT *RESTRICT y,     // data (observations)
	CMPLX * grid,          // grid
	CONSTANT int n0,     // data size
	CONSTANT int ng,      // grid size
	CONSTANT int nbatch, // number of grids
	CONSTANT int m,      // max filter radius
	CONSTANT FLT b,      // filter scaling
	CONSTANT FLT x0,     // min(x)
	CONSTANT FLT xf,     // max(x)
	CONSTANT FLT spp)    // samples per peak
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int batch = i / ng;

	if (batch < nbatch){
		FLT dx, dgi;



		// grid index for this thread
		int grid_index = i - ng * batch;

		// iterate through data
		for(int di = 0; di < n0; di ++){

			// scale
			FLT xval = (x[di] - x0) / (spp * (xf - x0));

			// grid index of datapoint (float)
			dgi = ng * xval;

			// "distance" between grid_index and datapoint
			dx = diffmod(dgi, grid_index, ng);

			// skip if datapoint too far away
			if (dx > m)
				continue;

			// add (weighted) datapoint to grid
			grid[i] += FILTER(dx, b) * y[di + n0 * batch];
		}
	}
}

__global__ void normalize(
	CMPLX *gin,
	CMPLX *gout,
	CONSTANT int ng, // sigma * nf
	CONSTANT int nf,     // number of desired frequency samples
	CONSTANT int nbatch, // number of transforms
	CONSTANT FLT b,      // filter scaling
	CONSTANT FLT x0,     // min(x)
	CONSTANT FLT xf,     // max(x)
	CONSTANT FLT spp,    // samples per peak
	CONSTANT FLT f0)     // first frequency
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	int batch = i / nf;

	if (batch < nbatch){
		int k = i % nf;

		FLT sT = spp * (xf - x0);
        FLT n0 = (x0 / sT) * ng;
		FLT k0 = f0 * sT;
		CMPLX G = gin[batch * ng + k];

		// *= exp(2pi i (k0 + k) * n0 / n)
		FLT theta_k = (2.f * PI * n0 * (k0 + k)) / ng;

		G *= CMPLX(cos(theta_k), sin(theta_k));

		// normalization factor from gridding kernel (gaussian)
		FLT khat = PI * (k0 + k) / ng;
		gout[i] = G * exp(b * khat * khat);
	}

}

