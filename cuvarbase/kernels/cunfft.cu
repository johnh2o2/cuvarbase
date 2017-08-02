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
	CONSTANT int n,
	CONSTANT int nbatch,
	CONSTANT FLT x0,
	CONSTANT FLT xf,
	CONSTANT FLT spp,
	CONSTANT FLT f0){

	int i = blockIdx.x *blockDim.x + threadIdx.x;

	int batch = i / n;

	if (batch < nbatch) {
		FLT phi0 = f0 * (xf - x0) * spp / n;
		FLT phi = 2 * PI * mod(i, n) * phi0;
		CMPLX shift = CMPLX(cos(phi), sin(phi))

		out[i] = shift * CMPLX(in[batch * n + k], 0.0f);
	}
}

__global__ void precompute_psi(
	FLT *RESTRICT x, // observation times
	FLT * q1,        // precomputed filter values (length n0)
	FLT * q2,        // precomputed filter values (length n0)
	FLT * q3,        // precomputed filter values (length 2 * m + 1)
	CONSTANT int n0,     // data size
	CONSTANT int n,      // grid size
	CONSTANT int m,      // max filter radius
	CONSTANT FLT b)      // filter scaling
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	FLT binv = 1.f/b;
	if (i < n0){

		FLT xg = m + modflt(n * x[i], 1.f);

		q1[i] = exp(-xg * (xg * binv)) / sqrt(b * PI);
		q2[i] = exp( 2.f * xg * binv);

	} else if (i - n0 < 2 * m + 1) {
		int l = i - n0;
		q3[l] = exp(-l * l * binv);
	}
}

__global__ void precompute_psi_noscale(
	FLT *RESTRICT x, // observation times
	FLT * q1,        // precomputed filter values (length n0)
	FLT * q2,        // precomputed filter values (length n0)
	FLT * q3,        // precomputed filter values (length 2 * m + 1)
	CONSTANT int n0,     // data size
	CONSTANT int n,      // grid size
	CONSTANT int m,      // max filter radius
	CONSTANT FLT b,      // filter scaling
	CONSTANT FLT x0,     // min(x)
	CONSTANT FLT xf,     // max(x)
	CONSTANT FLT spp)    // samples per peak
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	FLT binv = 1.f/b;
	if (i < n0){
		FLT xg = (x[i] - x0) / (spp * (xf - x0)) - 0.5f;

		xg = m + modflt(n * xg, 1.f);

		q1[i] = exp(-xg * (xg * binv)) / sqrt(b * PI);
		q2[i] = exp( 2.f * xg * binv);

	} else if (i - n0 < 2 * m + 1) {
		int l = i - n0;
		q3[l] = exp(-l * l * binv);
	}

}


__global__ void fast_gaussian_grid(
	FLT *RESTRICT x,     // data (observation times), length n0
	FLT *RESTRICT y,     // data (observations), length nbatch * n0
	CMPLX * grid,          // grid, length n * nbatch
	FLT *RESTRICT q1,	 // precomputed filter values
	FLT *RESTRICT q2,	 // precomputed filter values
	FLT *RESTRICT q3,	 // precomputed filter values
	CONSTANT int n0,     // data size
	CONSTANT int n,      // grid size
	CONSTANT int nbatch, // number of grids/datasets
	CONSTANT int m){     // max filter radius

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int batch = i / n0;

	if (batch < nbatch){
		// datapoint
		int di = i % n0;

		// observation
		FLT yi = y[i];

		// nearest gridpoint (rounding down)
		int u = (int) floorf(n * (x[di] + 0.5f) - m);

		// precomputed filter values
		FLT Q  = q1[di];
		FLT Q2 = q2[di];

		// add datapoint to grid
		for(int k = u; k < u + 2 * m + 1; k++){
			&(grid[mod(k, n) + batch * n]._M_re)
			ATOMIC_ADD(&(grid[mod(k, n) + batch * n]._M_re), Q * q3[k - u] * yi);
			Q *= Q2;
		}
	}
}

__global__ void fast_gaussian_grid_noscale(
	FLT *RESTRICT x,     // data (observation times), length n0
	FLT *RESTRICT y,     // data (observations), length (nbatch * n0)
	CMPLX * grid,          // grid, length n * nbatch
	FLT *RESTRICT q1,	 // precomputed filter values
	FLT *RESTRICT q2,	 // precomputed filter values
	FLT *RESTRICT q3,	 // precomputed filter values
	CONSTANT int n0,     // data size
	CONSTANT int n,      // grid size
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
		FLT xval = (x[di] - x0) / (spp * (xf - x0)) - 0.5f;

		// observation
		FLT yi = y[i];

		// nearest gridpoint (rounding down)
		int u = (int) floorf(n * (xval + 0.5f) - m);

		// precomputed filter values
		FLT Q  = q1[di];
		FLT Q2 = q2[di];

		// add datapoint to grid
		for(int k = u; k < u + 2 * m + 1; k++){
			ATOMIC_ADD(&(grid[mod(k, n) + batch * n]._M_re), Q * q3[k - u] * yi);
			Q *= Q2;
		}
	}
}



__global__ void slow_gaussian_grid(
	FLT *RESTRICT x,     // data (observation times)
	FLT *RESTRICT y,     // data (observations)
	CMPLX * grid,          // grid
	CONSTANT int n0,     // data size
	CONSTANT int n,      // grid size
	CONSTANT int nbatch, // number of grids
	CONSTANT int m,      // max filter radius
	CONSTANT FLT b)      // filter scaling
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int batch = i / n;

	if (batch < nbatch){
		FLT dx, dgi;

		// grid index for this thread
		int grid_index = i - n * batch;

		// iterate through data
		for(int di = 0; di < n0; di ++){

			// grid index of datapoint (float)
			dgi = n * (x[di] + 0.5f);

			// "distance" between grid_index and datapoint
			dx = diffmod(dgi, grid_index, n);

			// skip if datapoint too far away
			if (dx > m)
				continue;

			// add (weighted) datapoint to grid
			grid[i] += FILTER(dx, b) * y[di + n0 * batch];
		}
	}
}

__global__ void slow_gaussian_grid_noscale(
	FLT *RESTRICT x,     // data (observation times)
	FLT *RESTRICT y,     // data (observations)
	CMPLX * grid,          // grid
	CONSTANT int n0,     // data size
	CONSTANT int n,      // grid size
	CONSTANT int nbatch, // number of grids
	CONSTANT int m,      // max filter radius
	CONSTANT FLT b,      // filter scaling
	CONSTANT FLT x0,     // min(x)
	CONSTANT FLT xf,     // max(x)
	CONSTANT FLT spp)    // samples per peak 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int batch = i / n;

	if (batch < nbatch){
		FLT dx, dgi;



		// grid index for this thread
		int grid_index = i - n * batch;

		// iterate through data
		for(int di = 0; di < n0; di ++){

			// scale
			FLT xval = (x[di] - x0) / (spp * (xf - x0)) - 0.5f;

			// grid index of datapoint (float)
			dgi = n * (xval + 0.5f);

			// "distance" between grid_index and datapoint
			dx = diffmod(dgi, grid_index, n);

			// skip if datapoint too far away
			if (dx > m)
				continue;

			// add (weighted) datapoint to grid
			grid[i] += FILTER(dx, b) * y[di + n0 * batch];
		}
	}
}

__global__ void divide_phi_hat(
	CMPLX *gin,
	CMPLX *gout,
	CONSTANT int n, // sigma * N
	CONSTANT int N, // number of desired frequency samples
	CONSTANT int nbatch, // number of transforms
	CONSTANT FLT b,    // scale factor
	CONSTANT FLT phi0) // (unscaled) phase shift resulting from t[0] != 0
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	int batch = i / N;

	if (batch < nbatch){
		int m = i % N;

		int kprime = m - N/2;
		FLT Kprime = (PI * kprime) / n;
		int k = m + (n-N)/2;

		CMPLX G = gin[batch * n + k];

		// *= exp(i * (2 * pi * phi0) * (k - n / 2)) for t[0] != 0
		FLT theta_k = 2.f * PI * phi0 * kprime;
		G *= CMPLX(cos(theta_k), sin(theta_k));

		// Not sure why this is needed but necessary to be consistent
		// with jake vanderplas' NFFT (and I assume any other implementation)
		G *= (m % 2 == 0) ? 1.f : -1.f;

		// normalization factor from gridding kernel (gaussian)
		gout[i] = G * exp(b * Kprime * Kprime);
	}

}


__global__ void divide_phi_hat_noscale(
	CMPLX *gin,
	CMPLX *gout,
	CONSTANT int n, // sigma * N
	CONSTANT int N, // number of desired frequency samples
	CONSTANT int nbatch, // number of transforms
	CONSTANT FLT b,      // filter scaling
	CONSTANT FLT x0,     // min(x)
	CONSTANT FLT xf,     // max(x)
	CONSTANT FLT spp,    // samples per peak
	CONSTANT FLT f0)     // first frequency
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	int batch = i / N;

	if (batch < nbatch){
		int k = i % N;
		FLT kp = (PI * k) / n;

		CMPLX G = gin[batch * n + k];

		// *= exp(i * (2 * pi * phi0) * (k - n / 2)) for t[0] != 0
		FLT phi0 = x0 / (spp * (xf - x0));
		FLT k0 = f0 * spp * (xf - x0);

		FLT theta_k = 2.f * PI * phi0 * (k0 + k);

		G *= CMPLX(cos(theta_k), sin(theta_k));

		// Not sure why this is needed but necessary to be consistent
		// with jake vanderplas' NFFT (and I assume any other implementation)
		//G *= (m % 2 == 0) ? 1.f : -1.f;

		// normalization factor from gridding kernel (gaussian)
		gout[i] = G * exp(b * kp * kp);
	}

}

