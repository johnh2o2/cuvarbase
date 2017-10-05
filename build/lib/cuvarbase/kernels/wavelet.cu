#include<stdio.h>
#define WEIGHT(k) (w==NULL ? 1.0f : w[k])
#define GAUSSIAN(x) expf(-0.5f *x*x)
#define WEIGHTED_LININTERP true
#define SKIP_BIN(i) (bin_wtots[i] * NBINS < 0.01f)
//INSERT_NBINS_HERE
#define PHASE(x,f) (x * f - floorf(x * f))
#define TWOPI 6.28318530718f
#define RESTRICT __restrict__
#define CONSTANT const
#define MIN_NOBS 10
#define wavelet full_wavelet


__device__ float fast_wavelet(float dt, float sigma, float freq){
	float a = fabs(TWOPI * sigma * freq * dt);
	
	return a < 1.f ? 1.f - 3.f * a * a + 2.f * a * a * a : 0.f;
}

__device__ float full_wavelet(float dt, float sigma, float freq){
	float a = fabs(TWOPI * sigma * freq * dt);
	
	return expf(-a*a);
}

__device__ float cosine_wtransform(float *t, float *y, float *w, float freq, float tau, float sigma, 
									int imin, int imax){
	float pow = 0.f;
	float weight = 0.f;
	float tot_weight = 0.f;
	for(int i = imin; i <= imax; i++){
		weight = wavelet(t[i] - tau, sigma, freq) * (w == NULL ? 1.f : w[i]);
		tot_weight += weight;
		pow += y[i] * weight * cos(TWOPI * freq * t[i]);
	}
	return pow / tot_weight;
}

__device__ float sine_wtransform(float *t, float *y, float *w, float freq, float tau, float sigma, 
									int imin, int imax){
	float pow = 0.f;
	float weight = 0.f;
	float tot_weight = 0.f;
	for(int i = imin; i <= imax; i++){
		weight = wavelet(t[i] - tau, sigma, freq) * (w == NULL ? 1.f : w[i]);
		tot_weight += weight;
		pow += y[i] * weight * cos(TWOPI * freq * t[i]);
	}
	return pow / tot_weight;
}

__device__ float weighted_mean(float *t, float *y, float *w, float freq, float tau, 
							float sigma, int imin, int imax){
	float s = 0.f;
	float weight = 0.f;
	float total_weight = 0.f;
	for(int i = imin; i <= imax; i++){
		weight = wavelet(t[i] - tau, sigma, freq) * (w == NULL ? 1.f : w[i]);
		s += y[i] * weight;
		total_weight += weight;
	}
	return s / total_weight;
}

__device__ float weighted_var(float *t, float *y, float *w, float freq, float tau, 
							float sigma, int imin, int imax){
	float s = 0.f;
	float weight = 0.f;
	float total_weight = 0.f;
	for(int i = imin; i <= imax; i++){
		weight = wavelet(t[i] - tau, sigma, freq) * (w == NULL ? 1.f : w[i]);
		s += y[i] * y[i] * weight;
		total_weight += weight;
	}
	return s / total_weight;
}

__device__ float power(float *t, float *y, float *w, float freq, float tau, 
						float prec, float sigma, int nobs){

	// least squares (lomb scargle with floating mean)

	int imin = 0;
	int imax = nobs - 1;

	float wmin = pow(10.f, -prec);

	while( imin < nobs && wavelet(t[imin] - tau, sigma, freq) < wmin) imin ++;
	while( imax > 0    && wavelet(t[imax] - tau, sigma, freq) < wmin) imax --;

	if (imax - imin < MIN_NOBS) return 0.f;

	float Y = weighted_mean(t, y, w, freq, tau, sigma, imin, imax);
	float YY = weighted_var(t, y, w, freq, tau, sigma, imin, imax) - Y*Y;

	float C = cosine_wtransform(t, w, NULL, freq, tau, sigma, imin, imax);
	float S =   sine_wtransform(t, w, NULL, freq, tau, sigma, imin, imax);

	float C2 = cosine_wtransform(t, w, NULL, 2 * freq, tau, sigma, imin, imax);
	float S2 =   sine_wtransform(t, w, NULL, 2 * freq, tau, sigma, imin, imax);

	float YC = cosine_wtransform(t, y, w, freq, tau, sigma, imin, imax) - Y * C;
	float YS =   sine_wtransform(t, y, w, freq, tau, sigma, imin, imax) - Y * S;

	float CC = 0.5f * ( 1.f + C2 ) - C * C;
	float CS = 0.5f * S2 - C * S;
	float SS = 0.5f * ( 1.f - C2 ) - S * S;

	float D = CC * SS - CS * CS;

	float p = (SS * YC * YC + CC * YS * YS - 2 * CS * YC * YS) / (YY * D);

	// force 0 < p < 1
	return p < 0.f ? 0.f : (p > 1.f ? 0.f : p);
}


__device__ int sumint(int *arr, int len){
	int s = 0.f;
	for(int i = 0; i < len; i++)
		s += arr[i];
	return s;
}


__global__ void wavelet_spectrogram(float *t, float *y, float *w, float *spectrogram, 
										float *freqs, float *taus, int *ntaus, int nfreqs, 
										int nobs, float sigma, float prec){

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int tot_ntaus = sumint(ntaus, nfreqs);
	if (i < tot_ntaus){
		int fno = 0;
		int s = 0;
		while(s < i){
			fno ++;
			s += ntaus[fno];
		}

		float tau = taus[i];
		float freq = freqs[fno];

		spectrogram[i] = power(t, y, w, freq, tau, prec, sigma, nobs);
	}
}