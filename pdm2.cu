#include<stdio.h>
#define WEIGHT(k) (w==NULL ? 1.0f : w[k])
#define GAUSSIAN(x) expf(-0.5f *x*x)
__device__ float phase_diff(float dt, float freq){
	float dphi = dt * freq - floorf(dt * freq);
	return ((dphi > 0.5f) ? 1.0f - dphi : dphi);
}

__device__ float var_smooth_tophat(float *t, float *y, float *w, float freq, int ndata, float dphi){
	float mbar, tj, wtot, var, dphase;
	bool in_bin;
	var = 0.f;
	for(int j = 0; j < ndata; j++){
		mbar = 0.f;
		wtot = 0.f;
		tj = t[j];
		//printf("%d -- start\n", j);
		for(int k = 0; k < ndata; k++){
			in_bin = phase_diff(fabsf(t[k] - tj), freq) < dphi;
			wtot += in_bin ? w[k] : 0.f;
			mbar += in_bin ? w[k] * y[k] : 0.f;
		}
		//printf("%d -- end\n", j);
		mbar /= wtot;
		var += w[j] * (y[j] - mbar) * (y[j] - mbar);
	}
	return var;
}
__device__ float var_smooth_gauss(float *t, float *y, float *w, float freq, int ndata, float dphi){
        float mbar, tj, wtot, var, wgt, dphase;
	var = 0.f;
        for(int j = 0; j < ndata; j++){
                mbar = 0.f;
                wtot = 0.f;
                tj = t[j];
                //printf("%d -- start\n", j);
                for(int k = 0; k < ndata; k++){
			dphase = phase_diff(fabsf(t[k] - tj), freq);
			wgt   = w[k] * GAUSSIAN(dphase / dphi);
                        mbar += wgt * y[k];
                        wtot += wgt;
                }
                //printf("%d -- end\n", j);
                mbar /= wtot;
                var  += w[j] * (y[j] - mbar) * (y[j] - mbar);
        }
        return var;
}
__global__ void pdm_tophat(float *t, float *y, float *w, float *freqs, float *power, int ndata, int nfreqs, float dphi, float var){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nfreqs){
		power[i] = var/var_smooth_tophat(t, y, w, freqs[i], ndata, dphi);
	}
}

__global__ void pdm_gauss(float *t, float *y, float *w, float *freqs, float *power, int ndata, int nfreqs, float dphi, float var){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nfreqs){
		power[i] = var/var_smooth_gauss(t, y, w, freqs[i], ndata, dphi);
	}
}
