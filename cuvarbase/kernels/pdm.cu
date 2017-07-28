#include<stdio.h>
#define WEIGHT(k) (w==NULL ? 1.0f : w[k])
#define GAUSSIAN(x) expf(-0.5f *x*x)
#define WEIGHTED_LININTERP true
#define SKIP_BIN(i) (bin_wtots[i] * NBINS < 0.01f)
//INSERT_NBINS_HERE
#define PHASE(x,f) (x * f - floorf(x * f))

#define RESTRICT __restrict__
#define CONSTANT const


__device__ float phase_diff(
        CONSTANT float dt,
        CONSTANT float freq){
	float dphi = dt * freq - floorf(dt * freq);
	return ((dphi > 0.5f) ? 1.0f - dphi : dphi);
}

__device__ float var_step_function(
        float *RESTRICT t,
        float *RESTRICT y,
        float *RESTRICT w,
        CONSTANT float freq,
        CONSTANT int ndata){
    float bin_means[NBINS];
    float bin_wtots[NBINS];
    int bin;
    float var_tot = 0.f;
    for (int i = 0; i < NBINS; i++){
        bin_wtots[i] = 0.f;
        bin_means[i] = 0.f;
    }
    for(int i = 0; i < ndata; i++){
        bin = (int) (PHASE(t[i], freq) * NBINS);
        bin = bin % NBINS;
        bin_wtots[bin] += w[i];
        bin_means[bin] += y[i] * w[i];
    }

    for(int i = 0; i < NBINS; i++){
        if (bin_wtots[i] == 0.f)
            continue;
        bin_means[i] /= bin_wtots[i];
    }

    for(int i = 0; i < ndata; i++){
        bin = (int) (PHASE(t[i], freq) * NBINS);
        var_tot += w[i] * (y[i] - bin_means[bin]) * (y[i] - bin_means[bin]);
    }

    return var_tot;
}

__device__ float var_linear_interp(
        float *RESTRICT t,
        float *RESTRICT y,
        float *RESTRICT w,
        CONSTANT float freq,
        CONSTANT int ndata){

    float bin_means[NBINS];
    float bin_wtots[NBINS];
    int bin, bin0, bin1;
    float var_tot = 0.f;
    float phase, y0, alpha;
    for(int i = 0; i < NBINS; i++){
        bin_wtots[i] = 0.f;
        bin_means[i] = 0.f;
    }

    for(int i = 0; i < ndata; i++){
        bin = (int) (PHASE(t[i], freq) * NBINS);
        bin = bin % NBINS;
        bin_wtots[bin] += w[i];
        bin_means[bin] += w[i] * y[i];
    }

    for (int i = 0; i < NBINS; i++){
        if (bin_wtots[i] == 0.f)
            continue;
        bin_means[i] /= bin_wtots[i];
    }


    for (int i = 0; i < ndata; i++){
        phase = PHASE(t[i], freq);
        bin = (int) (phase * NBINS);
        bin = bin % NBINS;

        alpha = phase * NBINS - floorf(phase * NBINS) - 0.5f;
        bin0 = (alpha < 0) ? bin - 1 : bin;
        bin1 = (alpha < 0) ? bin : bin + 1;

        if (bin0 < 0)
            bin0 += NBINS;
        if (bin1 >= NBINS)
            bin1 -= NBINS;

        alpha += (alpha < 0) ? 1.f : 0.f;
        y0 = (1.f - alpha) * bin_means[bin0] + alpha * bin_means[bin1];
        var_tot += w[i] * (y[i] - y0) * (y[i] - y0);
    }

    return var_tot;
}


__device__ float var_binless_tophat(
        float *RESTRICT t,
        float *RESTRICT y,
        float *RESTRICT w,
        CONSTANT float freq,
        CONSTANT int ndata,
        CONSTANT float dphi){
	float mbar, tj, wtot, var, dphase;
	bool in_bin;
	var = 0.f;
	for(int j = 0; j < ndata; j++){
		mbar = 0.f;
		wtot = 0.f;
		tj = t[j];
		for(int k = 0; k < ndata; k++){
			in_bin = phase_diff(fabsf(t[k] - tj), freq) < dphi;
			wtot += in_bin ? w[k] : 0.f;
			mbar += in_bin ? w[k] * y[k] : 0.f;
		}
		mbar /= wtot;
		var += w[j] * (y[j] - mbar) * (y[j] - mbar);
	}
	return var;
}
__device__ float var_binless_gauss(
        float *RESTRICT t,
        float *RESTRICT y,
        float *RESTRICT w,
        CONSTANT float freq,
        CONSTANT int ndata,
        CONSTANT float dphi){
    float mbar, tj, wtot, var, wgt, dphase;
	var = 0.f;
    for(int j = 0; j < ndata; j++){
        mbar = 0.f;
        wtot = 0.f;
        tj = t[j];
        for(int k = 0; k < ndata; k++){
			dphase = phase_diff(fabsf(t[k] - tj), freq);
			wgt   = w[k] * GAUSSIAN(dphase / dphi);
            mbar += wgt * y[k];
            wtot += wgt;
        }
        mbar /= wtot;
        var  += w[j] * (y[j] - mbar) * (y[j] - mbar);
    }
    return var;
}
__global__ void pdm_binless_tophat(
        float *RESTRICT t,
        float *RESTRICT y,
        float *RESTRICT w,
        float *RESTRICT freqs,
        float *power,
        CONSTANT int ndata,
        CONSTANT int nfreqs,
        CONSTANT float dphi,
        CONSTANT float var){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nfreqs){
		power[i] = 1.f - var_binless_tophat(t, y, w, freqs[i], ndata, dphi) / var;
	}
}

__global__ void pdm_binless_gauss(
        float *RESTRICT t,
        float *RESTRICT y,
        float *RESTRICT w,
        float *RESTRICT freqs,
        float *power,
        CONSTANT int ndata,
        CONSTANT int nfreqs,
        CONSTANT float dphi,
        CONSTANT float var){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nfreqs){
		power[i] = 1.f - var_binless_gauss(t, y, w, freqs[i], ndata, dphi) / var;
	}
}

__global__ void pdm_binned_linterp(
        float *RESTRICT t,
        float *RESTRICT y,
        float *RESTRICT w,
        float *RESTRICT freqs,
        float *power,
        CONSTANT int ndata,
        CONSTANT int nfreqs,
        CONSTANT float dphi,
        CONSTANT float var){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nfreqs){
		power[i] = 1.f - var_linear_interp(t, y, w, freqs[i], ndata) / var;
	}
}
__global__ void pdm_binned_step(
        float *RESTRICT t,
        float *RESTRICT y,
        float *RESTRICT w,
        float *RESTRICT freqs,
        float *power,
        CONSTANT int ndata,
        CONSTANT int nfreqs,
        CONSTANT float dphi,
        CONSTANT float var){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nfreqs){
		power[i] = 1.f - var_step_function(t, y, w, freqs[i], ndata) / var;
	}
}
