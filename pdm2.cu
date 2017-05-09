#include<stdio.h>
#define WEIGHT(k) (w==NULL ? 1.0f : w[k])
#define GAUSSIAN(x) expf(-0.5f *x*x)
#define WEIGHTED_LININTERP true
#define SKIP_BIN(i) (bin_wtots[i] * NBINS < 0.01f)
#define NBINS 30
#define PHASE(x,f) (x * f - floorf(x * f))
__device__ float phase_diff(float dt, float freq){
	float dphi = dt * freq - floorf(dt * freq);
	return ((dphi > 0.5f) ? 1.0f - dphi : dphi);
}

__device__ float var_step_function(float *t, float *y, float *w, float freq, int ndata, float dphi){
    float bin_means[NBINS];
    float bin_vars[NBINS];
    float bin_wtots[NBINS];
    int bin;
    float wtot_skipped = 0.f;
    float var_tot = 0.f;
    for (int i = 0; i < NBINS; i++){
        bin_wtots[i] = 0.f;
        bin_means[i] = 0.f;
        bin_vars[i] = 0.f;
    }
    for(int i = 0; i < ndata; i++){
        bin = (int) (PHASE(t[i], freq) * NBINS);
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
        if (bin_wtots[bin] == 0.f){
            wtot_skipped += w[i];
            continue;
        }
        bin_vars[bin] += w[i] * (y[i] - bin_means[bin]) * (y[i] - bin_means[bin]);
    }
    
    for(int i = 0; i < NBINS; i++)
        var_tot += bin_vars[i];

    return var_tot / (1.f - wtot_skipped);
}

__device__ float var_linear_interp(float *t, float *y, float *w, float freq, int ndata){
    float bin_means[NBINS];
    float bin_vars[NBINS];
    float bin_wtots[NBINS];
    int bin, bin0, bin1;
    float dphi_f = 1.f/NBINS;
    float var_tot = 0.f;
    float wtot_skipped = 0.f;
    float phase, yprior, alpha;
    for(int i = 0; i < NBINS; i++){
        bin_wtots[i] = 0.f;
        bin_means[i] = 0.f;
        bin_vars[i] = 0.f;
    }

    for(int i = 0; i < ndata; i++){
        bin = (int) (PHASE(t[i], freq) * NBINS);
        bin = bin % NBINS;
        //printf("data[%d]; freq=%.3e; (%.3e -> %.3e, %.3e, %.3e) in bin %d\n", i, freq, t[i], phase_diff(t[i], freq), y[i], w[i], bin);
        bin_wtots[bin] += w[i];
        bin_means[bin] += w[i] * y[i];
    }
    
    for (int i = 0; i < NBINS; i++){
        //printf("bin %d (freq = %.3e) mean*wtot = %.5e, wtot = %.5e\n", i, freq, bin_means[i], bin_wtots[i]);
        if (bin_wtots[i] == 0.f)
            continue;
        bin_means[i] /= bin_wtots[i];
    }


    for (int i = 0; i < ndata; i++){
        phase = PHASE(t[i], freq);
        bin = (int) (phase * NBINS);
        bin = bin % NBINS;
        
        alpha = phase * NBINS - (bin + 0.5f) * dphi_f;
        bin0 = (alpha < 0) ? bin - 1 : bin;
        bin1 = (alpha < 0) ? bin : bin + 1;

        if (bin0 < 0)
            bin0 += NBINS;
        if (bin1 >= NBINS)
            bin1 -= NBINS;

        if (SKIP_BIN(bin0) || SKIP_BIN(bin1)){
            wtot_skipped += w[i];
            continue;
        }
        
        alpha = (alpha < 0) ? alpha + 1.f : alpha;
        yprior = (1.f - alpha) * bin_means[bin0] + alpha * bin_means[bin1];
        bin_vars[bin] += w[i] * (y[i] - yprior) * (y[i] - yprior);
    }

    for(int i = 0; i < NBINS; i++)
        var_tot += bin_vars[i];

    return var_tot / (1.f - wtot_skipped);
}


__device__ float var_binless_tophat(float *t, float *y, float *w, float freq, int ndata, float dphi){
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
__device__ float var_binless_gauss(float *t, float *y, float *w, float freq, int ndata, float dphi){
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
		power[i] = var/var_binless_tophat(t, y, w, freqs[i], ndata, dphi);
	}
}

__global__ void pdm_gauss(float *t, float *y, float *w, float *freqs, float *power, int ndata, int nfreqs, float dphi, float var){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nfreqs){
		power[i] = var/var_binless_gauss(t, y, w, freqs[i], ndata, dphi);
	}
}

__global__ void pdm_binned_linterp(float *t, float *y, float *w, float *freqs, float *power, int ndata, int nfreqs, float dphi, float var){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nfreqs){
		power[i] = var/var_linear_interp(t, y, w, freqs[i], ndata);
	}
}
