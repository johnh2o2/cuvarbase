#include<stdio.h>
#define GAUSSIAN(x) expf(-0.5f *x*x)
//INSERT_NBINS_HERE
// Fractional part of x*f in float32. For x*f in (-2^-25, 0) this rounds to
// exactly 1.0f, so every (int)(PHASE * NBINS) below must be wrapped with
// `% NBINS` before it indexes a bin array.
#define PHASE(x,f) (x * f - floorf(x * f))

#define RESTRICT __restrict__
#define CONSTANT const

#define MAX_BLOCK_SIZE 256

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
        bin = bin % NBINS;
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
	float mbar, tj, wtot, var;
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
    float mbar, tj, wtot, var, wgt;
	var = 0.f;
    for(int j = 0; j < ndata; j++){
        mbar = 0.f;
        wtot = 0.f;
        tj = t[j];
        for(int k = 0; k < ndata; k++){
			float dphase = phase_diff(fabsf(t[k] - tj), freq);
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


__global__ void pdm_binned_step_fast(
        const float *RESTRICT t,
        const float *RESTRICT y,
        const float *RESTRICT w,
        const float *RESTRICT freqs,
        float *power,
        CONSTANT int ndata,
        CONSTANT int nfreqs,
        CONSTANT float dphi,
        CONSTANT float var_tot_val){

    __shared__ float s_t[MAX_BLOCK_SIZE];
    __shared__ float s_y[MAX_BLOCK_SIZE];
    __shared__ float s_w[MAX_BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float freq = (i < nfreqs) ? freqs[i] : 0.0f;

    float bin_wtots[NBINS];
    float bin_sums[NBINS];

    for (int b = 0; b < NBINS; b++){
        bin_wtots[b] = 0.f;
        bin_sums[b] = 0.f;
    }

    for (int j = 0; j < ndata; j += blockDim.x) {
        int load_idx = j + tid;
        if (load_idx < ndata) {
            s_t[tid] = t[load_idx];
            s_y[tid] = y[load_idx];
            s_w[tid] = w[load_idx];
        }
        __syncthreads();

        if (i < nfreqs) {
            int n_in_tile = (ndata - j < blockDim.x) ? (ndata - j) : blockDim.x;
            for (int k = 0; k < n_in_tile; k++) {
                float phase = PHASE(s_t[k], freq);
                int bin = (int)(phase * NBINS);
                bin = bin % NBINS;
                bin_wtots[bin] += s_w[k];
                bin_sums[bin] += s_y[k] * s_w[k];
            }
        }
        __syncthreads();
    }

    if (i < nfreqs) {
        float ss_bin = 0.f;
        for (int b = 0; b < NBINS; b++) {
            if (bin_wtots[b] > 1e-10f) {
                ss_bin += (bin_sums[b] * bin_sums[b]) / bin_wtots[b];
            }
        }
        // Assumes y is zero-meaned (weighted)
        power[i] = ss_bin / var_tot_val;
    }
}

__global__ void pdm_binned_linterp_fast(
        const float *RESTRICT t,
        const float *RESTRICT y,
        const float *RESTRICT w,
        const float *RESTRICT freqs,
        float *power,
        CONSTANT int ndata,
        CONSTANT int nfreqs,
        CONSTANT float dphi,
        CONSTANT float var_tot_val){

    __shared__ float s_t[MAX_BLOCK_SIZE];
    __shared__ float s_y[MAX_BLOCK_SIZE];
    __shared__ float s_w[MAX_BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float freq = (i < nfreqs) ? freqs[i] : 0.0f;

    float bin_wtots[NBINS];
    float bin_means[NBINS];

    for (int b = 0; b < NBINS; b++){
        bin_wtots[b] = 0.f;
        bin_means[b] = 0.f;
    }

    // Pass 1: Accumulate bins
    for (int j = 0; j < ndata; j += blockDim.x) {
        int load_idx = j + tid;
        if (load_idx < ndata) {
            s_t[tid] = t[load_idx];
            s_y[tid] = y[load_idx];
            s_w[tid] = w[load_idx];
        }
        __syncthreads();

        if (i < nfreqs) {
            int n_in_tile = (ndata - j < blockDim.x) ? (ndata - j) : blockDim.x;
            for (int k = 0; k < n_in_tile; k++) {
                float phase = PHASE(s_t[k], freq);
                int bin = (int)(phase * NBINS);
                bin = bin % NBINS;
                bin_wtots[bin] += s_w[k];
                bin_means[bin] += s_y[k] * s_w[k];
            }
        }
        __syncthreads();
    }

    if (i < nfreqs) {
        for (int b = 0; b < NBINS; b++) {
            if (bin_wtots[b] > 1e-10f) {
                bin_means[b] /= bin_wtots[b];
            }
        }
    }

    float var_pdm = 0.f;
    // Pass 2: Calculate variance
    for (int j = 0; j < ndata; j += blockDim.x) {
        int load_idx = j + tid;
        if (load_idx < ndata) {
            s_t[tid] = t[load_idx];
            s_y[tid] = y[load_idx];
            s_w[tid] = w[load_idx];
        }
        __syncthreads();

        if (i < nfreqs) {
            int n_in_tile = (ndata - j < blockDim.x) ? (ndata - j) : blockDim.x;
            for (int k = 0; k < n_in_tile; k++) {
                float phase = PHASE(s_t[k], freq);
                float p_nbins = phase * NBINS;
                int bin = (int)(p_nbins);
                bin = bin % NBINS;

                float alpha = p_nbins - floorf(p_nbins) - 0.5f;
                int bin0 = (alpha < 0) ? bin - 1 : bin;
                int bin1 = (alpha < 0) ? bin : bin + 1;

                if (bin0 < 0) bin0 += NBINS;
                if (bin1 >= NBINS) bin1 -= NBINS;

                alpha += (alpha < 0) ? 1.f : 0.f;
                float y0 = (1.f - alpha) * bin_means[bin0] + alpha * bin_means[bin1];
                float dy = s_y[k] - y0;
                var_pdm += s_w[k] * dy * dy;
            }
        }
        __syncthreads();
    }

    if (i < nfreqs) {
        power[i] = 1.f - var_pdm / var_tot_val;
    }
}

__global__ void pdm_binless_tophat_fast(
        const float *RESTRICT t,
        const float *RESTRICT y,
        const float *RESTRICT w,
        const float *RESTRICT freqs,
        float *power,
        CONSTANT int ndata,
        CONSTANT int nfreqs,
        CONSTANT float dphi,
        CONSTANT float var_tot_val){

    __shared__ float s_t[MAX_BLOCK_SIZE];
    __shared__ float s_y[MAX_BLOCK_SIZE];
    __shared__ float s_w[MAX_BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float freq = (i < nfreqs) ? freqs[i] : 0.0f;

    float total_var_pdm = 0.f;

    for (int j = 0; j < ndata; j++) {
        float tj = t[j];
        float yj = y[j];
        float wj = w[j];

        float mbar = 0.f;
        float wtot = 0.f;

        for (int ks = 0; ks < ndata; ks += blockDim.x) {
            int load_idx = ks + tid;
            if (load_idx < ndata) {
                s_t[tid] = t[load_idx];
                s_y[tid] = y[load_idx];
                s_w[tid] = w[load_idx];
            }
            __syncthreads();

            if (i < nfreqs) {
                int n_in_tile = (ndata - ks < blockDim.x) ? (ndata - ks) : blockDim.x;
                for (int k = 0; k < n_in_tile; k++) {
                    float dph = phase_diff(fabsf(s_t[k] - tj), freq);
                    if (dph < dphi) {
                        mbar += s_w[k] * s_y[k];
                        wtot += s_w[k];
                    }
                }
            }
            __syncthreads();
        }

        if (i < nfreqs && wtot > 1e-10f) {
            float diff = yj - (mbar / wtot);
            total_var_pdm += wj * diff * diff;
        }
    }

    if (i < nfreqs) {
        power[i] = 1.f - total_var_pdm / var_tot_val;
    }
}

__global__ void pdm_binless_gauss_fast(
        const float *RESTRICT t,
        const float *RESTRICT y,
        const float *RESTRICT w,
        const float *RESTRICT freqs,
        float *power,
        CONSTANT int ndata,
        CONSTANT int nfreqs,
        CONSTANT float dphi,
        CONSTANT float var_tot_val){

    __shared__ float s_t[MAX_BLOCK_SIZE];
    __shared__ float s_y[MAX_BLOCK_SIZE];
    __shared__ float s_w[MAX_BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float freq = (i < nfreqs) ? freqs[i] : 0.0f;
    float inv_dphi = 1.0f / dphi;

    float total_var_pdm = 0.f;

    for (int j = 0; j < ndata; j++) {
        float tj = t[j];
        float yj = y[j];
        float wj = w[j];

        float mbar = 0.f;
        float wtot = 0.f;

        for (int ks = 0; ks < ndata; ks += blockDim.x) {
            int load_idx = ks + tid;
            if (load_idx < ndata) {
                s_t[tid] = t[load_idx];
                s_y[tid] = y[load_idx];
                s_w[tid] = w[load_idx];
            }
            __syncthreads();

            if (i < nfreqs) {
                int n_in_tile = (ndata - ks < blockDim.x) ? (ndata - ks) : blockDim.x;
                for (int k = 0; k < n_in_tile; k++) {
                    float dph = phase_diff(fabsf(s_t[k] - tj), freq);
                    float x = dph * inv_dphi;
                    float wgt = s_w[k] * expf(-0.5f * x * x);
                    mbar += wgt * s_y[k];
                    wtot += wgt;
                }
            }
            __syncthreads();
        }

        if (i < nfreqs && wtot > 1e-10f) {
            float diff = yj - (mbar / wtot);
            total_var_pdm += wj * diff * diff;
        }
    }

    if (i < nfreqs) {
        power[i] = 1.f - total_var_pdm / var_tot_val;
    }
}
