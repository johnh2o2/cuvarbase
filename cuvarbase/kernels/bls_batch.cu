#include <stdio.h>
//{CPP_DEFS}

// Multi-lightcurve BLS kernel for batch processing.
//
// Grid: (nfreqs, n_lcs)
//   blockIdx.x indexes over frequencies
//   blockIdx.y indexes over lightcurves
//
// Shared memory layout per block:
//   block_bins_yw[hist_size]  - binned weighted observations
//   block_bins_w[hist_size]   - binned weights
//   best_bls[blockDim.x]     - per-thread BLS maxima for reduction
//
// Data layout: all LC arrays padded to max_ndata and concatenated.
//   t_all[lc_idx * max_ndata + i]    for i < ndata_per_lc[lc_idx]
//   yw_all[lc_idx * max_ndata + i]
//   w_all[lc_idx * max_ndata + i]

__device__ unsigned int batch_get_id(){
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ float batch_mod1_fast(float a){
    return a - floorf(a);
}

__device__ int batch_mod(int a, int b){
    int r = a % b;
    return (r < 0) ? r + b : r;
}

__device__ float batch_bls_value(float ybar, float w, unsigned int ignore_neg){
    float bls = (w > 1e-10f && w < 1.f - 1e-10f) ? ybar * ybar / (w * (1.f - w)) : 0.f;
    return ((ignore_neg == 1) & (ybar > 0.f)) ? 0.f : bls;
}

__device__ int batch_divrndup(int a, int b){
    return (a % b > 0) ? a/b + 1 : a/b;
}


__global__ void full_bls_batch(
        const float* __restrict__ t_all,
        const float* __restrict__ yw_all,
        const float* __restrict__ w_all,
        float* __restrict__ bls_all,
        const float* __restrict__ freqs,
        const unsigned int* __restrict__ nbins0,
        const unsigned int* __restrict__ nbinsf,
        const unsigned int* __restrict__ ndata_per_lc,
        unsigned int max_ndata,
        unsigned int nfreq,
        unsigned int freq_offset,
        unsigned int hist_size,
        unsigned int noverlap,
        float dlogq,
        float dphi,
        unsigned int ignore_negative_delta_sols,
        unsigned int n_lcs){

    extern __shared__ float sh[];

    // Separate yw/w arrays in shared memory (avoid bank conflicts)
    float *block_bins_yw = sh;
    float *block_bins_w = (float *)&sh[hist_size];
    float *best_bls = (float *)&sh[2 * hist_size];

    __shared__ float f0;
    __shared__ int nb0, nbf, max_bin_width;
    __shared__ unsigned int ndata_lc;

    unsigned int lc_idx = blockIdx.y;
    if (lc_idx >= n_lcs)
        return;

    // Pointer offsets for this lightcurve
    unsigned int data_offset = lc_idx * max_ndata;
    const float *t = t_all + data_offset;
    const float *yw = yw_all + data_offset;
    const float *w = w_all + data_offset;

    // Output offset: bls_all[lc_idx * nfreq + freq_idx]
    float *bls_out = bls_all + lc_idx * nfreq;

    unsigned int s;
    int b;
    float phi, bls1, bls2, thread_max_bls, thread_yw, thread_w;

    unsigned int i_freq = blockIdx.x;
    while (i_freq < nfreq){

        thread_max_bls = 0.f;

        if (threadIdx.x == 0){
            f0 = freqs[i_freq + freq_offset];
            nb0 = nbins0[i_freq + freq_offset];
            nbf = nbinsf[i_freq + freq_offset];
            max_bin_width = batch_divrndup(nbf, nb0);
            ndata_lc = ndata_per_lc[lc_idx];
        }

        __syncthreads();

        // Initialize bins to 0
        for(unsigned int k = threadIdx.x; k < nbf; k += blockDim.x){
            block_bins_yw[k] = 0.f;
            block_bins_w[k] = 0.f;
        }

        __syncthreads();

        // Histogram the data for this LC
        for (unsigned int k = threadIdx.x; k < ndata_lc; k += blockDim.x){
            phi = batch_mod1_fast(t[k] * f0);
            b = batch_mod((int) floorf(((float) nbf) * phi - dphi), (int) nbf);

            atomicAdd(&(block_bins_yw[b]), yw[k]);
            atomicAdd(&(block_bins_w[b]), w[k]);
        }

        __syncthreads();

        // Scan q values and find best BLS
        for (unsigned int n = threadIdx.x; n < nbf; n += blockDim.x){

            thread_yw = 0.f;
            thread_w = 0.f;
            unsigned int m0 = 0;

            for (unsigned int m = 1; m < max_bin_width; m += 1){
                for (s = m0; s < m; s++){
                    thread_yw += block_bins_yw[(n + s) % nbf];
                    thread_w += block_bins_w[(n + s) % nbf];
                }
                m0 = m;

                bls1 = batch_bls_value(thread_yw, thread_w, ignore_negative_delta_sols);
                if (bls1 > thread_max_bls)
                    thread_max_bls = bls1;
            }
        }

        best_bls[threadIdx.x] = thread_max_bls;

        __syncthreads();

        // Standard tree reduction down to single warp
        for(unsigned int k = (blockDim.x / 2); k >= 32; k /= 2){
            if(threadIdx.x < k){
                bls1 = best_bls[threadIdx.x];
                bls2 = best_bls[threadIdx.x + k];
                best_bls[threadIdx.x] = (bls1 > bls2) ? bls1 : bls2;
            }
            __syncthreads();
        }

        // Final warp reduction using shuffle
        if (threadIdx.x < 32){
            float val = best_bls[threadIdx.x];

            for(int offset = 16; offset > 0; offset /= 2){
                float other = __shfl_down_sync(0xffffffff, val, offset);
                val = (val > other) ? val : other;
            }

            if (threadIdx.x == 0)
                best_bls[0] = val;
        }

        // Store result
        if (threadIdx.x == 0)
            bls_out[i_freq + freq_offset] = best_bls[0];

        i_freq += gridDim.x;
    }
}
