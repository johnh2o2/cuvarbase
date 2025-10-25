#include <stdio.h>
#define RESTRICT __restrict__
#define MIN_W 1E-9
#define MAX_W_COMPLEMENT 1E-9
//{CPP_DEFS}

/**
 * Simplified Sparse BLS CUDA Kernel for debugging
 *
 * This version uses a simpler O(N³) algorithm without fancy optimizations
 * to help identify the source of hangs in the full implementation.
 */

__device__ unsigned int get_id(){
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ float mod1(float a){
    return a - floorf(a);
}

__device__ float bls_power(float YW, float W, float YY,
                          unsigned int ignore_negative_delta_sols){
    if (ignore_negative_delta_sols && YW > 0.f)
        return 0.f;

    if (W < MIN_W || W > 1.f - MAX_W_COMPLEMENT)
        return 0.f;

    float bls = (YW * YW) / (W * (1.f - W) * YY);
    return bls;
}

/**
 * Simplified sparse BLS kernel - each block handles one frequency
 * Uses simple bubble sort and O(N³) algorithm to avoid complex synchronization
 */
__global__ void sparse_bls_kernel_simple(
    const float* __restrict__ t,
    const float* __restrict__ y,
    const float* __restrict__ dy,
    const float* __restrict__ freqs,
    unsigned int ndata,
    unsigned int nfreqs,
    unsigned int ignore_negative_delta_sols,
    float* __restrict__ bls_powers,
    float* __restrict__ best_q,
    float* __restrict__ best_phi)
{
    // Shared memory for this block
    extern __shared__ float shared_mem[];

    float* sh_phi = shared_mem;
    float* sh_y = &shared_mem[ndata];
    float* sh_w = &shared_mem[2 * ndata];
    float* sh_ybar_tmp = &shared_mem[3 * ndata];  // For reduction

    unsigned int freq_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;

    while (freq_idx < nfreqs) {
        float freq = freqs[freq_idx];

        // Step 1: Load data and compute phases
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            float phi = mod1(t[i] * freq);
            float weight = 1.f / (dy[i] * dy[i]);

            sh_phi[i] = phi;
            sh_y[i] = y[i];
            sh_w[i] = weight;
        }
        __syncthreads();

        // Step 2a: Compute sum of weights - parallel
        float local_sum_w = 0.f;
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            local_sum_w += sh_w[i];
        }
        sh_ybar_tmp[tid] = local_sum_w;
        __syncthreads();

        // Reduce to get total
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < blockDim.x) {
                sh_ybar_tmp[tid] += sh_ybar_tmp[tid + s];
            }
            __syncthreads();
        }

        float sum_w = sh_ybar_tmp[0];
        __syncthreads();

        // Step 2b: Normalize weights - parallel
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            sh_w[i] /= sum_w;
        }
        __syncthreads();

        // Step 3: Compute ybar - parallel reduction
        float local_ybar = 0.f;
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            local_ybar += sh_w[i] * sh_y[i];
        }
        sh_ybar_tmp[tid] = local_ybar;
        __syncthreads();

        // Reduce in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < blockDim.x) {
                sh_ybar_tmp[tid] += sh_ybar_tmp[tid + s];
            }
            __syncthreads();
        }

        float ybar = sh_ybar_tmp[0];
        __syncthreads();

        // Step 4: Compute YY - parallel reduction
        float local_YY = 0.f;
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            float diff = sh_y[i] - ybar;
            local_YY += sh_w[i] * diff * diff;
        }
        sh_ybar_tmp[tid] = local_YY;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < blockDim.x) {
                sh_ybar_tmp[tid] += sh_ybar_tmp[tid + s];
            }
            __syncthreads();
        }

        float YY = sh_ybar_tmp[0];
        __syncthreads();

        // Step 5: Simple bubble sort by phase (single thread)
        if (tid == 0) {
            for (unsigned int i = 0; i < ndata - 1; i++) {
                for (unsigned int j = 0; j < ndata - i - 1; j++) {
                    if (sh_phi[j] > sh_phi[j + 1]) {
                        // Swap all arrays
                        float tmp_phi = sh_phi[j];
                        sh_phi[j] = sh_phi[j + 1];
                        sh_phi[j + 1] = tmp_phi;

                        float tmp_y = sh_y[j];
                        sh_y[j] = sh_y[j + 1];
                        sh_y[j + 1] = tmp_y;

                        float tmp_w = sh_w[j];
                        sh_w[j] = sh_w[j + 1];
                        sh_w[j + 1] = tmp_w;
                    }
                }
            }
        }
        __syncthreads();

        // Step 6: Test all transit pairs (single thread for simplicity)
        if (tid == 0) {
            float max_bls = 0.f;
            float best_q_val = 0.f;
            float best_phi_val = 0.f;


            // Non-wrapped transits
            for (unsigned int i = 0; i < ndata; i++) {
                for (unsigned int j = i + 1; j <= ndata; j++) {  // Note: j == ndata is a special case for computing q, not for including observation j (which would be out of bounds)
                    float phi0 = sh_phi[i];
                    // Compute q properly - match CPU implementation
                    float q;
                    if (j < ndata) {
                        // Transit ends before observation j
                        if (j < ndata) {
                            q = 0.5f * (sh_phi[j] + sh_phi[j-1]) - phi0;
                        } else {
                            q = sh_phi[j] - phi0;
                        }
                    } else {
                        // Transit includes all remaining observations
                        q = sh_phi[ndata - 1] - phi0;
                    }

                    if (q <= 0.f || q > 0.5f) continue;

                    // Compute W and YW for observations i to j-1
                    float W = 0.f;
                    float YW = 0.f;
                    for (unsigned int k = i; k < j && k < ndata; k++) {
                        W += sh_w[k];
                        YW += sh_w[k] * sh_y[k];
                    }
                    YW -= ybar * W;

                    float bls = bls_power(YW, W, YY, ignore_negative_delta_sols);


                    if (bls > max_bls) {
                        max_bls = bls;
                        best_q_val = q;
                        best_phi_val = phi0;
                    }
                }

                // Wrapped transits: from i to end, then 0 to k
                for (unsigned int k = 0; k < i; k++) {
                    float phi0 = sh_phi[i];
                    float q;
                    if (k > 0) {
                        q = (1.f - sh_phi[i]) + 0.5f * (sh_phi[k-1] + sh_phi[k]);
                    } else {
                        q = 1.f - sh_phi[i];
                    }

                    if (q <= 0.f || q > 0.5f) continue;

                    // Compute W and YW: from i to end, plus 0 to k
                    float W = 0.f;
                    float YW = 0.f;
                    for (unsigned int m = i; m < ndata; m++) {
                        W += sh_w[m];
                        YW += sh_w[m] * sh_y[m];
                    }
                    for (unsigned int m = 0; m < k; m++) {
                        W += sh_w[m];
                        YW += sh_w[m] * sh_y[m];
                    }
                    YW -= ybar * W;

                    float bls = bls_power(YW, W, YY, ignore_negative_delta_sols);


                    if (bls > max_bls) {
                        max_bls = bls;
                        best_q_val = q;
                        best_phi_val = phi0;
                    }
                }
            }

            // Store results
            bls_powers[freq_idx] = max_bls;
            best_q[freq_idx] = best_q_val;
            best_phi[freq_idx] = best_phi_val;

        }
        __syncthreads();

        // Move to next frequency
        freq_idx += gridDim.x;
    }
}
