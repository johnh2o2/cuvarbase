#include <stdio.h>
#define RESTRICT __restrict__
#define MIN_W 1E-9
#define MAX_W_COMPLEMENT 1E-9
//{CPP_DEFS}

/**
 * Sparse BLS CUDA Kernel (simple version)
 *
 * Uses bubble sort on a single thread for simplicity,
 * then parallelizes pair testing across all threads in the block.
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
 * Sparse BLS kernel - each block handles one frequency.
 * Bubble sort on thread 0, then parallel pair testing across all threads.
 *
 * Shared memory layout:
 *   sh_phi[ndata], sh_y[ndata], sh_w[ndata],
 *   sh_bls[blockDim.x], sh_best_q[blockDim.x], sh_best_phi[blockDim.x]
 * Total: 3*ndata + 3*blockDim.x floats
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
    extern __shared__ float shared_mem[];

    float* sh_phi = shared_mem;
    float* sh_y = &shared_mem[ndata];
    float* sh_w = &shared_mem[2 * ndata];
    // Thread-local storage for reductions: 3 arrays of blockDim.x
    float* sh_bls = &shared_mem[3 * ndata];                  // blockDim.x
    float* sh_best_q = &shared_mem[3 * ndata + blockDim.x];  // blockDim.x
    float* sh_best_phi = &shared_mem[3 * ndata + 2 * blockDim.x]; // blockDim.x

    unsigned int freq_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;

    while (freq_idx < nfreqs) {
        float freq = freqs[freq_idx];

        // Step 1: Load data and compute phases (parallel)
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            float phi = mod1(t[i] * freq);
            float weight = 1.f / (dy[i] * dy[i]);

            sh_phi[i] = phi;
            sh_y[i] = y[i];
            sh_w[i] = weight;
        }
        __syncthreads();

        // Step 2: Compute sum of weights (parallel reduction)
        float local_sum_w = 0.f;
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            local_sum_w += sh_w[i];
        }
        sh_bls[tid] = local_sum_w;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < blockDim.x) {
                sh_bls[tid] += sh_bls[tid + s];
            }
            __syncthreads();
        }
        float sum_w = sh_bls[0];
        __syncthreads();

        // Step 2b: Normalize weights (parallel)
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            sh_w[i] /= sum_w;
        }
        __syncthreads();

        // Step 3: Compute ybar (parallel reduction)
        float local_ybar = 0.f;
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            local_ybar += sh_w[i] * sh_y[i];
        }
        sh_bls[tid] = local_ybar;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < blockDim.x) {
                sh_bls[tid] += sh_bls[tid + s];
            }
            __syncthreads();
        }
        float ybar = sh_bls[0];
        __syncthreads();

        // Step 4: Compute YY (parallel reduction)
        float local_YY = 0.f;
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            float diff = sh_y[i] - ybar;
            local_YY += sh_w[i] * diff * diff;
        }
        sh_bls[tid] = local_YY;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < blockDim.x) {
                sh_bls[tid] += sh_bls[tid + s];
            }
            __syncthreads();
        }
        float YY = sh_bls[0];
        __syncthreads();

        // Step 5: Bubble sort by phase (single thread - O(N^2), N <= 500)
        if (tid == 0) {
            for (unsigned int i = 0; i < ndata - 1; i++) {
                for (unsigned int jj = 0; jj < ndata - i - 1; jj++) {
                    if (sh_phi[jj] > sh_phi[jj + 1]) {
                        float tmp;
                        tmp = sh_phi[jj]; sh_phi[jj] = sh_phi[jj+1]; sh_phi[jj+1] = tmp;
                        tmp = sh_y[jj];   sh_y[jj]   = sh_y[jj+1];   sh_y[jj+1]   = tmp;
                        tmp = sh_w[jj];   sh_w[jj]   = sh_w[jj+1];   sh_w[jj+1]   = tmp;
                    }
                }
            }
        }
        __syncthreads();

        // Step 6: Parallel pair testing
        // Total pairs to test:
        //   Non-wrapped: for each i in [0,ndata), j in [i+1, ndata] -> obs i..j-1
        //   Wrapped: for each i in [0,ndata), k in [0, i) -> obs i..end + 0..k-1
        // We linearize: pair_idx encodes (i, j_or_k) across both non-wrapped and wrapped.
        // Non-wrapped pairs: N*(N+1)/2 pairs (i from 0..N-1, j from i+1..N)
        // Wrapped pairs: N*(N-1)/2 pairs (i from 0..N-1, k from 0..i-1)
        // Total = N^2 pairs. We index as pair_idx in [0, N^2).

        float thread_max_bls = 0.f;
        float thread_best_q = 0.f;
        float thread_best_phi = 0.f;

        unsigned int N = ndata;
        // Non-wrapped pairs: N*(N+1)/2
        // We encode: for i=0..N-1, j=i+1..N, linear index = i*(2*N-i+1)/2 + (j-i-1)
        // But simpler: just iterate with stride over a flat index space.
        // Total non-wrapped: sum_{i=0}^{N-1} (N-i) = N*(N+1)/2
        unsigned int total_nonwrap = N * (N + 1) / 2;
        // Total wrapped: sum_{i=0}^{N-1} i = N*(N-1)/2
        unsigned int total_wrap = N * (N - 1) / 2;
        unsigned int total_pairs = total_nonwrap + total_wrap;

        for (unsigned int p = tid; p < total_pairs; p += blockDim.x) {
            float phi0, q;
            float W = 0.f;
            float YW = 0.f;

            if (p < total_nonwrap) {
                // Decode non-wrapped pair (i, j) from flat index p
                // i*(2N-i+1)/2 + (j-i-1) = p
                // Find i by scanning (N is small)
                unsigned int idx = p;
                unsigned int i = 0;
                while (idx >= (N - i)) {
                    idx -= (N - i);
                    i++;
                }
                unsigned int j = i + 1 + idx; // j in [i+1, N]

                phi0 = sh_phi[i];

                if (j < N) {
                    // Transit ends before obs j: midpoint between j-1 and j
                    q = 0.5f * (sh_phi[j] + sh_phi[j-1]) - phi0;
                } else {
                    // j == N: all obs from i to end in transit
                    q = sh_phi[N - 1] - phi0 + 1e-7f;
                }

                if (q <= 0.f || q < qmin_f || q > qmax_f) continue;

                // Sum weights and yw for obs i..j-1
                for (unsigned int m = i; m < j && m < N; m++) {
                    W += sh_w[m];
                    YW += sh_w[m] * sh_y[m];
                }
                YW -= ybar * W;

            } else {
                // Decode wrapped pair (i, k) from flat index p - total_nonwrap
                unsigned int idx = p - total_nonwrap;
                // k ranges 0..i-1 for each i (starting from i=1)
                // i=1: 1 pair (k=0), i=2: 2 pairs, ...
                // Cumulative: i*(i-1)/2 + k = idx  -> find i
                unsigned int i = 1;
                while (idx >= i) {
                    idx -= i;
                    i++;
                }
                unsigned int k = idx; // k in [0, i)

                phi0 = sh_phi[i];

                if (k > 0) {
                    q = (1.f - phi0) + 0.5f * (sh_phi[k-1] + sh_phi[k]);
                } else {
                    // k=0: only tail obs, transit wraps past phase 1
                    q = 1.f - phi0 + 1e-7f;
                }

                if (q <= 0.f || q < qmin_f || q > qmax_f) continue;

                // Sum from i to end
                for (unsigned int m = i; m < N; m++) {
                    W += sh_w[m];
                    YW += sh_w[m] * sh_y[m];
                }
                // Sum from 0 to k-1
                for (unsigned int m = 0; m < k; m++) {
                    W += sh_w[m];
                    YW += sh_w[m] * sh_y[m];
                }
                YW -= ybar * W;
            }

            float bls = bls_power(YW, W, YY, ignore_negative_delta_sols);

            if (bls > thread_max_bls) {
                thread_max_bls = bls;
                thread_best_q = q;
                thread_best_phi = phi0;
            }
        }

        // Step 7: Tree reduction to find block maximum
        sh_bls[tid] = thread_max_bls;
        sh_best_q[tid] = thread_best_q;
        sh_best_phi[tid] = thread_best_phi;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride && tid + stride < blockDim.x) {
                if (sh_bls[tid + stride] > sh_bls[tid]) {
                    sh_bls[tid] = sh_bls[tid + stride];
                    sh_best_q[tid] = sh_best_q[tid + stride];
                    sh_best_phi[tid] = sh_best_phi[tid + stride];
                }
            }
            __syncthreads();
        }

        // Step 8: Write results
        if (tid == 0) {
            bls_powers[freq_idx] = sh_bls[0];
            best_q[freq_idx] = sh_best_q[0];
            best_phi[freq_idx] = sh_best_phi[0];
        }
        __syncthreads();

        // Move to next frequency
        freq_idx += gridDim.x;
    }
}
