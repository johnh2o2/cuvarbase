#include <stdio.h>
#define RESTRICT __restrict__
#define CONSTANT const
#define MIN_W 1E-9
#define MAX_W_COMPLEMENT 1E-9
//{CPP_DEFS}

/**
 * Sparse BLS CUDA Kernel (full version)
 *
 * Uses bitonic sort (parallel) and prefix sums for O(1) range queries.
 * Based on https://arxiv.org/abs/2103.06193
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
 * Bitonic sort with striding for ndata > blockDim.x
 *
 * Sorts sh_phi, sh_y, sh_w in parallel using bitonic merge network.
 * n_pow2 must be the next power of 2 >= ndata.
 * Elements beyond ndata are padded with large values (2.0f).
 */
__device__ void bitonic_sort_by_phase(float* sh_phi, float* sh_y, float* sh_w,
                                     unsigned int ndata, unsigned int n_pow2){
    unsigned int tid = threadIdx.x;

    for (unsigned int k = 2; k <= n_pow2; k *= 2) {
        for (unsigned int j = k / 2; j > 0; j /= 2) {
            // Each thread handles multiple elements with striding
            for (unsigned int idx = tid; idx < n_pow2; idx += blockDim.x) {
                unsigned int ixj = idx ^ j;

                if (ixj > idx) {
                    // Determine sort direction
                    bool ascending = ((idx & k) == 0);

                    // Bounds check: only compare valid elements
                    float phi_a = sh_phi[idx];
                    float phi_b = sh_phi[ixj];

                    bool swap = (phi_a > phi_b) == ascending;

                    if (swap) {
                        sh_phi[idx] = phi_b;
                        sh_phi[ixj] = phi_a;

                        float tmp;
                        tmp = sh_y[idx]; sh_y[idx] = sh_y[ixj]; sh_y[ixj] = tmp;
                        tmp = sh_w[idx]; sh_w[idx] = sh_w[ixj]; sh_w[ixj] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }
}

/**
 * Main sparse BLS kernel
 *
 * Each thread block handles one frequency. Within each block:
 * 1. Compute phases and weights for all observations
 * 2. Sort observations by phase using bitonic sort
 * 3. Build prefix sums for O(1) range queries
 * 4. Test all pairs of observations as transit boundaries (parallel)
 * 5. Tree reduce to find maximum BLS
 *
 * Shared memory layout:
 *   sh_phi[n_pow2]       - phases (padded to power of 2 for bitonic sort)
 *   sh_y[n_pow2]         - y values (padded)
 *   sh_w[n_pow2]         - weights (padded)
 *   sh_cumsum_w[ndata]   - prefix sum of weights
 *   sh_cumsum_yw[ndata]  - prefix sum of w*y
 *   thread_results[3*blockDim.x] - per-thread (bls, q, phi) for reduction
 *
 * Total: 3*n_pow2 + 2*ndata + 3*blockDim.x floats
 */
__global__ void sparse_bls_kernel(
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

    // Compute n_pow2 (next power of 2 >= ndata)
    unsigned int n_pow2 = 1;
    while (n_pow2 < ndata) n_pow2 *= 2;

    float* sh_phi = shared_mem;                               // n_pow2 floats
    float* sh_y = &shared_mem[n_pow2];                        // n_pow2 floats
    float* sh_w = &shared_mem[2 * n_pow2];                    // n_pow2 floats
    float* sh_cumsum_w = &shared_mem[3 * n_pow2];             // ndata floats
    float* sh_cumsum_yw = &shared_mem[3 * n_pow2 + ndata];    // ndata floats
    float* thread_results = &shared_mem[3 * n_pow2 + 2 * ndata]; // 3*blockDim.x

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

        // Pad arrays to n_pow2 for bitonic sort
        for (unsigned int i = ndata + tid; i < n_pow2; i += blockDim.x) {
            sh_phi[i] = 2.f; // Larger than any valid phase
            sh_y[i] = 0.f;
            sh_w[i] = 0.f;
        }
        __syncthreads();

        // Step 2: Normalize weights
        float local_sum = 0.f;
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            local_sum += sh_w[i];
        }

        // Use thread_results[0..blockDim-1] as scratch for reduction
        thread_results[tid] = local_sum;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < blockDim.x)
                thread_results[tid] += thread_results[tid + s];
            __syncthreads();
        }
        float sum_w = thread_results[0];
        __syncthreads();

        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            sh_w[i] /= sum_w;
        }
        __syncthreads();

        // Step 3: Compute ybar
        local_sum = 0.f;
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            local_sum += sh_w[i] * sh_y[i];
        }
        thread_results[tid] = local_sum;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < blockDim.x)
                thread_results[tid] += thread_results[tid + s];
            __syncthreads();
        }
        float ybar = thread_results[0];
        __syncthreads();

        // Step 4: Compute YY
        local_sum = 0.f;
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            float diff = sh_y[i] - ybar;
            local_sum += sh_w[i] * diff * diff;
        }
        thread_results[tid] = local_sum;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < blockDim.x)
                thread_results[tid] += thread_results[tid + s];
            __syncthreads();
        }
        float YY = thread_results[0];
        __syncthreads();

        // Step 5: Sort by phase using bitonic sort (parallel, with striding)
        bitonic_sort_by_phase(sh_phi, sh_y, sh_w, ndata, n_pow2);

        // Step 6: Compute prefix sums using serial scan on thread 0
        // This is O(N) which is fine for N <= 500 (sparse threshold)
        if (tid == 0) {
            sh_cumsum_w[0] = sh_w[0];
            sh_cumsum_yw[0] = sh_w[0] * sh_y[0];
            for (unsigned int i = 1; i < ndata; i++) {
                sh_cumsum_w[i] = sh_cumsum_w[i-1] + sh_w[i];
                sh_cumsum_yw[i] = sh_cumsum_yw[i-1] + sh_w[i] * sh_y[i];
            }
        }
        __syncthreads();

        // Step 7: Parallel pair testing with O(1) range queries
        float thread_max_bls = 0.f;
        float thread_q = 0.f;
        float thread_phi0 = 0.f;

        unsigned int N = ndata;
        unsigned int total_nonwrap = N * (N + 1) / 2;
        unsigned int total_wrap = N * (N - 1) / 2;
        unsigned int total_pairs = total_nonwrap + total_wrap;

        for (unsigned int p = tid; p < total_pairs; p += blockDim.x) {
            float phi0, q, W, YW;

            if (p < total_nonwrap) {
                // Decode non-wrapped pair (i, j) from flat index
                unsigned int idx = p;
                unsigned int i = 0;
                while (idx >= (N - i)) {
                    idx -= (N - i);
                    i++;
                }
                unsigned int j = i + 1 + idx; // j in [i+1, N]

                phi0 = sh_phi[i];

                if (j < N) {
                    q = 0.5f * (sh_phi[j] + sh_phi[j-1]) - phi0;
                } else {
                    q = sh_phi[N - 1] - phi0 + 1e-7f;
                }

                if (q <= 0.f || q > 0.5f) continue;

                // Use prefix sums for O(1) range query: sum of w[i..j-1]
                unsigned int last = (j < N) ? j - 1 : N - 1;
                W = (i == 0) ? sh_cumsum_w[last] : sh_cumsum_w[last] - sh_cumsum_w[i - 1];
                YW = (i == 0) ? sh_cumsum_yw[last] : sh_cumsum_yw[last] - sh_cumsum_yw[i - 1];
                YW -= ybar * W;

            } else {
                // Decode wrapped pair (i, k) from flat index
                unsigned int idx = p - total_nonwrap;
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
                    q = 1.f - phi0 + 1e-7f;
                }

                if (q <= 0.f || q > 0.5f) continue;

                // W = sum(w[i..N-1]) + sum(w[0..k-1])
                W = sh_cumsum_w[N - 1] - (i > 0 ? sh_cumsum_w[i - 1] : 0.f);
                YW = sh_cumsum_yw[N - 1] - (i > 0 ? sh_cumsum_yw[i - 1] : 0.f);

                if (k > 0) {
                    W += sh_cumsum_w[k - 1];
                    YW += sh_cumsum_yw[k - 1];
                }
                YW -= ybar * W;
            }

            float bls = bls_power(YW, W, YY, ignore_negative_delta_sols);

            if (bls > thread_max_bls) {
                thread_max_bls = bls;
                thread_q = q;
                thread_phi0 = phi0;
            }
        }

        // Step 8: Store thread results and reduce
        thread_results[tid] = thread_max_bls;
        thread_results[blockDim.x + tid] = thread_q;
        thread_results[2 * blockDim.x + tid] = thread_phi0;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                if (thread_results[tid + stride] > thread_results[tid]) {
                    thread_results[tid] = thread_results[tid + stride];
                    thread_results[blockDim.x + tid] = thread_results[blockDim.x + tid + stride];
                    thread_results[2 * blockDim.x + tid] = thread_results[2 * blockDim.x + tid + stride];
                }
            }
            __syncthreads();
        }

        // Step 9: Write results
        if (tid == 0) {
            bls_powers[freq_idx] = thread_results[0];
            best_q[freq_idx] = thread_results[blockDim.x];
            best_phi[freq_idx] = thread_results[2 * blockDim.x];
        }

        freq_idx += gridDim.x;
    }
}
