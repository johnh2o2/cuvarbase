#include <stdio.h>
#define RESTRICT __restrict__
#define CONSTANT const
#define MIN_W 1E-9
#define MAX_W_COMPLEMENT 1E-9
//{CPP_DEFS}

/**
 * Sparse BLS CUDA Kernel
 *
 * Implementation of sparse Box Least Squares algorithm based on
 * https://arxiv.org/abs/2103.06193
 *
 * Instead of binning, this algorithm tests all pairs of sorted observations
 * as potential transit boundaries. This is more efficient for small datasets
 * (ndata < ~500) where the O(N²) complexity per frequency is acceptable.
 */

__device__ unsigned int get_id(){
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ float mod1(float a){
    return a - floorf(a);
}

/**
 * Compute BLS power for given parameters
 *
 * @param YW: Weighted sum of y values in transit
 * @param W: Sum of weights in transit
 * @param YY: Total variance normalization
 * @param ignore_negative_delta_sols: If true, ignore inverted dips (YW > 0)
 * @return: BLS power value
 */
__device__ float bls_power(float YW, float W, float YY,
                          unsigned int ignore_negative_delta_sols){
    // Check if we should ignore this solution
    if (ignore_negative_delta_sols && YW > 0.f)
        return 0.f;

    // Check weight bounds
    if (W < MIN_W || W > 1.f - MAX_W_COMPLEMENT)
        return 0.f;

    // Compute BLS: (YW)² / (W * (1-W) * YY)
    float bls = (YW * YW) / (W * (1.f - W) * YY);
    return bls;
}

/**
 * Bitonic sort for sorting observations by phase within shared memory
 * Uses cooperative sorting across all threads in the block
 *
 * @param sh_phi: Shared memory array of phases
 * @param sh_y: Shared memory array of y values
 * @param sh_w: Shared memory array of weights
 * @param sh_indices: Shared memory array of original indices
 * @param n: Number of elements to sort
 */
__device__ void bitonic_sort_by_phase(float* sh_phi, float* sh_y, float* sh_w,
                                     int* sh_indices, unsigned int n){
    unsigned int tid = threadIdx.x;

    // Bitonic sort: repeatedly merge sorted sequences
    for (unsigned int k = 2; k <= n; k *= 2) {
        for (unsigned int j = k / 2; j > 0; j /= 2) {
            unsigned int ixj = tid ^ j;

            if (ixj > tid && tid < n && ixj < n) {
                // Determine sort direction
                bool ascending = ((tid & k) == 0);
                bool swap = (sh_phi[tid] > sh_phi[ixj]) == ascending;

                if (swap) {
                    // Swap all arrays in lockstep
                    float tmp_phi = sh_phi[tid];
                    float tmp_y = sh_y[tid];
                    float tmp_w = sh_w[tid];
                    int tmp_idx = sh_indices[tid];

                    sh_phi[tid] = sh_phi[ixj];
                    sh_y[tid] = sh_y[ixj];
                    sh_w[tid] = sh_w[ixj];
                    sh_indices[tid] = sh_indices[ixj];

                    sh_phi[ixj] = tmp_phi;
                    sh_y[ixj] = tmp_y;
                    sh_w[ixj] = tmp_w;
                    sh_indices[ixj] = tmp_idx;
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
 * 1. Compute phases for all observations at this frequency
 * 2. Sort observations by phase in shared memory
 * 3. Test all pairs of observations as potential transit boundaries
 * 4. Find maximum BLS power and corresponding (q, phi0)
 *
 * @param t: Observation times [ndata]
 * @param y: Observation values [ndata]
 * @param dy: Observation uncertainties [ndata]
 * @param freqs: Frequencies to test [nfreqs]
 * @param ndata: Number of observations
 * @param nfreqs: Number of frequencies
 * @param ignore_negative_delta_sols: Whether to ignore inverted dips
 * @param bls_powers: Output BLS powers [nfreqs]
 * @param best_q: Output best q values [nfreqs]
 * @param best_phi: Output best phi0 values [nfreqs]
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
    // Shared memory layout:
    // [phi, y, w, indices, cumsum_w, cumsum_yw, thread_max_bls, thread_best_q, thread_best_phi]
    extern __shared__ float shared_mem[];

    float* sh_phi = shared_mem;                           // ndata floats
    float* sh_y = &shared_mem[ndata];                     // ndata floats
    float* sh_w = &shared_mem[2 * ndata];                 // ndata floats
    int* sh_indices = (int*)&shared_mem[3 * ndata];      // ndata ints
    float* sh_cumsum_w = &shared_mem[3 * ndata + ndata]; // ndata floats
    float* sh_cumsum_yw = &shared_mem[4 * ndata + ndata];// ndata floats
    float* thread_results = &shared_mem[5 * ndata + ndata]; // blockDim.x * 3 floats

    unsigned int freq_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;

    // Loop over frequencies (in case we have more frequencies than blocks)
    while (freq_idx < nfreqs) {
        float freq = freqs[freq_idx];

        // Step 1: Load data and compute phases
        // Each thread loads multiple elements if ndata > blockDim.x
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            float phi = mod1(t[i] * freq);
            float weight = 1.f / (dy[i] * dy[i]);

            sh_phi[i] = phi;
            sh_y[i] = y[i];
            sh_w[i] = weight;
            sh_indices[i] = i;
        }
        __syncthreads();

        // Step 2: Normalize weights
        float sum_w = 0.f;
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            sum_w += sh_w[i];
        }

        // Reduce sum_w across threads
        __shared__ float block_sum_w;
        if (tid == 0) block_sum_w = 0.f;
        __syncthreads();

        atomicAdd(&block_sum_w, sum_w);
        __syncthreads();

        // Normalize weights
        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            sh_w[i] /= block_sum_w;
        }
        __syncthreads();

        // Step 3: Compute ybar and YY (normalization)
        float ybar = 0.f;
        float YY = 0.f;

        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            ybar += sh_w[i] * sh_y[i];
        }

        __shared__ float block_ybar;
        if (tid == 0) block_ybar = 0.f;
        __syncthreads();

        atomicAdd(&block_ybar, ybar);
        __syncthreads();

        ybar = block_ybar;

        for (unsigned int i = tid; i < ndata; i += blockDim.x) {
            float diff = sh_y[i] - ybar;
            YY += sh_w[i] * diff * diff;
        }

        __shared__ float block_YY;
        if (tid == 0) block_YY = 0.f;
        __syncthreads();

        atomicAdd(&block_YY, YY);
        __syncthreads();

        YY = block_YY;

        // Step 4: Sort by phase using bitonic sort
        // Pad to next power of 2 for bitonic sort
        unsigned int n_padded = 1;
        while (n_padded < ndata) n_padded *= 2;

        // Pad with large phase values
        for (unsigned int i = ndata + tid; i < n_padded; i += blockDim.x) {
            if (i < n_padded) {
                sh_phi[i] = 2.f; // Larger than any valid phase
                sh_y[i] = 0.f;
                sh_w[i] = 0.f;
                sh_indices[i] = -1;
            }
        }
        __syncthreads();

        bitonic_sort_by_phase(sh_phi, sh_y, sh_w, sh_indices, n_padded);

        // Step 5: Compute cumulative sums for fast range queries
        // Using prefix sum
        for (unsigned int stride = 1; stride < ndata; stride *= 2) {
            __syncthreads();
            for (unsigned int i = tid; i < ndata; i += blockDim.x) {
                if (i >= stride) {
                    float temp_w = sh_cumsum_w[i - stride];
                    float temp_yw = sh_cumsum_yw[i - stride];
                    __syncthreads();
                    sh_cumsum_w[i] = sh_w[i] + temp_w;
                    sh_cumsum_yw[i] = sh_w[i] * sh_y[i] + temp_yw;
                } else {
                    sh_cumsum_w[i] = sh_w[i];
                    sh_cumsum_yw[i] = sh_w[i] * sh_y[i];
                }
            }
        }
        __syncthreads();

        // Step 6: Each thread tests a subset of transit pairs
        float thread_max_bls = 0.f;
        float thread_q = 0.f;
        float thread_phi0 = 0.f;

        // Total number of pairs to test: ndata * ndata
        unsigned long long total_pairs = (unsigned long long)ndata * (unsigned long long)ndata;
        unsigned long long pairs_per_thread = (total_pairs + blockDim.x - 1) / blockDim.x;

        unsigned long long start_pair = (unsigned long long)tid * pairs_per_thread;
        unsigned long long end_pair = min(start_pair + pairs_per_thread, total_pairs);

        for (unsigned long long pair_idx = start_pair; pair_idx < end_pair; pair_idx++) {
            unsigned int i = pair_idx / ndata;
            unsigned int j = pair_idx % ndata;

            if (i >= ndata || j >= ndata) continue;

            float phi0, q, W, YW, bls;

            // Non-wrapped transits: from i to j
            if (j > i) {
                phi0 = sh_phi[i];

                // Compute q as midpoint to next excluded observation
                if (j < ndata - 1 && j > 0) {
                    q = 0.5f * (sh_phi[j] + sh_phi[j - 1]) - phi0;
                } else {
                    q = sh_phi[j] - phi0;
                }

                if (q > 0.5f) continue;

                // Compute W and YW for observations i to j-1 using cumulative sums
                W = (i == 0) ? sh_cumsum_w[j - 1] : sh_cumsum_w[j - 1] - sh_cumsum_w[i - 1];
                YW = (i == 0) ? sh_cumsum_yw[j - 1] : sh_cumsum_yw[j - 1] - sh_cumsum_yw[i - 1];
                YW -= ybar * W;

                bls = bls_power(YW, W, YY, ignore_negative_delta_sols);

                if (bls > thread_max_bls) {
                    thread_max_bls = bls;
                    thread_q = q;
                    thread_phi0 = phi0;
                }
            }

            // Wrapped transits: from i to end, then 0 to k
            if (j < i) {
                unsigned int k = j;
                phi0 = sh_phi[i];

                if (k > 0) {
                    q = (1.f - phi0) + 0.5f * (sh_phi[k - 1] + sh_phi[k]);
                } else {
                    q = 1.f - phi0;
                }

                if (q > 0.5f) continue;

                // W and YW = sum from i to end, plus 0 to k-1
                W = (sh_cumsum_w[ndata - 1] - sh_cumsum_w[i - 1]);
                YW = (sh_cumsum_yw[ndata - 1] - sh_cumsum_yw[i - 1]);

                if (k > 0) {
                    W += sh_cumsum_w[k - 1];
                    YW += sh_cumsum_yw[k - 1];
                }

                YW -= ybar * W;

                bls = bls_power(YW, W, YY, ignore_negative_delta_sols);

                if (bls > thread_max_bls) {
                    thread_max_bls = bls;
                    thread_q = q;
                    thread_phi0 = phi0;
                }
            }
        }

        // Store thread results
        thread_results[tid] = thread_max_bls;
        thread_results[blockDim.x + tid] = thread_q;
        thread_results[2 * blockDim.x + tid] = thread_phi0;
        __syncthreads();

        // Step 7: Reduce across threads to find maximum BLS
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                float bls1 = thread_results[tid];
                float bls2 = thread_results[tid + stride];

                if (bls2 > bls1) {
                    thread_results[tid] = bls2;
                    thread_results[blockDim.x + tid] = thread_results[blockDim.x + tid + stride];
                    thread_results[2 * blockDim.x + tid] = thread_results[2 * blockDim.x + tid + stride];
                }
            }
            __syncthreads();
        }

        // Step 8: Write results to global memory
        if (tid == 0) {
            bls_powers[freq_idx] = thread_results[0];
            best_q[freq_idx] = thread_results[blockDim.x];
            best_phi[freq_idx] = thread_results[2 * blockDim.x];
        }

        // Move to next frequency
        freq_idx += gridDim.x;
    }
}
