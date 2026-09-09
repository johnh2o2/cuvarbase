// Box Least Squares (BLS) transit-detection algorithm — CUDA implementation.
//
// Searches for periodic box-shaped (flat-bottom) dips in time-series data.
// Kovács, Zucker & Mazeh (2002).

#include "bls.h"

#include <algorithm>
#include <thread>

#include "cuda_runtime.h"
#include "math.h"

#include "errchk.cuh"

//
// Simple BLS Function Definitions
//

BLS::BLS(size_t n_bins, float qmin_val, float qmax_val) {
    num_bins = n_bins;
    bin_size = 1.0f / static_cast<float>(n_bins);
    qmin = qmin_val;
    qmax = qmax_val;
}

__host__ __device__ size_t BLS::NumBins() const {
    return num_bins;
}

__host__ __device__ size_t BLS::PhaseBin(float phase_val) const {
    return static_cast<size_t>(phase_val / bin_size);
}

__host__ __device__ float BLS::Qmin() const {
    return qmin;
}

__host__ __device__ float BLS::Qmax() const {
    return qmax;
}

//
// CUDA Kernels
//

// Tile size for shared-memory light curve tiling
#define BLS_TILE_SIZE 256

// Compile-time max for per-thread register bin arrays
#define BLS_MAX_BINS 64

// Hybrid threshold: use atomics for small point counts, privatization for large
#define BLS_HYBRID_THRESHOLD 8192

__global__ void BLSKernel(const float* __restrict__ times,
                          const float* __restrict__ ivar,
                          const float* __restrict__ ivar_yw,
                          const size_t length,
                          const float* __restrict__ periods,
                          const float* __restrict__ period_dts,
                          const size_t num_periods,
                          const size_t num_period_dts,
                          const BLS params,
                          float* __restrict__ bls_out) {
    // Shared memory layout:
    // Atomic path: [sh_times | sh_ivar | sh_ivar_yw | sh_w_bin | sh_yw_bin]
    //              = 3*TILE_SIZE + 2*n_bins (used simultaneously)
    // Privatization path: phase 1 tiles (3*TILE_SIZE), then reduction
    //                     (2*NUM_WARPS*n_bins), used sequentially
    // Both paths converge into prefix sums at sh_data[0..2*(n_bins+1)-1]
    extern __shared__ float sh_data[];
    float* sh_times = &sh_data[0];
    float* sh_ivar = &sh_data[BLS_TILE_SIZE];
    float* sh_ivar_yw = &sh_data[2 * BLS_TILE_SIZE];

    // One block per (period, period_dt) pair
    const size_t period_idx = blockIdx.x;
    const size_t pdt_idx = blockIdx.y;

    if (period_idx >= num_periods || pdt_idx >= num_period_dts) {
        return;
    }

    const float period = periods[period_idx];
    const float period_dt = period_dts[pdt_idx];
    const float pdt_corr = (period_dt / period) / 2;

    const size_t n_bins = params.NumBins();

    // Prefix sum arrays — both paths write here before convergence
    float* sh_w_prefix = &sh_data[0];
    float* sh_yw_prefix = &sh_data[n_bins + 1];

    const unsigned int FULL_MASK = 0xFFFFFFFF;
    const int NUM_WARPS = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (length <= BLS_HYBRID_THRESHOLD) {
        // === Atomic path: shared memory bins, no register pressure ===
        float* sh_w_bin = &sh_data[3 * BLS_TILE_SIZE];
        float* sh_yw_bin = &sh_data[3 * BLS_TILE_SIZE + n_bins];

        // Cooperatively zero bins
        for (size_t k = threadIdx.x; k < n_bins; k += blockDim.x) {
            sh_w_bin[k] = 0.0f;
            sh_yw_bin[k] = 0.0f;
        }
        __syncthreads();

        // Process the light curve in tiles
        for (size_t tile_start = 0; tile_start < length;
             tile_start += BLS_TILE_SIZE) {
            size_t tile_end = tile_start + BLS_TILE_SIZE;
            if (tile_end > length)
                tile_end = length;
            size_t tile_len = tile_end - tile_start;

            // Cooperatively load tile into shared memory
            for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
                sh_times[i] = times[tile_start + i];
                sh_ivar[i] = ivar[tile_start + i];
                sh_ivar_yw[i] = ivar_yw[tile_start + i];
            }
            __syncthreads();

            // Accumulate into shared bins via atomicAdd
            for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
                float t = sh_times[i];
                double t_corr_d = (double)t - (double)pdt_corr * (double)t * (double)t;
                double ratio_d = t_corr_d / (double)period;
                float folded = fabsf((float)(ratio_d - floor(ratio_d)));

                size_t bin = params.PhaseBin(folded);
                if (bin >= n_bins)
                    bin = n_bins - 1;

                atomicAdd(&sh_w_bin[bin], sh_ivar[i]);
                atomicAdd(&sh_yw_bin[bin], sh_ivar_yw[i]);
            }
            __syncthreads();
        }

        // Thread 0 computes prefix sums from shared bins
        if (threadIdx.x == 0) {
            sh_w_prefix[0] = 0.0f;
            sh_yw_prefix[0] = 0.0f;
            for (size_t k = 0; k < n_bins; k++) {
                sh_w_prefix[k + 1] = sh_w_prefix[k] + sh_w_bin[k];
                sh_yw_prefix[k + 1] = sh_yw_prefix[k] + sh_yw_bin[k];
            }
        }
        __syncthreads();
    } else {
        // === Privatization path: per-thread register arrays ===

        float my_w[BLS_MAX_BINS];
        float my_yw[BLS_MAX_BINS];
        for (size_t k = 0; k < n_bins; k++) {
            my_w[k] = 0.0f;
            my_yw[k] = 0.0f;
        }

        // Process the light curve in tiles
        for (size_t tile_start = 0; tile_start < length;
             tile_start += BLS_TILE_SIZE) {
            size_t tile_end = tile_start + BLS_TILE_SIZE;
            if (tile_end > length)
                tile_end = length;
            size_t tile_len = tile_end - tile_start;

            // Cooperatively load tile into shared memory
            for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
                sh_times[i] = times[tile_start + i];
                sh_ivar[i] = ivar[tile_start + i];
                sh_ivar_yw[i] = ivar_yw[tile_start + i];
            }
            __syncthreads();

            // All threads accumulate into private arrays (no atomics)
            for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
                float t = sh_times[i];
                double t_corr_d = (double)t - (double)pdt_corr * (double)t * (double)t;
                double ratio_d = t_corr_d / (double)period;
                float folded = fabsf((float)(ratio_d - floor(ratio_d)));

                size_t bin = params.PhaseBin(folded);
                if (bin >= n_bins)
                    bin = n_bins - 1;

                my_w[bin] += sh_ivar[i];
                my_yw[bin] += sh_ivar_yw[i];
            }
            __syncthreads();
        }

        // --- Reduce private bins across all threads ---

        // Step 1: Warp-shuffle reduce each bin within each warp
        for (size_t k = 0; k < n_bins; k++) {
            for (int offset = 16; offset > 0; offset >>= 1) {
                my_w[k] += __shfl_down_sync(FULL_MASK, my_w[k], offset);
                my_yw[k] += __shfl_down_sync(FULL_MASK, my_yw[k], offset);
            }
        }

        // Step 2: Warp leaders write partial sums to shared memory
        float* sh_reduce_w = &sh_data[0];
        float* sh_reduce_yw = &sh_data[NUM_WARPS * n_bins];

        if (lane_id == 0) {
            for (size_t k = 0; k < n_bins; k++) {
                sh_reduce_w[warp_id * n_bins + k] = my_w[k];
                sh_reduce_yw[warp_id * n_bins + k] = my_yw[k];
            }
        }
        __syncthreads();

        // Step 3: Thread 0 aggregates across warps and computes prefix sums
        if (threadIdx.x == 0) {
            float w_bin[BLS_MAX_BINS];
            float yw_bin[BLS_MAX_BINS];
            for (size_t k = 0; k < n_bins; k++) {
                w_bin[k] = 0.0f;
                yw_bin[k] = 0.0f;
                for (int w = 0; w < NUM_WARPS; w++) {
                    w_bin[k] += sh_reduce_w[w * n_bins + k];
                    yw_bin[k] += sh_reduce_yw[w * n_bins + k];
                }
            }

            sh_w_prefix[0] = 0.0f;
            sh_yw_prefix[0] = 0.0f;
            for (size_t k = 0; k < n_bins; k++) {
                sh_w_prefix[k + 1] = sh_w_prefix[k] + w_bin[k];
                sh_yw_prefix[k + 1] = sh_yw_prefix[k] + yw_bin[k];
            }
        }
        __syncthreads();
    }

    // === Shared code: Phase B search + Phase C max reduction ===

    float w_total = sh_w_prefix[n_bins];

    if (w_total <= 0.0f) {
        if (threadIdx.x == 0) {
            bls_out[period_idx * num_period_dts + pdt_idx] = 0.0f;
        }
        return;
    }

    // Transit duration range in bins
    size_t nb_min = max((size_t)1, (size_t)(params.Qmin() * n_bins));
    size_t nb_max = min(n_bins - 1, (size_t)ceilf(params.Qmax() * n_bins));

    // Phase B: All threads search the linearized (nb, phi) space
    size_t nb_range = nb_max - nb_min + 1;
    size_t total_pairs = nb_range * n_bins;

    float my_best_bls = 0.0f;

    for (size_t idx = threadIdx.x; idx < total_pairs; idx += blockDim.x) {
        size_t nb = nb_min + idx / n_bins;
        size_t phi = idx % n_bins;

        size_t end = phi + nb;
        float r, s;
        if (end <= n_bins) {
            r = sh_w_prefix[end] - sh_w_prefix[phi];
            s = sh_yw_prefix[end] - sh_yw_prefix[phi];
        } else {
            size_t wrap = end - n_bins;
            r = (sh_w_prefix[n_bins] - sh_w_prefix[phi]) + sh_w_prefix[wrap];
            s = (sh_yw_prefix[n_bins] - sh_yw_prefix[phi]) + sh_yw_prefix[wrap];
        }

        float r_frac = r / w_total;
        if (r_frac > 0.0f && r_frac < 1.0f) {
            float bls = (s * s) / (r * (w_total - r));
            if (bls > my_best_bls) {
                my_best_bls = bls;
            }
        }
    }

    // Phase C: Parallel max-reduction via warp shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(FULL_MASK, my_best_bls, offset);
        if (other > my_best_bls) my_best_bls = other;
    }

    // Warp leaders write to shared memory (reuse area after prefix sums)
    float* sh_reduce = &sh_data[2 * (n_bins + 1)];
    if (lane_id == 0) {
        sh_reduce[warp_id] = my_best_bls;
    }
    __syncthreads();

    // Thread 0 finds global max across warps
    if (threadIdx.x == 0) {
        float best_bls = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++) {
            if (sh_reduce[w] > best_bls) best_bls = sh_reduce[w];
        }
        bls_out[period_idx * num_period_dts + pdt_idx] = best_bls;
    }
}

//
// Wrapper Functions
//

// Per-device state for multi-GPU batched processing
struct BLSDeviceState {
    float* dev_periods;
    float* dev_period_dts;
    cudaStream_t streams[4];
    float* dev_times_buf[4];
    float* dev_ivar_buf[4];
    float* dev_ivar_yw_buf[4];
    float* dev_bls_buf[4];
    float* h_ivar[4];
    float* h_ivar_yw[4];
};

void BLS::CalcBLSBatched(const std::vector<float*>& times,
                         const std::vector<float*>& mags,
                         const std::vector<float*>& errs,
                         const std::vector<size_t>& lengths,
                         const float* periods,
                         const float* period_dts,
                         const size_t num_periods,
                         const size_t num_p_dts,
                         float* bls_out) const {
    size_t num_curves = lengths.size();
    if (num_curves == 0) return;

    size_t per_points = num_periods * num_p_dts;
    size_t per_out_size = per_points * sizeof(float);

    // Determine number of GPUs
    int num_devices = 1;
    if (cudaGetDeviceCount(&num_devices) != cudaSuccess) {
        num_devices = 1;
    }
    if (num_devices > (int)num_curves) {
        num_devices = (int)num_curves;
    }

    // Kernel launch information: one block per (period, period_dt) pair
    const size_t num_threads = 256;
    // Shared memory: atomic path needs 3*TILE_SIZE + 2*n_bins (simultaneous),
    // privatization needs max(3*TILE_SIZE, 2*NUM_WARPS*n_bins) (sequential).
    // The atomic layout always dominates, so use it for both paths.
    const size_t shared_bytes =
        (3 * BLS_TILE_SIZE + 2 * NumBins()) * sizeof(float);
    const dim3 grid_dim = dim3(num_periods, num_p_dts);

    const int NUM_STREAMS = 4;

    // Partition curves across devices
    size_t base_count = num_curves / num_devices;
    size_t remainder = num_curves % num_devices;

    std::vector<BLSDeviceState> dev_state(num_devices);

    // Phase 1: Allocate and enqueue on each device (one thread per GPU)
    std::vector<std::thread> dev_threads;
    for (int d = 0; d < num_devices; d++) {
        dev_threads.emplace_back([&, d]() {
        gpuErrchk(cudaSetDevice(d));

        size_t start = d * base_count + std::min((size_t)d, remainder);
        size_t count = base_count + ((size_t)d < remainder ? 1 : 0);

        // Per-device max_length for buffer sizing
        size_t dev_max_length = 0;
        for (size_t j = start; j < start + count; j++) {
            if (lengths[j] > dev_max_length) dev_max_length = lengths[j];
        }
        size_t buffer_bytes = sizeof(float) * dev_max_length;

        // Copy periods to this device
        gpuErrchk(cudaMalloc(&dev_state[d].dev_periods,
                             num_periods * sizeof(float)));
        gpuErrchk(cudaMalloc(&dev_state[d].dev_period_dts,
                             num_p_dts * sizeof(float)));
        gpuErrchk(cudaMemcpy(dev_state[d].dev_periods, periods,
                             num_periods * sizeof(float),
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_state[d].dev_period_dts, period_dts,
                             num_p_dts * sizeof(float),
                             cudaMemcpyHostToDevice));

        // Create streams and allocate per-stream buffers (device + host)
        for (int s = 0; s < NUM_STREAMS; s++) {
            gpuErrchk(cudaStreamCreate(&dev_state[d].streams[s]));
            gpuErrchk(
                cudaMalloc(&dev_state[d].dev_times_buf[s], buffer_bytes));
            gpuErrchk(
                cudaMalloc(&dev_state[d].dev_ivar_buf[s], buffer_bytes));
            gpuErrchk(
                cudaMalloc(&dev_state[d].dev_ivar_yw_buf[s], buffer_bytes));
            gpuErrchk(
                cudaMalloc(&dev_state[d].dev_bls_buf[s], per_out_size));
            dev_state[d].h_ivar[s] = (float*)malloc(buffer_bytes);
            dev_state[d].h_ivar_yw[s] = (float*)malloc(buffer_bytes);
        }

        // Enqueue work for this device's curves
        for (size_t j = 0; j < count; j++) {
            size_t i = start + j;
            int s = j % NUM_STREAMS;
            cudaStream_t stream = dev_state[d].streams[s];

            // Compute weighted mean for centering
            float total_w = 0.0f;
            float total_wy = 0.0f;
            for (size_t k = 0; k < lengths[i]; k++) {
                float e = errs[i][k];
                float w = 1.0f / (e * e);
                total_w += w;
                total_wy += w * mags[i][k];
            }
            float mean_y = (total_w > 0.0f) ? (total_wy / total_w) : 0.0f;

            // Precompute inverse variance and weighted centered data on CPU
            float* h_iv = dev_state[d].h_ivar[s];
            float* h_iyw = dev_state[d].h_ivar_yw[s];
            for (size_t k = 0; k < lengths[i]; k++) {
                float e = errs[i][k];
                h_iv[k] = 1.0f / (e * e);
                h_iyw[k] = h_iv[k] * (mags[i][k] - mean_y);
            }

            // Copy light curve data into device buffers (async)
            const size_t curve_bytes = lengths[i] * sizeof(float);
            gpuErrchk(cudaMemcpyAsync(dev_state[d].dev_times_buf[s],
                                      times[i], curve_bytes,
                                      cudaMemcpyHostToDevice, stream));
            gpuErrchk(cudaMemcpyAsync(dev_state[d].dev_ivar_buf[s],
                                      h_iv, curve_bytes,
                                      cudaMemcpyHostToDevice, stream));
            gpuErrchk(cudaMemcpyAsync(dev_state[d].dev_ivar_yw_buf[s],
                                      h_iyw, curve_bytes,
                                      cudaMemcpyHostToDevice, stream));

            gpuErrchk(cudaMemsetAsync(dev_state[d].dev_bls_buf[s], 0,
                                      per_out_size, stream));

            BLSKernel<<<grid_dim, num_threads, shared_bytes, stream>>>(
                dev_state[d].dev_times_buf[s],
                dev_state[d].dev_ivar_buf[s],
                dev_state[d].dev_ivar_yw_buf[s], lengths[i],
                dev_state[d].dev_periods, dev_state[d].dev_period_dts,
                num_periods, num_p_dts, *this,
                dev_state[d].dev_bls_buf[s]);

            gpuErrchk(cudaMemcpyAsync(&bls_out[i * per_points],
                                      dev_state[d].dev_bls_buf[s],
                                      per_out_size, cudaMemcpyDeviceToHost,
                                      stream));
        }
        });
    }
    for (auto& t : dev_threads) t.join();

    // Phase 2: Sync and free on each device
    for (int d = 0; d < num_devices; d++) {
        gpuErrchk(cudaSetDevice(d));

        for (int s = 0; s < NUM_STREAMS; s++) {
            gpuErrchk(cudaStreamSynchronize(dev_state[d].streams[s]));
        }

        gpuErrchk(cudaFree(dev_state[d].dev_periods));
        gpuErrchk(cudaFree(dev_state[d].dev_period_dts));
        for (int s = 0; s < NUM_STREAMS; s++) {
            gpuErrchk(cudaFree(dev_state[d].dev_times_buf[s]));
            gpuErrchk(cudaFree(dev_state[d].dev_ivar_buf[s]));
            gpuErrchk(cudaFree(dev_state[d].dev_ivar_yw_buf[s]));
            gpuErrchk(cudaFree(dev_state[d].dev_bls_buf[s]));
            gpuErrchk(cudaStreamDestroy(dev_state[d].streams[s]));
            free(dev_state[d].h_ivar[s]);
            free(dev_state[d].h_ivar_yw[s]);
        }
    }
}

float* BLS::CalcBLSBatched(const std::vector<float*>& times,
                           const std::vector<float*>& mags,
                           const std::vector<float*>& errs,
                           const std::vector<size_t>& lengths,
                           const float* periods,
                           const float* period_dts,
                           const size_t num_periods,
                           const size_t num_p_dts) const {
    size_t per_points = num_periods * num_p_dts;
    size_t per_out_size = per_points * sizeof(float);
    size_t per_size_total = per_out_size * lengths.size();

    float* bls_out = (float*)malloc(per_size_total);

    CalcBLSBatched(times, mags, errs, lengths, periods, period_dts, num_periods,
                   num_p_dts, bls_out);

    return bls_out;
}
