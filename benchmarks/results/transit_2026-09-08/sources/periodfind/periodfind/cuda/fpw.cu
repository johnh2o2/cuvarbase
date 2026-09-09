// Fast Phase-folding Weighted (FPW) algorithm — CUDA implementation.
//
// Implements the FPW statistic from Finkbeiner et al. 2025.

#include "fpw.h"

#include <algorithm>
#include <thread>

#include "cuda_runtime.h"
#include "math.h"

#include "errchk.cuh"

//
// Simple FPW Function Definitions
//

FPW::FPW(size_t n_bins) {
    num_bins = n_bins;
    bin_size = 1.0 / static_cast<float>(n_bins);
}

__host__ __device__ size_t FPW::NumBins() const {
    return num_bins;
}

__host__ __device__ size_t FPW::PhaseBin(float phase_val) const {
    return static_cast<size_t>(phase_val / bin_size);
}

//
// CUDA Kernels
//

// Tile size for shared-memory light curve tiling
#define FPW_TILE_SIZE 256

// Compile-time max for per-thread register bin arrays
#define FPW_MAX_BINS 64

// Hybrid threshold: use atomics for small point counts, privatization for large
#define FPW_HYBRID_THRESHOLD 2048

__global__ void FPWKernel(const float* __restrict__ times,
                          const float* __restrict__ mags,
                          const float* __restrict__ ivar,
                          const float* __restrict__ ivar_y,
                          const size_t length,
                          const float* __restrict__ periods,
                          const float* __restrict__ period_dts,
                          const size_t num_periods,
                          const size_t num_period_dts,
                          const FPW params,
                          float* __restrict__ fpw_out) {
    // Shared memory layout:
    // Atomic path: [sh_times | sh_ivar | sh_ivar_y | sh_v_bin | sh_y_bin]
    //              = 3*TILE_SIZE + 2*n_bins (used simultaneously)
    // Privatization path: phase 1 tiles (3*TILE_SIZE), then reduction
    //                     (2*NUM_WARPS*n_bins), used sequentially
    extern __shared__ float sh_data[];
    float* sh_times = &sh_data[0];
    float* sh_ivar = &sh_data[FPW_TILE_SIZE];
    float* sh_ivar_y = &sh_data[2 * FPW_TILE_SIZE];

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

    if (length <= FPW_HYBRID_THRESHOLD) {
        // === Atomic path: shared memory bins, no register pressure ===
        float* sh_v_bin = &sh_data[3 * FPW_TILE_SIZE];
        float* sh_y_bin = &sh_data[3 * FPW_TILE_SIZE + n_bins];

        // Cooperatively zero bins
        for (size_t k = threadIdx.x; k < n_bins; k += blockDim.x) {
            sh_v_bin[k] = 0.0f;
            sh_y_bin[k] = 0.0f;
        }
        __syncthreads();

        // Process the light curve in tiles
        for (size_t tile_start = 0; tile_start < length;
             tile_start += FPW_TILE_SIZE) {
            size_t tile_end = tile_start + FPW_TILE_SIZE;
            if (tile_end > length)
                tile_end = length;
            size_t tile_len = tile_end - tile_start;

            // Cooperatively load tile into shared memory
            for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
                sh_times[i] = times[tile_start + i];
                sh_ivar[i] = ivar[tile_start + i];
                sh_ivar_y[i] = ivar_y[tile_start + i];
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

                atomicAdd(&sh_v_bin[bin], sh_ivar[i]);
                atomicAdd(&sh_y_bin[bin], sh_ivar_y[i]);
            }
            __syncthreads();
        }

        // Thread 0 computes delta_chi from shared bins
        if (threadIdx.x == 0) {
            float delta_chi = 0.0f;
            for (size_t k = 0; k < n_bins; k++) {
                if (sh_v_bin[k] > 0.0f) {
                    delta_chi +=
                        sh_y_bin[k] * sh_y_bin[k] / (2.0f * sh_v_bin[k]);
                }
            }
            fpw_out[period_idx * num_period_dts + pdt_idx] = delta_chi;
        }
    } else {
        // === Privatization path: per-thread register arrays ===

        float my_v[FPW_MAX_BINS];
        float my_y[FPW_MAX_BINS];
        for (size_t k = 0; k < n_bins; k++) {
            my_v[k] = 0.0f;
            my_y[k] = 0.0f;
        }

        // Process the light curve in tiles
        for (size_t tile_start = 0; tile_start < length;
             tile_start += FPW_TILE_SIZE) {
            size_t tile_end = tile_start + FPW_TILE_SIZE;
            if (tile_end > length)
                tile_end = length;
            size_t tile_len = tile_end - tile_start;

            // Cooperatively load tile into shared memory
            for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
                sh_times[i] = times[tile_start + i];
                sh_ivar[i] = ivar[tile_start + i];
                sh_ivar_y[i] = ivar_y[tile_start + i];
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

                my_v[bin] += sh_ivar[i];
                my_y[bin] += sh_ivar_y[i];
            }
            __syncthreads();
        }

        // --- Reduce private bins across all threads ---

        const unsigned int FULL_MASK = 0xFFFFFFFF;
        const int NUM_WARPS = blockDim.x / 32;
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;

        // Step 1: Warp-shuffle reduce each bin within each warp
        for (size_t k = 0; k < n_bins; k++) {
            for (int offset = 16; offset > 0; offset >>= 1) {
                my_v[k] += __shfl_down_sync(FULL_MASK, my_v[k], offset);
                my_y[k] += __shfl_down_sync(FULL_MASK, my_y[k], offset);
            }
        }

        // Step 2: Warp leaders write partial sums to shared memory
        // Reuse sh_data for reduction workspace:
        // [0 .. NUM_WARPS*n_bins-1]               = v partial sums
        // [NUM_WARPS*n_bins .. 2*NUM_WARPS*n_bins-1] = y partial sums
        float* sh_reduce_v = &sh_data[0];
        float* sh_reduce_y = &sh_data[NUM_WARPS * n_bins];

        if (lane_id == 0) {
            for (size_t k = 0; k < n_bins; k++) {
                sh_reduce_v[warp_id * n_bins + k] = my_v[k];
                sh_reduce_y[warp_id * n_bins + k] = my_y[k];
            }
        }
        __syncthreads();

        // Step 3: Thread 0 sums across warps and computes delta_chi
        if (threadIdx.x == 0) {
            float delta_chi = 0.0f;
            for (size_t k = 0; k < n_bins; k++) {
                float total_v = 0.0f;
                float total_y = 0.0f;
                for (int w = 0; w < NUM_WARPS; w++) {
                    total_v += sh_reduce_v[w * n_bins + k];
                    total_y += sh_reduce_y[w * n_bins + k];
                }
                if (total_v > 0.0f) {
                    delta_chi += total_y * total_y / (2.0f * total_v);
                }
            }
            fpw_out[period_idx * num_period_dts + pdt_idx] = delta_chi;
        }
    }
}

//
// Wrapper Functions
//

// Per-device state for multi-GPU batched processing
struct FPWDeviceState {
    float* dev_periods;
    float* dev_period_dts;
    cudaStream_t streams[4];
    float* dev_times_buf[4];
    float* dev_ivar_buf[4];
    float* dev_ivar_y_buf[4];
    float* dev_fpw_buf[4];
    float* h_ivar[4];
    float* h_ivar_y[4];
};

void FPW::CalcFPWBatched(const std::vector<float*>& times,
                         const std::vector<float*>& mags,
                         const std::vector<float*>& errs,
                         const std::vector<size_t>& lengths,
                         const float* periods,
                         const float* period_dts,
                         const size_t num_periods,
                         const size_t num_p_dts,
                         float* fpw_out) const {
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
        (3 * FPW_TILE_SIZE + 2 * NumBins()) * sizeof(float);
    const dim3 grid_dim = dim3(num_periods, num_p_dts);

    const int NUM_STREAMS = 4;

    // Partition curves across devices
    size_t base_count = num_curves / num_devices;
    size_t remainder = num_curves % num_devices;

    std::vector<FPWDeviceState> dev_state(num_devices);

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
                cudaMalloc(&dev_state[d].dev_ivar_y_buf[s], buffer_bytes));
            gpuErrchk(
                cudaMalloc(&dev_state[d].dev_fpw_buf[s], per_out_size));
            dev_state[d].h_ivar[s] = (float*)malloc(buffer_bytes);
            dev_state[d].h_ivar_y[s] = (float*)malloc(buffer_bytes);
        }

        // Enqueue work for this device's curves
        for (size_t j = 0; j < count; j++) {
            size_t i = start + j;
            int s = j % NUM_STREAMS;
            cudaStream_t stream = dev_state[d].streams[s];

            // Precompute inverse variance and weighted data on CPU
            float* h_iv = dev_state[d].h_ivar[s];
            float* h_ivy = dev_state[d].h_ivar_y[s];
            for (size_t k = 0; k < lengths[i]; k++) {
                float e = errs[i][k];
                h_iv[k] = 1.0f / (e * e);
                h_ivy[k] = h_iv[k] * mags[i][k];
            }

            // Copy light curve data into device buffers (async)
            const size_t curve_bytes = lengths[i] * sizeof(float);
            gpuErrchk(cudaMemcpyAsync(dev_state[d].dev_times_buf[s],
                                      times[i], curve_bytes,
                                      cudaMemcpyHostToDevice, stream));
            gpuErrchk(cudaMemcpyAsync(dev_state[d].dev_ivar_buf[s],
                                      h_iv, curve_bytes,
                                      cudaMemcpyHostToDevice, stream));
            gpuErrchk(cudaMemcpyAsync(dev_state[d].dev_ivar_y_buf[s],
                                      h_ivy, curve_bytes,
                                      cudaMemcpyHostToDevice, stream));

            gpuErrchk(cudaMemsetAsync(dev_state[d].dev_fpw_buf[s], 0,
                                      per_out_size, stream));

            FPWKernel<<<grid_dim, num_threads, shared_bytes, stream>>>(
                dev_state[d].dev_times_buf[s], NULL,
                dev_state[d].dev_ivar_buf[s],
                dev_state[d].dev_ivar_y_buf[s], lengths[i],
                dev_state[d].dev_periods, dev_state[d].dev_period_dts,
                num_periods, num_p_dts, *this,
                dev_state[d].dev_fpw_buf[s]);

            gpuErrchk(cudaMemcpyAsync(&fpw_out[i * per_points],
                                      dev_state[d].dev_fpw_buf[s],
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
            gpuErrchk(cudaFree(dev_state[d].dev_ivar_y_buf[s]));
            gpuErrchk(cudaFree(dev_state[d].dev_fpw_buf[s]));
            gpuErrchk(cudaStreamDestroy(dev_state[d].streams[s]));
            free(dev_state[d].h_ivar[s]);
            free(dev_state[d].h_ivar_y[s]);
        }
    }
}

float* FPW::CalcFPWBatched(const std::vector<float*>& times,
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

    float* fpw_out = (float*)malloc(per_size_total);

    CalcFPWBatched(times, mags, errs, lengths, periods, period_dts, num_periods,
                   num_p_dts, fpw_out);

    return fpw_out;
}
