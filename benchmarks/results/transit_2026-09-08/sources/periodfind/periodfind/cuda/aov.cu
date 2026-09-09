// Copyright 2020 California Institute of Technology. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
// Author: Ethan Jaszewski

#include "aov.h"

#include <algorithm>
#include <thread>

#include "cuda_runtime.h"
#include "math.h"

#include "errchk.cuh"

//
// Simple AOV Function Definitions
//

AOV::AOV(size_t n_bins, size_t bin_overlap) {
    num_bins = n_bins;
    num_overlap = bin_overlap;

    bin_size = 1.0 / static_cast<float>(n_bins);
}

__host__ __device__ size_t AOV::NumPhaseBins() const {
    return num_bins;
}

__host__ __device__ size_t AOV::NumPhaseBinOverlap() const {
    return num_overlap;
}

__host__ __device__ size_t AOV::PhaseBin(float phase_val) const {
    return static_cast<size_t>(phase_val / bin_size);
}

//
// CUDA Kernels
//

// Compile-time max for per-thread register bin arrays
#define AOV_MAX_BINS 64

// Tile size for shared-memory light curve tiling (atomic path)
#define AOV_TILE_SIZE 256

// Hybrid threshold: use atomics for small point counts, privatization for large
#define AOV_HYBRID_THRESHOLD 4096

extern __shared__ float aov_sh_data[];

__global__ void FoldBinKernel(const float* __restrict__ times,
                              const float* __restrict__ mags,
                              const size_t length,
                              const float* __restrict__ periods,
                              const float* __restrict__ period_dts,
                              const AOV aov,
                              AOVData* __restrict__ data) {
    const size_t n_bins = aov.NumPhaseBins();

    // Period and period time derivative for this block.
    const float period = periods[blockIdx.x];
    const float period_dt = period_dts[blockIdx.y];

    // Time derivative correction factor.
    const float pdt_corr = (period_dt / period) / 2;

    // Output location for this block
    size_t block_id = blockIdx.x * gridDim.y + blockIdx.y;

    if (length <= AOV_HYBRID_THRESHOLD) {
        // === Atomic path: shared memory bins, no register pressure ===
        float* sh_times = &aov_sh_data[0];
        float* sh_mags = &aov_sh_data[AOV_TILE_SIZE];
        uint32_t* sh_count = (uint32_t*)&aov_sh_data[2 * AOV_TILE_SIZE];
        float* sh_sums = &aov_sh_data[2 * AOV_TILE_SIZE + n_bins];
        float* sh_sq_sums = &aov_sh_data[2 * AOV_TILE_SIZE + 2 * n_bins];

        // Cooperatively zero bins
        for (size_t k = threadIdx.x; k < n_bins; k += blockDim.x) {
            sh_count[k] = 0;
            sh_sums[k] = 0.0f;
            sh_sq_sums[k] = 0.0f;
        }
        __syncthreads();

        // Process the light curve in tiles
        for (size_t tile_start = 0; tile_start < length;
             tile_start += AOV_TILE_SIZE) {
            size_t tile_end = tile_start + AOV_TILE_SIZE;
            if (tile_end > length)
                tile_end = length;
            size_t tile_len = tile_end - tile_start;

            // Cooperatively load tile into shared memory
            for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
                sh_times[i] = times[tile_start + i];
                sh_mags[i] = mags[tile_start + i];
            }
            __syncthreads();

            // Accumulate into shared bins via atomicAdd
            for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
                float t = sh_times[i];
                double t_corr_d = (double)t - (double)pdt_corr * (double)t * (double)t;
                double ratio_d = t_corr_d / (double)period;
                float folded = fabsf((float)(ratio_d - floor(ratio_d)));

                float mag = sh_mags[i];
                float mag_sq = mag * mag;

                size_t bin = aov.PhaseBin(folded);

                for (size_t j = 0; j < aov.NumPhaseBinOverlap(); j++) {
                    size_t bin_idx = (bin + j) % n_bins;

                    atomicAdd(&sh_count[bin_idx], 1u);
                    atomicAdd(&sh_sums[bin_idx], mag);
                    atomicAdd(&sh_sq_sums[bin_idx], mag_sq);
                }
            }
            __syncthreads();
        }

        // Cooperatively write bin totals to global memory
        for (size_t k = threadIdx.x; k < n_bins; k += blockDim.x) {
            data[block_id * n_bins + k] = {sh_count[k], sh_sums[k],
                                           sh_sq_sums[k]};
        }
    } else {
        // === Privatization path: per-thread register arrays ===

        uint32_t my_count[AOV_MAX_BINS];
        float my_sums[AOV_MAX_BINS];
        float my_sq_sums[AOV_MAX_BINS];
        for (size_t k = 0; k < n_bins; k++) {
            my_count[k] = 0;
            my_sums[k] = 0.0f;
            my_sq_sums[k] = 0.0f;
        }

        // Compute the histogram statistics into private arrays.
        for (size_t idx = threadIdx.x; idx < length; idx += blockDim.x) {
            float t = times[idx];
            double t_corr_d = (double)t - (double)pdt_corr * (double)t * (double)t;
            double ratio_d = t_corr_d / (double)period;
            float folded = fabsf((float)(ratio_d - floor(ratio_d)));

            float mag = mags[idx];

            size_t bin = aov.PhaseBin(folded);

            for (size_t i = 0; i < aov.NumPhaseBinOverlap(); i++) {
                size_t bin_idx = (bin + i) % n_bins;

                my_count[bin_idx]++;
                my_sums[bin_idx] += mag;
                my_sq_sums[bin_idx] += mag * mag;
            }
        }

        // --- Reduce private bins across all threads ---

        const unsigned int FULL_MASK = 0xFFFFFFFF;
        const int NUM_WARPS = blockDim.x / 32;
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;

        // Step 1: Warp-shuffle reduce each bin within each warp
        for (size_t k = 0; k < n_bins; k++) {
            for (int offset = 16; offset > 0; offset >>= 1) {
                my_count[k] +=
                    __shfl_down_sync(FULL_MASK, my_count[k], offset);
                my_sums[k] +=
                    __shfl_down_sync(FULL_MASK, my_sums[k], offset);
                my_sq_sums[k] +=
                    __shfl_down_sync(FULL_MASK, my_sq_sums[k], offset);
            }
        }

        // Step 2: Warp leaders write partial sums to shared memory
        // Layout: 3 arrays of NUM_WARPS * n_bins values
        uint32_t* sh_reduce_count = (uint32_t*)&aov_sh_data[0];
        float* sh_reduce_sums = &aov_sh_data[NUM_WARPS * n_bins];
        float* sh_reduce_sq_sums = &aov_sh_data[2 * NUM_WARPS * n_bins];

        if (lane_id == 0) {
            for (size_t k = 0; k < n_bins; k++) {
                sh_reduce_count[warp_id * n_bins + k] = my_count[k];
                sh_reduce_sums[warp_id * n_bins + k] = my_sums[k];
                sh_reduce_sq_sums[warp_id * n_bins + k] = my_sq_sums[k];
            }
        }
        __syncthreads();

        // Step 3: Cooperatively sum across warps and write to global memory
        for (size_t k = threadIdx.x; k < n_bins; k += blockDim.x) {
            uint32_t total_count = 0;
            float total_sums = 0.0f;
            float total_sq_sums = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                total_count += sh_reduce_count[w * n_bins + k];
                total_sums += sh_reduce_sums[w * n_bins + k];
                total_sq_sums += sh_reduce_sq_sums[w * n_bins + k];
            }
            data[block_id * n_bins + k] = {total_count, total_sums,
                                           total_sq_sums};
        }
    }
}

__global__ void AOVKernel(const AOVData* __restrict__ data,
                          const size_t num_hists,
                          const float length,
                          const float avg,
                          const AOV aov,
                          float* __restrict__ aovs) {
    size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_id >= num_hists)
        return;

    float s1 = 0;
    float s2 = 0;

    for (size_t idx = 0; idx < aov.NumPhaseBins(); idx++) {
        AOVData a = data[thread_id * aov.NumPhaseBins() + idx];
        float n = static_cast<float>(a.count);
        float sum = a.sum;
        float sq_sum = a.sq_sum;

        if (n != 0) {
            float aux = sum / n;
            float residual = aux - avg;
            s1 += n * residual * residual;
            s2 += sq_sum - n * aux * aux;
        }
    }

    aovs[thread_id] = (static_cast<float>(length - aov.NumPhaseBins())
                       / static_cast<float>(aov.NumPhaseBins() - 1))
                      * (s1 / s2);
}

//
// Helper Functions
//

float ArrayMean(const float* data, const size_t length) {
    float sum = 0;

    for (size_t i = 0; i < length; i++) {
        sum += data[i];
    }

    return sum / static_cast<float>(length);
}

//
// Wrapper Functions
//

AOVData* AOV::DeviceFoldAndBin(const float* times,
                               const float* mags,
                               const size_t length,
                               const float* periods,
                               const float* period_dts,
                               const size_t num_periods,
                               const size_t num_p_dts) const {
    // Number of bytes of global memory required to store output
    size_t bytes = NumPhaseBins() * sizeof(AOVData) * num_periods * num_p_dts;

    // Allocate and zero global memory for output histograms
    AOVData* dev_hists;
    gpuErrchk(cudaMalloc(&dev_hists, bytes));

    // Number of threads and corresponding shared memory usage
    // Atomic path needs 2*TILE_SIZE + 3*n_bins (simultaneous),
    // privatization needs 3*NUM_WARPS*n_bins (sequential reduction).
    // Take the max for both paths.
    const size_t num_threads = 256;
    const size_t atomic_floats = 2 * AOV_TILE_SIZE + 3 * NumPhaseBins();
    const size_t reduce_floats = 3 * (num_threads / 32) * NumPhaseBins();
    const size_t shared_bytes =
        (atomic_floats > reduce_floats ? atomic_floats : reduce_floats)
        * sizeof(float);

    // Grid to search over periods and time derivatives
    const dim3 grid_dim = dim3(num_periods, num_p_dts);

    // NOTE: An AOV object is small enough that we can pass it in
    //       the registers by dereferencing it.
    FoldBinKernel<<<grid_dim, num_threads, shared_bytes>>>(
        times, mags, length, periods, period_dts, *this, dev_hists);

    return dev_hists;
}

AOVData* AOV::FoldAndBin(const float* times,
                         const float* mags,
                         const size_t length,
                         const float* periods,
                         const float* period_dts,
                         const size_t num_periods,
                         const size_t num_p_dts) const {
    // Number of bytes of input data
    const size_t data_bytes = length * sizeof(float);

    // Allocate device pointers
    float* dev_times;
    float* dev_mags;
    float* dev_periods;
    float* dev_period_dts;
    gpuErrchk(cudaMalloc(&dev_times, data_bytes));
    gpuErrchk(cudaMalloc(&dev_mags, data_bytes));
    gpuErrchk(cudaMalloc(&dev_periods, num_periods * sizeof(float)));
    gpuErrchk(cudaMalloc(&dev_period_dts, num_p_dts * sizeof(float)));

    // Copy data to device memory
    gpuErrchk(cudaMemcpy(dev_times, times, data_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_mags, mags, data_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_periods, periods, num_periods * sizeof(float),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_period_dts, period_dts, num_p_dts * sizeof(float),
                         cudaMemcpyHostToDevice));

    AOVData* dev_hists =
        DeviceFoldAndBin(dev_times, dev_mags, length, dev_periods,
                         dev_period_dts, num_periods, num_p_dts);

    // Allocate host histograms and copy from device
    size_t bytes = NumPhaseBins() * num_periods * num_p_dts * sizeof(AOVData);
    AOVData* hists = (AOVData*)malloc(bytes);
    gpuErrchk(cudaMemcpy(hists, dev_hists, bytes, cudaMemcpyDeviceToHost));

    // Free GPU memory
    gpuErrchk(cudaFree(dev_times));
    gpuErrchk(cudaFree(dev_mags));
    gpuErrchk(cudaFree(dev_periods));
    gpuErrchk(cudaFree(dev_period_dts));
    gpuErrchk(cudaFree(dev_hists));

    return hists;
}

float* AOV::DeviceCalcAOVFromHists(const AOVData* hists,
                                   const size_t num_hists,
                                   const float length,
                                   const float avg) const {
    // Allocate global memory for output conditional entropy values
    float* dev_aovs;
    gpuErrchk(cudaMalloc(&dev_aovs, num_hists * sizeof(float)));

    const size_t n_t = 512;
    const size_t n_b = (num_hists / n_t) + 1;

    // NOTE: An AOV object is small enough that we can pass it in
    //       the registers by dereferencing it.
    AOVKernel<<<n_b, n_t>>>(hists, num_hists, length, avg, *this, dev_aovs);

    return dev_aovs;
}

float* AOV::CalcAOVFromHists(const AOVData* hists,
                             const size_t num_hists,
                             const float length,
                             const float avg) const {
    // Number of bytes in the histogram
    const size_t bytes = num_hists * NumPhaseBins() * sizeof(AOVData);

    // Allocate device memory for histograms and copy over
    AOVData* dev_hists;
    gpuErrchk(cudaMalloc(&dev_hists, bytes));
    gpuErrchk(cudaMemcpy(dev_hists, hists, bytes, cudaMemcpyHostToDevice));

    float* dev_ces = DeviceCalcAOVFromHists(dev_hists, num_hists, length, avg);

    // Copy CEs to host
    float* ces = (float*)malloc(num_hists * sizeof(float));
    gpuErrchk(cudaMemcpy(ces, dev_ces, num_hists * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Free GPU memory
    gpuErrchk(cudaFree(dev_hists));
    gpuErrchk(cudaFree(dev_ces));

    return ces;
}

void AOV::CalcAOVVals(float* times,
                      float* mags,
                      size_t length,
                      const float* periods,
                      const float* period_dts,
                      const size_t num_periods,
                      const size_t num_p_dts,
                      float* aov_out) const {
    CalcAOVValsBatched(std::vector<float*>{times}, std::vector<float*>{mags},
                       std::vector<size_t>{length}, periods, period_dts,
                       num_periods, num_p_dts, aov_out);
}

float* AOV::CalcAOVVals(float* times,
                        float* mags,
                        size_t length,
                        const float* periods,
                        const float* period_dts,
                        const size_t num_periods,
                        const size_t num_p_dts) const {
    return CalcAOVValsBatched(std::vector<float*>{times},
                              std::vector<float*>{mags},
                              std::vector<size_t>{length}, periods, period_dts,
                              num_periods, num_p_dts);
}

// Per-device state for multi-GPU batched processing
struct AOVDeviceState {
    float* dev_periods;
    float* dev_period_dts;
    cudaStream_t streams[4];
    float* dev_times_buf[4];
    float* dev_mags_buf[4];
    AOVData* dev_hists_buf[4];
    float* dev_aovs_buf[4];
};

void AOV::CalcAOVValsBatched(const std::vector<float*>& times,
                             const std::vector<float*>& mags,
                             const std::vector<size_t>& lengths,
                             const float* periods,
                             const float* period_dts,
                             const size_t num_periods,
                             const size_t num_p_dts,
                             float* aov_out) const {
    size_t num_curves = lengths.size();
    if (num_curves == 0) return;

    size_t num_hists = num_periods * num_p_dts;
    size_t aov_out_size = num_hists * sizeof(float);
    size_t hist_bytes = NumPhaseBins() * sizeof(AOVData) * num_hists;

    // Determine number of GPUs
    int num_devices = 1;
    if (cudaGetDeviceCount(&num_devices) != cudaSuccess) {
        num_devices = 1;
    }
    if (num_devices > (int)num_curves) {
        num_devices = (int)num_curves;
    }

    // Kernel launch information for the fold & bin step
    // Atomic path needs 2*TILE_SIZE + 3*n_bins (simultaneous),
    // privatization needs 3*NUM_WARPS*n_bins (sequential reduction).
    const size_t num_threads_fb = 256;
    const size_t atomic_floats = 2 * AOV_TILE_SIZE + 3 * NumPhaseBins();
    const size_t reduce_floats = 3 * (num_threads_fb / 32) * NumPhaseBins();
    const size_t shared_bytes_fb =
        (atomic_floats > reduce_floats ? atomic_floats : reduce_floats)
        * sizeof(float);
    const dim3 grid_dim_fb = dim3(num_periods, num_p_dts);

    // Kernel launch information for the AOV calculation step
    const size_t num_threads_aov = 256;
    const size_t num_blocks_aov = (num_hists / num_threads_aov) + 1;
    const size_t shared_bytes_aov = num_threads_aov * sizeof(float);

    const int NUM_STREAMS = 4;

    // Partition curves across devices
    size_t base_count = num_curves / num_devices;
    size_t remainder = num_curves % num_devices;

    std::vector<AOVDeviceState> dev_state(num_devices);

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

        // Create streams and allocate per-stream buffers
        for (int s = 0; s < NUM_STREAMS; s++) {
            gpuErrchk(cudaStreamCreate(&dev_state[d].streams[s]));
            gpuErrchk(
                cudaMalloc(&dev_state[d].dev_times_buf[s], buffer_bytes));
            gpuErrchk(
                cudaMalloc(&dev_state[d].dev_mags_buf[s], buffer_bytes));
            gpuErrchk(
                cudaMalloc(&dev_state[d].dev_hists_buf[s], hist_bytes));
            gpuErrchk(
                cudaMalloc(&dev_state[d].dev_aovs_buf[s], aov_out_size));
        }

        // Enqueue work for this device's curves
        for (size_t j = 0; j < count; j++) {
            size_t i = start + j;
            int s = j % NUM_STREAMS;
            cudaStream_t stream = dev_state[d].streams[s];

            float mean_mag = ArrayMean(mags[i], lengths[i]);

            const size_t curve_bytes = lengths[i] * sizeof(float);
            gpuErrchk(cudaMemcpyAsync(dev_state[d].dev_times_buf[s],
                                      times[i], curve_bytes,
                                      cudaMemcpyHostToDevice, stream));
            gpuErrchk(cudaMemcpyAsync(dev_state[d].dev_mags_buf[s],
                                      mags[i], curve_bytes,
                                      cudaMemcpyHostToDevice, stream));

            gpuErrchk(cudaMemsetAsync(dev_state[d].dev_aovs_buf[s], 0,
                                      aov_out_size, stream));

            FoldBinKernel<<<grid_dim_fb, num_threads_fb, shared_bytes_fb,
                            stream>>>(
                dev_state[d].dev_times_buf[s],
                dev_state[d].dev_mags_buf[s], lengths[i],
                dev_state[d].dev_periods, dev_state[d].dev_period_dts,
                *this, dev_state[d].dev_hists_buf[s]);

            AOVKernel<<<num_blocks_aov, num_threads_aov, shared_bytes_aov,
                        stream>>>(
                dev_state[d].dev_hists_buf[s], num_hists, lengths[i],
                mean_mag, *this, dev_state[d].dev_aovs_buf[s]);

            gpuErrchk(cudaMemcpyAsync(&aov_out[i * num_hists],
                                      dev_state[d].dev_aovs_buf[s],
                                      aov_out_size, cudaMemcpyDeviceToHost,
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
            gpuErrchk(cudaFree(dev_state[d].dev_mags_buf[s]));
            gpuErrchk(cudaFree(dev_state[d].dev_hists_buf[s]));
            gpuErrchk(cudaFree(dev_state[d].dev_aovs_buf[s]));
            gpuErrchk(cudaStreamDestroy(dev_state[d].streams[s]));
        }
    }
}

float* AOV::CalcAOVValsBatched(const std::vector<float*>& times,
                               const std::vector<float*>& mags,
                               const std::vector<size_t>& lengths,
                               const float* periods,
                               const float* period_dts,
                               const size_t num_periods,
                               const size_t num_p_dts) const {
    // Size of one AOV out array, and total AOV output size.
    size_t aov_out_size = num_periods * num_p_dts * sizeof(float);
    size_t aov_size_total = aov_out_size * lengths.size();

    // Allocate the output AOV array so we can copy to it.
    float* aov_out = (float*)malloc(aov_size_total);

    // Perform AOV calculation.
    CalcAOVValsBatched(times, mags, lengths, periods, period_dts, num_periods,
                       num_p_dts, aov_out);

    return aov_out;
}
