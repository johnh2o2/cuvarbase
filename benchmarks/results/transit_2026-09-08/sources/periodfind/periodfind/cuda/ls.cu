// Copyright 2020 California Institute of Technology. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
// Author: Ethan Jaszewski

#include "ls.h"

#include <algorithm>
#include <thread>

#include <cstdio>

#include "cuda_runtime.h"
#include "math.h"

#include "errchk.cuh"

const float TWO_PI = M_PI * 2.0;

//
// Simple LombScargle Function Definitions
//

LombScargle::LombScargle() {}

//
// CUDA Kernels
//

// Tile size for shared-memory light curve tiling
#define LS_TILE_SIZE 256

__global__ void LombScargleKernel(const float* __restrict__ times,
                                  const float* __restrict__ mags,
                                  const size_t length,
                                  const float* __restrict__ periods,
                                  const float* __restrict__ period_dts,
                                  const size_t num_periods,
                                  const size_t num_period_dts,
                                  const LombScargle params,
                                  float* __restrict__ periodogram) {
    // Shared memory layout: tile data + reduction workspace
    // [0 .. LS_TILE_SIZE-1]              = sh_times
    // [LS_TILE_SIZE .. 2*LS_TILE_SIZE-1] = sh_mags
    // [2*LS_TILE_SIZE .. ]               = reduction workspace (4 * NUM_WARPS floats)
    extern __shared__ float sh_data[];
    float* sh_times = &sh_data[0];
    float* sh_mags = &sh_data[LS_TILE_SIZE];

    const int NUM_WARPS = blockDim.x / 32;
    float* sh_reduce = &sh_data[2 * LS_TILE_SIZE];
    // sh_reduce layout: [mag_cos * NUM_WARPS, mag_sin * NUM_WARPS,
    //                     cos_cos * NUM_WARPS, cos_sin * NUM_WARPS]

    // One block per (period, period_dt) pair
    const size_t period_idx = blockIdx.x;
    const size_t pdt_idx = blockIdx.y;

    if (period_idx >= num_periods || pdt_idx >= num_period_dts) {
        return;
    }

    const float period = periods[period_idx];
    const float period_dt = period_dts[pdt_idx];
    const float pdt_corr = (period_dt / period) / 2;

    // Per-thread accumulators — each thread processes a stripe
    float mag_cos = 0.0f;
    float mag_sin = 0.0f;
    float cos_cos = 0.0f;
    float cos_sin = 0.0f;

    float cos_val, sin_val;

    // Process the light curve in tiles
    for (size_t tile_start = 0; tile_start < length;
         tile_start += LS_TILE_SIZE) {
        size_t tile_end = tile_start + LS_TILE_SIZE;
        if (tile_end > length)
            tile_end = length;
        size_t tile_len = tile_end - tile_start;

        // Cooperatively load tile into shared memory
        for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
            sh_times[i] = times[tile_start + i];
            sh_mags[i] = mags[tile_start + i];
        }
        __syncthreads();

        // Each thread accumulates over its stripe of the tile
        for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
            float t = sh_times[i];
            float mag = sh_mags[i];

            double t_corr_d = (double)t - (double)pdt_corr * (double)t * (double)t;
            double ratio_d = t_corr_d / (double)period;
            float folded = fabsf((float)(ratio_d - floor(ratio_d)));

            sincosf(TWO_PI * folded, &sin_val, &cos_val);

            mag_cos += mag * cos_val;
            mag_sin += mag * sin_val;
            cos_cos += cos_val * cos_val;
            cos_sin += cos_val * sin_val;
        }
        __syncthreads();
    }

    // Parallel reduction: warp-level shuffle first
    const unsigned int FULL_MASK = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset >>= 1) {
        mag_cos += __shfl_down_sync(FULL_MASK, mag_cos, offset);
        mag_sin += __shfl_down_sync(FULL_MASK, mag_sin, offset);
        cos_cos += __shfl_down_sync(FULL_MASK, cos_cos, offset);
        cos_sin += __shfl_down_sync(FULL_MASK, cos_sin, offset);
    }

    // Warp leaders write to shared memory
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        sh_reduce[0 * NUM_WARPS + warp_id] = mag_cos;
        sh_reduce[1 * NUM_WARPS + warp_id] = mag_sin;
        sh_reduce[2 * NUM_WARPS + warp_id] = cos_cos;
        sh_reduce[3 * NUM_WARPS + warp_id] = cos_sin;
    }
    __syncthreads();

    // Final reduction across warps (thread 0 only)
    if (threadIdx.x != 0)
        return;

    mag_cos = 0.0f;
    mag_sin = 0.0f;
    cos_cos = 0.0f;
    cos_sin = 0.0f;
    for (int w = 0; w < NUM_WARPS; w++) {
        mag_cos += sh_reduce[0 * NUM_WARPS + w];
        mag_sin += sh_reduce[1 * NUM_WARPS + w];
        cos_cos += sh_reduce[2 * NUM_WARPS + w];
        cos_sin += sh_reduce[3 * NUM_WARPS + w];
    }

    float sin_sin = static_cast<float>(length) - cos_cos;

    float cos_tau, sin_tau;
    sincosf(0.5f * atan2f(2.0f * cos_sin, cos_cos - sin_sin), &sin_tau,
            &cos_tau);

    float numerator_l = cos_tau * mag_cos + sin_tau * mag_sin;
    numerator_l *= numerator_l;

    float numerator_r = cos_tau * mag_sin - sin_tau * mag_cos;
    numerator_r *= numerator_r;

    float denominator_l = cos_tau * cos_tau * cos_cos
                          + 2 * cos_tau * sin_tau * cos_sin
                          + sin_tau * sin_tau * sin_sin;

    float denominator_r = cos_tau * cos_tau * sin_sin
                          - 2 * cos_tau * sin_tau * cos_sin
                          + sin_tau * sin_tau * cos_cos;

    periodogram[period_idx * num_period_dts + pdt_idx] =
        0.5f * ((numerator_l / denominator_l) + (numerator_r / denominator_r));
}

//
// Wrapper Functions
//

float* LombScargle::DeviceCalcLS(const float* times,
                                 const float* mags,
                                 const size_t length,
                                 const float* periods,
                                 const float* period_dts,
                                 const size_t num_periods,
                                 const size_t num_p_dts) const {
    float* periodogram;
    gpuErrchk(
        cudaMalloc(&periodogram, num_periods * num_p_dts * sizeof(float)));

    // One block per (period, period_dt) pair
    const size_t num_threads = 256;
    const size_t num_warps = num_threads / 32;
    const size_t shared_bytes =
        (2 * LS_TILE_SIZE + 4 * num_warps) * sizeof(float);
    const dim3 grid_dim = dim3(num_periods, num_p_dts);

    LombScargleKernel<<<grid_dim, num_threads, shared_bytes>>>(
        times, mags, length, periods, period_dts, num_periods, num_p_dts, *this,
        periodogram);

    return periodogram;
}

void LombScargle::CalcLS(float* times,
                         float* mags,
                         size_t length,
                         const float* periods,
                         const float* period_dts,
                         const size_t num_periods,
                         const size_t num_p_dts,
                         float* per_out) const {
    CalcLSBatched(std::vector<float*>{times}, std::vector<float*>{mags},
                  std::vector<size_t>{length}, periods, period_dts, num_periods,
                  num_p_dts, per_out);
}

float* LombScargle::CalcLS(float* times,
                           float* mags,
                           size_t length,
                           const float* periods,
                           const float* period_dts,
                           const size_t num_periods,
                           const size_t num_p_dts) const {
    return CalcLSBatched(std::vector<float*>{times}, std::vector<float*>{mags},
                         std::vector<size_t>{length}, periods, period_dts,
                         num_periods, num_p_dts);
}

// Per-device state for multi-GPU batched processing
struct LSDeviceState {
    float* dev_periods;
    float* dev_period_dts;
    cudaStream_t streams[4];
    float* dev_times_buf[4];
    float* dev_mags_buf[4];
    float* dev_per_buf[4];
};

void LombScargle::CalcLSBatched(const std::vector<float*>& times,
                                const std::vector<float*>& mags,
                                const std::vector<size_t>& lengths,
                                const float* periods,
                                const float* period_dts,
                                const size_t num_periods,
                                const size_t num_p_dts,
                                float* per_out) const {
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
    const size_t num_warps = num_threads / 32;
    const size_t shared_bytes =
        (2 * LS_TILE_SIZE + 4 * num_warps) * sizeof(float);
    const dim3 grid_dim = dim3(num_periods, num_p_dts);

    const int NUM_STREAMS = 4;

    // Partition curves across devices
    size_t base_count = num_curves / num_devices;
    size_t remainder = num_curves % num_devices;

    std::vector<LSDeviceState> dev_state(num_devices);

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
                cudaMalloc(&dev_state[d].dev_per_buf[s], per_out_size));
        }

        // Enqueue work for this device's curves
        for (size_t j = 0; j < count; j++) {
            size_t i = start + j;
            int s = j % NUM_STREAMS;
            cudaStream_t stream = dev_state[d].streams[s];

            const size_t curve_bytes = lengths[i] * sizeof(float);
            gpuErrchk(cudaMemcpyAsync(dev_state[d].dev_times_buf[s],
                                      times[i], curve_bytes,
                                      cudaMemcpyHostToDevice, stream));
            gpuErrchk(cudaMemcpyAsync(dev_state[d].dev_mags_buf[s],
                                      mags[i], curve_bytes,
                                      cudaMemcpyHostToDevice, stream));

            gpuErrchk(cudaMemsetAsync(dev_state[d].dev_per_buf[s], 0,
                                      per_out_size, stream));

            LombScargleKernel<<<grid_dim, num_threads, shared_bytes,
                                stream>>>(
                dev_state[d].dev_times_buf[s],
                dev_state[d].dev_mags_buf[s], lengths[i],
                dev_state[d].dev_periods, dev_state[d].dev_period_dts,
                num_periods, num_p_dts, *this, dev_state[d].dev_per_buf[s]);

            gpuErrchk(cudaMemcpyAsync(&per_out[i * per_points],
                                      dev_state[d].dev_per_buf[s],
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
            gpuErrchk(cudaFree(dev_state[d].dev_mags_buf[s]));
            gpuErrchk(cudaFree(dev_state[d].dev_per_buf[s]));
            gpuErrchk(cudaStreamDestroy(dev_state[d].streams[s]));
        }
    }
}

float* LombScargle::CalcLSBatched(const std::vector<float*>& times,
                                  const std::vector<float*>& mags,
                                  const std::vector<size_t>& lengths,
                                  const float* periods,
                                  const float* period_dts,
                                  const size_t num_periods,
                                  const size_t num_p_dts) const {
    // Size of one periodogram out array, and total periodogram output size.
    size_t per_points = num_periods * num_p_dts;
    size_t per_out_size = per_points * sizeof(float);
    size_t per_size_total = per_out_size * lengths.size();

    // Allocate the output CE array so we can copy to it.
    float* per_out = (float*)malloc(per_size_total);

    CalcLSBatched(times, mags, lengths, periods, period_dts, num_periods,
                  num_p_dts, per_out);

    return per_out;
}
