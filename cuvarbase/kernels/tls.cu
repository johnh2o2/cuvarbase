/*
 * Transit Least Squares (TLS) GPU kernel
 *
 * Single optimized kernel using insertion sort for phase sorting.
 * Works correctly for datasets up to ~5000 points.
 *
 * References:
 * [1] Hippke & Heller (2019), A&A 623, A39
 * [2] Kovács et al. (2002), A&A 391, 369
 */

#include <stdio.h>

//{CPP_DEFS}

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

#define MAX_NDATA 10000
#define PI 3.141592653589793f
#define WARP_SIZE 32

// Device utility functions
__device__ inline float mod1(float x) {
    return x - floorf(x);
}

/**
 * Calculate optimal transit depth using weighted least squares
 */
__device__ float calculate_optimal_depth(
    const float* y_sorted,
    const float* dy_sorted,
    const float* phases_sorted,
    float duration_phase,
    float t0_phase,
    int ndata)
{
    float numerator = 0.0f;
    float denominator = 0.0f;

    for (int i = 0; i < ndata; i++) {
        float phase_rel = mod1(phases_sorted[i] - t0_phase + 0.5f) - 0.5f;

        if (fabsf(phase_rel) < duration_phase * 0.5f) {
            float sigma2 = dy_sorted[i] * dy_sorted[i] + 1e-10f;
            float model_depth = 1.0f;
            float y_residual = 1.0f - y_sorted[i];
            numerator += y_residual * model_depth / sigma2;
            denominator += model_depth * model_depth / sigma2;
        }
    }

    if (denominator < 1e-10f) return 0.0f;

    float depth = numerator / denominator;
    if (depth < 0.0f) depth = 0.0f;
    if (depth > 1.0f) depth = 1.0f;

    return depth;
}

/**
 * Calculate chi-squared for a given transit model fit
 */
__device__ float calculate_chi2(
    const float* y_sorted,
    const float* dy_sorted,
    const float* phases_sorted,
    float duration_phase,
    float t0_phase,
    float depth,
    int ndata)
{
    float chi2 = 0.0f;

    for (int i = 0; i < ndata; i++) {
        float phase_rel = mod1(phases_sorted[i] - t0_phase + 0.5f) - 0.5f;
        float model_val = (fabsf(phase_rel) < duration_phase * 0.5f) ? (1.0f - depth) : 1.0f;
        float residual = y_sorted[i] - model_val;
        float sigma2 = dy_sorted[i] * dy_sorted[i] + 1e-10f;
        chi2 += (residual * residual) / sigma2;
    }

    return chi2;
}

/**
 * TLS search kernel with Keplerian duration constraints
 * Grid: (nperiods, 1, 1), Block: (BLOCK_SIZE, 1, 1)
 *
 * This version uses per-period duration ranges based on Keplerian assumptions,
 * similar to BLS's qmin/qmax approach.
 */
extern "C" __global__ void tls_search_kernel_keplerian(
    const float* __restrict__ t,
    const float* __restrict__ y,
    const float* __restrict__ dy,
    const float* __restrict__ periods,
    const float* __restrict__ qmin,      // Minimum fractional duration per period
    const float* __restrict__ qmax,      // Maximum fractional duration per period
    const int ndata,
    const int nperiods,
    const int n_durations,               // Number of duration samples
    float* __restrict__ chi2_out,
    float* __restrict__ best_t0_out,
    float* __restrict__ best_duration_out,
    float* __restrict__ best_depth_out)
{
    extern __shared__ float shared_mem[];
    float* phases = shared_mem;
    float* y_sorted = &shared_mem[ndata];
    float* dy_sorted = &shared_mem[2 * ndata];
    float* thread_chi2 = &shared_mem[3 * ndata];
    float* thread_t0 = &thread_chi2[blockDim.x];
    float* thread_duration = &thread_t0[blockDim.x];
    float* thread_depth = &thread_duration[blockDim.x];

    int period_idx = blockIdx.x;
    if (period_idx >= nperiods) return;

    float period = periods[period_idx];
    float duration_phase_min = qmin[period_idx];
    float duration_phase_max = qmax[period_idx];

    // Phase fold
    for (int i = threadIdx.x; i < ndata; i += blockDim.x) {
        phases[i] = mod1(t[i] / period);
    }
    __syncthreads();

    // Insertion sort (works for ndata < 5000)
    if (threadIdx.x == 0 && ndata < 5000) {
        for (int i = 0; i < ndata; i++) {
            y_sorted[i] = y[i];
            dy_sorted[i] = dy[i];
        }
        for (int i = 1; i < ndata; i++) {
            float key_phase = phases[i];
            float key_y = y_sorted[i];
            float key_dy = dy_sorted[i];
            int j = i - 1;
            while (j >= 0 && phases[j] > key_phase) {
                phases[j + 1] = phases[j];
                y_sorted[j + 1] = y_sorted[j];
                dy_sorted[j + 1] = dy_sorted[j];
                j--;
            }
            phases[j + 1] = key_phase;
            y_sorted[j + 1] = key_y;
            dy_sorted[j + 1] = key_dy;
        }
    }
    __syncthreads();

    // Search over durations and T0 using Keplerian constraints
    float thread_min_chi2 = 1e30f;
    float thread_best_t0 = 0.0f;
    float thread_best_duration = 0.0f;
    float thread_best_depth = 0.0f;

    for (int d_idx = 0; d_idx < n_durations; d_idx++) {
        float log_dur_min = logf(duration_phase_min);
        float log_dur_max = logf(duration_phase_max);
        float log_duration = log_dur_min + (log_dur_max - log_dur_min) * d_idx / (n_durations - 1);
        float duration_phase = expf(log_duration);
        float duration = duration_phase * period;

        int n_t0 = 30;
        for (int t0_idx = threadIdx.x; t0_idx < n_t0; t0_idx += blockDim.x) {
            float t0_phase = (float)t0_idx / n_t0;
            float depth = calculate_optimal_depth(y_sorted, dy_sorted, phases, duration_phase, t0_phase, ndata);

            if (depth > 0.0f && depth < 0.5f) {
                float chi2 = calculate_chi2(y_sorted, dy_sorted, phases, duration_phase, t0_phase, depth, ndata);
                if (chi2 < thread_min_chi2) {
                    thread_min_chi2 = chi2;
                    thread_best_t0 = t0_phase;
                    thread_best_duration = duration;
                    thread_best_depth = depth;
                }
            }
        }
    }

    // Store results
    thread_chi2[threadIdx.x] = thread_min_chi2;
    thread_t0[threadIdx.x] = thread_best_t0;
    thread_duration[threadIdx.x] = thread_best_duration;
    thread_depth[threadIdx.x] = thread_best_depth;
    __syncthreads();

    // Reduction with warp optimization
    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride /= 2) {
        if (threadIdx.x < stride) {
            if (thread_chi2[threadIdx.x + stride] < thread_chi2[threadIdx.x]) {
                thread_chi2[threadIdx.x] = thread_chi2[threadIdx.x + stride];
                thread_t0[threadIdx.x] = thread_t0[threadIdx.x + stride];
                thread_duration[threadIdx.x] = thread_duration[threadIdx.x + stride];
                thread_depth[threadIdx.x] = thread_depth[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // Warp reduction (no sync needed)
    if (threadIdx.x < WARP_SIZE) {
        volatile float* vchi2 = thread_chi2;
        volatile float* vt0 = thread_t0;
        volatile float* vdur = thread_duration;
        volatile float* vdepth = thread_depth;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            if (vchi2[threadIdx.x + offset] < vchi2[threadIdx.x]) {
                vchi2[threadIdx.x] = vchi2[threadIdx.x + offset];
                vt0[threadIdx.x] = vt0[threadIdx.x + offset];
                vdur[threadIdx.x] = vdur[threadIdx.x + offset];
                vdepth[threadIdx.x] = vdepth[threadIdx.x + offset];
            }
        }
    }

    // Write final result
    if (threadIdx.x == 0) {
        chi2_out[period_idx] = thread_chi2[0];
        best_t0_out[period_idx] = thread_t0[0];
        best_duration_out[period_idx] = thread_duration[0];
        best_depth_out[period_idx] = thread_depth[0];
    }
}

/**
 * TLS search kernel
 * Grid: (nperiods, 1, 1), Block: (BLOCK_SIZE, 1, 1)
 */
extern "C" __global__ void tls_search_kernel(
    const float* __restrict__ t,
    const float* __restrict__ y,
    const float* __restrict__ dy,
    const float* __restrict__ periods,
    const int ndata,
    const int nperiods,
    float* __restrict__ chi2_out,
    float* __restrict__ best_t0_out,
    float* __restrict__ best_duration_out,
    float* __restrict__ best_depth_out)
{
    extern __shared__ float shared_mem[];
    float* phases = shared_mem;
    float* y_sorted = &shared_mem[ndata];
    float* dy_sorted = &shared_mem[2 * ndata];
    float* thread_chi2 = &shared_mem[3 * ndata];
    float* thread_t0 = &thread_chi2[blockDim.x];
    float* thread_duration = &thread_t0[blockDim.x];
    float* thread_depth = &thread_duration[blockDim.x];

    int period_idx = blockIdx.x;
    if (period_idx >= nperiods) return;

    float period = periods[period_idx];

    // Phase fold
    for (int i = threadIdx.x; i < ndata; i += blockDim.x) {
        phases[i] = mod1(t[i] / period);
    }
    __syncthreads();

    // Insertion sort (works for ndata < 5000)
    if (threadIdx.x == 0 && ndata < 5000) {
        for (int i = 0; i < ndata; i++) {
            y_sorted[i] = y[i];
            dy_sorted[i] = dy[i];
        }
        for (int i = 1; i < ndata; i++) {
            float key_phase = phases[i];
            float key_y = y_sorted[i];
            float key_dy = dy_sorted[i];
            int j = i - 1;
            while (j >= 0 && phases[j] > key_phase) {
                phases[j + 1] = phases[j];
                y_sorted[j + 1] = y_sorted[j];
                dy_sorted[j + 1] = dy_sorted[j];
                j--;
            }
            phases[j + 1] = key_phase;
            y_sorted[j + 1] = key_y;
            dy_sorted[j + 1] = key_dy;
        }
    }
    __syncthreads();

    // Search over durations and T0
    float thread_min_chi2 = 1e30f;
    float thread_best_t0 = 0.0f;
    float thread_best_duration = 0.0f;
    float thread_best_depth = 0.0f;

    int n_durations = 15;
    float duration_phase_min = 0.005f;
    float duration_phase_max = 0.15f;

    for (int d_idx = 0; d_idx < n_durations; d_idx++) {
        float log_dur_min = logf(duration_phase_min);
        float log_dur_max = logf(duration_phase_max);
        float log_duration = log_dur_min + (log_dur_max - log_dur_min) * d_idx / (n_durations - 1);
        float duration_phase = expf(log_duration);
        float duration = duration_phase * period;

        int n_t0 = 30;
        for (int t0_idx = threadIdx.x; t0_idx < n_t0; t0_idx += blockDim.x) {
            float t0_phase = (float)t0_idx / n_t0;
            float depth = calculate_optimal_depth(y_sorted, dy_sorted, phases, duration_phase, t0_phase, ndata);

            if (depth > 0.0f && depth < 0.5f) {
                float chi2 = calculate_chi2(y_sorted, dy_sorted, phases, duration_phase, t0_phase, depth, ndata);
                if (chi2 < thread_min_chi2) {
                    thread_min_chi2 = chi2;
                    thread_best_t0 = t0_phase;
                    thread_best_duration = duration;
                    thread_best_depth = depth;
                }
            }
        }
    }

    // Store results
    thread_chi2[threadIdx.x] = thread_min_chi2;
    thread_t0[threadIdx.x] = thread_best_t0;
    thread_duration[threadIdx.x] = thread_best_duration;
    thread_depth[threadIdx.x] = thread_best_depth;
    __syncthreads();

    // Reduction with warp optimization
    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride /= 2) {
        if (threadIdx.x < stride) {
            if (thread_chi2[threadIdx.x + stride] < thread_chi2[threadIdx.x]) {
                thread_chi2[threadIdx.x] = thread_chi2[threadIdx.x + stride];
                thread_t0[threadIdx.x] = thread_t0[threadIdx.x + stride];
                thread_duration[threadIdx.x] = thread_duration[threadIdx.x + stride];
                thread_depth[threadIdx.x] = thread_depth[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // Warp reduction (no sync needed)
    if (threadIdx.x < WARP_SIZE) {
        volatile float* vchi2 = thread_chi2;
        volatile float* vt0 = thread_t0;
        volatile float* vdur = thread_duration;
        volatile float* vdepth = thread_depth;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            if (vchi2[threadIdx.x + offset] < vchi2[threadIdx.x]) {
                vchi2[threadIdx.x] = vchi2[threadIdx.x + offset];
                vt0[threadIdx.x] = vt0[threadIdx.x + offset];
                vdur[threadIdx.x] = vdur[threadIdx.x + offset];
                vdepth[threadIdx.x] = vdepth[threadIdx.x + offset];
            }
        }
    }

    // Write final result
    if (threadIdx.x == 0) {
        chi2_out[period_idx] = thread_chi2[0];
        best_t0_out[period_idx] = thread_t0[0];
        best_duration_out[period_idx] = thread_duration[0];
        best_depth_out[period_idx] = thread_depth[0];
    }
}
