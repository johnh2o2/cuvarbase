/*
 * Transit Least Squares (TLS) GPU kernel
 *
 * This implements a GPU-accelerated version of the TLS algorithm for
 * detecting periodic planetary transits.
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

// Maximum number of data points (for shared memory allocation)
#define MAX_NDATA 10000

// Physical constants
#define PI 3.141592653589793f

// Device utility functions
__device__ inline float mod1(float x) {
    return x - floorf(x);
}

__device__ inline int get_global_id() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

/**
 * Calculate chi-squared for a given transit model fit
 *
 * chi2 = sum((y_i - model_i)^2 / sigma_i^2)
 */
__device__ float calculate_chi2(
    const float* y_sorted,
    const float* dy_sorted,
    const float* transit_model,
    float depth,
    int n_in_transit,
    int ndata)
{
    float chi2 = 0.0f;

    for (int i = 0; i < ndata; i++) {
        // Model: 1.0 out of transit, 1.0 - depth * model in transit
        float model_val = 1.0f;
        if (i < n_in_transit) {
            model_val = 1.0f - depth * (1.0f - transit_model[i]);
        }

        float residual = y_sorted[i] - model_val;
        float sigma2 = dy_sorted[i] * dy_sorted[i];

        chi2 += (residual * residual) / (sigma2 + 1e-10f);
    }

    return chi2;
}

/**
 * Calculate optimal transit depth using least squares
 *
 * depth_opt = sum(y_i * m_i) / sum(m_i^2)
 * where m_i is the transit model (0 out of transit, >0 in transit)
 */
__device__ float calculate_optimal_depth(
    const float* y_sorted,
    const float* transit_model,
    int n_in_transit)
{
    float numerator = 0.0f;
    float denominator = 0.0f;

    for (int i = 0; i < n_in_transit; i++) {
        float model_depth = 1.0f - transit_model[i];
        numerator += y_sorted[i] * model_depth;
        denominator += model_depth * model_depth;
    }

    if (denominator < 1e-10f) {
        return 0.0f;
    }

    return numerator / denominator;
}

/**
 * Simple phase folding
 */
__device__ inline float phase_fold(float t, float period) {
    return mod1(t / period);
}

/**
 * Simple trapezoidal transit model
 *
 * For Phase 1, we use a simple trapezoid instead of full Batman model.
 * This will be replaced with pre-computed limb-darkened models in Phase 2.
 */
__device__ float simple_transit_model(float phase, float duration_phase) {
    // Transit centered at phase = 0.0
    // Ingress/egress = 10% of total duration
    float ingress_frac = 0.1f;
    float t_ingress = duration_phase * ingress_frac;
    float t_flat = duration_phase * (1.0f - 2.0f * ingress_frac);

    // Wrap phase to [-0.5, 0.5]
    float p = phase;
    if (p > 0.5f) p -= 1.0f;

    float abs_p = fabsf(p);

    // Check if in transit (within +/- duration/2)
    if (abs_p > duration_phase * 0.5f) {
        return 1.0f; // Out of transit
    }

    // Distance from transit center
    float dist = abs_p;

    // Ingress region
    if (dist < t_ingress) {
        return 1.0f - dist / t_ingress;
    }

    // Flat bottom
    if (dist < t_ingress + t_flat) {
        return 0.0f; // Full depth
    }

    // Egress region
    float egress_start = t_ingress + t_flat;
    if (dist < duration_phase * 0.5f) {
        return 1.0f - (duration_phase * 0.5f - dist) / t_ingress;
    }

    return 1.0f; // Out of transit
}

/**
 * Comparison function for sorting (for use with thrust or manual sort)
 */
__device__ inline bool compare_phases(float a, float b) {
    return a < b;
}

/**
 * Simple bubble sort for small arrays (Phase 1 implementation)
 *
 * NOTE: This is inefficient for large arrays. In Phase 2, we'll use
 * CUB DeviceRadixSort or thrust::sort.
 */
__device__ void bubble_sort_phases(
    float* phases,
    float* y_sorted,
    float* dy_sorted,
    const float* y,
    const float* dy,
    int ndata)
{
    // Copy to sorted arrays
    for (int i = threadIdx.x; i < ndata; i += blockDim.x) {
        y_sorted[i] = y[i];
        dy_sorted[i] = dy[i];
    }
    __syncthreads();

    // Simple bubble sort (only works for small ndata in Phase 1)
    // Thread 0 does the sorting
    if (threadIdx.x == 0) {
        for (int i = 0; i < ndata - 1; i++) {
            for (int j = 0; j < ndata - i - 1; j++) {
                if (phases[j] > phases[j + 1]) {
                    // Swap phases
                    float temp = phases[j];
                    phases[j] = phases[j + 1];
                    phases[j + 1] = temp;

                    // Swap y
                    temp = y_sorted[j];
                    y_sorted[j] = y_sorted[j + 1];
                    y_sorted[j + 1] = temp;

                    // Swap dy
                    temp = dy_sorted[j];
                    dy_sorted[j] = dy_sorted[j + 1];
                    dy_sorted[j + 1] = temp;
                }
            }
        }
    }
    __syncthreads();
}

/**
 * Main TLS search kernel
 *
 * Each block processes one period. Threads within a block search over
 * different durations and T0 positions.
 *
 * Grid: (nperiods, 1, 1)
 * Block: (BLOCK_SIZE, 1, 1)
 */
extern "C" __global__ void tls_search_kernel(
    const float* __restrict__ t,           // Time array [ndata]
    const float* __restrict__ y,           // Flux array [ndata]
    const float* __restrict__ dy,          // Uncertainty array [ndata]
    const float* __restrict__ periods,     // Trial periods [nperiods]
    const int ndata,
    const int nperiods,
    float* __restrict__ chi2_out,          // Output: minimum chi2 [nperiods]
    float* __restrict__ best_t0_out,       // Output: best T0 [nperiods]
    float* __restrict__ best_duration_out, // Output: best duration [nperiods]
    float* __restrict__ best_depth_out)    // Output: best depth [nperiods]
{
    // Shared memory for this block's data
    extern __shared__ float shared_mem[];

    float* phases = shared_mem;
    float* y_sorted = &shared_mem[ndata];
    float* dy_sorted = &shared_mem[2 * ndata];
    float* transit_model = &shared_mem[3 * ndata];
    float* thread_chi2 = &shared_mem[4 * ndata];

    int period_idx = blockIdx.x;

    // Check bounds
    if (period_idx >= nperiods) {
        return;
    }

    float period = periods[period_idx];

    // Phase fold data (all threads participate)
    for (int i = threadIdx.x; i < ndata; i += blockDim.x) {
        phases[i] = phase_fold(t[i], period);
    }
    __syncthreads();

    // Sort by phase (Phase 1: simple sort by thread 0)
    // TODO Phase 2: Replace with CUB DeviceRadixSort
    bubble_sort_phases(phases, y_sorted, dy_sorted, y, dy, ndata);

    // Each thread will track its own minimum chi2
    float thread_min_chi2 = 1e30f;
    float thread_best_t0 = 0.0f;
    float thread_best_duration = 0.0f;
    float thread_best_depth = 0.0f;

    // Test different transit durations
    // For Phase 1, use a simple range of durations
    // TODO Phase 2: Use pre-computed duration grid per period

    int n_durations = 10; // Simple fixed number for Phase 1
    float duration_min = 0.01f;  // 1% of period
    float duration_max = 0.1f;   // 10% of period

    for (int d_idx = 0; d_idx < n_durations; d_idx++) {
        float duration = duration_min + (duration_max - duration_min) * d_idx / n_durations;
        float duration_phase = duration / period;

        // Generate transit model for this duration (all threads)
        for (int i = threadIdx.x; i < ndata; i += blockDim.x) {
            transit_model[i] = simple_transit_model(phases[i], duration_phase);
        }
        __syncthreads();

        // Test different T0 positions (each thread tests different T0)
        int n_t0 = 20; // Number of T0 positions to test

        for (int t0_idx = threadIdx.x; t0_idx < n_t0; t0_idx += blockDim.x) {
            float t0_phase = (float)t0_idx / n_t0;

            // Shift transit model by t0_phase
            // For simplicity in Phase 1, we recalculate the model
            // TODO Phase 2: Use more efficient array shifting

            float local_chi2 = 0.0f;

            // Calculate optimal depth for this configuration
            // Count how many points are "in transit"
            int n_in_transit = 0;
            for (int i = 0; i < ndata; i++) {
                float phase_shifted = mod1(phases[i] - t0_phase + 0.5f) - 0.5f;
                if (fabsf(phase_shifted) < duration_phase * 0.5f) {
                    n_in_transit++;
                }
            }

            if (n_in_transit > 2) {
                // Calculate optimal depth
                float depth = 0.1f; // For Phase 1, use fixed depth
                // TODO Phase 2: Calculate optimal depth

                // Calculate chi-squared
                local_chi2 = 0.0f;
                for (int i = 0; i < ndata; i++) {
                    float phase_shifted = mod1(phases[i] - t0_phase + 0.5f) - 0.5f;
                    float model_val = 1.0f;

                    if (fabsf(phase_shifted) < duration_phase * 0.5f) {
                        model_val = 1.0f - depth;
                    }

                    float residual = y_sorted[i] - model_val;
                    float sigma2 = dy_sorted[i] * dy_sorted[i];
                    local_chi2 += (residual * residual) / (sigma2 + 1e-10f);
                }

                // Update thread minimum
                if (local_chi2 < thread_min_chi2) {
                    thread_min_chi2 = local_chi2;
                    thread_best_t0 = t0_phase;
                    thread_best_duration = duration;
                    thread_best_depth = depth;
                }
            }
        }
        __syncthreads();
    }

    // Store thread results in shared memory
    thread_chi2[threadIdx.x] = thread_min_chi2;
    __syncthreads();

    // Parallel reduction to find minimum chi2 (tree reduction)
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            if (thread_chi2[threadIdx.x + stride] < thread_chi2[threadIdx.x]) {
                thread_chi2[threadIdx.x] = thread_chi2[threadIdx.x + stride];
                // Note: We're not tracking which thread had the minimum
                // TODO Phase 2: Properly track best parameters across threads
            }
        }
        __syncthreads();
    }

    // Thread 0 writes result
    if (threadIdx.x == 0) {
        chi2_out[period_idx] = thread_chi2[0];
        best_t0_out[period_idx] = thread_best_t0;
        best_duration_out[period_idx] = thread_best_duration;
        best_depth_out[period_idx] = thread_best_depth;
    }
}
