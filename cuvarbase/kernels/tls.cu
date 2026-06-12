/*
 * Transit Least Squares (TLS) GPU kernel
 *
 * Kernel using a limb-darkened transit template for physically
 * realistic fitting. Data are staged in shared memory; the depth and
 * chi2 accumulations are order-independent, so no phase sort is
 * needed (an earlier bitonic sort was pure wasted work and was also
 * incomplete for non-power-of-2 sizes).
 *
 * The transit template is a 1D array mapping transit_coord in [-1, 1]
 * to normalized depth in [0, 1], precomputed on the CPU using batman
 * (or a trapezoidal fallback) and loaded into shared memory.
 *
 * References:
 * [1] Hippke & Heller (2019), A&A 623, A39
 * [2] Kovacs et al. (2002), A&A 391, 369
 */

#include <stdio.h>

//{CPP_DEFS}

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

#define PI 3.141592653589793f
#define WARP_SIZE 32

/*
 * Epoch (t0) grid: the stride scales with the transit duration
 * (stride = duration_phase / T0_OVERSAMPLE) so that narrow transits
 * always overlap a tested epoch. The previous fixed 30-point grid
 * missed transits narrower than ~1/30 of the period entirely.
 * Mirrors cuvarbase.tls_grids.t0_grid_size().
 */
#ifndef T0_OVERSAMPLE
#define T0_OVERSAMPLE 3.0f
#endif
#ifndef MIN_N_T0
#define MIN_N_T0 30
#endif
#ifndef MAX_N_T0
#define MAX_N_T0 20000
#endif

__device__ inline int t0_grid_size(float duration_phase) {
    int n_t0 = (int)ceilf(T0_OVERSAMPLE / duration_phase);
    if (n_t0 < MIN_N_T0) n_t0 = MIN_N_T0;
    if (n_t0 > MAX_N_T0) n_t0 = MAX_N_T0;
    return n_t0;
}

// Device utility functions
__device__ inline float mod1(float x) {
    return x - floorf(x);
}

/**
 * Look up transit template value with linear interpolation.
 *
 * Maps transit_coord in [-1, 1] to template index, does linear
 * interpolation between adjacent samples. Returns 0 outside [-1, 1].
 *
 * s_template: shared memory pointer to template array
 * n_template: number of template samples
 * transit_coord: position within transit, [-1, 1]
 */
__device__ float lookup_template(const float* s_template, int n_template,
                                  float transit_coord)
{
    if (transit_coord < -1.0f || transit_coord > 1.0f)
        return 0.0f;

    // Map [-1, 1] to [0, n_template - 1]
    float idx_f = (transit_coord + 1.0f) * 0.5f * (float)(n_template - 1);

    int idx0 = (int)floorf(idx_f);
    int idx1 = idx0 + 1;

    // Clamp
    if (idx0 < 0) idx0 = 0;
    if (idx1 >= n_template) idx1 = n_template - 1;
    if (idx0 >= n_template) idx0 = n_template - 1;

    float frac = idx_f - floorf(idx_f);

    return s_template[idx0] * (1.0f - frac) + s_template[idx1] * frac;
}

/**
 * Calculate optimal transit depth using weighted least squares
 * with limb-darkened transit template.
 */
__device__ float calculate_optimal_depth(
    const float* y_sh,
    const float* dy_sh,
    const float* phases_sh,
    const float* s_template,
    int n_template,
    float duration_phase,
    float t0_phase,
    int ndata)
{
    float numerator = 0.0f;
    float denominator = 0.0f;

    float half_dur = duration_phase * 0.5f;

    for (int i = 0; i < ndata; i++) {
        float phase_rel = mod1(phases_sh[i] - t0_phase + 0.5f) - 0.5f;

        if (fabsf(phase_rel) < half_dur) {
            float transit_coord = phase_rel / half_dur;
            float template_val = lookup_template(s_template, n_template, transit_coord);
            float sigma2 = dy_sh[i] * dy_sh[i] + 1e-10f;
            float y_residual = 1.0f - y_sh[i];
            numerator += y_residual * template_val / sigma2;
            denominator += template_val * template_val / sigma2;
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
 * using limb-darkened transit template.
 */
__device__ float calculate_chi2(
    const float* y_sh,
    const float* dy_sh,
    const float* phases_sh,
    const float* s_template,
    int n_template,
    float duration_phase,
    float t0_phase,
    float depth,
    int ndata)
{
    float chi2 = 0.0f;
    float half_dur = duration_phase * 0.5f;

    for (int i = 0; i < ndata; i++) {
        float phase_rel = mod1(phases_sh[i] - t0_phase + 0.5f) - 0.5f;
        float model_val;
        if (fabsf(phase_rel) < half_dur) {
            float transit_coord = phase_rel / half_dur;
            float template_val = lookup_template(s_template, n_template, transit_coord);
            model_val = 1.0f - depth * template_val;
        } else {
            model_val = 1.0f;
        }
        float residual = y_sh[i] - model_val;
        float sigma2 = dy_sh[i] * dy_sh[i] + 1e-10f;
        chi2 += (residual * residual) / sigma2;
    }

    return chi2;
}

/**
 * TLS search kernel with Keplerian duration constraints
 * Grid: (nperiods, 1, 1), Block: (BLOCK_SIZE, 1, 1)
 *
 * Shared memory layout:
 *   phases[ndata] | y_sh[ndata] | dy_sh[ndata] |
 *   template[n_template] | thread_chi2[blockDim] | thread_t0[blockDim] |
 *   thread_dur[blockDim] | thread_depth[blockDim]
 */
extern "C" __global__ void tls_search_kernel_keplerian(
    const float* __restrict__ t,
    const float* __restrict__ y,
    const float* __restrict__ dy,
    const float* __restrict__ periods,
    const float* __restrict__ qmin,
    const float* __restrict__ qmax,
    const float* __restrict__ transit_template,
    const int ndata,
    const int nperiods,
    const int n_durations,
    const int n_template,
    float* __restrict__ chi2_out,
    float* __restrict__ best_t0_out,
    float* __restrict__ best_duration_out,
    float* __restrict__ best_depth_out)
{
    extern __shared__ float shared_mem[];
    float* phases = shared_mem;
    float* y_sh = &shared_mem[ndata];
    float* dy_sh = &shared_mem[2 * ndata];
    float* s_template = &shared_mem[3 * ndata];
    float* thread_chi2 = &s_template[n_template];
    float* thread_t0 = &thread_chi2[blockDim.x];
    float* thread_duration = &thread_t0[blockDim.x];
    float* thread_depth = &thread_duration[blockDim.x];

    int period_idx = blockIdx.x;
    if (period_idx >= nperiods) return;

    // Load template from global to shared memory (once per block)
    for (int i = threadIdx.x; i < n_template; i += blockDim.x) {
        s_template[i] = transit_template[i];
    }
    __syncthreads();

    float period = periods[period_idx];
    float duration_phase_min = qmin[period_idx];
    float duration_phase_max = qmax[period_idx];

    // Phase fold
    for (int i = threadIdx.x; i < ndata; i += blockDim.x) {
        phases[i] = mod1(t[i] / period);
    }
    __syncthreads();

    // Stage y and dy in shared memory
    for (int i = threadIdx.x; i < ndata; i += blockDim.x) {
        y_sh[i] = y[i];
        dy_sh[i] = dy[i];
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

        // Duration-scaled epoch grid (see t0_grid_size above)
        int n_t0 = t0_grid_size(duration_phase);
        for (int t0_idx = threadIdx.x; t0_idx < n_t0; t0_idx += blockDim.x) {
            float t0_phase = (float)t0_idx / n_t0;
            float depth = calculate_optimal_depth(y_sh, dy_sh, phases,
                                                   s_template, n_template,
                                                   duration_phase, t0_phase, ndata);

            if (depth > 0.0f && depth < 0.5f) {
                float chi2 = calculate_chi2(y_sh, dy_sh, phases,
                                             s_template, n_template,
                                             duration_phase, t0_phase, depth, ndata);
                if (chi2 < thread_min_chi2) {
                    thread_min_chi2 = chi2;
                    thread_best_t0 = t0_phase;
                    thread_best_duration = duration;
                    thread_best_depth = depth;
                }
            }
        }
    }

    // Store per-thread results to shared memory
    thread_chi2[threadIdx.x] = thread_min_chi2;
    thread_t0[threadIdx.x] = thread_best_t0;
    thread_duration[threadIdx.x] = thread_best_duration;
    thread_depth[threadIdx.x] = thread_best_depth;
    __syncthreads();

    // Block reduction down to warp size
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

    // Final warp reduction using shuffle (no sync needed)
    if (threadIdx.x < WARP_SIZE) {
        float val_chi2 = thread_chi2[threadIdx.x];
        float val_t0 = thread_t0[threadIdx.x];
        float val_dur = thread_duration[threadIdx.x];
        float val_dep = thread_depth[threadIdx.x];

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_chi2 = __shfl_down_sync(0xffffffff, val_chi2, offset);
            float other_t0 = __shfl_down_sync(0xffffffff, val_t0, offset);
            float other_dur = __shfl_down_sync(0xffffffff, val_dur, offset);
            float other_dep = __shfl_down_sync(0xffffffff, val_dep, offset);

            if (other_chi2 < val_chi2) {
                val_chi2 = other_chi2;
                val_t0 = other_t0;
                val_dur = other_dur;
                val_dep = other_dep;
            }
        }

        if (threadIdx.x == 0) {
            thread_chi2[0] = val_chi2;
            thread_t0[0] = val_t0;
            thread_duration[0] = val_dur;
            thread_depth[0] = val_dep;
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
 * TLS search kernel (standard, fixed duration range)
 * Grid: (nperiods, 1, 1), Block: (BLOCK_SIZE, 1, 1)
 *
 * Shared memory layout:
 *   phases[ndata] | y_sh[ndata] | dy_sh[ndata] |
 *   template[n_template] | thread_chi2[blockDim] | thread_t0[blockDim] |
 *   thread_dur[blockDim] | thread_depth[blockDim]
 */
extern "C" __global__ void tls_search_kernel(
    const float* __restrict__ t,
    const float* __restrict__ y,
    const float* __restrict__ dy,
    const float* __restrict__ periods,
    const float* __restrict__ transit_template,
    const int ndata,
    const int nperiods,
    const int n_template,
    float* __restrict__ chi2_out,
    float* __restrict__ best_t0_out,
    float* __restrict__ best_duration_out,
    float* __restrict__ best_depth_out)
{
    extern __shared__ float shared_mem[];
    float* phases = shared_mem;
    float* y_sh = &shared_mem[ndata];
    float* dy_sh = &shared_mem[2 * ndata];
    float* s_template = &shared_mem[3 * ndata];
    float* thread_chi2 = &s_template[n_template];
    float* thread_t0 = &thread_chi2[blockDim.x];
    float* thread_duration = &thread_t0[blockDim.x];
    float* thread_depth = &thread_duration[blockDim.x];

    int period_idx = blockIdx.x;
    if (period_idx >= nperiods) return;

    // Load template from global to shared memory (once per block)
    for (int i = threadIdx.x; i < n_template; i += blockDim.x) {
        s_template[i] = transit_template[i];
    }
    __syncthreads();

    float period = periods[period_idx];

    // Phase fold
    for (int i = threadIdx.x; i < ndata; i += blockDim.x) {
        phases[i] = mod1(t[i] / period);
    }
    __syncthreads();

    // Stage y and dy in shared memory
    for (int i = threadIdx.x; i < ndata; i += blockDim.x) {
        y_sh[i] = y[i];
        dy_sh[i] = dy[i];
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

        // Duration-scaled epoch grid (see t0_grid_size above)
        int n_t0 = t0_grid_size(duration_phase);
        for (int t0_idx = threadIdx.x; t0_idx < n_t0; t0_idx += blockDim.x) {
            float t0_phase = (float)t0_idx / n_t0;
            float depth = calculate_optimal_depth(y_sh, dy_sh, phases,
                                                   s_template, n_template,
                                                   duration_phase, t0_phase, ndata);

            if (depth > 0.0f && depth < 0.5f) {
                float chi2 = calculate_chi2(y_sh, dy_sh, phases,
                                             s_template, n_template,
                                             duration_phase, t0_phase, depth, ndata);
                if (chi2 < thread_min_chi2) {
                    thread_min_chi2 = chi2;
                    thread_best_t0 = t0_phase;
                    thread_best_duration = duration;
                    thread_best_depth = depth;
                }
            }
        }
    }

    // Store per-thread results to shared memory
    thread_chi2[threadIdx.x] = thread_min_chi2;
    thread_t0[threadIdx.x] = thread_best_t0;
    thread_duration[threadIdx.x] = thread_best_duration;
    thread_depth[threadIdx.x] = thread_best_depth;
    __syncthreads();

    // Block reduction down to warp size
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

    // Final warp reduction using shuffle (no sync needed)
    if (threadIdx.x < WARP_SIZE) {
        float val_chi2 = thread_chi2[threadIdx.x];
        float val_t0 = thread_t0[threadIdx.x];
        float val_dur = thread_duration[threadIdx.x];
        float val_dep = thread_depth[threadIdx.x];

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_chi2 = __shfl_down_sync(0xffffffff, val_chi2, offset);
            float other_t0 = __shfl_down_sync(0xffffffff, val_t0, offset);
            float other_dur = __shfl_down_sync(0xffffffff, val_dur, offset);
            float other_dep = __shfl_down_sync(0xffffffff, val_dep, offset);

            if (other_chi2 < val_chi2) {
                val_chi2 = other_chi2;
                val_t0 = other_t0;
                val_dur = other_dur;
                val_dep = other_dep;
            }
        }

        if (threadIdx.x == 0) {
            thread_chi2[0] = val_chi2;
            thread_t0[0] = val_t0;
            thread_duration[0] = val_dur;
            thread_depth[0] = val_dep;
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
