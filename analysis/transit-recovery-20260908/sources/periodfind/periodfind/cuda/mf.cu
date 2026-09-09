// Matched Filter morphology scoring — CUDA implementation.
//
// Phase-folds light curves, bins into profiles, computes R² and template
// correlations via circular cross-correlation.  Combined score =
// max_corr × R² × coverage serves as the periodogram statistic.

#include "mf.h"

#include <algorithm>
#include <cmath>
#include <thread>

#include "cuda_runtime.h"
#include "math.h"

#include "errchk.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// =========================================================================
// Host-side template generation
// =========================================================================

static void normalize_template(float* t, size_t n) {
    float mean = 0.0f;
    for (size_t i = 0; i < n; i++) mean += t[i];
    mean /= (float)n;
    for (size_t i = 0; i < n; i++) t[i] -= mean;
    float var = 0.0f;
    for (size_t i = 0; i < n; i++) var += t[i] * t[i];
    var /= (float)n;
    float std = sqrtf(var);
    if (std > 1e-12f) {
        for (size_t i = 0; i < n; i++) t[i] /= std;
    }
}

void MatchedFilter::GenerateTemplates() {
    num_templates = 0;
    size_t n = num_bins;

    // Sawtooth templates: 5 rise fractions
    float rise_fracs[] = {0.10f, 0.20f, 0.35f, 0.50f, 0.65f};
    for (int r = 0; r < 5; r++) {
        float rf = rise_fracs[r];
        float* t = &templates[num_templates * n];
        for (size_t k = 0; k < n; k++) {
            float phase = ((float)k + 0.5f) / (float)n;
            t[k] = (phase < rf) ? (phase / rf) : (1.0f - (phase - rf) / (1.0f - rf));
        }
        normalize_template(t, n);
        template_types[num_templates] = MF_TEMPLATE_SAWTOOTH;
        num_templates++;
    }

    // Sinusoidal template
    {
        float* t = &templates[num_templates * n];
        for (size_t k = 0; k < n; k++) {
            float phase = ((float)k + 0.5f) / (float)n;
            t[k] = sinf(2.0f * M_PI * phase);
        }
        normalize_template(t, n);
        template_types[num_templates] = MF_TEMPLATE_SINUSOIDAL;
        num_templates++;
    }

    // Eclipsing templates: 4 dip widths × 2 secondary options
    float dip_widths[] = {0.05f, 0.10f, 0.15f, 0.20f};
    for (int dw = 0; dw < 4; dw++) {
        for (int sec = 0; sec < 2; sec++) {
            float* t = &templates[num_templates * n];
            float half_dip = dip_widths[dw] / 2.0f;
            for (size_t k = 0; k < n; k++) {
                float phase = ((float)k + 0.5f) / (float)n;
                t[k] = 1.0f;
                // Primary eclipse at phase 0
                float dist_primary = (phase < 0.5f) ? phase : (1.0f - phase);
                if (dist_primary < half_dip) t[k] = 0.0f;
                // Secondary eclipse at phase 0.5 (half depth)
                if (sec == 1) {
                    float dist_secondary = fabsf(phase - 0.5f);
                    if (dist_secondary < half_dip) t[k] = 0.5f;
                }
            }
            normalize_template(t, n);
            template_types[num_templates] = MF_TEMPLATE_ECLIPSING;
            num_templates++;
        }
    }
}

// =========================================================================
// Constructor / accessors
// =========================================================================

MatchedFilter::MatchedFilter(size_t n_bins) {
    num_bins = n_bins;
    bin_size = 1.0f / static_cast<float>(n_bins);
    GenerateTemplates();
}

__host__ __device__ size_t MatchedFilter::NumBins() const {
    return num_bins;
}

__host__ __device__ size_t MatchedFilter::NumTemplates() const {
    return num_templates;
}

__host__ __device__ size_t MatchedFilter::PhaseBin(float phase_val) const {
    return static_cast<size_t>(phase_val / bin_size);
}

// =========================================================================
// CUDA Kernel
// =========================================================================

#define MF_TILE_SIZE 256
#define MF_HYBRID_THRESHOLD 2048

// Device-side template data (copied once per batch)
__constant__ float d_mf_templates[MF_MAX_TEMPLATES * MF_MAX_BINS];
__constant__ int d_mf_template_types[MF_MAX_TEMPLATES];
__constant__ int d_mf_num_templates;

__global__ void MFKernel(const float* __restrict__ times,
                         const float* __restrict__ ivar,
                         const float* __restrict__ ivar_y,
                         const float* __restrict__ ivar_y2,
                         const size_t length,
                         const float* __restrict__ periods,
                         const float* __restrict__ period_dts,
                         const size_t num_periods,
                         const size_t num_period_dts,
                         const size_t n_bins,
                         const float bin_size,
                         float* __restrict__ mf_out) {
    // Shared memory layout:
    // [sh_times | sh_ivar | sh_ivar_y | sh_ivar_y2 | sh_v_bin | sh_y_bin | sh_y2_bin]
    extern __shared__ float sh_data[];
    float* sh_times  = &sh_data[0];
    float* sh_ivar   = &sh_data[MF_TILE_SIZE];
    float* sh_ivar_y = &sh_data[2 * MF_TILE_SIZE];
    float* sh_ivar_y2 = &sh_data[3 * MF_TILE_SIZE];

    const size_t period_idx = blockIdx.x;
    const size_t pdt_idx = blockIdx.y;

    if (period_idx >= num_periods || pdt_idx >= num_period_dts)
        return;

    const float period = periods[period_idx];
    const float period_dt = period_dts[pdt_idx];
    const float pdt_corr = (period_dt / period) / 2.0f;

    // Use atomic path for moderate point counts (our typical case)
    float* sh_v_bin  = &sh_data[4 * MF_TILE_SIZE];
    float* sh_y_bin  = &sh_data[4 * MF_TILE_SIZE + n_bins];
    float* sh_y2_bin = &sh_data[4 * MF_TILE_SIZE + 2 * n_bins];

    // Cooperatively zero bins
    for (size_t k = threadIdx.x; k < n_bins; k += blockDim.x) {
        sh_v_bin[k] = 0.0f;
        sh_y_bin[k] = 0.0f;
        sh_y2_bin[k] = 0.0f;
    }
    __syncthreads();


    // Process light curve in tiles
    for (size_t tile_start = 0; tile_start < length; tile_start += MF_TILE_SIZE) {
        size_t tile_end = tile_start + MF_TILE_SIZE;
        if (tile_end > length) tile_end = length;
        size_t tile_len = tile_end - tile_start;

        // Cooperative load
        for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
            sh_times[i]   = times[tile_start + i];
            sh_ivar[i]    = ivar[tile_start + i];
            sh_ivar_y[i]  = ivar_y[tile_start + i];
            sh_ivar_y2[i] = ivar_y2[tile_start + i];
        }
        __syncthreads();

        // Accumulate into shared bins via atomicAdd
        for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
            float t = sh_times[i];
            double t_corr_d = (double)t - (double)pdt_corr * (double)t * (double)t;
            double ratio_d = t_corr_d / (double)period;
            float folded = fabsf((float)(ratio_d - floor(ratio_d)));

            size_t bin = static_cast<size_t>(folded / bin_size);
            if (bin >= n_bins) bin = n_bins - 1;

            atomicAdd(&sh_v_bin[bin],  sh_ivar[i]);
            atomicAdd(&sh_y_bin[bin],  sh_ivar_y[i]);
            atomicAdd(&sh_y2_bin[bin], sh_ivar_y2[i]);
        }
        __syncthreads();
    }

    // Thread 0 computes the full MF score
    if (threadIdx.x == 0) {
        // Build profile, compute R², coverage
        float profile[MF_MAX_BINS];
        int n_filled = 0;
        float total_wt = 0.0f;
        float total_wy = 0.0f;

        for (size_t k = 0; k < n_bins; k++) {
            if (sh_v_bin[k] > 0.0f) {
                profile[k] = sh_y_bin[k] / sh_v_bin[k];
                n_filled++;
                total_wt += sh_v_bin[k];
                total_wy += sh_y_bin[k];
            } else {
                profile[k] = 0.0f;
            }
        }

        float coverage = (float)n_filled / (float)n_bins;

        // Need at least 3 filled bins
        if (n_filled < 3) {
            mf_out[period_idx * num_period_dts + pdt_idx] = 0.0f;
            return;
        }

        // R²: 1 - SS_within / SS_total
        float grand_mean = total_wy / total_wt;
        float ss_total = 0.0f;
        float ss_within = 0.0f;

        // SS_total = sum(ivar_i * (mag_i - grand_mean)^2)
        // = sum(ivar_i * mag_i^2) - 2*grand_mean*sum(ivar_i*mag_i) + grand_mean^2*sum(ivar_i)
        // We don't have per-point access here, but we have per-bin aggregates.
        // SS_total needs global aggregates:
        float sum_wy2 = 0.0f;
        for (size_t k = 0; k < n_bins; k++) {
            sum_wy2 += sh_y2_bin[k];
        }
        ss_total = sum_wy2 - 2.0f * grand_mean * total_wy + grand_mean * grand_mean * total_wt;

        for (size_t k = 0; k < n_bins; k++) {
            if (sh_v_bin[k] > 0.0f) {
                ss_within += sh_y2_bin[k] - sh_y_bin[k] * sh_y_bin[k] / sh_v_bin[k];
            }
        }

        float r_squared = (ss_total > 0.0f) ? fmaxf(1.0f - ss_within / ss_total, 0.0f) : 0.0f;

        // Template correlation via circular cross-correlation
        // Profile statistics for Pearson correlation
        float p_sum = 0.0f;
        for (size_t k = 0; k < n_bins; k++) p_sum += profile[k];
        float p_mean = p_sum / (float)n_bins;
        float p_var = 0.0f;
        for (size_t k = 0; k < n_bins; k++) {
            float d = profile[k] - p_mean;
            p_var += d * d;
        }
        float p_std = sqrtf(p_var);
        float t_std = sqrtf((float)n_bins); // templates are unit-variance

        float max_corr = 0.0f;

        if (p_std > 1e-10f) {
            float denom = p_std * t_std;

            for (int ti = 0; ti < d_mf_num_templates; ti++) {
                const float* tmpl = &d_mf_templates[ti * n_bins];

                // Circular cross-correlation: try all shifts
                float best_r = -1.0f;
                for (size_t shift = 0; shift < n_bins; shift++) {
                    float dot = 0.0f;
                    for (size_t k = 0; k < n_bins; k++) {
                        size_t tk = (k + shift) % n_bins;
                        dot += (profile[k] - p_mean) * tmpl[tk];
                    }
                    float r = dot / denom;
                    if (r > best_r) best_r = r;
                }
                if (best_r > max_corr) max_corr = best_r;
            }
        }

        float combined = max_corr * r_squared * coverage;
        mf_out[period_idx * num_period_dts + pdt_idx] = combined;
    }
}

// =========================================================================
// Wrapper Functions
// =========================================================================

struct MFDeviceState {
    float* dev_periods;
    float* dev_period_dts;
    cudaStream_t streams[4];
    float* dev_times_buf[4];
    float* dev_ivar_buf[4];
    float* dev_ivar_y_buf[4];
    float* dev_ivar_y2_buf[4];
    float* dev_mf_buf[4];
    float* h_ivar[4];
    float* h_ivar_y[4];
    float* h_ivar_y2[4];
};

void MatchedFilter::CalcMFBatched(const std::vector<float*>& times,
                                   const std::vector<float*>& mags,
                                   const std::vector<float*>& errs,
                                   const std::vector<size_t>& lengths,
                                   const float* periods,
                                   const float* period_dts,
                                   const size_t num_periods,
                                   const size_t num_p_dts,
                                   float* mf_out) const {
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

    // Kernel config
    const size_t num_threads = 256;
    // Shared memory: 4*TILE_SIZE + 3*n_bins
    const size_t shared_bytes =
        (4 * MF_TILE_SIZE + 3 * NumBins()) * sizeof(float);
    const dim3 grid_dim = dim3(num_periods, num_p_dts);

    const int NUM_STREAMS = 4;

    // Partition curves across devices
    size_t base_count = num_curves / num_devices;
    size_t remainder = num_curves % num_devices;

    std::vector<MFDeviceState> dev_state(num_devices);

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

        // Copy templates to constant memory (once per device)
        gpuErrchk(cudaMemcpyToSymbol(d_mf_templates, templates,
                                      num_templates * num_bins * sizeof(float)));
        gpuErrchk(cudaMemcpyToSymbol(d_mf_template_types, template_types,
                                      num_templates * sizeof(int)));
        int nt = (int)num_templates;
        gpuErrchk(cudaMemcpyToSymbol(d_mf_num_templates, &nt, sizeof(int)));

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
            gpuErrchk(cudaMalloc(&dev_state[d].dev_times_buf[s], buffer_bytes));
            gpuErrchk(cudaMalloc(&dev_state[d].dev_ivar_buf[s], buffer_bytes));
            gpuErrchk(cudaMalloc(&dev_state[d].dev_ivar_y_buf[s], buffer_bytes));
            gpuErrchk(cudaMalloc(&dev_state[d].dev_ivar_y2_buf[s], buffer_bytes));
            gpuErrchk(cudaMalloc(&dev_state[d].dev_mf_buf[s], per_out_size));
            dev_state[d].h_ivar[s] = (float*)malloc(buffer_bytes);
            dev_state[d].h_ivar_y[s] = (float*)malloc(buffer_bytes);
            dev_state[d].h_ivar_y2[s] = (float*)malloc(buffer_bytes);
        }

        // Enqueue work
        for (size_t j = 0; j < count; j++) {
            size_t i = start + j;
            int s = j % NUM_STREAMS;
            cudaStream_t stream = dev_state[d].streams[s];

            // Precompute inverse variance and weighted data on CPU
            float* h_iv = dev_state[d].h_ivar[s];
            float* h_ivy = dev_state[d].h_ivar_y[s];
            float* h_ivy2 = dev_state[d].h_ivar_y2[s];
            for (size_t k = 0; k < lengths[i]; k++) {
                float e = errs[i][k];
                h_iv[k] = 1.0f / (e * e);
                h_ivy[k] = h_iv[k] * mags[i][k];
                h_ivy2[k] = h_iv[k] * mags[i][k] * mags[i][k];
            }

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
            gpuErrchk(cudaMemcpyAsync(dev_state[d].dev_ivar_y2_buf[s],
                                      h_ivy2, curve_bytes,
                                      cudaMemcpyHostToDevice, stream));

            gpuErrchk(cudaMemsetAsync(dev_state[d].dev_mf_buf[s], 0,
                                      per_out_size, stream));

            MFKernel<<<grid_dim, num_threads, shared_bytes, stream>>>(
                dev_state[d].dev_times_buf[s],
                dev_state[d].dev_ivar_buf[s],
                dev_state[d].dev_ivar_y_buf[s],
                dev_state[d].dev_ivar_y2_buf[s],
                lengths[i],
                dev_state[d].dev_periods, dev_state[d].dev_period_dts,
                num_periods, num_p_dts,
                num_bins, bin_size,
                dev_state[d].dev_mf_buf[s]);

            gpuErrchk(cudaMemcpyAsync(&mf_out[i * per_points],
                                      dev_state[d].dev_mf_buf[s],
                                      per_out_size, cudaMemcpyDeviceToHost,
                                      stream));
        }
        });
    }
    for (auto& t : dev_threads) t.join();

    // Sync and free
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
            gpuErrchk(cudaFree(dev_state[d].dev_ivar_y2_buf[s]));
            gpuErrchk(cudaFree(dev_state[d].dev_mf_buf[s]));
            gpuErrchk(cudaStreamDestroy(dev_state[d].streams[s]));
            free(dev_state[d].h_ivar[s]);
            free(dev_state[d].h_ivar_y[s]);
            free(dev_state[d].h_ivar_y2[s]);
        }
    }
}

float* MatchedFilter::CalcMFBatched(const std::vector<float*>& times,
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

    float* mf_out = (float*)malloc(per_size_total);

    CalcMFBatched(times, mags, errs, lengths, periods, period_dts, num_periods,
                  num_p_dts, mf_out);

    return mf_out;
}
