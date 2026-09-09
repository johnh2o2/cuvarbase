// Multi-Harmonic Fourier (MHF) periodogram — CUDA implementation.
//
// For each trial (period, period_dt), fits Fourier models with k = 0..max_harmonics
// terms using weighted least-squares, then uses BIC to select the best model.
// Score = ΔBIC = BIC_flat - BIC_best (higher = more periodic).

#include "mhf.h"

#include <algorithm>
#include <cmath>
#include <thread>

#include "cuda_runtime.h"
#include "math.h"

#include "errchk.cuh"

//
// Simple Method Definitions
//

MultiHarmonicFourier::MultiHarmonicFourier(size_t max_k) {
    max_harmonics = max_k;
    if (max_harmonics > 5) max_harmonics = 5;
    if (max_harmonics < 1) max_harmonics = 1;
}

__host__ __device__ size_t MultiHarmonicFourier::MaxHarmonics() const {
    return max_harmonics;
}

__host__ __device__ size_t MultiHarmonicFourier::NumParams() const {
    return 2 + 2 * max_harmonics;
}

//
// CUDA Kernel
//

// Tile size for shared-memory data loading
#define MHF_TILE_SIZE 256

// Maximum number of model parameters (offset + slope + 5 cos/sin pairs)
#define MHF_MAX_PARAMS 12

// One block per (period, period_dt) pair.
//
// Algorithm:
//   1. Thread 0 computes tmin by scanning the time array.
//   2. All threads cooperatively accumulate the normal equations
//      (A = X^T W X, b = X^T W y, yTWy = y^T W y) via shared-memory atomicAdd.
//   3. Thread 0 copies to local double arrays, then for each k = 0..max_harmonics:
//      - Extracts the (2+2k) × (2+2k) sub-system
//      - Cholesky solves in double precision
//      - Computes chi2_k = yTWy - beta_k^T * b_k
//      - Computes BIC_k
//   4. Outputs ΔBIC = BIC_flat - BIC_best.
//
// Shared memory layout:
//   [ata: np² floats] [atb: np floats] [yTWy: 1 float] [tmin: 1 float]
//   [tile_times: TILE floats] [tile_mags: TILE floats] [tile_ivar: TILE floats]
//
__global__ void MHFKernel(const float* __restrict__ times,
                           const float* __restrict__ mags,
                           const float* __restrict__ ivar,
                           const size_t length,
                           const float* __restrict__ periods,
                           const float* __restrict__ period_dts,
                           const size_t num_periods,
                           const size_t num_period_dts,
                           const size_t max_harmonics,
                           const size_t np_full,
                           float* __restrict__ mhf_out) {
    extern __shared__ float sh[];

    const size_t np2 = np_full * np_full;

    // Shared memory regions
    float* sh_ata   = sh;                                   // np² floats
    float* sh_atb   = sh + np2;                             // np floats
    float* sh_yTWy  = sh + np2 + np_full;                   // 1 float
    float* sh_tmin  = sh_yTWy + 1;                          // 1 float
    float* sh_times = sh_tmin + 1;                          // TILE_SIZE floats
    float* sh_mags  = sh_times + MHF_TILE_SIZE;             // TILE_SIZE floats
    float* sh_ivar  = sh_mags + MHF_TILE_SIZE;              // TILE_SIZE floats

    const size_t period_idx = blockIdx.x;
    const size_t pdt_idx = blockIdx.y;

    if (period_idx >= num_periods || pdt_idx >= num_period_dts) {
        return;
    }

    const float period = periods[period_idx];
    const float period_dt = period_dts[pdt_idx];
    const float pdt_corr = (period_dt / period) / 2.0f;

    // --- Step 1: Zero shared accumulators and compute tmin ---

    const size_t total_shared = np2 + np_full + 2;  // ata + atb + yTWy + tmin
    for (size_t k = threadIdx.x; k < total_shared; k += blockDim.x) {
        sh[k] = 0.0f;
    }
    __syncthreads();

    // Thread 0 computes tmin (n is small, ~100-500 points)
    if (threadIdx.x == 0) {
        float tmin_val = times[0];
        for (size_t i = 1; i < length; i++) {
            if (times[i] < tmin_val) tmin_val = times[i];
        }
        *sh_tmin = tmin_val;
    }
    __syncthreads();

    const float tmin = *sh_tmin;

    // --- Step 2: Accumulate normal equations via shared-memory atomicAdd ---

    for (size_t tile_start = 0; tile_start < length;
         tile_start += MHF_TILE_SIZE) {
        size_t tile_end = tile_start + MHF_TILE_SIZE;
        if (tile_end > length) tile_end = length;
        size_t tile_len = tile_end - tile_start;

        // Cooperatively load tile into shared memory
        for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
            sh_times[i] = times[tile_start + i];
            sh_mags[i]  = mags[tile_start + i];
            sh_ivar[i]  = ivar[tile_start + i];
        }
        __syncthreads();

        // Each thread processes some data points from the tile
        for (size_t i = threadIdx.x; i < tile_len; i += blockDim.x) {
            float t = sh_times[i];
            float y = sh_mags[i];
            float w = sh_ivar[i];

            // Phase fold (promoted to f64 for precision over long baselines)
            double t_corr_d = (double)t - (double)pdt_corr * (double)t * (double)t;
            double ratio_d = t_corr_d / (double)period;
            float phase = fabsf((float)(ratio_d - floor(ratio_d)));
            float phi = 2.0f * (float)M_PI * phase;

            // Build design matrix row
            float x_row[MHF_MAX_PARAMS];
            x_row[0] = 1.0f;               // constant
            x_row[1] = t - tmin;            // slope

            if (max_harmonics >= 1) {
                float s1, c1;
                sincosf(phi, &s1, &c1);
                x_row[2] = c1;
                x_row[3] = s1;

                // Trig recurrence for higher harmonics
                float c_prev2 = 1.0f, s_prev2 = 0.0f;
                float c_prev1 = c1,   s_prev1 = s1;
                float two_c1 = 2.0f * c1;

                for (size_t harm = 2; harm <= max_harmonics; harm++) {
                    float c_k = two_c1 * c_prev1 - c_prev2;
                    float s_k = two_c1 * s_prev1 - s_prev2;
                    x_row[2 * harm]     = c_k;
                    x_row[2 * harm + 1] = s_k;
                    c_prev2 = c_prev1;  s_prev2 = s_prev1;
                    c_prev1 = c_k;      s_prev1 = s_k;
                }
            }

            // Accumulate A = X^T W X (lower triangle only)
            for (size_t j = 0; j < np_full; j++) {
                float w_xj = w * x_row[j];
                for (size_t k = 0; k <= j; k++) {
                    atomicAdd(&sh_ata[j * np_full + k], w_xj * x_row[k]);
                }
            }

            // Accumulate b = X^T W y
            float wy = w * y;
            for (size_t j = 0; j < np_full; j++) {
                atomicAdd(&sh_atb[j], wy * x_row[j]);
            }

            // Accumulate y^T W y
            atomicAdd(sh_yTWy, w * y * y);
        }
        __syncthreads();
    }

    // --- Step 3: Thread 0 performs Cholesky solve + BIC selection ---

    if (threadIdx.x == 0) {
        double yTWy = (double)(*sh_yTWy);

        // Copy normal equations to local double arrays for numerical stability
        double ata_full[MHF_MAX_PARAMS * MHF_MAX_PARAMS];
        double atb_save[MHF_MAX_PARAMS];  // saved copy (Cholesky modifies in place)

        for (size_t j = 0; j < np_full; j++) {
            atb_save[j] = (double)sh_atb[j];
            for (size_t k = 0; k <= j; k++) {
                double val = (double)sh_ata[j * np_full + k];
                ata_full[j * np_full + k] = val;
                ata_full[k * np_full + j] = val;  // symmetrize (for sub-extraction)
            }
        }

        float n_f = (float)length;
        float ln_n = logf(n_f);

        float bic_flat = 1e30f;
        float best_bic = 1e30f;

        for (size_t km = 0; km <= max_harmonics; km++) {
            size_t np_k = 2 + 2 * km;

            if (length <= np_k) continue;

            // Extract sub-system
            double sub_a[MHF_MAX_PARAMS * MHF_MAX_PARAMS];
            double sub_b[MHF_MAX_PARAMS];

            for (size_t row = 0; row < np_k; row++) {
                sub_b[row] = atb_save[row];
                for (size_t col = 0; col < np_k; col++) {
                    sub_a[row * np_k + col] = ata_full[row * np_full + col];
                }
            }

            // Cholesky factorization: A = L L^T (in-place, lower triangle)
            bool ok = true;
            for (size_t ci = 0; ci < np_k && ok; ci++) {
                for (size_t cj = 0; cj <= ci; cj++) {
                    double s = sub_a[ci * np_k + cj];
                    for (size_t ck = 0; ck < cj; ck++) {
                        s -= sub_a[ci * np_k + ck] * sub_a[cj * np_k + ck];
                    }
                    if (ci == cj) {
                        if (s <= 0.0) { ok = false; break; }
                        sub_a[ci * np_k + cj] = sqrt(s);
                    } else {
                        sub_a[ci * np_k + cj] = s / sub_a[cj * np_k + cj];
                    }
                }
            }
            if (!ok) continue;

            // Forward substitution: L z = b
            for (size_t ci = 0; ci < np_k; ci++) {
                double s = sub_b[ci];
                for (size_t ck = 0; ck < ci; ck++) {
                    s -= sub_a[ci * np_k + ck] * sub_b[ck];
                }
                sub_b[ci] = s / sub_a[ci * np_k + ci];
            }

            // Back substitution: L^T x = z
            for (int ci = (int)np_k - 1; ci >= 0; ci--) {
                double s = sub_b[ci];
                for (size_t ck = ci + 1; ck < np_k; ck++) {
                    s -= sub_a[ck * np_k + ci] * sub_b[ck];
                }
                sub_b[ci] = s / sub_a[ci * np_k + ci];
            }

            // chi2 = y^T W y - beta^T b
            double chi2 = yTWy;
            for (size_t j = 0; j < np_k; j++) {
                chi2 -= sub_b[j] * atb_save[j];
            }
            if (chi2 < 0.0) chi2 = 0.0;

            float bic = (float)chi2 + ln_n * (float)np_k;

            if (km == 0) bic_flat = bic;
            if (bic < best_bic) best_bic = bic;
        }

        float delta_bic = bic_flat - best_bic;
        if (delta_bic < 0.0f) delta_bic = 0.0f;

        mhf_out[period_idx * num_period_dts + pdt_idx] = delta_bic;
    }
}

//
// Wrapper Functions
//

// Per-device state for multi-GPU batched processing
struct MHFDeviceState {
    float* dev_periods;
    float* dev_period_dts;
    cudaStream_t streams[4];
    float* dev_times_buf[4];
    float* dev_mags_buf[4];
    float* dev_ivar_buf[4];
    float* dev_mhf_buf[4];
    float* h_ivar[4];
};

void MultiHarmonicFourier::CalcMHFBatched(
    const std::vector<float*>& times,
    const std::vector<float*>& mags,
    const std::vector<float*>& errs,
    const std::vector<size_t>& lengths,
    const float* periods,
    const float* period_dts,
    const size_t num_periods,
    const size_t num_p_dts,
    float* mhf_out) const {

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

    // Kernel launch configuration
    const size_t num_threads = 256;
    const size_t np_full = NumParams();
    const size_t np2 = np_full * np_full;

    // Shared memory: ata(np²) + atb(np) + yTWy(1) + tmin(1) + 3×TILE_SIZE
    const size_t shared_bytes =
        (np2 + np_full + 2 + 3 * MHF_TILE_SIZE) * sizeof(float);
    const dim3 grid_dim = dim3(num_periods, num_p_dts);

    const int NUM_STREAMS = 4;

    // Partition curves across devices
    size_t base_count = num_curves / num_devices;
    size_t remainder = num_curves % num_devices;

    std::vector<MHFDeviceState> dev_state(num_devices);

    // Phase 1: Allocate and enqueue on each device
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
                cudaMalloc(&dev_state[d].dev_ivar_buf[s], buffer_bytes));
            gpuErrchk(
                cudaMalloc(&dev_state[d].dev_mhf_buf[s], per_out_size));
            dev_state[d].h_ivar[s] = (float*)malloc(buffer_bytes);
        }

        // Enqueue work for this device's curves
        for (size_t j = 0; j < count; j++) {
            size_t i = start + j;
            int s = j % NUM_STREAMS;
            cudaStream_t stream = dev_state[d].streams[s];

            // Skip degenerate curves
            if (lengths[i] < 3) {
                memset(&mhf_out[i * per_points], 0, per_out_size);
                continue;
            }

            // Precompute inverse variance on CPU
            float* h_iv = dev_state[d].h_ivar[s];
            for (size_t k = 0; k < lengths[i]; k++) {
                float e = errs[i][k];
                h_iv[k] = 1.0f / (e * e);
            }

            // Async copy to device
            const size_t curve_bytes = lengths[i] * sizeof(float);
            gpuErrchk(cudaMemcpyAsync(dev_state[d].dev_times_buf[s],
                                      times[i], curve_bytes,
                                      cudaMemcpyHostToDevice, stream));
            gpuErrchk(cudaMemcpyAsync(dev_state[d].dev_mags_buf[s],
                                      mags[i], curve_bytes,
                                      cudaMemcpyHostToDevice, stream));
            gpuErrchk(cudaMemcpyAsync(dev_state[d].dev_ivar_buf[s],
                                      h_iv, curve_bytes,
                                      cudaMemcpyHostToDevice, stream));

            gpuErrchk(cudaMemsetAsync(dev_state[d].dev_mhf_buf[s], 0,
                                      per_out_size, stream));

            MHFKernel<<<grid_dim, num_threads, shared_bytes, stream>>>(
                dev_state[d].dev_times_buf[s],
                dev_state[d].dev_mags_buf[s],
                dev_state[d].dev_ivar_buf[s],
                lengths[i],
                dev_state[d].dev_periods,
                dev_state[d].dev_period_dts,
                num_periods, num_p_dts,
                max_harmonics, np_full,
                dev_state[d].dev_mhf_buf[s]);

            gpuErrchk(cudaMemcpyAsync(&mhf_out[i * per_points],
                                      dev_state[d].dev_mhf_buf[s],
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
            gpuErrchk(cudaFree(dev_state[d].dev_ivar_buf[s]));
            gpuErrchk(cudaFree(dev_state[d].dev_mhf_buf[s]));
            gpuErrchk(cudaStreamDestroy(dev_state[d].streams[s]));
            free(dev_state[d].h_ivar[s]);
        }
    }
}

float* MultiHarmonicFourier::CalcMHFBatched(
    const std::vector<float*>& times,
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

    float* mhf_out = (float*)malloc(per_size_total);

    CalcMHFBatched(times, mags, errs, lengths, periods, period_dts,
                   num_periods, num_p_dts, mhf_out);

    return mhf_out;
}
