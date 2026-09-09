#include "vn.h"

#include <algorithm>
#include <thread>
#include <iostream>

#include "cuda_runtime.h"
#include "math.h"

#include "errchk.cuh"

//
// ViterbiNarrowband Function Definitions
//

ViterbiNarrowband::ViterbiNarrowband(size_t n_phase,
                                     size_t n_mag,
                                     size_t p_overlap,
                                     size_t m_overlap,
                                     size_t bw,
                                     size_t mg) {
    num_phase_bins = n_phase;
    num_mag_bins = n_mag;
    num_phase_overlap = p_overlap;
    num_mag_overlap = m_overlap;
    bandwidth = bw;
    margin = mg;
    phase_bin_size = 1.0f / static_cast<float>(n_phase);
    mag_bin_size = 1.0f / static_cast<float>(n_mag);
}

__host__ __device__ size_t ViterbiNarrowband::NumBins() const {
    return num_phase_bins * num_mag_bins;
}

__host__ __device__ size_t ViterbiNarrowband::NumPhaseBins() const {
    return num_phase_bins;
}

__host__ __device__ size_t ViterbiNarrowband::NumMagBins() const {
    return num_mag_bins;
}

__host__ __device__ size_t ViterbiNarrowband::NumPhaseBinOverlap() const {
    return num_phase_overlap;
}

__host__ __device__ size_t ViterbiNarrowband::NumMagBinOverlap() const {
    return num_mag_overlap;
}

__host__ __device__ size_t ViterbiNarrowband::Bandwidth() const {
    return bandwidth;
}

__host__ __device__ size_t ViterbiNarrowband::Margin() const {
    return margin;
}

__host__ __device__ float ViterbiNarrowband::PhaseBinSize() const {
    return phase_bin_size;
}

__host__ __device__ float ViterbiNarrowband::MagBinSize() const {
    return mag_bin_size;
}

__host__ __device__ size_t ViterbiNarrowband::PhaseBin(float phase_val) const {
    return static_cast<size_t>(phase_val / phase_bin_size);
}

__host__ __device__ size_t ViterbiNarrowband::MagBin(float mag_val) const {
    return static_cast<size_t>(mag_val / mag_bin_size);
}

__host__ __device__ size_t ViterbiNarrowband::BinIndex(size_t phase_bin,
                                                       size_t mag_bin) const {
    return phase_bin * num_mag_bins + mag_bin;
}

//
// CUDA Kernels
//

// Maximum supported histogram dimensions.
#define VN_MAX_PHASE_BINS 64
#define VN_MAX_MAG_BINS   32

/**
 * Fused FoldBin + Viterbi Narrowband kernel.
 *
 * One block per (period, pdt) pair.  All threads cooperate on folding &
 * binning, then the first warp runs the circular Viterbi via __shfl_sync.
 *
 * Shared memory layout (all reinterpreted on the same base):
 *   [0 .. n_bins)             uint32_t histogram counts
 *   [0 .. n_bins)             float    normalized histogram (reuse same space)
 *   [n_bins .. 2*n_bins)      float    log-emission table
 *
 * Total shared: 2 * n_bins * sizeof(float)   (since sizeof(uint32_t)==4)
 */
__global__ void VNFusedKernel(const float* __restrict__ times,
                              const float* __restrict__ mags,
                              const size_t length,
                              const float* __restrict__ periods,
                              const float* __restrict__ period_dts,
                              const ViterbiNarrowband h_params,
                              float* __restrict__ vn_vals) {
    const size_t n_phase = h_params.NumPhaseBins();
    const size_t n_mag = h_params.NumMagBins();
    const size_t n_bins = n_phase * n_mag;
    const size_t bw = h_params.Bandwidth();
    const size_t mg = h_params.Margin();

    const size_t period_idx = blockIdx.x;
    const size_t pdt_idx = blockIdx.y;
    const float period = periods[period_idx];
    const float period_dt = period_dts[pdt_idx];
    const float pdt_corr = (period_dt / period) / 2.0f;

    // Shared memory: first half = histogram (uint32 then float), second half = log-emit
    extern __shared__ char sh_raw[];
    uint32_t* sh_hist_u = reinterpret_cast<uint32_t*>(sh_raw);
    float*    sh_hist_f = reinterpret_cast<float*>(sh_raw);
    float*    sh_emit   = reinterpret_cast<float*>(sh_raw) + n_bins;

    // ---- Phase 1: Cooperative fold & bin (all threads) ----
    for (size_t i = threadIdx.x; i < n_bins; i += blockDim.x) {
        sh_hist_u[i] = 0;
    }
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < length; idx += blockDim.x) {
        float t = times[idx];
        double t_corr_d = (double)t - (double)pdt_corr * (double)t * (double)t;
        double ratio_d = t_corr_d / (double)period;
        float folded = fabsf((float)(ratio_d - floor(ratio_d)));

        size_t phase_bin = h_params.PhaseBin(folded);
        size_t mag_bin = h_params.MagBin(mags[idx]);

        for (size_t i = 0; i < h_params.NumPhaseBinOverlap(); i++) {
            for (size_t j = 0; j < h_params.NumMagBinOverlap(); j++) {
                size_t bin_idx =
                    h_params.BinIndex((phase_bin + i) % n_phase,
                                      (mag_bin + j) % n_mag);
                atomicAdd(&sh_hist_u[bin_idx], 1);
            }
        }
    }
    __syncthreads();

    // ---- Phase 2: Normalize histogram + compute log-emissions (all threads) ----
    const float eps = 1e-10f;
    float div_f = static_cast<float>(
        length * h_params.NumPhaseBinOverlap() * h_params.NumMagBinOverlap());

    for (size_t i = threadIdx.x; i < n_bins; i += blockDim.x) {
        float h = static_cast<float>(sh_hist_u[i]) / div_f;
        sh_hist_f[i] = h;                   // normalized histogram
        sh_emit[i]   = logf(h + eps);       // log-emission
    }
    __syncthreads();

    // ---- Phase 3: Warp-cooperative circular Viterbi (first warp only) ----
    //
    // IMPORTANT: __shfl_sync(FULL_MASK, ...) requires ALL 32 threads in the
    // warp to execute the instruction.  Therefore all shuffles are placed
    // outside of `if (active)` guards.  Inactive threads (m >= n_mag) carry
    // v = -1e30f which never wins a max comparison.
    //
    const unsigned FULL_MASK = 0xFFFFFFFF;
    const unsigned tid = threadIdx.x;
    const unsigned m = tid;  // each thread in first warp handles one mag bin

    if (tid < 32) {
        const bool active = (m < (unsigned)n_mag);

        // === Pass A: Find best starting state (no backpointers) ===
        float best_start_score = -1e30f;
        unsigned best_start_state = 0;

        for (unsigned start_state = 0; start_state < (unsigned)n_mag; start_state++) {
            float v = (active && m == start_state) ? sh_emit[m] : -1e30f;

            for (unsigned phi = 1; phi < (unsigned)n_phase; phi++) {
                float best_prev = -1e30f;
                for (unsigned d = 0; d <= 2 * (unsigned)bw; d++) {
                    int src = (int)m - (int)bw + (int)d;
                    // All 32 threads execute shuffle; result discarded for bad src
                    float neighbor = __shfl_sync(FULL_MASK, v, src);
                    if (active && src >= 0 && src < (int)n_mag && neighbor > best_prev) {
                        best_prev = neighbor;
                    }
                }
                if (active) {
                    v = sh_emit[phi * (unsigned)n_mag + m] + best_prev;
                }
            }

            // Circular constraint: read end score from thread start_state
            float end_score = __shfl_sync(FULL_MASK, v, (int)start_state);
            if (tid == 0 && end_score > best_start_score) {
                best_start_score = end_score;
                best_start_state = start_state;
            }
            // Broadcast running best to all threads
            best_start_state = __shfl_sync(FULL_MASK, best_start_state, 0);
        }

        // === Pass B: Retrace winner with backpointers ===
        unsigned my_back[VN_MAX_PHASE_BINS];
        float v = (active && m == best_start_state) ? sh_emit[m] : -1e30f;

        for (unsigned phi = 1; phi < (unsigned)n_phase; phi++) {
            float best_prev = -1e30f;
            unsigned best_prev_m = 0;
            for (unsigned d = 0; d <= 2 * (unsigned)bw; d++) {
                int src = (int)m - (int)bw + (int)d;
                float neighbor = __shfl_sync(FULL_MASK, v, src);
                if (active && src >= 0 && src < (int)n_mag && neighbor > best_prev) {
                    best_prev = neighbor;
                    best_prev_m = (unsigned)src;
                }
            }
            if (active) {
                v = sh_emit[phi * (unsigned)n_mag + m] + best_prev;
            }
            my_back[phi] = best_prev_m;
        }

        // Backtrace: all threads maintain cur in lockstep via shuffle
        unsigned path[VN_MAX_PHASE_BINS];
        unsigned cur = best_start_state;
        path[(unsigned)n_phase - 1] = cur;
        for (int phi = (int)n_phase - 2; phi >= 0; phi--) {
            unsigned my_bp = my_back[phi + 1];
            unsigned bp = __shfl_sync(FULL_MASK, my_bp, (int)cur);
            cur = bp;
            path[phi] = cur;
        }

        // ---- Phase 4: Concentration ratio (thread 0) ----
        if (tid == 0) {
            float total_mass = 0.0f;
            for (unsigned i = 0; i < (unsigned)n_bins; i++) {
                total_mass += sh_hist_f[i];
            }

            if (total_mass <= 0.0f) {
                vn_vals[period_idx * gridDim.y + pdt_idx] = 0.0f;
            } else {
                float path_mass = 0.0f;
                for (unsigned phi = 0; phi < (unsigned)n_phase; phi++) {
                    unsigned path_m = path[phi];
                    unsigned lo = (path_m >= (unsigned)mg) ? path_m - (unsigned)mg : 0;
                    unsigned hi = (path_m + (unsigned)mg < (unsigned)n_mag)
                                      ? path_m + (unsigned)mg
                                      : (unsigned)n_mag - 1;
                    for (unsigned mm = lo; mm <= hi; mm++) {
                        path_mass += sh_hist_f[phi * (unsigned)n_mag + mm];
                    }
                }
                vn_vals[period_idx * gridDim.y + pdt_idx] = path_mass / total_mass;
            }
        }
    }
}

//
// Batched Wrapper
//

struct VNDeviceState {
    float* dev_periods;
    float* dev_period_dts;
    cudaStream_t streams[4];
    float* dev_times_buf[4];
    float* dev_mags_buf[4];
    float* dev_vn_buf[4];
};

void ViterbiNarrowband::CalcVNValsBatched(
    const std::vector<float*>& times,
    const std::vector<float*>& mags,
    const std::vector<size_t>& lengths,
    const float* periods,
    const float* period_dts,
    const size_t num_periods,
    const size_t num_p_dts,
    float* vn_out) const {

    size_t num_curves = lengths.size();
    if (num_curves == 0) return;

    size_t num_hists = num_periods * num_p_dts;
    size_t vn_out_size = num_hists * sizeof(float);

    // Determine number of GPUs
    int num_devices = 1;
    if (cudaGetDeviceCount(&num_devices) != cudaSuccess) {
        num_devices = 1;
    }
    if (num_devices > (int)num_curves) {
        num_devices = (int)num_curves;
    }

    // Fused kernel: one block per (period, pdt) pair
    const size_t num_threads = 256;
    // Shared: histogram (n_bins floats) + log-emissions (n_bins floats)
    const size_t shared_bytes = 2 * NumBins() * sizeof(float);
    const dim3 grid_dim = dim3(num_periods, num_p_dts);

    const int NUM_STREAMS = 4;

    // Partition curves across devices
    size_t base_count = num_curves / num_devices;
    size_t remainder = num_curves % num_devices;

    std::vector<VNDeviceState> dev_state(num_devices);

    std::vector<std::thread> dev_threads;
    for (int d = 0; d < num_devices; d++) {
        dev_threads.emplace_back([&, d]() {
        gpuErrchk(cudaSetDevice(d));

        size_t start = d * base_count + std::min((size_t)d, remainder);
        size_t count = base_count + ((size_t)d < remainder ? 1 : 0);

        size_t dev_max_length = 0;
        for (size_t j = start; j < start + count; j++) {
            if (lengths[j] > dev_max_length) dev_max_length = lengths[j];
        }
        size_t buffer_bytes = sizeof(float) * dev_max_length;

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

        for (int s = 0; s < NUM_STREAMS; s++) {
            gpuErrchk(cudaStreamCreate(&dev_state[d].streams[s]));
            gpuErrchk(
                cudaMalloc(&dev_state[d].dev_times_buf[s], buffer_bytes));
            gpuErrchk(
                cudaMalloc(&dev_state[d].dev_mags_buf[s], buffer_bytes));
            gpuErrchk(
                cudaMalloc(&dev_state[d].dev_vn_buf[s], vn_out_size));
        }

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

            VNFusedKernel<<<grid_dim, num_threads, shared_bytes, stream>>>(
                dev_state[d].dev_times_buf[s],
                dev_state[d].dev_mags_buf[s], lengths[i],
                dev_state[d].dev_periods, dev_state[d].dev_period_dts,
                *this, dev_state[d].dev_vn_buf[s]);

            gpuErrchk(cudaMemcpyAsync(&vn_out[i * num_hists],
                                      dev_state[d].dev_vn_buf[s],
                                      vn_out_size, cudaMemcpyDeviceToHost,
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
            gpuErrchk(cudaFree(dev_state[d].dev_mags_buf[s]));
            gpuErrchk(cudaFree(dev_state[d].dev_vn_buf[s]));
            gpuErrchk(cudaStreamDestroy(dev_state[d].streams[s]));
        }
    }
}

float* ViterbiNarrowband::CalcVNValsBatched(
    const std::vector<float*>& times,
    const std::vector<float*>& mags,
    const std::vector<size_t>& lengths,
    const float* periods,
    const float* period_dts,
    const size_t num_periods,
    const size_t num_p_dts) const {

    size_t vn_out_size = num_periods * num_p_dts * sizeof(float);
    size_t vn_size_total = vn_out_size * lengths.size();

    float* vn_out = (float*)malloc(vn_size_total);

    CalcVNValsBatched(times, mags, lengths, periods, period_dts, num_periods,
                      num_p_dts, vn_out);

    return vn_out;
}
