#ifndef __PF_VN_H__
#define __PF_VN_H__

#include <cstddef>
#include <cstdint>
#include <vector>

// Required to avoid errors during Python API compilation.
#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

class ViterbiNarrowband {
   private:
    size_t num_phase_bins;
    size_t num_mag_bins;
    size_t num_phase_overlap;
    size_t num_mag_overlap;
    size_t bandwidth;
    size_t margin;
    float phase_bin_size;
    float mag_bin_size;

   public:
    ViterbiNarrowband()
        : ViterbiNarrowband(20, 20, 1, 1, 2, 1) {};

    ViterbiNarrowband(size_t n_phase,
                      size_t n_mag,
                      size_t p_overlap,
                      size_t m_overlap,
                      size_t bw,
                      size_t mg);

    __host__ __device__ size_t NumBins() const;
    __host__ __device__ size_t NumPhaseBins() const;
    __host__ __device__ size_t NumMagBins() const;
    __host__ __device__ size_t NumPhaseBinOverlap() const;
    __host__ __device__ size_t NumMagBinOverlap() const;
    __host__ __device__ size_t Bandwidth() const;
    __host__ __device__ size_t Margin() const;
    __host__ __device__ float PhaseBinSize() const;
    __host__ __device__ float MagBinSize() const;
    __host__ __device__ size_t PhaseBin(float phase_val) const;
    __host__ __device__ size_t MagBin(float mag_val) const;
    __host__ __device__ size_t BinIndex(size_t phase_bin, size_t mag_bin) const;

    void CalcVNValsBatched(const std::vector<float*>& times,
                           const std::vector<float*>& mags,
                           const std::vector<size_t>& lengths,
                           const float* periods,
                           const float* period_dts,
                           const size_t num_periods,
                           const size_t num_p_dts,
                           float* vn_out) const;

    float* CalcVNValsBatched(const std::vector<float*>& times,
                             const std::vector<float*>& mags,
                             const std::vector<size_t>& lengths,
                             const float* periods,
                             const float* period_dts,
                             const size_t num_periods,
                             const size_t num_p_dts) const;
};

#endif
