#ifndef __PF_FPW_H__
#define __PF_FPW_H__

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

class FPW {
   private:
    size_t num_bins;
    float bin_size;

   public:
    FPW() : FPW(10) {};

    FPW(size_t n_bins);

    __host__ __device__ size_t NumBins() const;

    __host__ __device__ size_t PhaseBin(float phase_val) const;

    void CalcFPWBatched(const std::vector<float*>& times,
                        const std::vector<float*>& mags,
                        const std::vector<float*>& errs,
                        const std::vector<size_t>& lengths,
                        const float* periods,
                        const float* period_dts,
                        const size_t num_periods,
                        const size_t num_p_dts,
                        float* fpw_out) const;

    float* CalcFPWBatched(const std::vector<float*>& times,
                          const std::vector<float*>& mags,
                          const std::vector<float*>& errs,
                          const std::vector<size_t>& lengths,
                          const float* periods,
                          const float* period_dts,
                          const size_t num_periods,
                          const size_t num_p_dts) const;
};

#endif
