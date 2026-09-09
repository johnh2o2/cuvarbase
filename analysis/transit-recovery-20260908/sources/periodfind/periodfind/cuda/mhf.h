#ifndef __PF_MHF_H__
#define __PF_MHF_H__

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

class MultiHarmonicFourier {
   private:
    size_t max_harmonics;

   public:
    MultiHarmonicFourier() : MultiHarmonicFourier(5) {};

    MultiHarmonicFourier(size_t max_k);

    __host__ __device__ size_t MaxHarmonics() const;

    __host__ __device__ size_t NumParams() const;

    void CalcMHFBatched(const std::vector<float*>& times,
                        const std::vector<float*>& mags,
                        const std::vector<float*>& errs,
                        const std::vector<size_t>& lengths,
                        const float* periods,
                        const float* period_dts,
                        const size_t num_periods,
                        const size_t num_p_dts,
                        float* mhf_out) const;

    float* CalcMHFBatched(const std::vector<float*>& times,
                          const std::vector<float*>& mags,
                          const std::vector<float*>& errs,
                          const std::vector<size_t>& lengths,
                          const float* periods,
                          const float* period_dts,
                          const size_t num_periods,
                          const size_t num_p_dts) const;
};

#endif
