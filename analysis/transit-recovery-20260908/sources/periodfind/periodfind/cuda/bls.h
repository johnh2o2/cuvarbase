#ifndef __PF_BLS_H__
#define __PF_BLS_H__

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

class BLS {
   private:
    size_t num_bins;
    float bin_size;
    float qmin;
    float qmax;

   public:
    BLS() : BLS(10, 0.01f, 0.5f) {};

    BLS(size_t n_bins, float qmin, float qmax);

    __host__ __device__ size_t NumBins() const;

    __host__ __device__ size_t PhaseBin(float phase_val) const;

    __host__ __device__ float Qmin() const;

    __host__ __device__ float Qmax() const;

    void CalcBLSBatched(const std::vector<float*>& times,
                        const std::vector<float*>& mags,
                        const std::vector<float*>& errs,
                        const std::vector<size_t>& lengths,
                        const float* periods,
                        const float* period_dts,
                        const size_t num_periods,
                        const size_t num_p_dts,
                        float* bls_out) const;

    float* CalcBLSBatched(const std::vector<float*>& times,
                          const std::vector<float*>& mags,
                          const std::vector<float*>& errs,
                          const std::vector<size_t>& lengths,
                          const float* periods,
                          const float* period_dts,
                          const size_t num_periods,
                          const size_t num_p_dts) const;
};

#endif
