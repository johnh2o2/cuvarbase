#ifndef __PF_MF_H__
#define __PF_MF_H__

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

// Template types for matched filter
#define MF_TEMPLATE_SAWTOOTH   0
#define MF_TEMPLATE_SINUSOIDAL 1
#define MF_TEMPLATE_ECLIPSING  2

// Maximum number of templates and bins (compile-time limits)
#define MF_MAX_TEMPLATES 14
#define MF_MAX_BINS 64

class MatchedFilter {
   private:
    size_t num_bins;
    float bin_size;

    // Template data (host-side, copied to device constant memory on first use)
    size_t num_templates;
    float templates[MF_MAX_TEMPLATES * MF_MAX_BINS];  // flattened
    int template_types[MF_MAX_TEMPLATES];

    void GenerateTemplates();

   public:
    MatchedFilter() : MatchedFilter(20) {};

    MatchedFilter(size_t n_bins);

    __host__ __device__ size_t NumBins() const;

    __host__ __device__ size_t NumTemplates() const;

    __host__ __device__ size_t PhaseBin(float phase_val) const;

    const float* GetTemplates() const { return templates; }
    const int* GetTemplateTypes() const { return template_types; }

    void CalcMFBatched(const std::vector<float*>& times,
                       const std::vector<float*>& mags,
                       const std::vector<float*>& errs,
                       const std::vector<size_t>& lengths,
                       const float* periods,
                       const float* period_dts,
                       const size_t num_periods,
                       const size_t num_p_dts,
                       float* mf_out) const;

    float* CalcMFBatched(const std::vector<float*>& times,
                         const std::vector<float*>& mags,
                         const std::vector<float*>& errs,
                         const std::vector<size_t>& lengths,
                         const float* periods,
                         const float* period_dts,
                         const size_t num_periods,
                         const size_t num_p_dts) const;
};

#endif
