// Copyright 2020 California Institute of Technology. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
// Author: Ethan Jaszewski

#ifndef __PF_AOV_H__
#define __PF_AOV_H__

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

struct AOVData {
    uint32_t count;
    float sum;
    float sq_sum;
};

class AOV {
   private:
    size_t num_bins;
    size_t num_overlap;
    float bin_size;

   public:
    /**
     * Constructs an AOV object with 10 phase bins.
     */
    AOV() : AOV(10) {};

    /**
     * Constructs an AOV object with the given number of phase bins.
     *
     * @param n_bins number of bins
     */
    AOV(size_t n_bins) : AOV(n_bins, 1) {};

    /**
     * Constructs an AOV object with the given number of phase bins and using
     * the specified number of bins overlapping.
     *
     * @param n_bins number of phase bins
     * @param bin_overlap number of phase bins to overlap
     */
    AOV(size_t n_bins, size_t bin_overlap);

    /**
     * Returns the number of phase bins in the histogram.
     *
     * @return number of phase bins in histogram
     */
    __host__ __device__ size_t NumPhaseBins() const;

    /**
     * Returns the amount of phase bin overlap in the histogram.
     *
     * @return amount of phase bin overlap in histogram
     */
    __host__ __device__ size_t NumPhaseBinOverlap() const;

    /**
     * Gives the first phase bin for a given phase value.
     *
     * @param phase_val phase value
     *
     * @return phase bin index
     */
    __host__ __device__ size_t PhaseBin(float phase_val) const;

    /**
     * Folds and bins a light curve across all trial periods and time
     * derivatives.
     *
     * Computes a histogram by folding and binning the times and magnitudes
     * supplied. Times are folded according to the given trial periods and
     * period time derivatives. A histogram is created for each trial period,
     * time derivative pair, then output in an on-device array indexed first by
     * period index, then time derivative index (i.e. (period 0, td 0), (period
     * 0, td 1), ... (period 1, td 0), (period 1, td 1) etc.) which is returned.
     *
     * Arguments should all be device pointers.
     *
     * @param times light curve datapoint times
     * @param mags light curve datapoint magnitudes
     * @param periods list of trial periods
     * @param period_dts list of trial period time derivatives
     * @param num_periods number of trial periods
     * @param num_p_dts number of trial time derivatives
     *
     * @return on-device array of histograms
     */
    AOVData* DeviceFoldAndBin(const float* times,
                              const float* mags,
                              const size_t length,
                              const float* periods,
                              const float* period_dts,
                              const size_t num_periods,
                              const size_t num_p_dts) const;

    /**
     * Folds and bins a light curve across all trial periods and time
     * derivatives.
     *
     * Computes a histogram by folding and binning the times and magnitudes
     * supplied. Times are folded according to the given trial periods and
     * period time derivatives. A histogram is created for each trial period,
     * time derivative pair, then output in a host array indexed first by period
     * index, then time derivative index (i.e. (period 0, td 0), (period 0, td
     * 1), ... (period 1, td 0), (period 1, td 1) etc.) which is returned.
     *
     * Arguments should all be host pointers.
     *
     * @param times light curve datapoint times
     * @param mags light curve datapoint magnitudes
     * @param periods list of trial periods
     * @param period_dts list of trial period time derivatives
     * @param num_periods number of trial periods
     * @param num_p_dts number of trial time derivatives
     *
     * @return host array of histograms
     */
    AOVData* FoldAndBin(const float* times,
                        const float* mags,
                        const size_t length,
                        const float* periods,
                        const float* period_dts,
                        const size_t num_periods,
                        const size_t num_p_dts) const;

    /**
     * Computes the AOV values for an array of histograms.
     *
     * Computes the AOV score for all input histograms, which are
     * assumed to have the same dimensions. Computed AOV values
     * are output in a device-allocated array which is returned, in the same
     * order as the input histograms.
     *
     * Arguments should all be device pointers.
     *
     * @param hists array of input histograms
     * @param num_hists number of input histograms
     *
     * @return on-device array of AOV values
     */
    float* DeviceCalcAOVFromHists(const AOVData* hists,
                                  const size_t num_hists,
                                  const float length,
                                  const float avg) const;

    /**
     * Computes the AOV values for an array of histograms.
     *
     * Computes the AOV value for all input histograms, which are
     * assumed to have the same dimensions. Computed AOV values
     * are output in a host array, in the same order as the input histograms.
     *
     * Arguments should all be host pointers.
     *
     * @param hists array of input histograms
     * @param num_hists number of input histograms
     *
     * @return host array of AOV values
     */
    float* CalcAOVFromHists(const AOVData* hists,
                            const size_t num_hists,
                            const float length,
                            const float avg) const;

    /**
     * Computes the AOV values for the input light curve.
     *
     * Computes the AOV values for the input light curve (times
     * and mags). The magnitudes are assumed to have been scaled into a [0, 1)
     * range. The AOV score is calculated for each period and time
     * derivative, then written to the output array indexed first by period
     * index, then time derivative index (i.e. (period 0, td 0),
     * (period 0, td 1), ... (period 1, td 0), (period 1, td 1) etc.).
     *
     * Arguments should all be host pointers.
     *
     * @param times light curve datapoint times
     * @param mags light curve datapoint magnitudes
     * @param length length of the input light curve
     * @param periods list of trial periods
     * @param period_dts list of trial period time derivatives
     * @param num_periods number of trial periods
     * @param num_p_dts number of trial time derivatives
     * @param aov_out output AOV values
     */
    void CalcAOVVals(float* times,
                     float* mags,
                     size_t length,
                     const float* periods,
                     const float* period_dts,
                     const size_t num_periods,
                     const size_t num_p_dts,
                     float* aov_out) const;

    /**
     * Computes the AOV values for the input light curve.
     *
     * Computes the AOV values for the input light curve (times
     * and mags). The magnitudes are assumed to have been scaled into a [0, 1)
     * range. The AOV score is calculated for each period and time
     * derivative, then output in a host array indexed first by period index,
     * then time derivative index (i.e. (period 0, td 0), (period 0, td 1), ...
     * (period 1, td 0), (period 1, td 1) etc.) which is returned.
     *
     * Arguments should all be host pointers.
     *
     * @param times light curve datapoint times
     * @param mags light curve datapoint magnitudes
     * @param length length of the input light curve
     * @param periods list of trial periods
     * @param period_dts list of trial period time derivatives
     * @param num_periods number of trial periods
     * @param num_p_dts number of trial time derivatives
     *
     * @return host array of AOV values
     */
    float* CalcAOVVals(float* times,
                       float* mags,
                       size_t length,
                       const float* periods,
                       const float* period_dts,
                       const size_t num_periods,
                       const size_t num_p_dts) const;

    /**
     * Computes the AOV values for all input light curves.
     *
     * Computes the AOV values for each input light curve (times
     * and mags). The magnitudes are assumed to have been scaled into a [0, 1)
     * range. The light curves should have lengths as specified (in order) in
     * the lengths vector. For each light curve, the AOV value is
     * calculated for each period and time derivative, indexed first by period
     * index then time derivative index (i.e. (period 0, td 0), (period 0, td
     * 1), ... (period 1, td 0), (period 1, td 1) etc.). The AOV
     * arrays are stored consecutively and written to the provided output array.
     *
     * Arguments should all be host pointers.
     *
     * @param times ordered vector of light curve times
     * @param mags orderded vector of light curve magnitudes
     * @param lengths ordered vector of light curve lengths
     * @param periods list of trial periods
     * @param period_dts list of trial period time derivatives
     * @param num_periods number of trial periods
     * @param num_p_dts number of trial time derivatives
     * @param aov_out output AOV values
     */
    void CalcAOVValsBatched(const std::vector<float*>& times,
                            const std::vector<float*>& mags,
                            const std::vector<size_t>& lengths,
                            const float* periods,
                            const float* period_dts,
                            const size_t num_periods,
                            const size_t num_p_dts,
                            float* aov_out) const;

    /**
     * Computes the AOV values for all input light curves.
     *
     * Computes the AOV values for each input light curve (times
     * and mags). The magnitudes are assumed to have been scaled into a [0, 1)
     * range. The light curves should have lengths as specified (in order) in
     * the lengths vector. For each light curve, the AOV value is
     * calculated for each period and time derivative, indexed first by period
     * index then time derivative index (i.e. (period 0, td 0), (period 0, td
     * 1), ... (period 1, td 0), (period 1, td 1) etc.). The AOV
     * arrays are stored consecutively and output as a single host array.
     *
     * Arguments should all be host pointers.
     *
     * @param times ordered vector of light curve times
     * @param mags orderded vector of light curve magnitudes
     * @param lengths ordered vector of light curve lengths
     * @param periods list of trial periods
     * @param period_dts list of trial period time derivatives
     * @param num_periods number of trial periods
     * @param num_p_dts number of trial time derivatives
     *
     * @return host array of conditional entropy values
     */
    float* CalcAOVValsBatched(const std::vector<float*>& times,
                              const std::vector<float*>& mags,
                              const std::vector<size_t>& lengths,
                              const float* periods,
                              const float* period_dts,
                              const size_t num_periods,
                              const size_t num_p_dts) const;
};

#endif
