// Copyright 2020 California Institute of Technology. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
// Author: Ethan Jaszewski

#ifndef __PF_CE_H__
#define __PF_CE_H__

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

class ConditionalEntropy {
   private:
    size_t num_phase_bins;
    size_t num_mag_bins;
    size_t num_phase_overlap;
    size_t num_mag_overlap;
    float phase_bin_size;
    float mag_bin_size;

   public:
    /**
     * Constructs a ConditionalEntropy object with 10 phase bins and 10
     * magnitude bins.
     */
    ConditionalEntropy() : ConditionalEntropy(10, 10) {};

    /**
     * Constructs a ConditionalEntropy object with the given number of phase and
     * magnitude bins.
     *
     * @param n_phase number of phase bins
     * @param n_mag number of magnitude bins
     */
    ConditionalEntropy(size_t n_phase, size_t n_mag)
        : ConditionalEntropy(n_phase, n_mag, 1, 1) {}

    /**
     * Constructs a ConditionalEntropy object with the given number of phase and
     * magnitude bins, using the specified number of bins overlapping.
     *
     * @param n_phase number of phase bins
     * @param n_mag number of magnitude bins
     * @param p_overlap number of phase bins to overlap
     * @param m_overlap number of magnitude bins to overlap
     */
    ConditionalEntropy(size_t n_phase,
                       size_t n_mag,
                       size_t p_overlap,
                       size_t m_overlap);

    /**
     * Returns the number of total bins in the histogram.
     *
     * @return number of bins in histogram
     */
    __host__ __device__ size_t NumBins() const;

    /**
     * Returns the number of phase bins in the histogram.
     *
     * @return number of phase bins in histogram
     */
    __host__ __device__ size_t NumPhaseBins() const;

    /**
     * Returns the number of magnitude bins in the histogram.
     *
     * @return number of magnitude bins in histogram
     */
    __host__ __device__ size_t NumMagBins() const;

    /**
     * Returns the amount of phase bin overlap in the histogram.
     *
     * @return amount of phase bin overlap in histogram
     */
    __host__ __device__ size_t NumPhaseBinOverlap() const;

    /**
     * Returns the amount of magnitude bin overlap in the histogram.
     *
     * @return amount of magnitude bin overlap in histogram
     */
    __host__ __device__ size_t NumMagBinOverlap() const;

    /**
     * Returns the size of the phase bins in the histogram.
     *
     * @return size of the phase bins
     */
    __host__ __device__ float PhaseBinSize() const;

    /**
     * Returns the size of the magnitude bins in the histogram.
     *
     * @return size of the magnitude bins
     */
    __host__ __device__ float MagBinSize() const;

    /**
     * Gives the first phase bin for a given phase value.
     *
     * @param phase_val phase value
     *
     * @return phase bin index
     */
    __host__ __device__ size_t PhaseBin(float phase_val) const;

    /**
     * Gives the first magnitude bin for a given magnitude value.
     *
     * @param mag_val magnitude value
     *
     * @return magnitude bin index
     */
    __host__ __device__ size_t MagBin(float mag_val) const;

    /**
     * Gives the index of a given bin in a row-major histogram.
     *
     * @param phase_bin phase bin index
     * @param mag_bin magnitude bin index
     *
     * @return row-major histogram index
     */
    __host__ __device__ size_t BinIndex(size_t phase_bin, size_t mag_bin) const;

    /**
     * Folds and bins a light curve across all trial periods and time
     * derivatives.
     *
     * Computes a histogram by folding and binning the times and magnitudes
     * supplied. The magnitudes are assumed to have been scaled into a [0, 1)
     * range. Times are then folded according to the given trial periods and
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
    float* DeviceFoldAndBin(const float* times,
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
     * supplied. The magnitudes are assumed to have been scaled into a [0, 1)
     * range. Times are then folded according to the given trial periods and
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
    float* FoldAndBin(const float* times,
                      const float* mags,
                      const size_t length,
                      const float* periods,
                      const float* period_dts,
                      const size_t num_periods,
                      const size_t num_p_dts) const;

    /**
     * Computes the conditional entropy for an array of histograms.
     *
     * Computes the conditional entropy for all input histograms, which are
     * assumed to have the same dimensions. Computed conditional entropy values
     * are output in a device-allocated array which is returned, in the same
     * order as the input histograms.
     *
     * Arguments should all be device pointers.
     *
     * @param hists array of input histograms
     * @param num_hists number of input histograms
     *
     * @return on-device array of conditional entropy values
     */
    float* DeviceCalcCEFromHists(const float* hists,
                                 const size_t num_hists) const;

    /**
     * Computes the conditional entropy for an array of histograms.
     *
     * Computes the conditional entropy for all input histograms, which are
     * assumed to have the same dimensions. Computed conditional entropy values
     * are output in a host array, in the same order as the input histograms.
     *
     * Arguments should all be host pointers.
     *
     * @param hists array of input histograms
     * @param num_hists number of input histograms
     *
     * @return host array of conditional entropy values
     */
    float* CalcCEFromHists(const float* hists, const size_t num_hists) const;

    /**
     * Computes the conditional entropy values for the input light curve.
     *
     * Computes the conditional entropy values for the input light curve (times
     * and mags). The magnitudes are assumed to have been scaled into a [0, 1)
     * range. Conditional entropy is calculated for each period and time
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
     * @param ce_out output conditional entropy values
     */
    void CalcCEVals(float* times,
                    float* mags,
                    size_t length,
                    const float* periods,
                    const float* period_dts,
                    const size_t num_periods,
                    const size_t num_p_dts,
                    float* ce_out) const;

    /**
     * Computes the conditional entropy values for the input light curve.
     *
     * Computes the conditional entropy values for the input light curve (times
     * and mags). The magnitudes are assumed to have been scaled into a [0, 1)
     * range. Conditional entropy is calculated for each period and time
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
     * @return host array of conditional entropy values
     */
    float* CalcCEVals(float* times,
                      float* mags,
                      size_t length,
                      const float* periods,
                      const float* period_dts,
                      const size_t num_periods,
                      const size_t num_p_dts) const;

    /**
     * Computes the conditional entropy values for all input light curves.
     *
     * Computes the conditional entropy values for each input light curve (times
     * and mags). The magnitudes are assumed to have been scaled into a [0, 1)
     * range. The light curves should be stored consecutively in the input times
     * and magnitudes arrays, and should have lengths as specified (in order) in
     * the lengths vector. For each light curve, conditional entropy is
     * calculated for each period and time derivative, indexed first by period
     * index then time derivative index (i.e. (period 0, td 0), (period 0, td
     * 1), ... (period 1, td 0), (period 1, td 1) etc.). The conditional entropy
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
     * @param ce_out output conditional entropy values
     */
    void CalcCEValsBatched(const std::vector<float*>& times,
                           const std::vector<float*>& mags,
                           const std::vector<size_t>& lengths,
                           const float* periods,
                           const float* period_dts,
                           const size_t num_periods,
                           const size_t num_p_dts,
                           float* ce_out) const;

    /**
     * Computes the conditional entropy values for all input light curves.
     *
     * Computes the conditional entropy values for each input light curve (times
     * and mags). The magnitudes are assumed to have been scaled into a [0, 1)
     * range. The light curves should be stored consecutively in the input times
     * and magnitudes arrays, and should have lengths as specified (in order) in
     * the lengths vector. For each light curve, conditional entropy is
     * calculated for each period and time derivative, indexed first by period
     * index then time derivative index (i.e. (period 0, td 0), (period 0, td
     * 1), ... (period 1, td 0), (period 1, td 1) etc.). The conditional entropy
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
    float* CalcCEValsBatched(const std::vector<float*>& times,
                             const std::vector<float*>& mags,
                             const std::vector<size_t>& lengths,
                             const float* periods,
                             const float* period_dts,
                             const size_t num_periods,
                             const size_t num_p_dts) const;
};

#endif
