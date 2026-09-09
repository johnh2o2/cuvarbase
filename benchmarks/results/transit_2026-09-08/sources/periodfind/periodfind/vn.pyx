#cython: language_level=3

"""
Provides an interface for analyzing light curves using the Viterbi Narrowband
algorithm.
"""

import numpy as np
from periodfind import Statistics, Periodogram
from periodfind._utils import prepare_magnitudes, validate_inputs, ensure_float32

cimport numpy as np
from libc.stddef cimport size_t
from libcpp.vector cimport vector

# Include numpy <-> c array interop
np.import_array()

# Define the C++ VN class so we can use it
cdef extern from "./cuda/vn.h":
    cdef cppclass CppViterbiNarrowband "ViterbiNarrowband":
        CppViterbiNarrowband(size_t num_phase,
                             size_t num_mag,
                             size_t num_phase_overlap,
                             size_t num_mag_overlap,
                             size_t bandwidth,
                             size_t margin)

        void CalcVNValsBatched(const vector[float*]& times,
                               const vector[float*]& mags,
                               const vector[size_t]& lengths,
                               const float* periods,
                               const float* period_dts,
                               const size_t num_periods,
                               const size_t num_p_dts,
                               float* vn_out) const;

cdef class ViterbiNarrowband:
    """Viterbi Narrowband period-finding score.

    Builds the same 2D phase-magnitude histogram as Conditional Entropy,
    then runs a circular Viterbi algorithm to find the most likely narrow
    path through phase-mag space.  The score is the fraction of histogram
    mass concentrated within ``margin`` bins of the optimal path.

    Parameters
    ----------
    n_phase : int, default=20
        The number of phase bins in the histogram.

    n_mag : int, default=20
        The number of magnitude bins in the histogram.

    phase_bin_extent : int, default=1
        Effective width (in bins) of each phase bin (overlap/smoothing).

    mag_bin_extent : int, default=1
        Effective width (in bins) of each magnitude bin (overlap/smoothing).

    bandwidth : int, default=2
        Maximum magnitude-bin shift between adjacent phase bins.

    margin : int, default=1
        Number of bins around the Viterbi path counted for the
        concentration ratio.
    """

    cdef CppViterbiNarrowband* vn

    def __cinit__(self,
                  n_phase=20,
                  n_mag=20,
                  phase_bin_extent=1,
                  mag_bin_extent=1,
                  bandwidth=2,
                  margin=1):
        self.vn = new CppViterbiNarrowband(
            n_phase,
            n_mag,
            phase_bin_extent,
            mag_bin_extent,
            bandwidth,
            margin)

    def __dealloc__(self):
        if self.vn is not NULL:
            del self.vn

    def calc(self,
             list times,
             list mags,
             np.ndarray[ndim=1, dtype=np.float32_t] periods,
             np.ndarray[ndim=1, dtype=np.float32_t] period_dts,
             output="stats",
             normalize=True,
             center=False,
             n_stats=1,
             significance_type='stdmean'):
        """Runs Viterbi Narrowband calculations on a list of light curves.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times.

        mags : list of ndarray
            List of light curve magnitudes.

        periods : ndarray
            Array of trial periods.

        period_dts : ndarray
            Array of trial period time derivatives.

        output : {'stats', 'periodogram'}, default='stats'
            Type of output to return.

        normalize : bool, default=True
            Whether to normalize magnitudes to (0, 1).

        center : bool, default=False
            Whether to center magnitudes to zero mean.

        n_stats : int, default=1
            Number of output Statistics to return.

        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Significance metric.

        Returns
        -------
        data : list of Statistics or list of Periodogram
        """

        validate_inputs(times, mags)
        ensure_float32(times, 'times')
        ensure_float32(mags, 'mags')

        cdef np.ndarray[ndim=1, dtype=np.float32_t] time_arr
        cdef vector[float*] times_ptrs
        cdef vector[size_t] times_lens
        for time_obj in times:
            time_arr = time_obj
            times_ptrs.push_back(&time_arr[0])
            times_lens.push_back(len(time_arr))

        mags_use = prepare_magnitudes(mags, center, normalize)

        cdef np.ndarray[ndim=1, dtype=np.float32_t] mag_arr
        cdef vector[float*] mags_ptrs
        cdef vector[size_t] mags_lens
        for mag_obj in mags_use:
            mag_arr = mag_obj
            mags_ptrs.push_back(&mag_arr[0])
            mags_lens.push_back(len(mag_arr))

        n_per = len(periods)
        n_pdt = len(period_dts)

        vn_ndarr = np.zeros([len(times), n_per, n_pdt], dtype=np.float32)
        cdef float[:, :, ::1] vn_view = vn_ndarr

        self.vn.CalcVNValsBatched(
            times_ptrs, mags_ptrs, times_lens,
            &periods[0], &period_dts[0], n_per, n_pdt,
            &vn_view[0, 0, 0]
        )

        if output == 'stats':
            all_stats = []
            for i in range(len(times)):
                stats = Statistics.statistics_from_data(
                    vn_ndarr[i],
                    [periods, period_dts],
                    True,
                    n=n_stats,
                    significance_type=significance_type,
                )

                all_stats.append(stats)

            return all_stats
        elif output == 'periodogram':
            return [Periodogram(data, [periods, period_dts], True)
                    for data in vn_ndarr]
        else:
            raise NotImplementedError(
                f'Output type "{output}" is not implemented. '
                f'Use "stats" or "periodogram".')
