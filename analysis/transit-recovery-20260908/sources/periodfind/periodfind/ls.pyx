#cython: language_level=3

# Copyright 2020 California Institute of Technology. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
# Author: Ethan Jaszewski

"""
Provides an interface for analyzing light curves using Lomb-Scargle
periodograms.
"""

import numpy as np
from periodfind import Statistics, Periodogram
from periodfind._utils import prepare_magnitudes, validate_inputs, ensure_float32

cimport numpy as np
from libc.stddef cimport size_t
from libcpp.vector cimport vector

# Include numpy <-> c array interop
np.import_array()

cdef extern from "./cuda/ls.h":
    cdef cppclass CppLombScargle "LombScargle":
        CppLombScargle()

        void CalcLSBatched(const vector[float*]& times,
                           const vector[float*]& mags,
                           const vector[size_t]& lengths,
                           const float* periods,
                           const float* period_dts,
                           const size_t num_periods,
                           const size_t num_p_dts,
                           float* per_out) const;

cdef class LombScargle:
    """Lomb-Scargle periodogram light curve analysis.

    Attempts to determine the period of a light curve by computing a Lomb-
    Scargle periodogram for the input light curve.
    """

    cdef CppLombScargle* ls

    def __cinit__(self):
        self.ls = new CppLombScargle()

    def __dealloc__(self):
        if self.ls is not NULL:
            del self.ls

    def calc(self,
             list times,
             list mags,
             np.ndarray[ndim=1, dtype=np.float32_t] periods,
             np.ndarray[ndim=1, dtype=np.float32_t] period_dts,
             output="stats",
             normalize=False,
             center=True,
             n_stats=1,
             significance_type='stdmean'):
        """Runs Lomb-Scargle calculations on a list of light curves.

        Computes an Lomb-Scargle periodogram for each of the input
        light curves, then returns either statistics or a full periodogram,
        depending on what is requested.

        Parameters
        ----------
        times : list of ndarray
            List of light curve times.

        mags : list of ndarray
            List of light curve magnitudes.

        periods : ndarray
            Array of trial periods

        period_dts : ndarray
            Array of trial period time derivatives

        output : {'stats', 'periodogram'}, default='stats'
            Type of output that should be returned

        normalize : bool, default=False
            Whether to normalize the light curve magnitudes. If true, light
            curve magnitudes will be normalized to a (0, 1) range

        center : bool, default=True
            Whether to center the light curve magnitutes. If true, light curve
            magnitudes will be shifted so that the data have zero mean.

        n_stats : int, default=1
            Number of output `Statistics` to return if `output='stats'`

        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Specifies the significance statistic that should be used. See the
            documentation for the `Statistics` class for more information.
            Used only if `output='stats'`.

        Returns
        -------
        data : list of Statistics or list of Periodogram
            If `output='stats'`, then returns a list of `Statistics` objects,
            one for each light curve.

            If `output='periodogram'`, then returns a list of `Periodogram`
            objects, one for each light curve.

        Notes
        -----
        The times and magnitudes arrays must be given such that the pair
        `(times[i], magnitudes[i])` gives the `i`th light curve. As such,
        `times[i]` and `magnitudes[i]` must have the same length for all `i`.

        Although normalization is not required for the Lomb-Scargle
        calculation, it can help reduce floating point error, so it is
        recommended for light curves with large magnitude values.
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

        ls_ndarr = np.zeros([len(times), n_per, n_pdt], dtype=np.float32)
        cdef float[:, :, ::1] ls_view = ls_ndarr

        self.ls.CalcLSBatched(
            times_ptrs, mags_ptrs, times_lens,
            &periods[0], &period_dts[0], n_per, n_pdt,
            &ls_view[0, 0, 0]
        )

        if output == 'stats':
            all_stats = []
            for i in range(len(times)):
                stats = Statistics.statistics_from_data(
                    ls_ndarr[i],
                    [periods, period_dts],
                    True,
                    n=n_stats,
                    significance_type=significance_type,
                )

                all_stats.append(stats)

            return all_stats
        elif output == 'periodogram':
            return [Periodogram(data, [periods, period_dts], True)
                    for data in ls_ndarr]
        else:
            raise NotImplementedError(
                f'Output type "{output}" is not implemented. '
                f'Use "stats" or "periodogram".')
