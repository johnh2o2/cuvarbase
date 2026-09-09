#cython: language_level=3

"""
Provides an interface for analyzing light curves using the Box Least Squares
(BLS) transit-detection algorithm (Kovács, Zucker & Mazeh 2002).
"""

import numpy as np
from periodfind import Statistics, Periodogram
from periodfind._utils import validate_inputs, ensure_float32

cimport numpy as np
from libc.stddef cimport size_t
from libcpp.vector cimport vector

# Include numpy <-> c array interop
np.import_array()

cdef extern from "./cuda/bls.h":
    cdef cppclass CppBLS "BLS":
        CppBLS(size_t num_bins, float qmin, float qmax)

        void CalcBLSBatched(const vector[float*]& times,
                            const vector[float*]& mags,
                            const vector[float*]& errs,
                            const vector[size_t]& lengths,
                            const float* periods,
                            const float* period_dts,
                            const size_t num_periods,
                            const size_t num_p_dts,
                            float* bls_out) const;

cdef class BoxLeastSquares:
    """Box Least Squares (BLS) transit-detection light curve analysis.

    Searches for periodic box-shaped (flat-bottom) dips in time-series data,
    as described by Kovács, Zucker & Mazeh (2002).  Supports per-point
    uncertainties for inverse-variance weighting.

    Parameters
    ----------
    n_bins : int, default=50
        The number of phase bins.
    qmin : float, default=0.01
        Minimum transit duration as a fraction of the period.
    qmax : float, default=0.5
        Maximum transit duration as a fraction of the period.
    """

    cdef CppBLS* bls

    def __cinit__(self, n_bins=50, qmin=0.01, qmax=0.5):
        self.bls = new CppBLS(n_bins, qmin, qmax)

    def __dealloc__(self):
        if self.bls is not NULL:
            del self.bls

    def calc(self,
             list times,
             list mags,
             np.ndarray[ndim=1, dtype=np.float32_t] periods,
             np.ndarray[ndim=1, dtype=np.float32_t] period_dts,
             errs=None,
             output="stats",
             normalize=False,
             center=False,
             n_stats=1,
             significance_type='stdmean'):
        """Runs BLS calculations on a list of light curves.

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

        errs : list of ndarray or None, default=None
            List of per-point uncertainties (standard deviations).
            If None, uniform uncertainties of 1.0 are assumed.

        output : {'stats', 'periodogram'}, default='stats'
            Type of output that should be returned

        normalize : bool, default=False
            Unused (accepted for API consistency).

        center : bool, default=False
            Unused (accepted for API consistency).

        n_stats : int, default=1
            Number of output `Statistics` to return if `output='stats'`

        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Specifies the significance statistic that should be used.

        Returns
        -------
        data : list of Statistics or list of Periodogram
        """

        validate_inputs(times, mags)
        ensure_float32(times, 'times')
        ensure_float32(mags, 'mags')

        # Handle uncertainties
        if errs is None:
            errs = [np.ones(len(t), dtype=np.float32) for t in times]
        else:
            ensure_float32(errs, 'errs')
            if len(errs) != len(times):
                raise ValueError(
                    f"errs must have the same number of arrays as times, "
                    f"got {len(errs)} and {len(times)}")
            for i, (e, t) in enumerate(zip(errs, times)):
                if len(e) != len(t):
                    raise ValueError(
                        f"errs[{i}] and times[{i}] have different lengths: "
                        f"{len(e)} vs {len(t)}")

        cdef np.ndarray[ndim=1, dtype=np.float32_t] time_arr
        cdef vector[float*] times_ptrs
        cdef vector[size_t] times_lens
        for time_obj in times:
            time_arr = time_obj
            times_ptrs.push_back(&time_arr[0])
            times_lens.push_back(len(time_arr))

        cdef np.ndarray[ndim=1, dtype=np.float32_t] mag_arr
        cdef vector[float*] mags_ptrs
        for mag_obj in mags:
            mag_arr = mag_obj
            mags_ptrs.push_back(&mag_arr[0])

        cdef np.ndarray[ndim=1, dtype=np.float32_t] err_arr
        cdef vector[float*] errs_ptrs
        for err_obj in errs:
            err_arr = err_obj
            errs_ptrs.push_back(&err_arr[0])

        n_per = len(periods)
        n_pdt = len(period_dts)

        bls_ndarr = np.zeros([len(times), n_per, n_pdt], dtype=np.float32)
        cdef float[:, :, ::1] bls_view = bls_ndarr

        self.bls.CalcBLSBatched(
            times_ptrs, mags_ptrs, errs_ptrs, times_lens,
            &periods[0], &period_dts[0], n_per, n_pdt,
            &bls_view[0, 0, 0]
        )

        if output == 'stats':
            all_stats = []
            for i in range(len(times)):
                stats = Statistics.statistics_from_data(
                    bls_ndarr[i],
                    [periods, period_dts],
                    True,
                    n=n_stats,
                    significance_type=significance_type,
                )

                all_stats.append(stats)

            return all_stats
        elif output == 'periodogram':
            return [Periodogram(data, [periods, period_dts], True)
                    for data in bls_ndarr]
        else:
            raise NotImplementedError(
                f'Output type "{output}" is not implemented. '
                f'Use "stats" or "periodogram".')
