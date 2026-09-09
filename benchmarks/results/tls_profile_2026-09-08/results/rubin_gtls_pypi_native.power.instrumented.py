def power(self, **kwargs):
    with _TLS_PROFILE.segment('GTLS input/template setup'):
        'Compute the periodogram for a set of user-defined parameters'
        self, kwargs = validate_args(self, kwargs)
        if self.verbose:
            print(constants.TLS_VERSION)
        if len(self.periods) == 0:
            periods = period_grid(R_star=self.R_star, M_star=self.M_star, time_span=np.max(self.t) - np.min(self.t), period_min=self.period_min, period_max=self.period_max, oversampling_factor=self.oversampling_factor, n_transits_min=self.n_transits_min)
        else:
            periods = self.periods
        durations = duration_grid(periods, shortest=1 / len(self.t), log_step=self.duration_grid_step)
        maxwidth_in_samples = int(np.max(durations) * np.size(self.y))
        if maxwidth_in_samples % 2 != 0:
            maxwidth_in_samples = maxwidth_in_samples + 1
        self.maxwidth_in_samples = maxwidth_in_samples
        self.lc_cache_overview, self.lc_arr = get_cache(durations=durations, maxwidth_in_samples=maxwidth_in_samples, per=self.per, rp=self.rp, a=self.a, inc=self.inc, ecc=self.ecc, w=self.w, u=self.u, limb_dark=self.limb_dark, verbose=self.verbose)
        if self.verbose:
            print('Searching ' + str(len(self.y)) + ' data points, ' + str(len(periods)) + ' periods from ' + str(round(min(periods), 3)) + ' to ' + str(round(max(periods), 3)) + ' days')
        periods = np.sort(periods)
        if self.fast == False:
            self.periods, self.period, self.rawDuration, durationPoints, self.duration, self.Depth, self.bestT0, SDE, chi2, self.transitTimes, power, snr, snrPink, snrFit, snrFitPink, raw_power, raw_chi2, possiblePeriodsIndices, possiblePeriods = core.search_multi_periods(periods=periods, t=self.t, y=self.y, dy=self.dy, transit_depth_min=self.transit_depth_min, R_star_min=self.R_star_min, R_star_max=self.R_star_max, M_star_min=self.M_star_min, M_star_max=self.M_star_max, lc_arr=self.lc_arr, lc_cache_overview=self.lc_cache_overview, T0_fit_margin=self.T0_fit_margin, oversampling_factor=self.oversampling_factor, verbose=self.verbose, useLocalPTXCUBIN=self.useLocalPTXCUBIN, GPUDeviceID=self.GPUDeviceID, legacy=self.legacy, bar_location=self.bar_location)
        else:
            periods, power = core.search_multi_periods(periods=periods, t=self.t, y=self.y, dy=self.dy, transit_depth_min=self.transit_depth_min, R_star_min=self.R_star_min, R_star_max=self.R_star_max, M_star_min=self.M_star_min, M_star_max=self.M_star_max, lc_arr=self.lc_arr, lc_cache_overview=self.lc_cache_overview, T0_fit_margin=self.T0_fit_margin, oversampling_factor=self.oversampling_factor, verbose=self.verbose, useLocalPTXCUBIN=self.useLocalPTXCUBIN, GPUDeviceID=self.GPUDeviceID, legacy=self.legacy, fast=True, bar_location=self.bar_location)
            return (periods, power)
        self.rawDurations = durations
        return gtlsResult(self.periods, self.period, self.rawDuration, durationPoints, durations, self.duration, self.Depth, self.bestT0, SDE, chi2, self.transitTimes, power, snr, snrPink, snrFit, snrFitPink, raw_power, raw_chi2, possiblePeriodsIndices, possiblePeriods)
