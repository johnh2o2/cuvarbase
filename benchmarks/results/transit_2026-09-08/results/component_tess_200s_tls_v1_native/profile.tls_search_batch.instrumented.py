def tls_search_batch(lightcurves, *, R_star=1.0, M_star=1.0, R_planet=1.0, periods=None, qmin=None, qmax=None, period_min=None, period_max=None, n_transits_min=2, oversampling_factor=3, qmin_fac=0.5, qmax_fac=2.0, n_durations=15, t0_oversample=3.0, refine_top_k=50, refine_oversample=33.0, block_size=None, nbins=None, limb_dark='quadratic', u=None, return_arrays=False, sde_kernel_size=None, fap_null_draws=0, fap_seed=None, _warn_failed=False):
    with _TLS_PROFILE.segment('v1 grid/configuration'):
        "\n    Survey-scale Transit Least Squares search over a batch of\n    lightcurves sharing one trial-period grid.\n\n    This is the fast path for N >> 1 lightcurves: a single kernel\n    launch (per chunk) searches every (lightcurve, period) pair with a\n    phase-binned scan, then an exact per-point refinement kernel\n    re-fits the ``refine_top_k`` best candidate periods per lightcurve\n    on a finer local (duration, t0) grid. There is no cap on ndata.\n\n    Parameters\n    ----------\n    lightcurves : list of (t, y, dy) tuples\n        Times (days), fluxes (normalized to a baseline of 1.0), and\n        flux uncertainties. Each lightcurve's epoch floor(min(t)) is\n        subtracted internally (float64), so BJD-scale times are safe.\n    R_star, M_star : float\n        Stellar radius/mass in solar units; set the period grid and the\n        Keplerian duration window (shared by all lightcurves).\n    R_planet : float\n        Fiducial planet radius (Earth radii) for the duration window.\n    periods, qmin, qmax : array_like, optional\n        Explicit trial grid: periods (days, any order -- sorted\n        internally, per-period output arrays come back in the caller's\n        order) and per-period fractional duration bounds aligned with\n        ``periods``. Auto-generated (Ofir 2014 grid + Keplerian\n        durations from :func:`cuvarbase.tls_grids.duration_window`)\n        when omitted.\n    period_min, period_max : float, optional\n        Period search range for the auto grid.\n    n_transits_min, oversampling_factor : optional\n        Auto period-grid parameters (see tls_grids.period_grid_ofir).\n    qmin_fac, qmax_fac : float\n        Keplerian duration window factors (search [qmin_fac*q,\n        qmax_fac*q] at each period).\n    n_durations : int\n        Trial durations per period (log-spaced), max 64.\n    t0_oversample : float\n        Coarse epoch oversampling; t0 stride = duration / t0_oversample.\n    refine_top_k : int\n        Number of best candidate periods per lightcurve re-fit exactly\n        (default 50; 0 disables refinement).\n    refine_oversample : float\n        Refinement epoch stride = duration / refine_oversample (the\n        reference transitleastsquares package uses ~100).\n    block_size : int, optional\n        CUDA block size override (power of two). By default each\n        bin-count band picks its own (256, or 512 for bands with 4096+\n        bins, shrunk to fit the device's shared-memory cap).\n    nbins : int, optional\n        Phase bins (power of two). Auto-sized so a bin is no wider than\n        the narrowest trial duration / t0_oversample, within the\n        device's shared-memory limit.\n    limb_dark, u : optional\n        Limb-darkening law/coefficients for the transit template\n        (defaults: ``'quadratic'``, ``[0.4804, 0.1867]``).\n    return_arrays : bool\n        Also return the per-period chi2/t0/duration/depth arrays and\n        derived spectra for each lightcurve (adds D2H transfer time).\n    sde_kernel_size : int, optional\n        Median-detrend window for the SDE statistic (see tls_stats).\n    fap_null_draws : int, optional (default: 0)\n        Opt-in empirical false-alarm probability. For each lightcurve,\n        ``fap_null_draws`` null realizations are built by randomly\n        permuting the (y, dy) pairs over the observation times (a\n        white-noise null that keeps the sampling, the point count and\n        the noise distribution but destroys any coherent signal and\n        any red noise), searched on the identical trial grid and\n        settings (coarse scan only; the SDE never uses the\n        refinement), and the result gets ``'FAP' = (1 + n_exceed) /\n        (fap_null_draws + 1)`` where ``n_exceed`` counts null SDEs\n        >= the observed SDE, plus the null SDEs under ``'SDE_null'``.\n        Cost: ``fap_null_draws`` extra searches per lightcurve\n        (measured 400 pure-noise searches of 2880 points x 6157\n        periods in 1.6 s on an A40). The smallest resolvable FAP is\n        ``1 / (fap_null_draws + 1)``. No 'FAP' key is returned\n        otherwise: the pre-1.0 value was an uncalibrated function of\n        the SDE.\n    fap_seed : int or None, optional\n        Seed of the ``numpy.random.RandomState`` used for the null\n        permutations (None: fresh entropy).\n\n    Returns\n    -------\n    results : list of dict\n        One dict per lightcurve:\n        'period', 'period_uncertainty', 't0_phase' (fold phase of the\n        mid-transit relative to floor(min t)), 'T0' (absolute\n        mid-transit time of the first transit at or after min(t), so\n        ``min(t) <= T0 < min(t) + period``; fold with\n        ``((t - T0) / period) % 1``), 'duration', 'depth', 'chi2_min',\n        'SDE', 'SDE_raw' (``SR = chi2_min / chi2`` statistic, see\n        :mod:`cuvarbase.tls_stats`), 'SNR' (``sqrt(chi2_0 -\n        chi2_min)``), 'n_transits', 'n_failed_periods'; plus the\n        per-period arrays (in the caller's period order) when\n        ``return_arrays`` is set, and 'FAP'/'SDE_null' when\n        ``fap_null_draws`` > 0.\n\n        A lightcurve with no valid solution at any trial period (flat\n        or noiseless flux) gets the same keys with SDE = 0, NaN best-fit\n        parameters and the message under 'error' (a warning is raised).\n\n        The best-fit parameters (including 'chi2_min') come from the\n        exact refinement pass, so 'chi2_min' is generally slightly\n        below the minimum of the returned coarse 'chi2' spectrum; the\n        SDE statistics are computed from the uniform coarse spectrum\n        only, keeping the detection statistic's scale consistent\n        across periods.\n    "
        if u is None:
            u = [0.4804, 0.1867]
        tls_grids.validate_stellar_parameters(R_star, M_star)
        tls_models.validate_limb_darkening_coeffs(u, limb_dark)
        if len(lightcurves) == 0:
            return []
        for i, lc in enumerate(lightcurves):
            if len(lc) != 3:
                raise ValueError('tls_search_batch: lightcurve %d must be a (t, y, dy) tuple; got %d elements' % (i, len(lc)))
            _check_tls_lightcurve(lc[0], lc[1], lc[2], name='tls_search_batch lightcurve %d' % i)
        n_durations = _validate_n_durations(n_durations)
        if n_durations > _TLS_FAST_MAX_DURATIONS:
            raise ValueError('n_durations must be in [2, %d] (got %d)' % (_TLS_FAST_MAX_DURATIONS, n_durations))
        if refine_top_k is not None and refine_top_k < 0:
            raise ValueError('refine_top_k must be >= 0 (got %r)' % (refine_top_k,))
        if refine_top_k and (not refine_oversample > 0):
            raise ValueError('refine_oversample must be > 0 (got %r)' % (refine_oversample,))
        if periods is None:
            spans_probe = [np.max(lc[0]) - np.min(lc[0]) for lc in lightcurves]
            t_ref = lightcurves[int(np.argmax(spans_probe))][0]
            periods = tls_grids.period_grid_ofir(t_ref, R_star=R_star, M_star=M_star, oversampling_factor=oversampling_factor, period_min=period_min, period_max=period_max, n_transits_min=n_transits_min)
        periods_in = np.asarray(_validate_periods(periods), dtype=np.float32)
        nperiods = len(periods_in)
        if (qmin is None) != (qmax is None):
            raise ValueError('provide both qmin and qmax, or neither')
        if qmin is None:
            qmin, qmax = tls_grids.duration_window(periods_in.astype(np.float64), R_star=R_star, M_star=M_star, R_planet=R_planet, qmin_fac=qmin_fac, qmax_fac=qmax_fac)
        qmin = np.ascontiguousarray(qmin, dtype=np.float32)
        qmax = np.ascontiguousarray(qmax, dtype=np.float32)
        if len(qmin) != nperiods or len(qmax) != nperiods:
            raise ValueError('qmin and qmax must have same length as periods (%d)' % nperiods)
        _validate_q_window(qmin, qmax, periods=periods_in)
        periods, order = _sort_period_grid(periods_in)
        if order is not None:
            qmin = np.ascontiguousarray(qmin[order])
            qmax = np.ascontiguousarray(qmax[order])
        qmin_global = float(np.min(qmin))
        max_dev_shared = _device_max_shared()
        ensure_context()
        cc_major = cuda.Context.get_device().compute_capability()[0]

        def _band_block_size(nb):
            if block_size is not None:
                return block_size
            big_bin_threshold = 4096 if cc_major >= 8 else 8192
            bs = 512 if nb >= big_bin_threshold else 256
            while bs > 64 and _tls_fast_shared_size(bs, nb) > max_dev_shared:
                bs //= 2
            return bs
        need = t0_oversample / np.maximum(qmin.astype(np.float64), 1e-06)
        if nbins is None:
            nbins_per = np.power(2, np.ceil(np.log2(np.clip(need, 256, None)))).astype(np.int64)
            nbins_per = np.minimum(nbins_per, _TLS_FAST_MAX_NBINS)
            while _tls_fast_shared_size(_band_block_size(int(nbins_per.max())), int(nbins_per.max())) > max_dev_shared:
                cap = int(nbins_per.max()) // 2
                nbins_per = np.minimum(nbins_per, cap)
                if cap <= 256:
                    break
            short = need > nbins_per
            if np.any(short):
                warnings.warn('TLS fast path: %d of %d trial periods have their narrowest durations under-resolved by the phase bins (device shared-memory cap); their coarse scan is smeared and recovery there relies on the exact refinement pass.' % (int(short.sum()), nperiods))
            bands = [(int(nb), np.flatnonzero(nbins_per == nb).astype(np.int32)) for nb in np.unique(nbins_per)]
            smear = float(np.max(need / nbins_per))
        else:
            bands = [(int(nbins), np.arange(nperiods, dtype=np.int32))]
            smear = float(np.max(need / nbins))
        smear = max(1.0, smear)
        refine_nd = 3 if smear <= 1.3 else 5
    with _TLS_PROFILE.segment('v1 kernel lookup/grid transfers'):
        band_launches = []
        for nb, idx in bands:
            bs = _band_block_size(nb)
            kern = _get_cached_fast_kernels(bs, nb, t0_oversample, refine_nd=refine_nd)
            band_launches.append((kern, bs, _tls_fast_shared_size(bs, nb), len(idx), gpuarray.to_gpu(periods[idx]), gpuarray.to_gpu(qmin[idx]), gpuarray.to_gpu(qmax[idx]), gpuarray.to_gpu(idx)))
        refine_bs = _band_block_size(bands[0][0])
        refine_kern = band_launches[0][0]
        refine_smem = _tls_refine_shared_size(refine_bs)
        dur_ratio = float(np.median(qmax / qmin))
        dur_span = dur_ratio ** (1.0 / (2.0 * max(n_durations - 1, 1)))
        dur_span *= min(smear, 4.0)
        nbins_finest = bands[-1][0]
        t0_halfwidth = min(3.0, max(0.5, 1.5 / (nbins_finest * qmin_global)))
    with _TLS_PROFILE.segment('v1 template tables'):
        T_tab, S1_tab, S2_tab = tls_models.generate_template_tables(n_table=_TLS_FAST_NTEMPLATE, limb_dark=limb_dark, u=u)
    with _TLS_PROFILE.segment('v1 host lightcurve preparation'):
        t_hi_c, t_lo_c, a_c, b_c, offs, lens, chi2_0, epochs, spans = _preprocess_batch(lightcurves)
        n_lc = len(lightcurves)
        tmins = np.array([np.min(np.asarray(lc[0], dtype=np.float64)) for lc in lightcurves], dtype=np.float64)
    with _TLS_PROFILE.segment('v1 buffer allocations/transfers'):
        periods_g = gpuarray.to_gpu(periods)
        T_g = gpuarray.to_gpu(T_tab)
        S1_g = gpuarray.to_gpu(S1_tab)
        S2_g = gpuarray.to_gpu(S2_tab)
        max_lcs_by_out = max(1, _TLS_FAST_MAX_OUT_FLOATS // max(nperiods, 1))
        chunks = []
        i0 = 0
        while i0 < n_lc:
            i1 = i0 + 1
            pts = int(lens[i0])
            while i1 < n_lc and i1 - i0 < max_lcs_by_out and (i1 - i0 < _TLS_FAST_MAX_GRID_Y) and (pts + int(lens[i1]) <= _TLS_FAST_MAX_POINTS):
                pts += int(lens[i1])
                i1 += 1
            chunks.append((i0, i1))
            i0 = i1
        max_chunk_lcs = max((i1 - i0 for i0, i1 in chunks))
        max_chunk_pts = max((int(lens[i0:i1].sum()) for i0, i1 in chunks))
        thi_g = gpuarray.empty(max_chunk_pts, np.float32)
        tlo_g = gpuarray.empty(max_chunk_pts, np.float32)
        a_g = gpuarray.empty(max_chunk_pts, np.float32)
        b_g = gpuarray.empty(max_chunk_pts, np.float32)
        off_g = gpuarray.empty(max_chunk_lcs, np.int32)
        len_g = gpuarray.empty(max_chunk_lcs, np.int32)
        out_n = max_chunk_lcs * nperiods
        score_g = gpuarray.empty(out_n, np.float32)
        t0_g = gpuarray.empty(out_n, np.float32)
        dur_g = gpuarray.empty(out_n, np.float32)
        depth_g = gpuarray.empty(out_n, np.float32)
        K = int(min(refine_top_k, max(16, nperiods // 10), nperiods)) if refine_top_k else 0
        if K:
            cand_g = gpuarray.empty(max_chunk_lcs * K, np.int32)
            rscore_g = gpuarray.empty(max_chunk_lcs * K, np.float32)
            rt0_g = gpuarray.empty(max_chunk_lcs * K, np.float32)
            rdur_g = gpuarray.empty(max_chunk_lcs * K, np.float32)
            rdepth_g = gpuarray.empty(max_chunk_lcs * K, np.float32)
        results = [None] * n_lc
    for i0, i1 in chunks:
        with _TLS_PROFILE.segment('v1 lightcurve transfers'):
            nc = i1 - i0
            p0 = int(offs[i0])
            pts = int(lens[i0:i1].sum())
            thi_g[:pts].set(t_hi_c[p0:p0 + pts])
            tlo_g[:pts].set(t_lo_c[p0:p0 + pts])
            a_g[:pts].set(a_c[p0:p0 + pts])
            b_g[:pts].set(b_c[p0:p0 + pts])
            off_g[:nc].set((offs[i0:i1] - p0).astype(np.int32))
            len_g[:nc].set(lens[i0:i1].astype(np.int32))
        with _TLS_PROFILE.segment('v1 coarse search kernel'):
            for kern, bs, smem, band_n, per_g, qmn_g, qmx_g, map_g in band_launches:
                kern['search'](thi_g, tlo_g, a_g, b_g, off_g, len_g, per_g, qmn_g, qmx_g, map_g, S1_g, S2_g, np.int32(band_n), np.int32(nperiods), np.int32(n_durations), score_g, t0_g, dur_g, depth_g, block=(bs, 1, 1), grid=(band_n, nc, 1), shared=smem)
        with _TLS_PROFILE.segment('v1 coarse spectrum transfer'):
            score_h = score_g[:nc * nperiods].get().reshape(nc, nperiods)
        with _TLS_PROFILE.segment('v1 candidate selection/refinement/transfers'):
            rscore_h = rt0_h = rdur_h = rdepth_h = cand = None
            if K:
                cand = np.empty((nc, K), dtype=np.int32)
                for j in range(nc):
                    if K < nperiods:
                        cand[j] = np.argpartition(-score_h[j], K)[:K]
                    else:
                        cand[j] = np.arange(nperiods)
                cand_g[:nc * K].set(cand.ravel())
                refine_kern['refine'](thi_g, tlo_g, a_g, b_g, off_g, len_g, periods_g, cand_g, T_g, np.int32(nperiods), np.int32(K), np.float32(dur_span), np.float32(t0_halfwidth), np.float32(refine_oversample), t0_g, dur_g, rscore_g, rt0_g, rdur_g, rdepth_g, block=(refine_bs, 1, 1), grid=(K, nc, 1), shared=refine_smem)
                rscore_h = rscore_g[:nc * K].get().reshape(nc, K)
                rt0_h = rt0_g[:nc * K].get().reshape(nc, K)
                rdur_h = rdur_g[:nc * K].get().reshape(nc, K)
                rdepth_h = rdepth_g[:nc * K].get().reshape(nc, K)
        with _TLS_PROFILE.segment('v1 parameter-spectrum transfers'):
            if return_arrays or not K or bool((rscore_h.max(axis=1) <= 0.0).any()):
                t0_h = t0_g[:nc * nperiods].get().reshape(nc, nperiods)
                dur_h = dur_g[:nc * nperiods].get().reshape(nc, nperiods)
                depth_h = depth_g[:nc * nperiods].get().reshape(nc, nperiods)
        with _TLS_PROFILE.segment('v1 CPU statistics/results'):

            def _finish_lc(j):
                lc_idx = i0 + j
                srow = score_h[j]
                valid = srow > 0.0
                n_failed = int(nperiods - valid.sum())
                if n_failed == nperiods:
                    msg = _NO_SOLUTION_MSG % nperiods
                    warnings.warn('lightcurve %d: %s; returning a null result (SDE = 0)' % (lc_idx, msg))
                    return (lc_idx, _null_result(nperiods, chi2_0[lc_idx], msg, periods=periods_in, arrays=return_arrays))
                if n_failed and _warn_failed:
                    warnings.warn('%d of %d trial periods returned no valid TLS solution (chi2 sentinel); they are excluded from the best-fit search and the SDE statistics and appear as NaN in the returned arrays' % (n_failed, nperiods))
                row = chi2_0[lc_idx] - srow.astype(np.float64)
                chi2_valid = row[valid]
                periods_valid = periods[valid]
                slot = int(np.argmax(rscore_h[j])) if K else 0
                if K and rscore_h[j, slot] > 0.0:
                    best_idx = int(cand[j, slot])
                    best_t0 = float(rt0_h[j, slot])
                    best_duration = float(rdur_h[j, slot])
                    best_depth = float(rdepth_h[j, slot])
                    chi2_min = float(chi2_0[lc_idx] - rscore_h[j, slot])
                    best_valid_idx = int(np.searchsorted(np.flatnonzero(valid), best_idx))
                else:
                    best_valid_idx = int(np.argmin(chi2_valid))
                    best_idx = int(np.flatnonzero(valid)[best_valid_idx])
                    chi2_min = float(row[best_idx])
                    best_t0 = float(t0_h[j, best_idx])
                    best_duration = float(dur_h[j, best_idx])
                    best_depth = float(depth_h[j, best_idx])
                best_period = float(periods[best_idx])
                n_transits = int(spans[lc_idx] / best_period)
                stats = tls_stats.compute_all_statistics(chi2_valid, periods_valid, best_valid_idx, best_depth, best_duration, n_transits, kernel_size=sde_kernel_size, chi2_null=float(chi2_0[lc_idx]), chi2_best=chi2_min)
                period_uncertainty = tls_stats.compute_period_uncertainty(periods_valid, chi2_valid, best_valid_idx)
                T0 = _first_transit_at_or_after(epochs[lc_idx] + best_t0 * best_period, best_period, tmins[lc_idx])
                res = {'period': best_period, 'period_uncertainty': period_uncertainty, 't0_phase': best_t0, 'T0': float(T0), 'duration': best_duration, 'depth': best_depth, 'chi2_min': chi2_min, 'SDE': stats['SDE'], 'SDE_raw': stats['SDE_raw'], 'SNR': stats['SNR'], 'n_transits': n_transits, 'n_failed_periods': n_failed}
                if return_arrays:

                    def _expand(values):
                        full = np.full(nperiods, np.nan)
                        full[valid] = values
                        return _to_caller_order(full, order)
                    res.update({'periods': periods_in, 'chi2': _to_caller_order(np.where(valid, row, np.nan), order), 'best_t0_per_period': _to_caller_order(t0_h[j].copy(), order), 'best_duration_per_period': _to_caller_order(dur_h[j].copy(), order), 'best_depth_per_period': _to_caller_order(depth_h[j].copy(), order), 'valid_periods': _to_caller_order(valid, order), 'power': _expand(stats['power']), 'SR': _expand(stats['SR'])})
                return (lc_idx, res)
            for j in range(nc):
                lc_idx, res = _finish_lc(j)
                results[lc_idx] = res
    with _TLS_PROFILE.segment('v1 final packaging'):
        if fap_null_draws:
            try:
                n_null_draws = operator.index(fap_null_draws)
            except TypeError:
                raise ValueError('fap_null_draws must be an integer >= 1 (got %r)' % (fap_null_draws,))
            _attach_null_fap(results, lightcurves, n_null_draws, fap_seed, dict(periods=periods, qmin=qmin, qmax=qmax, n_durations=n_durations, t0_oversample=t0_oversample, refine_top_k=0, block_size=block_size, nbins=nbins, limb_dark=limb_dark, u=u, R_star=R_star, M_star=M_star, sde_kernel_size=sde_kernel_size))
        return results
