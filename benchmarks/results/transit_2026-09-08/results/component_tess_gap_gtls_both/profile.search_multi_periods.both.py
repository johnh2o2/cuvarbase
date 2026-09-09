def search_multi_periods(periods, t, y, dy, transit_depth_min, R_star_min, R_star_max, M_star_min, M_star_max, lc_arr, lc_cache_overview, T0_fit_margin, oversampling_factor, verbose, useLocalPTXCUBIN=False, GPUDeviceID=0, fast=False, legacy=False, SimplifyEdgeEffect=True, bar_location=0):
    set_cuda_device(GPUDeviceID)
    GPUCode = GPUFun.getGPUCode()
    if T0_fit_margin == 0:
        GPUCode = GPUCode.replace('#define SKIP_POINT 8', '#define SKIP_POINT ' + '0x7f800000')
    else:
        GPUCode = GPUCode.replace('#define SKIP_POINT 8', '#define SKIP_POINT ' + str(int(1 / T0_fit_margin)))
    module = cp.RawModule(code=GPUCode)
    module.compile()
    durations, indices = np.unique(lc_cache_overview['width_in_samples'], return_index=True)
    lc_arr = lc_arr[indices]
    lc_cache_overview = lc_cache_overview[indices]
    maxDuration = int(max(durations))
    if maxDuration % 2 != 0:
        maxDuration = maxDuration + 1
    durations = np.sort(durations)
    tSize = len(t)
    patchedDatasSize = int(tSize + maxDuration)
    patchedDatasSizeGPU = cp.asarray(np.array([patchedDatasSize])).astype(cp.int32)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(cp.cuda.Device().id)
    nvmlinfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    singleCalcPeriods_max = nvmlinfo.free / (5 * (patchedDatasSize * 2 + 2 + len(durations) * patchedDatasSize * 4 + 2 * len(durations)))
    singleCalcPeriods = int(np.min([np.floor(singleCalcPeriods_max), len(periods) / 30]))
    if singleCalcPeriods < 15:
        singleCalcPeriods = int(singleCalcPeriods / 1.1)
    TotalIter = int(np.ceil(len(periods) / singleCalcPeriods))
    if verbose:
        pbar = tqdm.tqdm(total=TotalIter, position=bar_location)
    periodsGPU = cp.empty((singleCalcPeriods,), dtype=cp.float64)
    durationsMaxGPU = cp.empty((singleCalcPeriods,), dtype=cp.int32)
    durationsMinGPU = cp.empty((singleCalcPeriods,), dtype=cp.int32)
    locationGPU = cp.empty(len(periods), dtype=cp.int32)
    LowestResidualsEachPeriodGPU = cp.empty(len(periods), dtype=cp.float32)
    iterFlagGPU = cp.int32(0)
    fulldurationsMaxGPU = cp.empty((len(periods),), dtype=cp.int32)
    fulldurationsMinGPU = cp.empty((len(periods),), dtype=cp.int32)
    fullperiodsSizeGPU = cp.asarray(np.array([len(periods)])).astype(cp.int32)
    tSizeGPU = cp.asarray(np.array([tSize])).astype(cp.int32)
    tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)
    periodsGPU = cp.asarray(periods).astype(cp.float64)
    durationsGridGPU = module.get_function('durationsGrid')
    blockSize, gridSizeX = calcGridBlockSize(len(periods))
    durationsGridGPU((gridSizeX, 1, 1), (blockSize,), (periodsGPU, fulldurationsMaxGPU, fulldurationsMinGPU, tLengthGPU, tSizeGPU, fullperiodsSizeGPU))
    fulldurationsSizeGPU = cp.asarray(np.array([len(durations)])).astype(cp.int32)
    fulldurationsGPU = cp.asarray(durations).astype(cp.int32)
    durationBoolArrayGPU = cp.empty((len(periods), len(durations)), dtype=cp.bool_)
    durationBoolFunGPU = module.get_function('durationBool')
    blockSize, gridSizeX = calcGridBlockSize(len(periods))
    durationBoolFunGPU((gridSizeX, len(durations), 1), (blockSize, 1, 1), (fulldurationsMaxGPU, fulldurationsMinGPU, fulldurationsSizeGPU, fullperiodsSizeGPU, fulldurationsGPU, durationBoolArrayGPU))
    durationsGridCollectionGPU = cp.empty((TotalIter, len(durations)), dtype=cp.bool_)
    yGPU = cp.asarray(y).astype(cp.float32)
    dyGPU = cp.asarray(dy).astype(cp.float32)
    tGPU_cached = cp.asarray(t).astype(cp.float64)
    tSizeGPU_cached = cp.asarray(np.array([tSize])).astype(cp.int32)
    tLengthGPU_cached = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)
    periodsSizeGPU_cached = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
    maxDurationGPU_cached = cp.asarray(np.array([maxDuration])).astype(cp.int32)
    periodSizeGPU_cached = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
    datapointsGPU_cached = cp.array([len(y)]).astype(cp.int32)
    transitDepthMinGPU_cached = cp.array([transit_depth_min]).astype(cp.float32)
    phasesGPU_cached = cp.empty((singleCalcPeriods, tSize), dtype=cp.float64)
    sortIndexGPU_cached = cp.empty((singleCalcPeriods, tSize), dtype=cp.int32)
    patchedDatasGPU_cached = cp.empty((singleCalcPeriods, tSize + maxDuration), dtype=cp.float32)
    patchedDysGPU_cached = cp.empty((singleCalcPeriods, tSize + maxDuration), dtype=cp.float32)
    edgeEffectCorrectionsGPU_cached = cp.empty(singleCalcPeriods, dtype=cp.float32)
    inverseSquaredPatchedDysGPU_cached = cp.empty((singleCalcPeriods, tSize + maxDuration), dtype=cp.float32)
    cumsumGPU_cached = cp.empty((singleCalcPeriods, patchedDatasSize), dtype=cp.float32)
    base_error_cached = cp.empty((singleCalcPeriods, patchedDatasSize), dtype=cp.float32)
    for iterFlag in range(TotalIter):
        if iterFlag == TotalIter - 1:
            SinglePeriods = periods[iterFlag * singleCalcPeriods:]
            actual_period_count = len(SinglePeriods)
            if actual_period_count < singleCalcPeriods:
                SinglePeriods = np.append(SinglePeriods, np.full(singleCalcPeriods - actual_period_count, SinglePeriods[-1]))
            start_idx = iterFlag * singleCalcPeriods
            end_idx = min(start_idx + actual_period_count, len(periods))
            temp_bool = durationBoolArrayGPU[start_idx]
            temp_bool = cp.any(durationBoolArrayGPU[start_idx:end_idx], axis=0)
            durationsGridCollectionGPU[iterFlag] = temp_bool
        else:
            SinglePeriods = periods[iterFlag * singleCalcPeriods:(iterFlag + 1) * singleCalcPeriods]
            start_idx = iterFlag * singleCalcPeriods
            end_idx = (iterFlag + 1) * singleCalcPeriods
            temp_bool = durationBoolArrayGPU[start_idx]
            temp_bool = cp.any(durationBoolArrayGPU[start_idx:end_idx], axis=0)
            durationsGridCollectionGPU[iterFlag] = temp_bool
        durationsBoolGrid = durationsGridCollectionGPU[iterFlag].get()
        singleDurations = durations[durationsBoolGrid]
        if len(singleDurations) == 0:
            start_idx = iterFlag * singleCalcPeriods
            valid_range = min(start_idx + singleCalcPeriods, len(periods)) - start_idx
            LowestResidualsEachPeriodGPU[start_idx:start_idx + valid_range] = cp.nan
            if verbose:
                pbar.update(1)
            continue
        single_lc_arr = lc_arr[durationsBoolGrid]
        single_lc_cache_overview = lc_cache_overview[durationsBoolGrid]
        overshootGPU = cp.array(single_lc_cache_overview['overshoot']).astype(cp.float32)
        periodsGPU = cp.asarray(SinglePeriods).astype(cp.float64)
        durationsMaxGPU = cp.asarray(SinglePeriods).astype(cp.int32)
        durationsMinGPU = cp.asarray(SinglePeriods).astype(cp.int32)
        lowestResidualsGPU = cp.empty((singleCalcPeriods, len(singleDurations), tSize), dtype=cp.float32)
        phasesGPU = phasesGPU_cached
        sortIndexGPU = sortIndexGPU_cached
        durationsGridGPU = module.get_function('durationsGrid')
        blockSize, gridSizeX = calcGridBlockSize(singleCalcPeriods)
        durationsGridGPU((gridSizeX, 1, 1), (blockSize,), (periodsGPU, durationsMaxGPU, durationsMinGPU, tLengthGPU_cached, tSizeGPU_cached, periodsSizeGPU_cached))
        patchedDatasGPU = patchedDatasGPU_cached
        patchedDysGPU = patchedDysGPU_cached
        lc_arr_max_len = np.array([np.max(singleDurations)]).astype(np.int32)
        lc_arr_full_length = 1 - np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') for x in single_lc_arr])
        lcArrMaxLenGPU = cp.asarray(lc_arr_max_len).astype(cp.int32)
        lcArrFullLengthGPU = cp.asarray(lc_arr_full_length).astype(cp.float32)
        edgeEffectCorrectionsGPU = edgeEffectCorrectionsGPU_cached
        inverseSquaredPatchedDysGPU = inverseSquaredPatchedDysGPU_cached
        durationsGPU = cp.asarray(singleDurations).astype(cp.int32)
        durationsSizeGPU = cp.asarray(np.array([len(singleDurations)])).astype(cp.int32)
        fullSumGPU = cp.empty((singleCalcPeriods, len(singleDurations)), dtype=cp.float32)
        cumsumGPU = cumsumGPU_cached
        ootrGPU = cp.empty((singleCalcPeriods, len(singleDurations), tSize), dtype=cp.float32)
        fastFoldGPU = module.get_function('foldFast')
        blockSize, gridSizeX = calcGridBlockSize(tSize)
        fastFoldGPU((gridSizeX, singleCalcPeriods), (blockSize,), (tGPU_cached, periodsGPU, phasesGPU, periodsSizeGPU_cached, tSizeGPU_cached))
        i_max = 10
        for i in range(1, i_max + 1):
            sortIndexGPU[(i - 1) * singleCalcPeriods / i_max:i * singleCalcPeriods / i_max] = phasesGPU[(i - 1) * singleCalcPeriods / i_max:i * singleCalcPeriods / i_max].argsort()
        patchDataGPU = module.get_function('patchData')
        blockSize, gridSizeX = calcGridBlockSize(tSize + maxDuration)
        patchDataGPU((gridSizeX, singleCalcPeriods), (blockSize,), (patchedDatasGPU, patchedDysGPU, patchedDatasSizeGPU, sortIndexGPU, maxDurationGPU_cached, yGPU, dyGPU, tSizeGPU_cached))
        calcInverseSquaredPatchedDyGPU = module.get_function('calcInverseSquaredPatchedDy')
        blockSize, gridSizeX = calcGridBlockSize(patchedDatasSize)
        calcInverseSquaredPatchedDyGPU((gridSizeX, singleCalcPeriods, 1), (blockSize, 1, 1), (inverseSquaredPatchedDysGPU, patchedDysGPU, patchedDatasSizeGPU))
        calcEdgeEffectCorrectionsGPU = module.get_function('calcEdgeEffectCorrections')
        blockSize, gridSizeX = calcGridBlockSize(singleCalcPeriods)
        calcEdgeEffectCorrectionsGPU((gridSizeX, 1, 1), (blockSize, 1, 1), (edgeEffectCorrectionsGPU, patchedDatasGPU, inverseSquaredPatchedDysGPU, patchedDatasSizeGPU, maxDurationGPU_cached, periodSizeGPU_cached))
        cumsumGPU[:] = cp.cumsum(patchedDatasGPU, axis=1)
        patchedDatasSize_local = patchedDatasGPU.shape[1]
        base_error = base_error_cached
        kernel_calc_error = module.get_function('calculate_base_error')
        block_dim_1d = (256,)
        grid_dim_2d = ((patchedDatasSize_local + block_dim_1d[0] - 1) // block_dim_1d[0], singleCalcPeriods)
        kernel_calc_error(grid=grid_dim_2d, block=block_dim_1d, args=(base_error, patchedDatasGPU, inverseSquaredPatchedDysGPU, patchedDatasSize_local, singleCalcPeriods))
        error_prefix_sum = cp.cumsum(base_error, axis=1)
        calcAllFullSumGPU_v2 = module.get_function('calcAllFullSum_v2')
        blockSize, gridSizeX = calcGridBlockSize(len(singleDurations))
        calcAllFullSumGPU_v2((gridSizeX, singleCalcPeriods, 1), (blockSize, 1, 1), (fullSumGPU, error_prefix_sum, np.int32(patchedDatasSize_local), durationsGPU, np.int32(len(singleDurations)), np.int32(singleCalcPeriods)))
        kernel_final_ootr = module.get_function('calculate_final_ootr_v3')
        grid_dim_3d = ((tSize + 255) // 256, len(singleDurations), singleCalcPeriods)
        block_dim_3d = (256, 1, 1)
        kernel_final_ootr(grid=grid_dim_3d, block=block_dim_3d, args=(ootrGPU, error_prefix_sum, fullSumGPU, durationsGPU, tSize, patchedDatasSize_local, len(singleDurations), singleCalcPeriods))
        calcAllLowestResidualsGPU = module.get_function('calcAllLowestResidualsGPUB_SignalTiled_v2')
        blockSize, gridSizeX = calcGridBlockSize(tSize)
        calcAllLowestResidualsGPU((gridSizeX, len(singleDurations), singleCalcPeriods), (blockSize, 1, 1), (lowestResidualsGPU, tSizeGPU_cached, patchedDatasGPU, patchedDatasSizeGPU, durationsGPU, durationsSizeGPU, lcArrFullLengthGPU, lcArrMaxLenGPU, inverseSquaredPatchedDysGPU, overshootGPU, ootrGPU, fullSumGPU, edgeEffectCorrectionsGPU, datapointsGPU_cached, cumsumGPU, transitDepthMinGPU_cached))
        start_idx = iterFlag * singleCalcPeriods
        end_idx = start_idx + singleCalcPeriods
        valid_range = min(end_idx, len(periods)) - start_idx
        valid_lowest_residuals = lowestResidualsGPU[:valid_range]
        flattened_residuals = valid_lowest_residuals.reshape(valid_range, -1)
        min_indices = cp.argmin(flattened_residuals, axis=-1)
        min_values = flattened_residuals[cp.arange(valid_range), min_indices]
        locationGPU[start_idx:start_idx + valid_range] = min_indices
        LowestResidualsEachPeriodGPU[start_idx:start_idx + valid_range] = min_values
        iterFlagGPU = iterFlagGPU + 1
        del lowestResidualsGPU, periodsGPU, durationsMaxGPU, durationsMinGPU
        del overshootGPU, durationsGPU, durationsSizeGPU
        del fullSumGPU, ootrGPU
        if 'lcArrFullLengthGPU' in dir():
            del lcArrFullLengthGPU, lcArrMaxLenGPU
        if verbose:
            pbar.update(1)
    chi2 = LowestResidualsEachPeriodGPU.get()
    raw_chi2 = chi2.copy()
    median = np.median(raw_chi2)
    chi2_mask = raw_chi2 > 100 * median
    chi2 = ma.array(raw_chi2, mask=chi2_mask)
    periods = ma.array(periods, mask=chi2_mask)
    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    raw_power = power.copy()
    if fast:
        return (periods, power)
    combined = list(enumerate(zip(periods, -power)))
    sorted_combined = sorted(combined, key=lambda x: x[1][1])
    top_100_indices = [item[0] for item in sorted_combined[:100]]
    top_100_periods = [item[1][0] for item in sorted_combined[:100]]
    remaining_combined = [item for item in sorted_combined if item[0] not in top_100_indices]
    remaining_combined_greater_than_1 = [item for item in remaining_combined if item[1][0] > 1]
    sorted_remaining_combined_greater_than_1 = sorted(remaining_combined_greater_than_1, key=lambda x: x[1][1])
    next_100_indices = [item[0] for item in sorted_remaining_combined_greater_than_1[:100]]
    next_100_periods = [item[1][0] for item in sorted_remaining_combined_greater_than_1[:100]]
    possiblePeriodsIndices = top_100_indices + next_100_indices
    possiblePeriods = top_100_periods + next_100_periods
    del phasesGPU_cached, sortIndexGPU_cached, patchedDatasGPU_cached, patchedDysGPU_cached
    del edgeEffectCorrectionsGPU_cached, inverseSquaredPatchedDysGPU_cached
    del cumsumGPU_cached, base_error_cached
    del tGPU_cached, tSizeGPU_cached, tLengthGPU_cached, periodsSizeGPU_cached
    del maxDurationGPU_cached, periodSizeGPU_cached, datapointsGPU_cached, transitDepthMinGPU_cached
    del yGPU, dyGPU, locationGPU, LowestResidualsEachPeriodGPU
    del durationBoolArrayGPU, durationsGridCollectionGPU
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    chi2_again = search_multi_periods_again(possiblePeriods, t, y, dy, transit_depth_min, lc_arr, lc_cache_overview, GPUDeviceID, singleCalcPeriods)
    chi2[possiblePeriodsIndices] = chi2_again
    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    power_again = power[possiblePeriodsIndices]
    periodIndex = possiblePeriodsIndices[np.argmax(power_again)]
    period = periods[periodIndex]
    possiblePeriodsTimesRate = [0.5, 1, 2, 2 / 3, 3 / 2]
    possiblePeriodsTemp = [period * rate for rate in possiblePeriodsTimesRate]
    possiblePeriodsIndices_multi, possiblePeriods_multi = find_nearest_indices(possiblePeriodsTemp, periods)
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    chi2_again = search_multi_periods_again(possiblePeriods_multi, t, y, dy, transit_depth_min, lc_arr, lc_cache_overview, GPUDeviceID, singleCalcPeriods)
    chi2[possiblePeriodsIndices_multi] = chi2_again
    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    power_again = power[possiblePeriodsIndices_multi]
    periodIndex = possiblePeriodsIndices_multi[np.argmax(power_again)]
    period = periods[periodIndex]
    rawDuration, durationPointsNum, transit_duration_in_days, transitDepth, T0, transit_times, snr, snr_pink, snrFit, snrFitPink = search_single_periods(period, t, y, dy, transit_depth_min, lc_arr, lc_cache_overview, GPUDeviceID)
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return (periods, period, rawDuration, durationPointsNum, transit_duration_in_days, transitDepth, T0, SDE, chi2, transit_times, power, snr, snr_pink, snrFit, snrFitPink, raw_power, raw_chi2, possiblePeriodsIndices, possiblePeriods)
