#!/usr/bin/env python
"""
Standalone worker script for multi-GPU processing.
This script is invoked via subprocess to avoid multiprocessing spawn issues.
"""
import sys
import pickle
import numpy as np
import cupy as cp
import pynvml
import tqdm

def set_cuda_device(device_id):
    """Set the CUDA device."""
    cp.cuda.Device(device_id).use()

def calcGridBlockSize(size):
    MAX_BLOCK_SIZE = 128
    blockSize = size
    if blockSize > MAX_BLOCK_SIZE:
        blockSize = MAX_BLOCK_SIZE
    gridSizeX = int((size / blockSize) + 1)
    return blockSize, gridSizeX

def run_worker(input_file, output_file):
    # Load input data
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    periods_chunk = data['periods_chunk']
    chunk_start_idx = data['chunk_start_idx']
    total_periods = data.get('total_periods', len(periods_chunk))  # For consistent singleCalcPeriods
    t = data['t']
    y = data['y']
    dy = data['dy']
    transit_depth_min = data['transit_depth_min']
    lc_arr = data['lc_arr']
    lc_cache_overview = data['lc_cache_overview']
    T0_fit_margin = data['T0_fit_margin']
    GPUDeviceID = data['GPUDeviceID']
    verbose = data.get('verbose', True)
    bar_location = data.get('bar_location', 0)
    
    # Import GPUFun from gputls package
    from gputls import GPUFun
    
    # Set GPU device
    set_cuda_device(GPUDeviceID)
    
    # Build GPU module
    GPUCode = GPUFun.getGPUCode()
    if T0_fit_margin == 0:
        GPUCode = GPUCode.replace('#define SKIP_POINT 8', '#define SKIP_POINT ' + '0x7f800000')
    else:
        GPUCode = GPUCode.replace('#define SKIP_POINT 8', '#define SKIP_POINT ' + str(int(1/T0_fit_margin)))
    module = cp.RawModule(code=GPUCode)
    module.compile()

    durations, indices = np.unique(lc_cache_overview["width_in_samples"], return_index=True)
    lc_arr_local = lc_arr[indices]
    lc_cache_overview_local = lc_cache_overview[indices]
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
    singleCalcPeriods_max = (nvmlinfo.free) / (5*(patchedDatasSize * 2 + 2 + len(durations)*patchedDatasSize*4 + 2*len(durations)))

    # Use total_periods / 30 as the base (same as core.py's singleCalcPeriods_estimate)
    # Only reduce if GPU memory is insufficient
    singleCalcPeriods_base = max(1, int(total_periods / 30))
    singleCalcPeriods = int(np.min([np.floor(singleCalcPeriods_max), singleCalcPeriods_base]))

    if singleCalcPeriods < 15:
        singleCalcPeriods = int(singleCalcPeriods / 1.1)
    
    if singleCalcPeriods < 1:
        singleCalcPeriods = 1

    # Calculate global batch boundaries that this chunk intersects with
    # Global batch i covers periods [i*singleCalcPeriods, (i+1)*singleCalcPeriods)
    chunk_end_idx = chunk_start_idx + len(periods_chunk)
    
    # Find the first and last global batch that this chunk intersects
    first_global_batch = chunk_start_idx // singleCalcPeriods
    last_global_batch = (chunk_end_idx - 1) // singleCalcPeriods
    
    TotalIter = last_global_batch - first_global_batch + 1

    # Initialize the variables
    locationGPU = cp.empty(len(periods_chunk), dtype=cp.int32)
    LowestResidualsEachPeriodGPU = cp.empty(len(periods_chunk), dtype=cp.float32)

    fulldurationsMaxGPU = cp.empty((len(periods_chunk),), dtype=cp.int32)
    fulldurationsMinGPU = cp.empty((len(periods_chunk),), dtype=cp.int32)
    fullperiodsSizeGPU = cp.asarray(np.array([len(periods_chunk)])).astype(cp.int32)
    tSizeGPU = cp.asarray(np.array([tSize])).astype(cp.int32)
    tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)
    periodsGPU = cp.asarray(periods_chunk).astype(cp.float64)
    
    durationsGridGPU = module.get_function('durationsGrid')
    blockSize, gridSizeX = calcGridBlockSize(len(periods_chunk))
    durationsGridGPU((gridSizeX,1,1), (blockSize,),
                    (periodsGPU, fulldurationsMaxGPU, fulldurationsMinGPU, tLengthGPU, tSizeGPU, fullperiodsSizeGPU))

    fulldurationsSizeGPU = cp.asarray(np.array([len(durations)])).astype(cp.int32)
    fulldurationsGPU = cp.asarray(durations).astype(cp.int32)
    durationBoolArrayGPU = cp.empty((len(periods_chunk), len(durations)), dtype=cp.bool_)

    durationBoolFunGPU = module.get_function('durationBool')
    blockSize, gridSizeX = calcGridBlockSize(len(periods_chunk))
    durationBoolFunGPU((gridSizeX, len(durations), 1), (blockSize, 1, 1),
                    (fulldurationsMaxGPU, fulldurationsMinGPU, fulldurationsSizeGPU, fullperiodsSizeGPU, fulldurationsGPU, durationBoolArrayGPU))

    durationsGridCollectionGPU = cp.empty((TotalIter, len(durations)), dtype=cp.bool_)

    yGPU = cp.asarray(y).astype(cp.float32)
    dyGPU = cp.asarray(dy).astype(cp.float32)

    # Cached variables
    tGPU_cached = cp.asarray(t).astype(cp.float64)
    tSizeGPU_cached = cp.asarray(np.array([tSize])).astype(cp.int32)
    tLengthGPU_cached = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)
    periodsSizeGPU_cached = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
    maxDurationGPU_cached = cp.asarray(np.array([maxDuration])).astype(cp.int32)
    periodSizeGPU_cached = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
    datapointsGPU_cached = cp.array([len(y)]).astype(cp.int32)
    transitDepthMinGPU_cached = cp.array([transit_depth_min]).astype(cp.float32)
    
    # Pre-allocate reusable arrays
    phasesGPU_cached = cp.empty((singleCalcPeriods, tSize), dtype=cp.float64)
    sortIndexGPU_cached = cp.empty((singleCalcPeriods, tSize), dtype=cp.int32)
    patchedDatasGPU_cached = cp.empty((singleCalcPeriods, tSize + maxDuration), dtype=cp.float32)
    patchedDysGPU_cached = cp.empty((singleCalcPeriods, tSize + maxDuration), dtype=cp.float32)
    edgeEffectCorrectionsGPU_cached = cp.empty((singleCalcPeriods), dtype=cp.float32)
    inverseSquaredPatchedDysGPU_cached = cp.empty((singleCalcPeriods, tSize + maxDuration), dtype=cp.float32)
    cumsumGPU_cached = cp.empty((singleCalcPeriods, patchedDatasSize), dtype=cp.float32)
    base_error_cached = cp.empty((singleCalcPeriods, patchedDatasSize), dtype=cp.float32)

    # Initialize progress bar if verbose
    pbar = None
    if verbose:
        pbar = tqdm.tqdm(total=TotalIter, position=bar_location, desc=f"GPU {GPUDeviceID}", leave=True)

    for iterFlag in range(TotalIter):
        # Calculate global batch index
        global_batch = first_global_batch + iterFlag
        
        # Global batch covers periods [global_start, global_end) in the full periods array
        global_start = global_batch * singleCalcPeriods
        global_end = min((global_batch + 1) * singleCalcPeriods, total_periods)
        
        # Map to local chunk indices
        local_start = max(0, global_start - chunk_start_idx)
        local_end = min(len(periods_chunk), global_end - chunk_start_idx)
        
        # Get periods for this batch (need exactly singleCalcPeriods for GPU arrays)
        actual_period_count = local_end - local_start
        SinglePeriods = periods_chunk[local_start:local_end]
        
        if actual_period_count < singleCalcPeriods:
            # Pad to singleCalcPeriods for GPU computation
            SinglePeriods = np.append(SinglePeriods, 
                                     np.full(singleCalcPeriods - actual_period_count, SinglePeriods[-1]))
        
        # Compute duration bool OR for this batch (using local indices)
        temp_bool = durationBoolArrayGPU[local_start]
        for i in range(local_start + 1, local_end):
            temp_bool = cp.logical_or(temp_bool, durationBoolArrayGPU[i])
        durationsGridCollectionGPU[iterFlag] = temp_bool

        durationsBoolGrid = durationsGridCollectionGPU[iterFlag].get()
        singleDurations = durations[durationsBoolGrid]
        
        if len(singleDurations) == 0:
            LowestResidualsEachPeriodGPU[local_start:local_end] = cp.nan
            continue
            
        single_lc_arr = lc_arr_local[durationsBoolGrid]
        single_lc_cache_overview = lc_cache_overview_local[durationsBoolGrid]
        overshootGPU = cp.array(single_lc_cache_overview["overshoot"]).astype(cp.float32)

        periodsGPU = cp.asarray(SinglePeriods).astype(cp.float64)
        durationsMaxGPU = cp.asarray(SinglePeriods).astype(cp.int32)
        durationsMinGPU = cp.asarray(SinglePeriods).astype(cp.int32)

        lowestResidualsGPU = cp.empty((singleCalcPeriods, len(singleDurations), tSize), dtype=cp.float32)

        phasesGPU = phasesGPU_cached
        sortIndexGPU = sortIndexGPU_cached

        durationsGridGPU = module.get_function('durationsGrid')
        blockSize, gridSizeX = calcGridBlockSize(singleCalcPeriods)
        durationsGridGPU((gridSizeX,1,1), (blockSize,),
                        (periodsGPU, durationsMaxGPU, durationsMinGPU, tLengthGPU_cached, tSizeGPU_cached, periodsSizeGPU_cached))

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
        fastFoldGPU((gridSizeX, singleCalcPeriods,), (blockSize,), (tGPU_cached, periodsGPU, phasesGPU, periodsSizeGPU_cached, tSizeGPU_cached))

        i_max = 10
        for i in range(1, i_max + 1):
            sortIndexGPU[int((i-1)*singleCalcPeriods/i_max):int(i*singleCalcPeriods/i_max)] = phasesGPU[int((i-1)*singleCalcPeriods/i_max):int(i*singleCalcPeriods/i_max)].argsort()

        patchDataGPU = module.get_function('patchData')
        blockSize, gridSizeX = calcGridBlockSize(tSize + maxDuration)
        patchDataGPU((gridSizeX, singleCalcPeriods,), (blockSize,),
        (patchedDatasGPU, patchedDysGPU, patchedDatasSizeGPU, sortIndexGPU,
        maxDurationGPU_cached, yGPU, dyGPU, tSizeGPU_cached))

        calcInverseSquaredPatchedDyGPU = module.get_function('calcInverseSquaredPatchedDy')
        blockSize, gridSizeX = calcGridBlockSize(patchedDatasSize)
        calcInverseSquaredPatchedDyGPU((gridSizeX, singleCalcPeriods, 1), (blockSize, 1, 1),
        (inverseSquaredPatchedDysGPU, patchedDysGPU, patchedDatasSizeGPU,))

        calcEdgeEffectCorrectionsGPU = module.get_function('calcEdgeEffectCorrections')
        blockSize, gridSizeX = calcGridBlockSize(singleCalcPeriods)
        calcEdgeEffectCorrectionsGPU((gridSizeX, 1, 1), (blockSize, 1, 1),
        (edgeEffectCorrectionsGPU, patchedDatasGPU, inverseSquaredPatchedDysGPU,
        patchedDatasSizeGPU, maxDurationGPU_cached, periodSizeGPU_cached,))
        
        for i in range(singleCalcPeriods):
            cumsumGPU[i] = cp.cumsum(patchedDatasGPU[i])

        patchedDatasSize_local = patchedDatasGPU.shape[1]

        base_error = base_error_cached
        kernel_calc_error = module.get_function('calculate_base_error')
        block_dim_1d = (256,)
        grid_dim_2d = ((patchedDatasSize_local + block_dim_1d[0] - 1) // block_dim_1d[0], singleCalcPeriods)

        kernel_calc_error(
            grid=grid_dim_2d, block=block_dim_1d,
            args=(base_error, patchedDatasGPU, inverseSquaredPatchedDysGPU, patchedDatasSize_local, singleCalcPeriods)
        )

        error_prefix_sum = cp.cumsum(base_error, axis=1)

        calcAllFullSumGPU_v2 = module.get_function('calcAllFullSum_v2')
        blockSize, gridSizeX = calcGridBlockSize(len(singleDurations))
        calcAllFullSumGPU_v2((gridSizeX, singleCalcPeriods, 1), (blockSize, 1, 1),
        (fullSumGPU, error_prefix_sum,
        np.int32(patchedDatasSize_local), durationsGPU, np.int32(len(singleDurations)),
        np.int32(singleCalcPeriods),))

        kernel_final_ootr = module.get_function('calculate_final_ootr_v3')
        grid_dim_3d = ((tSize + 255) // 256, len(singleDurations), singleCalcPeriods)
        block_dim_3d = (256, 1, 1)

        kernel_final_ootr(
            grid=grid_dim_3d, block=block_dim_3d,
            args=(
                ootrGPU, 
                error_prefix_sum,
                fullSumGPU,
                durationsGPU,
                tSize,
                patchedDatasSize_local,
                len(singleDurations),
                singleCalcPeriods
            )
        )
        
        calcAllLowestResidualsGPU = module.get_function('calcAllLowestResidualsGPUB_SignalTiled_v2')
        blockSize, gridSizeX = calcGridBlockSize(tSize)
        calcAllLowestResidualsGPU((gridSizeX, len(singleDurations), singleCalcPeriods),
        (blockSize, 1, 1), (lowestResidualsGPU, tSizeGPU_cached,
        patchedDatasGPU, patchedDatasSizeGPU,
        durationsGPU, durationsSizeGPU,
        lcArrFullLengthGPU,
        lcArrMaxLenGPU, inverseSquaredPatchedDysGPU,
        overshootGPU, ootrGPU, fullSumGPU, edgeEffectCorrectionsGPU, datapointsGPU_cached, cumsumGPU,
        transitDepthMinGPU_cached
        ))

        # Store results using local indices
        valid_lowest_residuals = lowestResidualsGPU[:actual_period_count]
        flattened_residuals = valid_lowest_residuals.reshape(actual_period_count, -1)
        min_indices = cp.argmin(flattened_residuals, axis=-1)
        min_values = flattened_residuals[cp.arange(actual_period_count), min_indices]
        
        locationGPU[local_start:local_end] = min_indices
        LowestResidualsEachPeriodGPU[local_start:local_end] = min_values

        # Update progress bar
        if pbar is not None:
            pbar.update(1)

    # Close progress bar
    if pbar is not None:
        pbar.close()

    chi2_chunk = LowestResidualsEachPeriodGPU.get()
    
    # Clean up GPU memory before saving
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    
    # Save output
    with open(output_file, 'wb') as f:
        pickle.dump(chi2_chunk, f)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python _worker.py <input_file> <output_file>")
        sys.exit(1)
    
    run_worker(sys.argv[1], sys.argv[2])
