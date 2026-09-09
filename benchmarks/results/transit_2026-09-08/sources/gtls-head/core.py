import numpy as np
import numpy.ma as ma
import cupy as cp
from .stats import spectra,all_transit_times,calculate_transit_duration_in_days,intransit_stats,snr_stats,calcDurationDays
from .helpers import transit_mask
from .transit import mutipleTransitFit
from . import GPUFun
import pynvml
import tqdm
import warnings
import subprocess
import tempfile
import pickle
import os

def foldCPU(time,flux,dy,period):
    """Fold time series data."""
    phase = (time % period) / period
    rank = np.argsort(phase)
    return np.array(time[rank]),flux[rank],dy[rank]

def set_cuda_device(device_id):
    """Set the CUDA device."""
    cp.cuda.Device(device_id).use()

def calcGridBlockSize(size):
    MAX_BLOCK_SIZE = 128
    blockSize = size
    if blockSize > MAX_BLOCK_SIZE:
        blockSize = MAX_BLOCK_SIZE
    gridSizeX = int((size / blockSize) + 1)
    return blockSize,gridSizeX

def find_nearest_indices(a, b):
    a = np.array(a)
    b = np.array(b)

    nearest_indices = np.zeros(a.shape, dtype=int)
    nearest_elements = np.zeros(a.shape)

    for i, val in enumerate(a):
        nearest_index = np.argmin(np.abs(b - val))
        nearest_indices[i] = nearest_index
        nearest_elements[i] = b[nearest_index]
    
    return nearest_indices, nearest_elements

# ============== Multi-GPU Support Functions ==============

def _search_periods_chunk_worker(args):
    """
    Worker function for multi-GPU parallel processing.
    Each worker processes a chunk of periods on a specific GPU.
    
    This function must be at module level for multiprocessing to work.
    """
    (periods_chunk, chunk_start_idx, t, y, dy, transit_depth_min, R_star_min, R_star_max,
     M_star_min, M_star_max, lc_arr, lc_cache_overview, T0_fit_margin,
     oversampling_factor, GPUDeviceID, SimplifyEdgeEffect) = args
    
    # Import cupy inside worker to ensure fresh CUDA context
    import cupy as cp
    
    # Set GPU device for this worker
    set_cuda_device(GPUDeviceID)
    
    # Call the original search function with fast=True to only get chi2
    GPUCode = GPUFun.getGPUCode()
    if T0_fit_margin == 0:
        GPUCode = GPUCode.replace('#define SKIP_POINT 8','#define SKIP_POINT ' + '0x7f800000')
    else:
        GPUCode = GPUCode.replace('#define SKIP_POINT 8','#define SKIP_POINT ' + str(int(1/T0_fit_margin)))
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

    singleCalcPeriods = int(np.min([np.floor(singleCalcPeriods_max), len(periods_chunk) / 30]))

    if singleCalcPeriods < 15:
        singleCalcPeriods = int(singleCalcPeriods / 1.1)
    
    if singleCalcPeriods < 1:
        singleCalcPeriods = 1

    TotalIter = int(np.ceil(len(periods_chunk) / singleCalcPeriods))

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

    for iterFlag in range(TotalIter):
        if iterFlag == TotalIter - 1:
            SinglePeriods = periods_chunk[iterFlag*singleCalcPeriods:]
            actual_period_count = len(SinglePeriods)
            if actual_period_count < singleCalcPeriods:
                SinglePeriods = np.append(SinglePeriods, 
                                         np.full(singleCalcPeriods - actual_period_count, SinglePeriods[-1]))
            start_idx = iterFlag*singleCalcPeriods
            end_idx = min(start_idx + actual_period_count, len(periods_chunk))
            temp_bool = durationBoolArrayGPU[start_idx]
            for i in range(start_idx + 1, end_idx):
                temp_bool = cp.logical_or(temp_bool, durationBoolArrayGPU[i])
            durationsGridCollectionGPU[iterFlag] = temp_bool
        else:
            SinglePeriods = periods_chunk[iterFlag*singleCalcPeriods:(iterFlag+1)*singleCalcPeriods]
            start_idx = iterFlag*singleCalcPeriods
            end_idx = (iterFlag+1)*singleCalcPeriods
            temp_bool = durationBoolArrayGPU[start_idx]
            for i in range(start_idx + 1, end_idx):
                temp_bool = cp.logical_or(temp_bool, durationBoolArrayGPU[i])
            durationsGridCollectionGPU[iterFlag] = temp_bool

        durationsBoolGrid = durationsGridCollectionGPU[iterFlag].get()
        singleDurations = durations[durationsBoolGrid]
        
        if len(singleDurations) == 0:
            start_idx = iterFlag * singleCalcPeriods
            valid_range = min(start_idx + singleCalcPeriods, len(periods_chunk)) - start_idx
            LowestResidualsEachPeriodGPU[start_idx:start_idx + valid_range] = cp.nan
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
            sortIndexGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max] = phasesGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max].argsort()

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

        start_idx = iterFlag * singleCalcPeriods
        end_idx = start_idx + singleCalcPeriods
        valid_range = min(end_idx, len(periods_chunk)) - start_idx
        valid_lowest_residuals = lowestResidualsGPU[:valid_range]
        flattened_residuals = valid_lowest_residuals.reshape(valid_range, -1)
        min_indices = cp.argmin(flattened_residuals, axis=-1)
        min_values = flattened_residuals[cp.arange(valid_range), min_indices]
        
        locationGPU[start_idx:start_idx + valid_range] = min_indices
        LowestResidualsEachPeriodGPU[start_idx:start_idx + valid_range] = min_values

    chi2_chunk = LowestResidualsEachPeriodGPU.get()
    
    return chunk_start_idx, chi2_chunk


def search_multi_periods_multiGPU(
    periods,
    t,
    y,
    dy,
    transit_depth_min,
    R_star_min,
    R_star_max,
    M_star_min,
    M_star_max,
    lc_arr,
    lc_cache_overview,
    T0_fit_margin,
    oversampling_factor,
    verbose,
    useLocalPTXCUBIN = False,
    GPUDeviceIDs = [0, 1],  # List of GPU IDs to use
    fast = False,
    legacy = False,
    SimplifyEdgeEffect = True,
    bar_location = 0
):
    """
    Multi-GPU version of search_multi_periods.
    Splits the periods array across multiple GPUs and processes in parallel.
    
    Parameters
    ----------
    GPUDeviceIDs : list of int
        List of GPU device IDs to use for parallel processing.
        Default is [0, 1] for 2 GPUs.
    
    Other parameters are the same as search_multi_periods.
    """
    n_gpus = len(GPUDeviceIDs)
    
    if n_gpus == 1:
        # Fall back to single GPU version
        return search_multi_periods(
            periods, t, y, dy, transit_depth_min, R_star_min, R_star_max,
            M_star_min, M_star_max, lc_arr, lc_cache_overview, T0_fit_margin,
            oversampling_factor, verbose, useLocalPTXCUBIN, GPUDeviceIDs[0],
            fast, legacy, SimplifyEdgeEffect, bar_location
        )
    
    # Pre-calculate singleCalcPeriods to align chunk boundaries with batch boundaries
    # This ensures that each GPU processes complete batches, avoiding duration bool inconsistencies
    durations_temp, _ = np.unique(lc_cache_overview["width_in_samples"], return_index=True)
    maxDuration_temp = int(max(durations_temp))
    if maxDuration_temp % 2 != 0:
        maxDuration_temp = maxDuration_temp + 1
    tSize_temp = len(t)
    patchedDatasSize_temp = int(tSize_temp + maxDuration_temp)
    
    # Estimate singleCalcPeriods (we'll use a conservative estimate)
    # The actual value depends on GPU memory, but we use len(periods)/30 as the formula
    singleCalcPeriods_estimate = max(1, int(len(periods) / 30))
    
    # Calculate total number of batches
    total_batches = int(np.ceil(len(periods) / singleCalcPeriods_estimate))
    
    # Distribute batches evenly across GPUs
    batches_per_gpu = int(np.ceil(total_batches / n_gpus))
    
    # Calculate chunk boundaries aligned to batch boundaries
    chunk_start_indices = []
    periods_chunks = []
    for i in range(n_gpus):
        start_batch = i * batches_per_gpu
        end_batch = min((i + 1) * batches_per_gpu, total_batches)
        
        start_idx = start_batch * singleCalcPeriods_estimate
        end_idx = min(end_batch * singleCalcPeriods_estimate, len(periods))
        
        if start_idx < len(periods):
            chunk_start_indices.append(start_idx)
            periods_chunks.append(periods[start_idx:end_idx])
    
    # Update n_gpus if some GPUs got no work
    n_gpus = len(periods_chunks)
    GPUDeviceIDs = GPUDeviceIDs[:n_gpus]
    
    if verbose:
        print(f"Multi-GPU: Splitting {len(periods)} periods across {n_gpus} GPUs")
        print(f"  (batch size = {singleCalcPeriods_estimate}, aligned to batch boundaries)")
        for i, (gpu_id, chunk, start_idx) in enumerate(zip(GPUDeviceIDs, periods_chunks, chunk_start_indices)):
            print(f"  GPU {gpu_id}: {len(chunk)} periods (starting at index {start_idx})")
    
    # Use subprocess to run workers in completely separate Python processes
    # This avoids the multiprocessing spawn issues entirely
    chi2_results = {}
    processes = []
    temp_files = []
    
    try:
        # Create temporary files for input/output
        for i, (gpu_id, chunk, start_idx) in enumerate(zip(GPUDeviceIDs, periods_chunks, chunk_start_indices)):
            # Save input data to temp file
            input_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
            temp_files.extend([input_file.name, output_file.name])
            
            input_data = {
                'periods_chunk': chunk,
                'chunk_start_idx': start_idx,
                'total_periods': len(periods),  # Pass total periods count for consistent singleCalcPeriods
                't': t, 'y': y, 'dy': dy,
                'transit_depth_min': transit_depth_min,
                'R_star_min': R_star_min, 'R_star_max': R_star_max,
                'M_star_min': M_star_min, 'M_star_max': M_star_max,
                'lc_arr': lc_arr, 'lc_cache_overview': lc_cache_overview,
                'T0_fit_margin': T0_fit_margin,
                'oversampling_factor': oversampling_factor,
                'GPUDeviceID': gpu_id,
                'SimplifyEdgeEffect': SimplifyEdgeEffect,
                'verbose': verbose,
                'bar_location': i  # Each GPU gets its own progress bar position
            }
            pickle.dump(input_data, input_file)
            input_file.close()
            
            # Launch subprocess - don't capture stdout/stderr so progress bars show
            worker_script = os.path.join(os.path.dirname(__file__), '_worker.py')
            proc = subprocess.Popen(
                ['python', worker_script, input_file.name, output_file.name],
                stdout=None, stderr=None  # Let output go to terminal
            )
            processes.append((proc, start_idx, output_file.name))
        
        # Wait for all processes and collect results
        for proc, start_idx, output_file in processes:
            proc.wait()  # Wait for process to complete
            if proc.returncode != 0:
                raise RuntimeError(f"Worker process on GPU failed with code {proc.returncode}")
            
            with open(output_file, 'rb') as f:
                chi2_chunk = pickle.load(f)
            chi2_results[start_idx] = chi2_chunk
            
    except Exception as e:
        if verbose:
            warnings.warn(f"Multi-GPU subprocess execution failed: {e}. Falling back to sequential execution.")
        # Fallback to sequential execution
        chi2_results = {}
        for gpu_id, chunk, start_idx in zip(GPUDeviceIDs, periods_chunks, chunk_start_indices):
            args = (chunk, start_idx, t, y, dy, transit_depth_min, R_star_min, R_star_max,
                    M_star_min, M_star_max, lc_arr, lc_cache_overview, T0_fit_margin,
                    oversampling_factor, gpu_id, SimplifyEdgeEffect)
            chunk_start_idx, chi2_chunk = _search_periods_chunk_worker(args)
            chi2_results[chunk_start_idx] = chi2_chunk
    finally:
        # Cleanup temp files
        for f in temp_files:
            try:
                os.unlink(f)
            except:
                pass
    
    # Merge results in correct order
    chi2 = np.concatenate([chi2_results[idx] for idx in sorted(chi2_results.keys())])
    
    raw_chi2 = chi2.copy()
    median = np.median(raw_chi2)
    chi2_mask = raw_chi2 > (100 * median)
    chi2 = ma.array(raw_chi2, mask=chi2_mask)
    periods_masked = ma.array(periods, mask=chi2_mask)

    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    raw_power = power.copy()

    if fast:
        return periods_masked, power

    # Use first GPU for the remaining processing
    primary_gpu = GPUDeviceIDs[0]
    
    # Calculate singleCalcPeriods for search_multi_periods_again
    tSize = len(t)
    durations = np.unique(lc_cache_overview["width_in_samples"])
    maxDuration = int(max(durations))
    if maxDuration % 2 != 0:
        maxDuration = maxDuration + 1
    patchedDatasSize = int(tSize + maxDuration)
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(primary_gpu)
    nvmlinfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    singleCalcPeriods_max = (nvmlinfo.free) / (5*(patchedDatasSize * 2 + 2 + len(durations)*patchedDatasSize*4 + 2*len(durations)))
    singleCalcPeriods = int(np.min([np.floor(singleCalcPeriods_max), len(periods) / 30]))
    if singleCalcPeriods < 15:
        singleCalcPeriods = int(singleCalcPeriods / 1.1)
    if singleCalcPeriods < 1:
        singleCalcPeriods = 1

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

    chi2_again = search_multi_periods_again(
        possiblePeriods, t, y, dy, transit_depth_min,
        lc_arr, lc_cache_overview, primary_gpu, singleCalcPeriods
    )

    chi2[possiblePeriodsIndices] = chi2_again

    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    power_again = power[possiblePeriodsIndices]
    periodIndex = possiblePeriodsIndices[np.argmax(power_again)]
    period = periods[periodIndex]

    possiblePeriodsTimesRate = [0.5, 1, 2, 2/3, 3/2]
    possiblePeriodsTemp = [period * rate for rate in possiblePeriodsTimesRate]

    possiblePeriodsIndices_multi, possiblePeriods_multi = find_nearest_indices(possiblePeriodsTemp, periods)

    chi2_again = search_multi_periods_again(
        possiblePeriods_multi, t, y, dy, transit_depth_min,
        lc_arr, lc_cache_overview, primary_gpu, singleCalcPeriods
    )

    chi2[possiblePeriodsIndices_multi] = chi2_again

    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    power_again = power[possiblePeriodsIndices_multi]
    periodIndex = possiblePeriodsIndices_multi[np.argmax(power_again)]
    period = periods[periodIndex]

    rawDuration, durationPointsNum, transit_duration_in_days, transitDepth, T0, transit_times, snr, snr_pink, snrFit, snrFitPink = search_single_periods(
        period, t, y, dy, transit_depth_min,
        lc_arr, lc_cache_overview, primary_gpu
    )

    # Clean up GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return periods_masked, period, rawDuration, durationPointsNum, transit_duration_in_days, transitDepth, T0,\
            SDE, chi2, transit_times, power, snr, snr_pink, snrFit, snrFitPink, raw_power, raw_chi2, possiblePeriodsIndices, possiblePeriods

# ============== End of Multi-GPU Support Functions ==============

def search_multi_periods(
    periods,
    t,
    y,
    dy,
    transit_depth_min,
    R_star_min,
    R_star_max,
    M_star_min,
    M_star_max,
    lc_arr,
    lc_cache_overview,
    T0_fit_margin,
    oversampling_factor,
    verbose,
    useLocalPTXCUBIN = False,
    GPUDeviceID = 0,
    #fast: just return periods and power, not the full result
    fast = False,
    #legacy: Skip-points search, like the original TLS,not implemented yet.
    legacy = False,
    # SimplifyEdgeEffect if dy is nearly uniform, we can simplify the edge effect correction
    SimplifyEdgeEffect = True,
    bar_location = 0
):
    
    # Choose the GPU device
    set_cuda_device(GPUDeviceID)

    GPUCode = GPUFun.getGPUCode()
    if T0_fit_margin == 0:
        GPUCode = GPUCode.replace('#define SKIP_POINT 8','#define SKIP_POINT ' + '0x7f800000')
    else:
        GPUCode = GPUCode.replace('#define SKIP_POINT 8','#define SKIP_POINT ' + str(int(1/T0_fit_margin)))
    module = cp.RawModule(code=GPUCode)
    module.compile()

    durations,indices = np.unique(lc_cache_overview["width_in_samples"],return_index=True)
    lc_arr = lc_arr[indices]
    lc_cache_overview = lc_cache_overview[indices]
    maxDuration = int(max(durations))

    # why?
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

    singleCalcPeriods = int(np.min([np.floor(singleCalcPeriods_max),len(periods) / 30]))

    if singleCalcPeriods < 15:
        singleCalcPeriods = int(singleCalcPeriods / 1.1)

    #Due to GPU memory size limitation, GPU can only do several periods at a time.
    TotalIter = int(np.ceil(len(periods) / singleCalcPeriods))

    if verbose:
        pbar = tqdm.tqdm(total=TotalIter,position=bar_location)

    #Initialize the variables
    periodsGPU = cp.empty((singleCalcPeriods,),dtype=cp.float64)
    durationsMaxGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)
    durationsMinGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)
    locationGPU = cp.empty(len(periods),dtype=cp.int32)
    LowestResidualsEachPeriodGPU = cp.empty(len(periods),dtype=cp.float32)

    iterFlagGPU = cp.int32(0)

    fulldurationsMaxGPU = cp.empty((len(periods),),dtype=cp.int32)
    fulldurationsMinGPU = cp.empty((len(periods),),dtype=cp.int32)
    fullperiodsSizeGPU = cp.asarray(np.array([len(periods)])).astype(cp.int32)
    tSizeGPU = cp.asarray(np.array([tSize])).astype(cp.int32)
    tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)
    periodsGPU = cp.asarray(periods).astype(cp.float64)
    
    durationsGridGPU = module.get_function('durationsGrid')
    blockSize,gridSizeX = calcGridBlockSize(len(periods))
    durationsGridGPU((gridSizeX,1,1),(blockSize,),
                    (periodsGPU,fulldurationsMaxGPU, fulldurationsMinGPU,tLengthGPU,tSizeGPU, fullperiodsSizeGPU))

    fulldurationsSizeGPU = cp.asarray(np.array([len(durations)])).astype(cp.int32)
    fulldurationsGPU = cp.asarray(durations).astype(cp.int32)
    durationBoolArrayGPU = cp.empty((len(periods),len(durations)),dtype=cp.bool_)

    durationBoolFunGPU = module.get_function('durationBool')
    blockSize,gridSizeX = calcGridBlockSize(len(periods))
    durationBoolFunGPU((gridSizeX,len(durations),1),(blockSize,1,1),
                    (fulldurationsMaxGPU,fulldurationsMinGPU,fulldurationsSizeGPU,fullperiodsSizeGPU,fulldurationsGPU,durationBoolArrayGPU))

    durationsGridCollectionGPU = cp.empty((TotalIter,len(durations)),dtype=cp.bool_)

    yGPU = cp.asarray(y).astype(cp.float32)
    dyGPU = cp.asarray(dy).astype(cp.float32)

    # === 缓存循环内不变的 GPU 变量 (search_multi_periods) ===
    tGPU_cached = cp.asarray(t).astype(cp.float64)
    tSizeGPU_cached = cp.asarray(np.array([tSize])).astype(cp.int32)
    tLengthGPU_cached = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)
    periodsSizeGPU_cached = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
    maxDurationGPU_cached = cp.asarray(np.array([maxDuration])).astype(cp.int32)
    periodSizeGPU_cached = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
    datapointsGPU_cached = cp.array([len(y)]).astype(cp.int32)
    transitDepthMinGPU_cached = cp.array([transit_depth_min]).astype(cp.float32)
    
    # 预分配循环内可复用的数组
    phasesGPU_cached = cp.empty((singleCalcPeriods, tSize), dtype=cp.float64)
    sortIndexGPU_cached = cp.empty((singleCalcPeriods, tSize), dtype=cp.int32)
    patchedDatasGPU_cached = cp.empty((singleCalcPeriods, tSize + maxDuration), dtype=cp.float32)
    patchedDysGPU_cached = cp.empty((singleCalcPeriods, tSize + maxDuration), dtype=cp.float32)
    edgeEffectCorrectionsGPU_cached = cp.empty((singleCalcPeriods), dtype=cp.float32)
    inverseSquaredPatchedDysGPU_cached = cp.empty((singleCalcPeriods, tSize + maxDuration), dtype=cp.float32)
    cumsumGPU_cached = cp.empty((singleCalcPeriods, patchedDatasSize), dtype=cp.float32)
    base_error_cached = cp.empty((singleCalcPeriods, patchedDatasSize), dtype=cp.float32)

    for iterFlag in range(TotalIter):

        if iterFlag == TotalIter - 1:
            SinglePeriods = periods[iterFlag*singleCalcPeriods:]
            actual_period_count = len(SinglePeriods)
            # 用最后一个有效period填充而不是0，避免除零错误
            if actual_period_count < singleCalcPeriods:
                SinglePeriods = np.append(SinglePeriods, 
                                         np.full(singleCalcPeriods - actual_period_count, SinglePeriods[-1]))
            # 正确处理duration bool: 使用所有有效periods的duration范围
            start_idx = iterFlag*singleCalcPeriods
            end_idx = min(start_idx + actual_period_count, len(periods))
            # 对所有有效的periods进行 logical_or 操作，获取这批periods需要的所有durations
            temp_bool = durationBoolArrayGPU[start_idx]
            for i in range(start_idx + 1, end_idx):
                temp_bool = cp.logical_or(temp_bool, durationBoolArrayGPU[i])
            durationsGridCollectionGPU[iterFlag] = temp_bool
        else:
            SinglePeriods = periods[iterFlag*singleCalcPeriods:(iterFlag+1)*singleCalcPeriods]
            # 修复：对当前批次的所有periods做logical_or，而不是只对比首尾两个
            start_idx = iterFlag*singleCalcPeriods
            end_idx = (iterFlag+1)*singleCalcPeriods
            temp_bool = durationBoolArrayGPU[start_idx]
            for i in range(start_idx + 1, end_idx):
                temp_bool = cp.logical_or(temp_bool, durationBoolArrayGPU[i])
            durationsGridCollectionGPU[iterFlag] = temp_bool

        durationsBoolGrid = durationsGridCollectionGPU[iterFlag].get()
        singleDurations = durations[durationsBoolGrid]
        
        # Check if any durations are selected
        if len(singleDurations) == 0:
            # Skip this batch and fill with NaN
            start_idx = iterFlag * singleCalcPeriods
            valid_range = min(start_idx + singleCalcPeriods, len(periods)) - start_idx
            LowestResidualsEachPeriodGPU[start_idx:start_idx + valid_range] = cp.nan
            if verbose:
                pbar.update(1)
            continue
        single_lc_arr = lc_arr[durationsBoolGrid]
        single_lc_cache_overview = lc_cache_overview[durationsBoolGrid]
        overshootGPU = cp.array(single_lc_cache_overview["overshoot"]).astype(cp.float32)

        periodsGPU = cp.asarray(SinglePeriods).astype(cp.float64)
        durationsMaxGPU = cp.asarray(SinglePeriods).astype(cp.int32)
        durationsMinGPU = cp.asarray(SinglePeriods).astype(cp.int32)

        lowestResidualsGPU = cp.empty((singleCalcPeriods,len(singleDurations),tSize),dtype=cp.float32)

        # 使用缓存的变量
        phasesGPU = phasesGPU_cached
        sortIndexGPU = sortIndexGPU_cached

        durationsGridGPU = module.get_function('durationsGrid')
        blockSize,gridSizeX = calcGridBlockSize(singleCalcPeriods)
        durationsGridGPU((gridSizeX,1,1),(blockSize,),
                        (periodsGPU,durationsMaxGPU, durationsMinGPU,tLengthGPU_cached,tSizeGPU_cached, periodsSizeGPU_cached))

        # 使用缓存的数组
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

        #GPU variables for the loop
        fullSumGPU = cp.empty((singleCalcPeriods,len(singleDurations)),dtype=cp.float32)
        cumsumGPU = cumsumGPU_cached
        ootrGPU = cp.empty((singleCalcPeriods,len(singleDurations),(tSize)),dtype=cp.float32)

        fastFoldGPU = module.get_function('foldFast')
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        fastFoldGPU((gridSizeX,singleCalcPeriods,),(blockSize,), (tGPU_cached, periodsGPU,phasesGPU,periodsSizeGPU_cached,tSizeGPU_cached))

        # # incase gpu memory is not enough, split the phasesGPU into several parts
        i_max = 10
        for i in range(1,i_max + 1):
            sortIndexGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max] = phasesGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max].argsort()
        # todo: change to below way, seems a bug here
        # sortIndexGPU = phasesGPU.argsort()

        #calculate patched data
        patchDataGPU = module.get_function('patchData')
        blockSize,gridSizeX = calcGridBlockSize(tSize + maxDuration)
        patchDataGPU((gridSizeX,singleCalcPeriods,),(blockSize,),
        (patchedDatasGPU,patchedDysGPU,patchedDatasSizeGPU,sortIndexGPU,
        maxDurationGPU_cached,yGPU,dyGPU,tSizeGPU_cached))

        calcInverseSquaredPatchedDyGPU = module.get_function('calcInverseSquaredPatchedDy')
        blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize)
        calcInverseSquaredPatchedDyGPU((gridSizeX,singleCalcPeriods,1),(blockSize,1,1),
        (inverseSquaredPatchedDysGPU,patchedDysGPU,patchedDatasSizeGPU,))

        calcEdgeEffectCorrectionsGPU = module.get_function('calcEdgeEffectCorrections')
        blockSize,gridSizeX = calcGridBlockSize(singleCalcPeriods)
        calcEdgeEffectCorrectionsGPU((gridSizeX,1,1),(blockSize,1,1),
        (edgeEffectCorrectionsGPU,patchedDatasGPU,inverseSquaredPatchedDysGPU,
        patchedDatasSizeGPU,maxDurationGPU_cached,periodSizeGPU_cached,))
        
        for i in range(singleCalcPeriods):
            # if((iterFlag * singleCalcPeriods + i) < singleCalcPeriods):
            cumsumGPU[i] = cp.cumsum(patchedDatasGPU[i])

        patchedDatasSize_local = patchedDatasGPU.shape[1] 

        # --- STAGE 1: Pre-compute Error Prefix Sum ---
        # This is shared between calcAllFullSum_v2 and calculate_final_ootr_v3

        # 1a. Use cached base_error array
        base_error = base_error_cached

        # 1b. Launch kernel to calculate base error.
        kernel_calc_error = module.get_function('calculate_base_error')
        block_dim_1d = (256,)
        grid_dim_2d = ((patchedDatasSize_local + block_dim_1d[0] - 1) // block_dim_1d[0], singleCalcPeriods)

        kernel_calc_error(
            grid=grid_dim_2d, block=block_dim_1d,
            args=(base_error, patchedDatasGPU, inverseSquaredPatchedDysGPU, patchedDatasSize_local, singleCalcPeriods)
        )

        # 1c. Perform cumsum on the ENTIRE base_error array.
        error_prefix_sum = cp.cumsum(base_error, axis=1)

        # --- OPTIMIZED: Use error_prefix_sum for fullsum calculation (O(1) instead of O(N)) ---
        calcAllFullSumGPU_v2 = module.get_function('calcAllFullSum_v2')
        blockSize,gridSizeX = calcGridBlockSize(len(singleDurations))
        calcAllFullSumGPU_v2((gridSizeX,singleCalcPeriods,1),(blockSize,1,1),
        (fullSumGPU, error_prefix_sum,
        np.int32(patchedDatasSize_local), durationsGPU, np.int32(len(singleDurations)),
        np.int32(singleCalcPeriods),))

        # --- STAGE 2: Calculate Final OOTR using the corrected kernel ---

        kernel_final_ootr = module.get_function('calculate_final_ootr_v3')
        # Grid is based on the final output dimensions
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
                patchedDatasSize_local, # Pass the full data size for correct boundary checks
                len(singleDurations),
                singleCalcPeriods
            )
        )
        
        # Use v2 kernel (fixed version without shared memory issues)
        calcAllLowestResidualsGPU = module.get_function('calcAllLowestResidualsGPUB_SignalTiled_v2')
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        calcAllLowestResidualsGPU((gridSizeX,len(singleDurations),singleCalcPeriods),
        (blockSize,1,1),(lowestResidualsGPU,tSizeGPU_cached,
        patchedDatasGPU,patchedDatasSizeGPU,
        durationsGPU,durationsSizeGPU,
        lcArrFullLengthGPU,
        lcArrMaxLenGPU,inverseSquaredPatchedDysGPU,
        overshootGPU,ootrGPU,fullSumGPU,edgeEffectCorrectionsGPU,datapointsGPU_cached,cumsumGPU,
        transitDepthMinGPU_cached
        ))

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

        # Clean up iteration-specific temporary arrays
        del lowestResidualsGPU, periodsGPU, durationsMaxGPU, durationsMinGPU
        del overshootGPU, durationsGPU, durationsSizeGPU
        del fullSumGPU, ootrGPU
        if 'lcArrFullLengthGPU' in dir():
            del lcArrFullLengthGPU, lcArrMaxLenGPU
        # patchedDatasSizeGPU is defined outside loop and reused

        if verbose:
            pbar.update(1)

    chi2 = LowestResidualsEachPeriodGPU.get()

    raw_chi2 = chi2.copy()
    median = np.median(raw_chi2)
    chi2_mask = raw_chi2 > (100 * median)
    chi2 = ma.array(raw_chi2, mask=chi2_mask)
    periods = ma.array(periods, mask=chi2_mask)

    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    raw_power = power.copy()

    if fast:
        return periods,power

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

    # Delete pre-allocated cached arrays
    del phasesGPU_cached, sortIndexGPU_cached, patchedDatasGPU_cached, patchedDysGPU_cached
    del edgeEffectCorrectionsGPU_cached, inverseSquaredPatchedDysGPU_cached
    del cumsumGPU_cached, base_error_cached
    del tGPU_cached, tSizeGPU_cached, tLengthGPU_cached, periodsSizeGPU_cached
    del maxDurationGPU_cached, periodSizeGPU_cached, datapointsGPU_cached, transitDepthMinGPU_cached
    del yGPU, dyGPU, locationGPU, LowestResidualsEachPeriodGPU
    del durationBoolArrayGPU, durationsGridCollectionGPU

    # Clean up GPU memory before next phase
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    chi2_again = search_multi_periods_again(
        possiblePeriods,
        t,
        y,
        dy,
        transit_depth_min,
        lc_arr,
        lc_cache_overview,
        GPUDeviceID,
        singleCalcPeriods
    )

    chi2[possiblePeriodsIndices] = chi2_again
    # chi2_median = np.median(chi2)
    # # replace the extreme outliers
    # chi2 = np.where(np.abs(chi2 - chi2_median) > 50 * chi2_median, chi2_median, chi2)

    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    power_again = power[possiblePeriodsIndices]
    periodIndex = possiblePeriodsIndices[np.argmax(power_again)]
    period = periods[periodIndex]

    possiblePeriodsTimesRate = [0.5,1,2,2/3,3/2]
    possiblePeriodsTemp = [period * rate for rate in possiblePeriodsTimesRate]

    possiblePeriodsIndices_multi, possiblePeriods_multi = find_nearest_indices(possiblePeriodsTemp, periods)

    # Clean up GPU memory before next phase
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    chi2_again = search_multi_periods_again(
        possiblePeriods_multi,
        t,
        y,
        dy,
        transit_depth_min,
        lc_arr,
        lc_cache_overview,
        GPUDeviceID,
        singleCalcPeriods
    )

    chi2[possiblePeriodsIndices_multi] = chi2_again
    # chi2_median = np.ma.median(chi2)
    # replace the extreme outliers
    # chi2 = np.where(np.abs(chi2 - chi2_median) > 50 * chi2_median, chi2_median, chi2)

    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    power_again = power[possiblePeriodsIndices_multi]
    periodIndex = possiblePeriodsIndices_multi[np.argmax(power_again)]
    period = periods[periodIndex]

    rawDuration,durationPointsNum,transit_duration_in_days,transitDepth,T0,transit_times,snr,snr_pink,snrFit,snrFitPink = search_single_periods(
        period,
        t,
        y,
        dy,
        transit_depth_min,
        lc_arr,
        lc_cache_overview,
        GPUDeviceID
    )

    # Clean up GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return periods,period,rawDuration,durationPointsNum,transit_duration_in_days,transitDepth,T0,\
            SDE,chi2,transit_times,power,snr,snr_pink,snrFit,snrFitPink,raw_power,raw_chi2,possiblePeriodsIndices,possiblePeriods

def search_multi_periods_again(
    periods,
    t,
    y,
    dy,
    transit_depth_min,
    lc_arr,
    lc_cache_overview,
    GPUDeviceID,
    singleCalcPeriods,
):
    
    # Choose the GPU device
    set_cuda_device(GPUDeviceID)

    GPUCode = GPUFun.getGPUCode()
    module = cp.RawModule(code=GPUCode,options=('-lineinfo', ))

    durations,indices = np.unique(lc_cache_overview["width_in_samples"],return_index=True)
    lc_arr = lc_arr[indices]
    lc_cache_overview = lc_cache_overview[indices]
    maxDuration = int(max(durations))

    # why?
    if maxDuration % 2 != 0:
        maxDuration = maxDuration + 1
    
    durations = np.sort(durations)
    
    tSize = len(t)
    patchedDatasSize = int(tSize + maxDuration)
    patchedDatasSizeGPU = cp.asarray(np.array([patchedDatasSize])).astype(cp.int32)

    # pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(cp.cuda.Device().id)
    # nvmlinfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # singleCalcPeriods_max = (nvmlinfo.free) / (5*(patchedDatasSize * 2 + 2 + len(durations)*patchedDatasSize*4 + 2*len(durations)))

    # # singleCalcPeriods = int(np.min([np.floor(singleCalcPeriods_max),len(periods)]))

    # # if singleCalcPeriods < 15:
    # #     singleCalcPeriods = int(singleCalcPeriods / 1.1)

    TotalIter = int(np.ceil(len(periods) / singleCalcPeriods))

    #Initialize the variables
    periodsGPU = cp.empty((singleCalcPeriods,),dtype=cp.float64)
    durationsMaxGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)
    durationsMinGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)
    locationGPU = cp.empty(len(periods),dtype=cp.int32)
    LowestResidualsEachPeriodGPU = cp.empty(len(periods),dtype=cp.float32)

    iterFlagGPU = cp.int32(0)

    fulldurationsMaxGPU = cp.empty((len(periods),),dtype=cp.int32)
    fulldurationsMinGPU = cp.empty((len(periods),),dtype=cp.int32)
    fullperiodsSizeGPU = cp.asarray(np.array([len(periods)])).astype(cp.int32)
    tSizeGPU = cp.asarray(np.array([tSize])).astype(cp.int32)
    tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)
    periodsGPU = cp.asarray(periods).astype(cp.float64)
    
    durationsGridGPU = module.get_function('durationsGrid')
    blockSize,gridSizeX = calcGridBlockSize(len(periods))
    durationsGridGPU((gridSizeX,1,1),(blockSize,),
                    (periodsGPU,fulldurationsMaxGPU, fulldurationsMinGPU,tLengthGPU,tSizeGPU, fullperiodsSizeGPU))

    fulldurationsSizeGPU = cp.asarray(np.array([len(durations)])).astype(cp.int32)
    fulldurationsGPU = cp.asarray(durations).astype(cp.int32)
    durationBoolArrayGPU = cp.empty((len(periods),len(durations)),dtype=cp.bool_)

    durationBoolFunGPU = module.get_function('durationBool')
    blockSize,gridSizeX = calcGridBlockSize(len(periods))
    durationBoolFunGPU((gridSizeX,len(durations),1),(blockSize,1,1),
                    (fulldurationsMaxGPU,fulldurationsMinGPU,fulldurationsSizeGPU,fullperiodsSizeGPU,fulldurationsGPU,durationBoolArrayGPU))

    durationsGridCollectionGPU = cp.empty((TotalIter,len(durations)),dtype=cp.bool_)

    yGPU = cp.asarray(y).astype(cp.float32)
    dyGPU = cp.asarray(dy).astype(cp.float32)

    for iterFlag in range(TotalIter):

        if iterFlag == TotalIter - 1:
            SinglePeriods = periods[iterFlag*singleCalcPeriods:]
            actual_period_count = len(SinglePeriods)
            # 用最后一个有效period填充而不是0，避免除零错误
            if actual_period_count < singleCalcPeriods:
                SinglePeriods = np.append(SinglePeriods, 
                                         np.full(singleCalcPeriods - actual_period_count, SinglePeriods[-1]))
            # 正确处理duration bool: 使用所有有效periods的duration范围
            start_idx = iterFlag*singleCalcPeriods
            end_idx = min(start_idx + actual_period_count, len(periods))
            # 对所有有效的periods进行logical_or操作，获取这批periods需要的所有durations
            temp_bool = durationBoolArrayGPU[start_idx]
            for i in range(start_idx + 1, end_idx):
                temp_bool = cp.logical_or(temp_bool, durationBoolArrayGPU[i])
            durationsGridCollectionGPU[iterFlag] = temp_bool
        else:
            SinglePeriods = periods[iterFlag*singleCalcPeriods:(iterFlag+1)*singleCalcPeriods]
            # 修复：对当前批次的所有periods做logical_or，而不是只对比首尾两个
            start_idx = iterFlag*singleCalcPeriods
            end_idx = (iterFlag+1)*singleCalcPeriods
            temp_bool = durationBoolArrayGPU[start_idx]
            for i in range(start_idx + 1, end_idx):
                temp_bool = cp.logical_or(temp_bool, durationBoolArrayGPU[i])
            durationsGridCollectionGPU[iterFlag] = temp_bool

        durationsBoolGrid = durationsGridCollectionGPU[iterFlag].get()
        singleDurations = durations[durationsBoolGrid]
        single_lc_arr = lc_arr[durationsBoolGrid]
        single_lc_cache_overview = lc_cache_overview[durationsBoolGrid]
        overshootGPU = cp.array(single_lc_cache_overview["overshoot"]).astype(cp.float32)

        periodsGPU = cp.asarray(SinglePeriods).astype(cp.float64)
        durationsMaxGPU = cp.asarray(SinglePeriods).astype(cp.int32)
        durationsMinGPU = cp.asarray(SinglePeriods).astype(cp.int32)

        lowestResidualsGPU = cp.empty((singleCalcPeriods,len(singleDurations),tSize),dtype=cp.float32)

        # Phase fold
        phasesGPU = cp.empty((singleCalcPeriods,tSize),dtype=cp.float64)
        sortIndexGPU = cp.empty((singleCalcPeriods,tSize),dtype=cp.int32)
        tGPU = cp.asarray(t).astype(cp.float64)
        periodsSizeGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
        tSizeGPU = cp.asarray(np.array([tSize])).astype(cp.int32)
        tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)

        durationsGridGPU = module.get_function('durationsGrid')
        blockSize,gridSizeX = calcGridBlockSize(singleCalcPeriods)
        durationsGridGPU((gridSizeX,1,1),(blockSize,),
                        (periodsGPU,durationsMaxGPU, durationsMinGPU,tLengthGPU,tSizeGPU, periodsSizeGPU))

        patchedDatasGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
        patchedDysGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)

        lc_arr_max_len = np.array([np.max(singleDurations)]).astype(np.int32)
        lc_arr_full_length = 1 - np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') for x in single_lc_arr])

        lcArrMaxLenGPU = cp.asarray(lc_arr_max_len).astype(cp.int32)
        lcArrFullLengthGPU = cp.asarray(lc_arr_full_length).astype(cp.float32)
        
        edgeEffectCorrectionsGPU = cp.empty((singleCalcPeriods),dtype=cp.float32)

        inverseSquaredPatchedDysGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
        maxDurationGPU = cp.asarray(np.array([maxDuration])).astype(cp.int32)
        periodSizeGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
        durationsGPU = cp.asarray(singleDurations).astype(cp.int32)
        durationsSizeGPU = cp.asarray(np.array([len(singleDurations)])).astype(cp.int32)
        datapointsGPU = cp.array([len(y)]).astype(cp.int32)
        transitDepthMinGPU = cp.array([transit_depth_min]).astype(cp.float32)

        #GPU variables for the loop
        fullSumGPU = cp.empty((singleCalcPeriods,len(singleDurations)),dtype=cp.float32)
        cumsumGPU = cp.empty((singleCalcPeriods,patchedDatasSize),dtype=cp.float32)
        ootrGPU = cp.empty((singleCalcPeriods,len(singleDurations),(tSize)),dtype=cp.float32)

        fastFoldGPU = module.get_function('foldFast')
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        fastFoldGPU((gridSizeX,singleCalcPeriods,),(blockSize,), (tGPU, periodsGPU,phasesGPU,periodsSizeGPU,tSizeGPU))
        i_max = 10
        for i in range(1,i_max + 1):
            sortIndexGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max] = phasesGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max].argsort()

        #calculate patched data
        patchDataGPU = module.get_function('patchData')
        blockSize,gridSizeX = calcGridBlockSize(tSize + maxDuration)
        patchDataGPU((gridSizeX,singleCalcPeriods,),(blockSize,),
        (patchedDatasGPU,patchedDysGPU,patchedDatasSizeGPU,sortIndexGPU,
        maxDurationGPU,yGPU,dyGPU,tSizeGPU))

        calcInverseSquaredPatchedDyGPU = module.get_function('calcInverseSquaredPatchedDy')
        blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize)
        calcInverseSquaredPatchedDyGPU((gridSizeX,singleCalcPeriods,1),(blockSize,1,1),
        (inverseSquaredPatchedDysGPU,patchedDysGPU,patchedDatasSizeGPU,))

        calcEdgeEffectCorrectionsGPU = module.get_function('calcEdgeEffectCorrections')
        blockSize,gridSizeX = calcGridBlockSize(singleCalcPeriods)
        calcEdgeEffectCorrectionsGPU((gridSizeX,1,1),(blockSize,1,1),
        (edgeEffectCorrectionsGPU,patchedDatasGPU,inverseSquaredPatchedDysGPU,
        patchedDatasSizeGPU,maxDurationGPU,periodSizeGPU,))
        
        for i in range(singleCalcPeriods):
            # if((iterFlag * singleCalcPeriods + i) < singleCalcPeriods):
            cumsumGPU[i] = cp.cumsum(patchedDatasGPU[i])

        calcAllFullSumGPU = module.get_function('calcAllFullSum')
        blockSize,gridSizeX = calcGridBlockSize(len(singleDurations))
        calcAllFullSumGPU((gridSizeX,singleCalcPeriods,1),(blockSize,1,1),
        (fullSumGPU,patchedDatasGPU,inverseSquaredPatchedDysGPU,
        patchedDatasSizeGPU,durationsGPU,durationsSizeGPU,
        periodSizeGPU,))

        calcAllOutOfTransitResiduals_step1_2GPU = module.get_function('calcAllOutOfTransitResiduals_step1_2GPU')
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        calcAllOutOfTransitResiduals_step1_2GPU((gridSizeX,len(singleDurations),singleCalcPeriods),
        (blockSize,1,1),(ootrGPU,patchedDatasGPU,durationsGPU,durationsSizeGPU,
        inverseSquaredPatchedDysGPU,patchedDatasSizeGPU,tSizeGPU,))

        ootrGPU = cp.cumsum(ootrGPU,axis=-1)
        calcAllOutOfTransitResiduals_step2_2GPU = module.get_function('calcAllOutOfTransitResiduals_step2_2GPU')
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        calcAllOutOfTransitResiduals_step2_2GPU((gridSizeX,len(singleDurations),singleCalcPeriods),
        (blockSize,1,1),(ootrGPU,
        durationsSizeGPU,patchedDatasSizeGPU,
        durationsGPU,tSizeGPU,fullSumGPU,))

        calcAllLowestResidualsGPU = module.get_function('calcAllLowestResidualsGPUBNoSkipTemp')
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        calcAllLowestResidualsGPU((gridSizeX,len(singleDurations),singleCalcPeriods),
        (blockSize,1,1),(lowestResidualsGPU,tSizeGPU,
        patchedDatasGPU,patchedDatasSizeGPU,
        durationsGPU,durationsSizeGPU,
        lcArrFullLengthGPU,
        lcArrMaxLenGPU,inverseSquaredPatchedDysGPU,
        overshootGPU,ootrGPU,fullSumGPU,edgeEffectCorrectionsGPU,datapointsGPU,cumsumGPU,
        transitDepthMinGPU
        ))

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

    chi2 = LowestResidualsEachPeriodGPU.get()

    # Clean up GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return chi2

# This function is used for refind Duration and T0 since we skip some points in "search_multi_periods"
def search_single_periods(
    period,
    t,
    y,
    dy,
    transit_depth_min,
    lc_arr,
    lc_cache_overview,
    GPUDeviceID = 0
):

    # Choose the GPU device
    set_cuda_device(GPUDeviceID)

    GPUCode = GPUFun.getGPUCode()
    module = cp.RawModule(code=GPUCode)

    durations = np.unique(lc_cache_overview["width_in_samples"])
    maxDuration = int(max(durations))

    # why?
    if maxDuration % 2 != 0:
        maxDuration = maxDuration + 1
    
    durations = np.sort(durations)
    
    tSize = len(t)
    patchedDatasSize = int(tSize + maxDuration)
    patchedDatasSizeGPU = cp.asarray(np.array([patchedDatasSize])).astype(cp.int32)

    singleCalcPeriods = 1

    #Initialize the variables
    periodsGPU = cp.asarray(np.array([period])).astype(cp.float64)

    durationsMaxGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)
    durationsMinGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)

    SinglePeriods = np.array([period])
    durationsMaxGPU = cp.asarray(SinglePeriods).astype(cp.int32)
    durationsMinGPU = cp.asarray(SinglePeriods).astype(cp.int32)

    # Phase fold
    phasesGPU = cp.empty((singleCalcPeriods,tSize),dtype=cp.float64)
    sortIndexGPU = cp.empty((singleCalcPeriods,tSize),dtype=cp.int32)
    tGPU = cp.asarray(t).astype(cp.float64)
    periodsSizeGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
    tSizeGPU = cp.asarray(np.array([tSize])).astype(cp.int32)
    tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)

    durationsGridGPU = module.get_function('durationsGrid')
    blockSize,gridSizeX = calcGridBlockSize(singleCalcPeriods)
    durationsGridGPU((gridSizeX,1,1),(blockSize,),
                    (periodsGPU,durationsMaxGPU, durationsMinGPU,tLengthGPU,tSizeGPU, periodsSizeGPU))

    durationMax = durationsMaxGPU.get().item()
    durationMin = durationsMinGPU.get().item()
    durationsBoolList = np.logical_and(durations <= durationMax, durations >= durationMin)
    durations = durations[durationsBoolList]
    single_lc_arr = lc_arr[durationsBoolList]
    single_lc_cache_overview = lc_cache_overview[durationsBoolList]

    lowestResidualsGPU = cp.empty((len(durations),tSize),dtype=cp.float32)

    fastFoldGPU = module.get_function('foldFast')
    blockSize,gridSizeX = calcGridBlockSize(tSize)
    fastFoldGPU((gridSizeX,singleCalcPeriods,),(blockSize,), (tGPU, periodsGPU,phasesGPU,periodsSizeGPU,tSizeGPU))
    # if singleCalcPeriods > 100:
    i_max = 10
    for i in range(1,i_max + 1):
        sortIndexGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max] = phasesGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max].argsort()

    patchedDatasGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
    patchedDysGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
    yGPU = cp.asarray(y).astype(cp.float32)
    dyGPU = cp.asarray(dy).astype(cp.float32)

    lc_arr_max_len = np.array([np.max(durations)]).astype(np.int32)
    lc_arr_full_length = 1 - np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') for x in single_lc_arr])

    lcArrMaxLenGPU = cp.asarray(lc_arr_max_len).astype(cp.int32)
    lcArrFullLengthGPU = cp.asarray(lc_arr_full_length).astype(cp.float32)

    edgeEffectCorrectionsGPU = cp.empty((singleCalcPeriods),dtype=cp.float32)
    
    inverseSquaredPatchedDysGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
    maxDurationGPU = cp.asarray(np.array([maxDuration])).astype(cp.int32)
    periodSizeGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
    durationsGPU = cp.asarray(durations).astype(cp.int32)
    durationsSizeGPU = cp.asarray(np.array([len(durations)])).astype(cp.int32)

    overshootGPU = cp.array(single_lc_cache_overview["overshoot"]).astype(cp.float32)
    datapointsGPU = cp.array([len(y)]).astype(cp.int32)

    ## depth variable is not needed anymore because we trapezoid fit the transit to get the depth
    ## This method is much more faster and accurate than producing a depth from the transit model
    # depthsEachPeriodGPU = cp.empty((singleCalcPeriods),dtype=cp.float32)
    transitDepthMinGPU = cp.array([transit_depth_min]).astype(cp.float32)

    #GPU variables for the loop
    fullSumGPU = cp.empty((singleCalcPeriods,len(durations)),dtype=cp.float32)
    cumsumGPU = cp.empty((singleCalcPeriods,patchedDatasSize),dtype=cp.float32)
    
    # resultArrayXAxisSizeGPU is the maximum size of the result of a function possible use to restore the result of a patched light curve
    # resultArrayXAxisSizeGPU = cp.asarray(np.array([int(patchedDatasSize) - (np.min(durations)) + 1])).astype(cp.int32)
    # resultArrayXAxisSizeGPU = cp.asarray(np.array([tSize])).astype(cp.int32)

    # ootrGPU = cp.empty((singleCalcPeriods,len(durations),(int(patchedDatasSize) - (np.min(durations)) + 1)),dtype=cp.float32)
    ootrGPU = cp.empty((singleCalcPeriods,len(durations),(tSize)),dtype=cp.float32)

    #calculate patched data
    patchDataGPU = module.get_function('patchData')
    blockSize,gridSizeX = calcGridBlockSize(tSize + maxDuration)
    patchDataGPU((gridSizeX,singleCalcPeriods,),(blockSize,),
    (patchedDatasGPU,patchedDysGPU,patchedDatasSizeGPU,sortIndexGPU,
    maxDurationGPU,yGPU,dyGPU,tSizeGPU))

    calcInverseSquaredPatchedDyGPU = module.get_function('calcInverseSquaredPatchedDy')
    blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize)
    calcInverseSquaredPatchedDyGPU((gridSizeX,singleCalcPeriods,1),(blockSize,1,1),
    (inverseSquaredPatchedDysGPU,patchedDysGPU,patchedDatasSizeGPU,))

    calcEdgeEffectCorrectionsGPU = module.get_function('calcEdgeEffectCorrections')
    blockSize,gridSizeX = calcGridBlockSize(singleCalcPeriods)
    calcEdgeEffectCorrectionsGPU((gridSizeX,1,1),(blockSize,1,1),
    (edgeEffectCorrectionsGPU,patchedDatasGPU,inverseSquaredPatchedDysGPU,
    patchedDatasSizeGPU,maxDurationGPU,periodSizeGPU,))
    
    for i in range(singleCalcPeriods):
        # if((iterFlag * singleCalcPeriods + i) < singleCalcPeriods):
        cumsumGPU[i] = cp.cumsum(patchedDatasGPU[i])

    calcAllFullSumGPU = module.get_function('calcAllFullSum')
    blockSize,gridSizeX = calcGridBlockSize(len(durations))
    calcAllFullSumGPU((gridSizeX,singleCalcPeriods,1),(blockSize,1,1),
    (fullSumGPU,patchedDatasGPU,inverseSquaredPatchedDysGPU,
    patchedDatasSizeGPU,durationsGPU,durationsSizeGPU,
    periodSizeGPU,))

    calcAllOutOfTransitResiduals_step1_2GPU = module.get_function('calcAllOutOfTransitResiduals_step1_2GPU')
    # blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize - (np.min(durations)) + 1)
    blockSize,gridSizeX = calcGridBlockSize(tSize)
    calcAllOutOfTransitResiduals_step1_2GPU((gridSizeX,len(durations),singleCalcPeriods),
    (blockSize,1,1),(ootrGPU,patchedDatasGPU,durationsGPU,durationsSizeGPU,
    inverseSquaredPatchedDysGPU,patchedDatasSizeGPU,tSizeGPU,))

    # ootrGPU = np.cumsum(ootrGPU,axis=-1)
    ootrGPU = cp.cumsum(ootrGPU,axis=-1)
    calcAllOutOfTransitResiduals_step2_2GPU = module.get_function('calcAllOutOfTransitResiduals_step2_2GPU')
    # blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize - (np.min(durations)) + 1)
    blockSize,gridSizeX = calcGridBlockSize(tSize)
    calcAllOutOfTransitResiduals_step2_2GPU((gridSizeX,len(durations),singleCalcPeriods),
    (blockSize,1,1),(ootrGPU,
    durationsSizeGPU,patchedDatasSizeGPU,
    durationsGPU,tSizeGPU,fullSumGPU,))

    calcAllLowestResidualsGPU = module.get_function('calcAllLowestResidualsGPUBNoSkip')
    blockSize,gridSizeX = calcGridBlockSize(tSize)
    calcAllLowestResidualsGPU((gridSizeX,len(durations),singleCalcPeriods),
    (blockSize,1,1),(lowestResidualsGPU,tSizeGPU,
    patchedDatasGPU,patchedDatasSizeGPU,
    durationsGPU,durationsSizeGPU,
    lcArrFullLengthGPU,
    lcArrMaxLenGPU,inverseSquaredPatchedDysGPU,
    overshootGPU,ootrGPU,fullSumGPU,edgeEffectCorrectionsGPU,datapointsGPU,cumsumGPU,
    durationsMaxGPU,durationsMinGPU,transitDepthMinGPU
    ))

    bestLocation = lowestResidualsGPU.argmin().get()

    durationIndex = np.floor(bestLocation / (tSize)).astype(int)    
    durationPointsNum = durations[durationIndex]

    find = np.where(single_lc_cache_overview["width_in_samples"] == durationPointsNum)[0]
    if len(find) > 1:
        find = find[0]
    bestRow = find.item()
    rawDuration = single_lc_cache_overview['duration'][bestRow]

    bestTime,bestFlux,bestFluxDy = foldCPU(t,y,dy,period)
    bestFlux = np.concatenate((bestFlux,bestFlux[:maxDuration]))
    bestFluxDy = np.concatenate((bestFluxDy,bestFluxDy[:maxDuration]))

    bestRowT0 = bestLocation % (tSize)

    transitMean = bestFlux[bestRowT0:bestRowT0+durationPointsNum].mean()

    # Transit Depth
    overshoot = single_lc_cache_overview["overshoot"][durationIndex]
    transitDepth =  ((1-transitMean) * overshoot).item()

    dataOutTransit = np.concatenate((bestFlux[0:bestRowT0],bestFlux[bestRowT0+durationPointsNum:]))

    if bestRowT0 > tSize - 1:
        bestRowT0 = bestRowT0 - tSize 

    snrFit = (1 - transitDepth)*(durationPointsNum ** 0.5)/np.std(dataOutTransit)
    DataCumsum = np.cumsum(dataOutTransit)
    DataSlideAvg = (DataCumsum[durationPointsNum:] - DataCumsum[:-durationPointsNum])/durationPointsNum
    redNoise = np.std(DataSlideAvg)

    Tx = bestTime[bestRowT0]
    T0 = Tx - int((Tx-min(t)) / period) * period - period
    transit_times = all_transit_times(T0, t, period)

    snrFitPink = (1 - transitDepth)/((np.std(dataOutTransit)**2/(durationPointsNum)) + (redNoise**2/(len(transit_times))))**0.5

    # if legacy:
    #     #Raw TLS Calculate transit duration(days) Method
    #     transit_duration_in_days = calculate_transit_duration_in_days(
    #         t, period, transit_times, rawDuration
    #     )
    # else:
    ## Alternative TLS Calculate transit duration(days) Method
    transit_duration_in_days = calcDurationDays(t, period, T0, rawDuration)
    
    T0 = T0 + transit_duration_in_days / 2
    transit_times = transit_times + transit_duration_in_days / 2
    if(T0 < min(t)):
        T0 = T0 + period
    else:
        T0 = T0

    #SNR
    depth_mean_odd, depth_mean_even, depth_mean_odd_std, depth_mean_even_std, all_flux_intransit_odd, all_flux_intransit_even, per_transit_count, transit_depths, transit_depths_uncertainties = intransit_stats(
    t, y, transit_times, transit_duration_in_days
    )
    snr_per_transit, snr_pink_per_transit = snr_stats(
        t=t,
        y=y,
        period=period,
        duration=rawDuration,
        T0=T0,
        transit_times=transit_times,
        transit_duration_in_days=transit_duration_in_days,
        per_transit_count=per_transit_count,
    )

    all_flux_intransit = np.concatenate(
        [all_flux_intransit_odd, all_flux_intransit_even]
    )
    intransit = transit_mask(t, period, 2 * rawDuration, T0)
    flux_ootr = y[~intransit]
    depth_mean = np.mean(all_flux_intransit)
    # depth_mean_std = np.std(all_flux_intransit) / np.sum(
    #     per_transit_count
    # ) ** (0.5)
    snr = ((1 - depth_mean) / np.std(flux_ootr)) * len(all_flux_intransit) ** (0.5)

    snr_pink = np.mean(snr_pink_per_transit) * (len(transit_times)**(0.5))
    
    # Clean up GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    
    return rawDuration,durationPointsNum,transit_duration_in_days,transitDepth,T0,transit_times,snr,snr_pink,snrFit,snrFitPink