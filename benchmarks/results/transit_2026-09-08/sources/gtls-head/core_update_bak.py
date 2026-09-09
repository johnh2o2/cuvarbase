import numpy as np
import numpy.ma as ma
import cupy as cp
from .stats import spectra, all_transit_times, calculate_transit_duration_in_days, intransit_stats, snr_stats, calcDurationDays
from .helpers import transit_mask
from .transit import mutipleTransitFit
from . import GPUFun
import pynvml
import tqdm

# Constants for optimization
MAX_BLOCK_SIZE = 128
MEMORY_SAFETY_FACTOR = 5
PERIODS_DIVISION_FACTOR = 30
MEMORY_ADJUSTMENT_THRESHOLD = 15
MEMORY_ADJUSTMENT_FACTOR = 1.1
MAX_ARGSORT_ITERATIONS = 10
SKIP_POINT_INFINITY = '0x7f800000'
DEFAULT_SKIP_POINT = '8'

def calcGridBlockSize(size):
    """Calculate grid and block size for CUDA kernels."""
    block_size = size
    if block_size > MAX_BLOCK_SIZE:
        block_size = MAX_BLOCK_SIZE
    grid_size_x = int((size / block_size) + 1)
    return block_size, grid_size_x

def find_nearest_indices(target_periods, periods_array):
    """Find nearest indices of target periods in periods array."""
    target_periods = np.array(target_periods)
    periods_array = np.array(periods_array)
    
    indices = []
    values = []
    
    for target in target_periods:
        distances = np.abs(periods_array - target)
        nearest_idx = np.argmin(distances)
        indices.append(nearest_idx)
        values.append(periods_array[nearest_idx])
    
    return indices, values

def foldCPU(time, flux, dy, period):
    """Fold time series data with improved readability."""
    phase = (time % period) / period
    sorted_indices = np.argsort(phase)
    return np.array(time[sorted_indices]), flux[sorted_indices], dy[sorted_indices]

def set_cuda_device(device_id):
    """Set the CUDA device with error handling."""
    try:
        cp.cuda.Device(device_id).use()
    except Exception as e:
        raise RuntimeError(f"Failed to set CUDA device {device_id}: {e}")

def calcGridBlockSize(size):
    """Calculate optimal grid and block size for CUDA kernels."""
    block_size = min(size, MAX_BLOCK_SIZE)
    grid_size_x = int((size + block_size - 1) // block_size)  # Ceiling division
    return block_size, grid_size_x

def find_nearest_indices(target_array, search_array):
    """Find nearest indices with improved variable naming."""
    target_array = np.array(target_array)
    search_array = np.array(search_array)

    nearest_indices = np.zeros(target_array.shape, dtype=int)
    nearest_elements = np.zeros(target_array.shape)

    for i, target_val in enumerate(target_array):
        nearest_idx = np.argmin(np.abs(search_array - target_val))
        nearest_indices[i] = nearest_idx
        nearest_elements[i] = search_array[nearest_idx]
    
    return nearest_indices, nearest_elements

def _setup_gpu_environment(GPUDeviceID, T0_fit_margin):
    """Setup GPU environment and compile CUDA code."""
    set_cuda_device(GPUDeviceID)
    
    gpu_code = GPUFun.getGPUCode()
    skip_point_value = SKIP_POINT_INFINITY if T0_fit_margin == 0 else str(int(1/T0_fit_margin))
    gpu_code = gpu_code.replace(f'#define SKIP_POINT {DEFAULT_SKIP_POINT}', 
                               f'#define SKIP_POINT {skip_point_value}')
    
    module = cp.RawModule(code=gpu_code)
    module.compile()
    return module

def _prepare_duration_data(lc_cache_overview, lc_arr):
    """Prepare and process duration-related data."""
    durations, indices = np.unique(lc_cache_overview["width_in_samples"], return_index=True)
    lc_arr = lc_arr[indices]
    lc_cache_overview = lc_cache_overview[indices]
    max_duration = int(max(durations))

    # Ensure even number for memory alignment
    if max_duration % 2 != 0:
        max_duration += 1
    
    durations = np.sort(durations)
    return durations, lc_arr, lc_cache_overview, max_duration

def _calculate_memory_constraints(t, max_duration, durations, periods):
    """Calculate GPU memory constraints and optimal batch size."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(cp.cuda.Device().id)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    t_size = len(t)
    patched_data_size = t_size + max_duration
    
    # Memory calculation for single period processing
    memory_per_period = (MEMORY_SAFETY_FACTOR * 
                        (patched_data_size * 2 + 2 + 
                         len(durations) * patched_data_size * 4 + 2 * len(durations)))
    
    single_calc_periods_max = memory_info.free // memory_per_period
    single_calc_periods = int(min(single_calc_periods_max, len(periods) // PERIODS_DIVISION_FACTOR))
    
    # Memory adjustment for small batch sizes
    if single_calc_periods < MEMORY_ADJUSTMENT_THRESHOLD:
        single_calc_periods = int(single_calc_periods // MEMORY_ADJUSTMENT_FACTOR)
    
    return single_calc_periods, patched_data_size

def _initialize_gpu_arrays(periods, durations, single_calc_periods, t_size, patched_data_size):
    """Initialize all GPU arrays and constants."""
    # Period-related arrays
    periods_gpu = cp.empty((single_calc_periods,), dtype=cp.float64)
    durations_max_gpu = cp.empty((single_calc_periods,), dtype=cp.int32)
    durations_min_gpu = cp.empty((single_calc_periods,), dtype=cp.int32)
    
    # Result arrays
    location_gpu = cp.empty(len(periods), dtype=cp.int32)
    lowest_residuals_each_period_gpu = cp.empty(len(periods), dtype=cp.float32)
    
    # Full duration arrays
    full_durations_max_gpu = cp.empty((len(periods),), dtype=cp.int32)
    full_durations_min_gpu = cp.empty((len(periods),), dtype=cp.int32)
    
    # Size and parameter arrays
    arrays_dict = {
        'full_periods_size_gpu': cp.asarray([len(periods)], dtype=cp.int32),
        't_size_gpu': cp.asarray([t_size], dtype=cp.int32),
        'patched_data_size_gpu': cp.asarray([patched_data_size], dtype=cp.int32),
        'full_durations_size_gpu': cp.asarray([len(durations)], dtype=cp.int32),
        'full_durations_gpu': cp.asarray(durations, dtype=cp.int32),
        'iter_flag_gpu': cp.int32(0)
    }
    
    return (periods_gpu, durations_max_gpu, durations_min_gpu, location_gpu, 
            lowest_residuals_each_period_gpu, full_durations_max_gpu, full_durations_min_gpu, 
            arrays_dict)

def _setup_duration_grids(module, periods, t, arrays_dict, full_durations_max_gpu, 
                         full_durations_min_gpu, durations):
    """Setup duration grids and boolean arrays."""
    # Convert input data to GPU arrays
    t_length_gpu = cp.asarray([max(t) - min(t)], dtype=cp.float32)
    periods_gpu = cp.asarray(periods, dtype=cp.float64)
    
    # Calculate duration grids
    durations_grid_gpu = module.get_function('durationsGrid')
    block_size, grid_size_x = calcGridBlockSize(len(periods))
    durations_grid_gpu((grid_size_x, 1, 1), (block_size,),
                      (periods_gpu, full_durations_max_gpu, full_durations_min_gpu,
                       t_length_gpu, arrays_dict['t_size_gpu'], arrays_dict['full_periods_size_gpu']))
    
    # Setup boolean array for duration filtering
    duration_bool_array_gpu = cp.empty((len(periods), len(durations)), dtype=cp.bool_)
    duration_bool_fun_gpu = module.get_function('durationBool')
    duration_bool_fun_gpu((grid_size_x, len(durations), 1), (block_size, 1, 1),
                         (full_durations_max_gpu, full_durations_min_gpu, 
                          arrays_dict['full_durations_size_gpu'], arrays_dict['full_periods_size_gpu'],
                          arrays_dict['full_durations_gpu'], duration_bool_array_gpu))
    
    return duration_bool_array_gpu

def search_multi_periods(
    periods, t, y, dy, transit_depth_min, R_star_min, R_star_max, M_star_min, M_star_max,
    lc_arr, lc_cache_overview, T0_fit_margin, oversampling_factor, verbose,
    useLocalPTXCUBIN=False, GPUDeviceID=0, fast=False, legacy=False, 
    SimplifyEdgeEffect=True, bar_location=0
):
    """
    Optimized multi-period search function with improved code structure.
    
    Args:
        periods: Array of periods to search
        t, y, dy: Time series data and uncertainties
        transit_depth_min: Minimum transit depth threshold
        R_star_min, R_star_max: Stellar radius bounds (solar radii)
        M_star_min, M_star_max: Stellar mass bounds (solar masses)
        lc_arr: Light curve array
        lc_cache_overview: Cache overview data
        T0_fit_margin: T0 fitting margin
        oversampling_factor: Oversampling factor for spectra
        verbose: Enable progress bar
        useLocalPTXCUBIN: Use local PTX binary (unused)
        GPUDeviceID: GPU device ID
        fast: Return only periods and power
        legacy: Use legacy method (not implemented)
        SimplifyEdgeEffect: Simplify edge effect correction
        bar_location: Progress bar position
    """
    
    # Step 1: Setup GPU environment
    module = _setup_gpu_environment(GPUDeviceID, T0_fit_margin)
    
    # Step 2: Prepare duration data
    durations, lc_arr, lc_cache_overview, max_duration = _prepare_duration_data(lc_cache_overview, lc_arr)
    
    # Step 3: Calculate memory constraints
    single_calc_periods, patched_data_size = _calculate_memory_constraints(t, max_duration, durations, periods)
    total_iterations = int(np.ceil(len(periods) / single_calc_periods))
    
    # Step 4: Initialize GPU arrays
    (periods_gpu, durations_max_gpu, durations_min_gpu, location_gpu, 
     lowest_residuals_each_period_gpu, full_durations_max_gpu, full_durations_min_gpu, 
     arrays_dict) = _initialize_gpu_arrays(periods, durations, single_calc_periods, 
                                          len(t), patched_data_size)
    
    # Step 5: Setup duration grids
    duration_bool_array_gpu = _setup_duration_grids(module, periods, t, arrays_dict, 
                                                   full_durations_max_gpu, full_durations_min_gpu, durations)
    
    # Step 6: Initialize progress bar
    if verbose:
        progress_bar = tqdm.tqdm(total=total_iterations, position=bar_location)
    
    # Step 7: Prepare input data arrays
    durations_grid_collection_gpu = cp.empty((total_iterations, len(durations)), dtype=cp.bool_)
    y_gpu = cp.asarray(y, dtype=cp.float32)
    dy_gpu = cp.asarray(dy, dtype=cp.float32)

    
    # Step 8: Main processing loop
    for iter_flag in range(total_iterations):
        # Prepare current batch of periods
        current_periods = _prepare_current_batch(periods, iter_flag, single_calc_periods, total_iterations)
        
        # Setup duration grids for current batch
        durations_grid_current = _setup_current_duration_grid(
            duration_bool_array_gpu, durations_grid_collection_gpu, iter_flag, 
            total_iterations, single_calc_periods)
        
        # Process current batch
        single_durations, single_lc_arr, single_lc_cache_overview = _process_current_batch(
            durations, lc_arr, lc_cache_overview, durations_grid_current)
        
        # Execute GPU computations for current batch
        batch_results = _execute_gpu_batch_computation(
            module, current_periods, t, y_gpu, dy_gpu, single_durations, single_lc_arr, 
            single_lc_cache_overview, single_calc_periods, len(t), max_duration, 
            patched_data_size, arrays_dict, transit_depth_min)
        
        # Store results
        _store_batch_results(batch_results, iter_flag, single_calc_periods, 
                           location_gpu, lowest_residuals_each_period_gpu, periods)
        
        arrays_dict['iter_flag_gpu'] += 1
        
        if verbose:
            progress_bar.update(1)
    
    # Step 9: Process results and perform post-processing
    chi2 = lowest_residuals_each_period_gpu.get()
    return _post_process_results(chi2, periods, oversampling_factor, fast, t, y, dy, 
                                transit_depth_min, lc_arr, lc_cache_overview, 
                                GPUDeviceID, single_calc_periods)

def _prepare_current_batch(periods, iter_flag, single_calc_periods, total_iterations):
    """Prepare current batch of periods for processing."""
    if iter_flag == total_iterations - 1:
        # Last iteration - handle remaining periods
        current_periods = periods[iter_flag * single_calc_periods:]
        # Pad with zeros to maintain array size
        padding_size = single_calc_periods - len(current_periods)
        current_periods = np.append(current_periods, np.zeros(padding_size))
    else:
        start_idx = iter_flag * single_calc_periods
        end_idx = (iter_flag + 1) * single_calc_periods
        current_periods = periods[start_idx:end_idx]
    
    return current_periods

def _setup_current_duration_grid(duration_bool_array_gpu, durations_grid_collection_gpu, 
                                iter_flag, total_iterations, single_calc_periods):
    """Setup duration grid for current iteration."""
    if iter_flag == total_iterations - 1:
        start_idx = iter_flag * single_calc_periods
        durations_grid_collection_gpu[iter_flag] = cp.logical_or(
            duration_bool_array_gpu[start_idx], duration_bool_array_gpu[-1])
    else:
        start_idx = iter_flag * single_calc_periods
        end_idx = (iter_flag + 1) * single_calc_periods
        durations_grid_collection_gpu[iter_flag] = cp.logical_or(
            duration_bool_array_gpu[start_idx], duration_bool_array_gpu[end_idx])
    
    return durations_grid_collection_gpu[iter_flag].get()

def _process_current_batch(durations, lc_arr, lc_cache_overview, durations_grid_current):
    """Process current batch data filtering."""
    single_durations = durations[durations_grid_current]
    single_lc_arr = lc_arr[durations_grid_current]
    single_lc_cache_overview = lc_cache_overview[durations_grid_current]
    return single_durations, single_lc_arr, single_lc_cache_overview

def _execute_gpu_batch_computation(module, current_periods, t, y_gpu, dy_gpu, 
                                  single_durations, single_lc_arr, single_lc_cache_overview,
                                  single_calc_periods, t_size, max_duration, patched_data_size,
                                  arrays_dict, transit_depth_min):
    """Execute GPU computations for current batch."""
    
    # Initialize batch-specific GPU arrays
    periods_gpu = cp.asarray(current_periods, dtype=cp.float64)
    durations_max_gpu = cp.asarray(current_periods, dtype=cp.int32)
    durations_min_gpu = cp.asarray(current_periods, dtype=cp.int32)
    
    lowest_residuals_gpu = cp.empty((single_calc_periods, len(single_durations), t_size), dtype=cp.float32)
    
    # Phase folding
    phases_gpu = cp.empty((single_calc_periods, t_size), dtype=cp.float64)
    sort_index_gpu = cp.empty((single_calc_periods, t_size), dtype=cp.int32)
    t_gpu = cp.asarray(t, dtype=cp.float64)
    
    # Setup GPU parameters for this batch
    periods_size_gpu = cp.asarray([single_calc_periods], dtype=cp.int32)
    t_length_gpu = cp.asarray([max(t) - min(t)], dtype=cp.float32)
    
    # Duration grid calculation
    durations_grid_gpu = module.get_function('durationsGrid')
    block_size, grid_size_x = calcGridBlockSize(single_calc_periods)
    durations_grid_gpu((grid_size_x, 1, 1), (block_size,),
                      (periods_gpu, durations_max_gpu, durations_min_gpu, 
                       t_length_gpu, arrays_dict['t_size_gpu'], periods_size_gpu))
    
    # Data preparation
    patched_datas_gpu = cp.empty((single_calc_periods, t_size + max_duration), dtype=cp.float32)
    patched_dys_gpu = cp.empty((single_calc_periods, t_size + max_duration), dtype=cp.float32)
    
    # Light curve processing
    lc_arr_max_len = np.array([np.max(single_durations)], dtype=np.int32)
    lc_arr_full_length = 1 - np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') 
                                      for x in single_lc_arr])
    
    lc_arr_max_len_gpu = cp.asarray(lc_arr_max_len, dtype=cp.int32)
    lc_arr_full_length_gpu = cp.asarray(lc_arr_full_length, dtype=cp.float32)
    
    # Additional GPU arrays
    edge_effect_corrections_gpu = cp.empty(single_calc_periods, dtype=cp.float32)
    inverse_squared_patched_dys_gpu = cp.empty((single_calc_periods, t_size + max_duration), dtype=cp.float32)
    max_duration_gpu = cp.asarray([max_duration], dtype=cp.int32)
    period_size_gpu = cp.asarray([single_calc_periods], dtype=cp.int32)
    durations_gpu = cp.asarray(single_durations, dtype=cp.int32)
    durations_size_gpu = cp.asarray([len(single_durations)], dtype=cp.int32)
    overshoot_gpu = cp.array(single_lc_cache_overview["overshoot"], dtype=cp.float32)
    datapoints_gpu = cp.array([len(y_gpu)], dtype=cp.int32)
    transit_depth_min_gpu = cp.array([transit_depth_min], dtype=cp.float32)
    
    # Main computation arrays
    full_sum_gpu = cp.empty((single_calc_periods, len(single_durations)), dtype=cp.float32)
    cumsum_gpu = cp.empty((single_calc_periods, patched_data_size), dtype=cp.float32)
    ootr_gpu = cp.empty((single_calc_periods, len(single_durations), t_size), dtype=cp.float32)
    
    # Execute GPU kernels
    _execute_gpu_kernels(module, t_gpu, periods_gpu, phases_gpu, sort_index_gpu,
                        periods_size_gpu, arrays_dict, patched_datas_gpu, patched_dys_gpu,
                        y_gpu, dy_gpu, max_duration_gpu, inverse_squared_patched_dys_gpu,
                        edge_effect_corrections_gpu, period_size_gpu, cumsum_gpu,
                        full_sum_gpu, durations_gpu, durations_size_gpu, ootr_gpu,
                        lowest_residuals_gpu, lc_arr_full_length_gpu, lc_arr_max_len_gpu,
                        overshoot_gpu, datapoints_gpu, transit_depth_min_gpu,
                        single_calc_periods, t_size, max_duration, patched_data_size, 
                        single_durations)
    
    return lowest_residuals_gpu

def _execute_gpu_kernels(module, t_gpu, periods_gpu, phases_gpu, sort_index_gpu,
                        periods_size_gpu, arrays_dict, patched_datas_gpu, patched_dys_gpu,
                        y_gpu, dy_gpu, max_duration_gpu, inverse_squared_patched_dys_gpu,
                        edge_effect_corrections_gpu, period_size_gpu, cumsum_gpu,
                        full_sum_gpu, durations_gpu, durations_size_gpu, ootr_gpu,
                        lowest_residuals_gpu, lc_arr_full_length_gpu, lc_arr_max_len_gpu,
                        overshoot_gpu, datapoints_gpu, transit_depth_min_gpu,
                        single_calc_periods, t_size, max_duration, patched_data_size,
                        single_durations):
    """Execute all GPU kernels in sequence."""
    
    # 1. Phase folding
    fast_fold_gpu = module.get_function('foldFast')
    block_size, grid_size_x = calcGridBlockSize(t_size)
    fast_fold_gpu((grid_size_x, single_calc_periods), (block_size,), 
                 (t_gpu, periods_gpu, phases_gpu, periods_size_gpu, arrays_dict['t_size_gpu']))
    
    # 2. Sorting (split into chunks to avoid memory issues)
    for i in range(1, MAX_ARGSORT_ITERATIONS + 1):
        start_idx = (i - 1) * single_calc_periods // MAX_ARGSORT_ITERATIONS
        end_idx = i * single_calc_periods // MAX_ARGSORT_ITERATIONS
        sort_index_gpu[start_idx:end_idx] = phases_gpu[start_idx:end_idx].argsort()
    
    # 3. Patch data
    patch_data_gpu = module.get_function('patchData')
    block_size, grid_size_x = calcGridBlockSize(t_size + max_duration)
    patch_data_gpu((grid_size_x, single_calc_periods), (block_size,),
                  (patched_datas_gpu, patched_dys_gpu, arrays_dict['patched_data_size_gpu'],
                   sort_index_gpu, max_duration_gpu, y_gpu, dy_gpu, arrays_dict['t_size_gpu']))
    
    # 4. Calculate inverse squared patched dy
    calc_inverse_squared_patched_dy_gpu = module.get_function('calcInverseSquaredPatchedDy')
    block_size, grid_size_x = calcGridBlockSize(patched_data_size)
    calc_inverse_squared_patched_dy_gpu((grid_size_x, single_calc_periods, 1), (block_size, 1, 1),
                                       (inverse_squared_patched_dys_gpu, patched_dys_gpu, 
                                        arrays_dict['patched_data_size_gpu']))
    
    # 5. Calculate edge effect corrections
    calc_edge_effect_corrections_gpu = module.get_function('calcEdgeEffectCorrections')
    block_size, grid_size_x = calcGridBlockSize(single_calc_periods)
    calc_edge_effect_corrections_gpu((grid_size_x, 1, 1), (block_size, 1, 1),
                                    (edge_effect_corrections_gpu, patched_datas_gpu, 
                                     inverse_squared_patched_dys_gpu, arrays_dict['patched_data_size_gpu'],
                                     max_duration_gpu, period_size_gpu))
    
    # 6. Calculate cumulative sum
    for i in range(single_calc_periods):
        cumsum_gpu[i] = cp.cumsum(patched_datas_gpu[i])
    
    # 7. Calculate full sum
    calc_all_full_sum_gpu = module.get_function('calcAllFullSum')
    block_size, grid_size_x = calcGridBlockSize(len(single_durations))
    calc_all_full_sum_gpu((grid_size_x, single_calc_periods, 1), (block_size, 1, 1),
                         (full_sum_gpu, patched_datas_gpu, inverse_squared_patched_dys_gpu,
                          arrays_dict['patched_data_size_gpu'], durations_gpu, durations_size_gpu, period_size_gpu))
    
    # 8. Calculate out-of-transit residuals - step 1
    calc_ootr_step1_gpu = module.get_function('calcAllOutOfTransitResiduals_step1_2GPU')
    block_size, grid_size_x = calcGridBlockSize(t_size)
    calc_ootr_step1_gpu((grid_size_x, len(single_durations), single_calc_periods), (block_size, 1, 1),
                       (ootr_gpu, patched_datas_gpu, durations_gpu, durations_size_gpu,
                        inverse_squared_patched_dys_gpu, arrays_dict['patched_data_size_gpu'], 
                        arrays_dict['t_size_gpu']))
    
    # 9. Calculate cumulative OOTR
    ootr_gpu = cp.cumsum(ootr_gpu, axis=-1)
    
    # 10. Calculate out-of-transit residuals - step 2
    calc_ootr_step2_gpu = module.get_function('calcAllOutOfTransitResiduals_step2_2GPU')
    calc_ootr_step2_gpu((grid_size_x, len(single_durations), single_calc_periods), (block_size, 1, 1),
                       (ootr_gpu, durations_size_gpu, arrays_dict['patched_data_size_gpu'],
                        durations_gpu, arrays_dict['t_size_gpu'], full_sum_gpu))
    
    # 11. Calculate lowest residuals
    calc_lowest_residuals_gpu = module.get_function('calcAllLowestResidualsGPUB')
    calc_lowest_residuals_gpu((grid_size_x, len(single_durations), single_calc_periods), (block_size, 1, 1),
                             (lowest_residuals_gpu, arrays_dict['t_size_gpu'], patched_datas_gpu,
                              arrays_dict['patched_data_size_gpu'], durations_gpu, durations_size_gpu,
                              lc_arr_full_length_gpu, lc_arr_max_len_gpu, inverse_squared_patched_dys_gpu,
                              overshoot_gpu, ootr_gpu, full_sum_gpu, edge_effect_corrections_gpu,
                              datapoints_gpu, cumsum_gpu, transit_depth_min_gpu))

def _store_batch_results(batch_results, iter_flag, single_calc_periods, 
                        location_gpu, lowest_residuals_each_period_gpu, periods):
    """Store results from current batch processing."""
    start_idx = iter_flag * single_calc_periods
    end_idx = start_idx + single_calc_periods
    valid_range = min(end_idx, len(periods)) - start_idx
    
    valid_lowest_residuals = batch_results[:valid_range]
    flattened_residuals = valid_lowest_residuals.reshape(valid_range, -1)
    
    min_indices = cp.argmin(flattened_residuals, axis=-1)
    min_values = cp.min(flattened_residuals, axis=-1)
    
    location_gpu[start_idx:start_idx + valid_range] = min_indices
    lowest_residuals_each_period_gpu[start_idx:start_idx + valid_range] = min_values

def _post_process_results(chi2, periods, oversampling_factor, fast, t, y, dy, 
                         transit_depth_min, lc_arr, lc_cache_overview, 
                         GPUDeviceID, single_calc_periods):
    """Post-process results and perform additional analysis."""
    
    # Filter outliers
    raw_chi2 = chi2.copy()
    median_chi2 = np.median(raw_chi2)
    chi2_mask = raw_chi2 > (100 * median_chi2)
    chi2 = ma.array(raw_chi2, mask=chi2_mask)
    periods = ma.array(periods, mask=chi2_mask)
    
    # Calculate power spectrum
    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    raw_power = power.copy()
    
    if fast:
        return periods, power
    
    # Find best period candidates
    best_period_candidates = _find_best_period_candidates(periods, power)
    
    # Refine results with additional searches
    chi2, periods, SDE, period = _refine_period_search(
        chi2, periods, power, best_period_candidates, t, y, dy, 
        transit_depth_min, lc_arr, lc_cache_overview, GPUDeviceID, 
        single_calc_periods, oversampling_factor)
    
    # Final single period analysis
    transit_results = search_single_periods(
        period, t, y, dy, transit_depth_min, lc_arr, 
        lc_cache_overview, GPUDeviceID)
    
    return _format_final_results(periods, period, transit_results, SDE, chi2, 
                                power, raw_power, raw_chi2, best_period_candidates)

def _find_best_period_candidates(periods, power):
    """Find the best period candidates for further analysis."""
    # Combine periods with negative power for sorting
    combined = list(enumerate(zip(periods, -power)))
    sorted_combined = sorted(combined, key=lambda x: x[1][1])
    
    # Get top 100 candidates
    top_100_indices = [item[0] for item in sorted_combined[:100]]
    top_100_periods = [item[1][0] for item in sorted_combined[:100]]
    
    # Get next 100 candidates with period > 1
    remaining_combined = [item for item in sorted_combined if item[0] not in top_100_indices]
    remaining_gt_1 = [item for item in remaining_combined if item[1][0] > 1]
    sorted_remaining_gt_1 = sorted(remaining_gt_1, key=lambda x: x[1][1])
    next_100_indices = [item[0] for item in sorted_remaining_gt_1[:100]]
    next_100_periods = [item[1][0] for item in sorted_remaining_gt_1[:100]]
    
    possible_periods_indices = top_100_indices + next_100_indices
    possible_periods = top_100_periods + next_100_periods
    
    return {'indices': possible_periods_indices, 'periods': possible_periods}

def _refine_period_search(chi2, periods, power, candidates, t, y, dy, 
                         transit_depth_min, lc_arr, lc_cache_overview, 
                         GPUDeviceID, single_calc_periods, oversampling_factor):
    """Refine period search with additional iterations."""
    
    # First refinement
    chi2_refined = search_multi_periods_again(
        candidates['periods'], t, y, dy, transit_depth_min, 
        lc_arr, lc_cache_overview, GPUDeviceID, single_calc_periods)
    
    chi2[candidates['indices']] = chi2_refined
    
    # Recalculate power spectrum
    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    power_refined = power[candidates['indices']]
    period_index = candidates['indices'][np.argmax(power_refined)]
    period = periods[period_index]
    
    # Second refinement with harmonic periods
    harmonic_rates = [0.5, 1, 2, 2/3, 3/2]
    harmonic_periods = [period * rate for rate in harmonic_rates]
    harmonic_indices, harmonic_periods = find_nearest_indices(harmonic_periods, periods)
    
    chi2_harmonic = search_multi_periods_again(
        harmonic_periods, t, y, dy, transit_depth_min, 
        lc_arr, lc_cache_overview, GPUDeviceID, single_calc_periods)
    
    chi2[harmonic_indices] = chi2_harmonic
    
    # Final power spectrum calculation
    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    power_harmonic = power[harmonic_indices]
    period_index = harmonic_indices[np.argmax(power_harmonic)]
    period = periods[period_index]
    
    return chi2, periods, SDE, period

def _format_final_results(periods, period, transit_results, SDE, chi2, 
                         power, raw_power, raw_chi2, candidates):
    """Format final results for return."""
    (raw_duration, duration_points_num, transit_duration_in_days, 
     transit_depth, T0, transit_times, snr, snr_pink, snr_fit, snr_fit_pink) = transit_results
    
    return (periods, period, raw_duration, duration_points_num, transit_duration_in_days,
            transit_depth, T0, SDE, chi2, transit_times, power, snr, snr_pink, 
            snr_fit, snr_fit_pink, raw_power, raw_chi2, candidates['indices'], candidates['periods'])

def search_multi_periods_again(
    periods,
    t,
    y,
    dy,
    transit_depth_min,
    lc_arr,
    lc_cache_overview,
    GPUDeviceID,
    single_calc_periods,
):
    """
    Optimized second pass multi-period search with improved code structure.
    Reuses helper functions from the main search function.
    
    Args:
        periods: Array of periods to refine
        t, y, dy: Time series data and uncertainties
        transit_depth_min: Minimum transit depth threshold
        lc_arr: Light curve array
        lc_cache_overview: Cache overview data
        GPUDeviceID: GPU device ID
        single_calc_periods: Number of periods to calculate simultaneously
    
    Returns:
        chi2: Array of chi-squared values for input periods
    """
    
    # Step 1: Setup GPU environment and compile CUDA code
    set_cuda_device(GPUDeviceID)
    gpu_code = GPUFun.getGPUCode()
    module = cp.RawModule(code=gpu_code)
    
    # Step 2: Prepare duration data (reuse helper function)
    durations, lc_arr, lc_cache_overview, max_duration = _prepare_duration_data(lc_cache_overview, lc_arr)
    
    # Step 3: Initialize processing parameters
    t_size = len(t)
    patched_data_size = t_size + max_duration
    total_iterations = int(np.ceil(len(periods) / single_calc_periods))
    
    # Step 4: Initialize GPU arrays for refinement search
    (location_gpu, lowest_residuals_each_period_gpu, 
     arrays_dict) = _initialize_refinement_arrays(periods, durations, single_calc_periods, 
                                                  t_size, patched_data_size)
    
    # Step 5: Setup duration grids for all periods
    duration_bool_array_gpu = _setup_refinement_duration_grids(
        module, periods, t, arrays_dict, durations)
    
    # Step 6: Prepare input data arrays
    durations_grid_collection_gpu = cp.empty((total_iterations, len(durations)), dtype=cp.bool_)
    y_gpu = cp.asarray(y, dtype=cp.float32)
    dy_gpu = cp.asarray(dy, dtype=cp.float32)
    
    # Step 7: Main refinement processing loop
    for iter_flag in range(total_iterations):
        # Prepare current batch of periods (reuse helper)
        current_periods = _prepare_current_batch(periods, iter_flag, single_calc_periods, total_iterations)
        
        # Setup duration grids for current batch (reuse helper)
        durations_grid_current = _setup_refinement_current_duration_grid(
            duration_bool_array_gpu, durations_grid_collection_gpu, iter_flag, 
            total_iterations, single_calc_periods)
        
        # Process current batch (reuse helper)
        single_durations, single_lc_arr, single_lc_cache_overview = _process_current_batch(
            durations, lc_arr, lc_cache_overview, durations_grid_current)
        
        # Execute GPU computations for current batch with refinement kernel
        batch_results = _execute_refinement_gpu_computation(
            module, current_periods, t, y_gpu, dy_gpu, single_durations, single_lc_arr, 
            single_lc_cache_overview, single_calc_periods, t_size, max_duration, 
            patched_data_size, arrays_dict, transit_depth_min)
        
        # Store results (reuse helper)
        _store_batch_results(batch_results, iter_flag, single_calc_periods, 
                           location_gpu, lowest_residuals_each_period_gpu, periods)
    
    # Step 8: Return final chi-squared values
    return lowest_residuals_each_period_gpu.get()

def _initialize_refinement_arrays(periods, durations, single_calc_periods, t_size, patched_data_size):
    """Initialize GPU arrays specifically for refinement search."""
    # Result arrays
    location_gpu = cp.empty(len(periods), dtype=cp.int32)
    lowest_residuals_each_period_gpu = cp.empty(len(periods), dtype=cp.float32)
    
    # Size and parameter arrays for refinement
    arrays_dict = {
        'full_periods_size_gpu': cp.asarray([len(periods)], dtype=cp.int32),
        't_size_gpu': cp.asarray([t_size], dtype=cp.int32),
        'patched_data_size_gpu': cp.asarray([patched_data_size], dtype=cp.int32),
        'full_durations_size_gpu': cp.asarray([len(durations)], dtype=cp.int32),
        'full_durations_gpu': cp.asarray(durations, dtype=cp.int32),
    }
    
    return location_gpu, lowest_residuals_each_period_gpu, arrays_dict

def _setup_refinement_duration_grids(module, periods, t, arrays_dict, durations):
    """Setup duration grids specifically for refinement search."""
    # Full duration arrays for all periods
    full_durations_max_gpu = cp.empty((len(periods),), dtype=cp.int32)
    full_durations_min_gpu = cp.empty((len(periods),), dtype=cp.int32)
    
    # Convert input data to GPU arrays
    t_length_gpu = cp.asarray([max(t) - min(t)], dtype=cp.float32)
    periods_gpu = cp.asarray(periods, dtype=cp.float64)
    
    # Calculate duration grids for all periods
    durations_grid_gpu = module.get_function('durationsGrid')
    block_size, grid_size_x = calcGridBlockSize(len(periods))
    durations_grid_gpu((grid_size_x, 1, 1), (block_size,),
                      (periods_gpu, full_durations_max_gpu, full_durations_min_gpu,
                       t_length_gpu, arrays_dict['t_size_gpu'], arrays_dict['full_periods_size_gpu']))
    
    # Setup boolean array for duration filtering
    duration_bool_array_gpu = cp.empty((len(periods), len(durations)), dtype=cp.bool_)
    duration_bool_fun_gpu = module.get_function('durationBool')
    duration_bool_fun_gpu((grid_size_x, len(durations), 1), (block_size, 1, 1),
                         (full_durations_max_gpu, full_durations_min_gpu, 
                          arrays_dict['full_durations_size_gpu'], arrays_dict['full_periods_size_gpu'],
                          arrays_dict['full_durations_gpu'], duration_bool_array_gpu))
    
    return duration_bool_array_gpu

def _setup_refinement_current_duration_grid(duration_bool_array_gpu, durations_grid_collection_gpu, 
                                            iter_flag, total_iterations, single_calc_periods):
    """Setup duration grid for current iteration in refinement search."""
    if iter_flag == total_iterations - 1:
        start_idx = iter_flag * single_calc_periods
        durations_grid_collection_gpu[iter_flag] = cp.logical_or(
            duration_bool_array_gpu[start_idx], duration_bool_array_gpu[-1])
    else:
        start_idx = iter_flag * single_calc_periods
        end_idx = (iter_flag + 1) * single_calc_periods
        durations_grid_collection_gpu[iter_flag] = cp.logical_or(
            duration_bool_array_gpu[start_idx], duration_bool_array_gpu[end_idx - 1])
    
    return durations_grid_collection_gpu[iter_flag].get()

def _execute_refinement_gpu_computation(module, current_periods, t, y_gpu, dy_gpu, 
                                       single_durations, single_lc_arr, single_lc_cache_overview,
                                       single_calc_periods, t_size, max_duration, patched_data_size,
                                       arrays_dict, transit_depth_min):
    """Execute GPU computations for refinement search using the NoSkipTemp kernel."""
    
    # Initialize batch-specific GPU arrays
    periods_gpu = cp.asarray(current_periods, dtype=cp.float64)
    durations_max_gpu = cp.asarray(current_periods, dtype=cp.int32)
    durations_min_gpu = cp.asarray(current_periods, dtype=cp.int32)
    
    lowest_residuals_gpu = cp.empty((single_calc_periods, len(single_durations), t_size), dtype=cp.float32)
    
    # Phase folding setup
    phases_gpu = cp.empty((single_calc_periods, t_size), dtype=cp.float64)
    sort_index_gpu = cp.empty((single_calc_periods, t_size), dtype=cp.int32)
    t_gpu = cp.asarray(t, dtype=cp.float64)
    
    # GPU parameters for this batch
    periods_size_gpu = cp.asarray([single_calc_periods], dtype=cp.int32)
    t_length_gpu = cp.asarray([max(t) - min(t)], dtype=cp.float32)
    
    # Duration grid calculation
    durations_grid_gpu = module.get_function('durationsGrid')
    block_size, grid_size_x = calcGridBlockSize(single_calc_periods)
    durations_grid_gpu((grid_size_x, 1, 1), (block_size,),
                      (periods_gpu, durations_max_gpu, durations_min_gpu, 
                       t_length_gpu, arrays_dict['t_size_gpu'], periods_size_gpu))
    
    # Prepare data processing arrays
    patched_datas_gpu = cp.empty((single_calc_periods, t_size + max_duration), dtype=cp.float32)
    patched_dys_gpu = cp.empty((single_calc_periods, t_size + max_duration), dtype=cp.float32)
    
    # Light curve processing
    lc_arr_max_len = np.array([np.max(single_durations)], dtype=np.int32)
    lc_arr_full_length = 1 - np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') 
                                      for x in single_lc_arr])
    
    lc_arr_max_len_gpu = cp.asarray(lc_arr_max_len, dtype=cp.int32)
    lc_arr_full_length_gpu = cp.asarray(lc_arr_full_length, dtype=cp.float32)
    
    # Additional computation arrays
    edge_effect_corrections_gpu = cp.empty(single_calc_periods, dtype=cp.float32)
    inverse_squared_patched_dys_gpu = cp.empty((single_calc_periods, t_size + max_duration), dtype=cp.float32)
    max_duration_gpu = cp.asarray([max_duration], dtype=cp.int32)
    period_size_gpu = cp.asarray([single_calc_periods], dtype=cp.int32)
    durations_gpu = cp.asarray(single_durations, dtype=cp.int32)
    durations_size_gpu = cp.asarray([len(single_durations)], dtype=cp.int32)
    overshoot_gpu = cp.array(single_lc_cache_overview["overshoot"], dtype=cp.float32)
    datapoints_gpu = cp.array([len(y_gpu)], dtype=cp.int32)
    transit_depth_min_gpu = cp.array([transit_depth_min], dtype=cp.float32)
    
    # Main computation arrays
    full_sum_gpu = cp.empty((single_calc_periods, len(single_durations)), dtype=cp.float32)
    cumsum_gpu = cp.empty((single_calc_periods, patched_data_size), dtype=cp.float32)
    ootr_gpu = cp.empty((single_calc_periods, len(single_durations), t_size), dtype=cp.float32)
    
    # Execute GPU kernels with refinement-specific kernel call
    _execute_refinement_gpu_kernels(module, t_gpu, periods_gpu, phases_gpu, sort_index_gpu,
                                   periods_size_gpu, arrays_dict, patched_datas_gpu, patched_dys_gpu,
                                   y_gpu, dy_gpu, max_duration_gpu, inverse_squared_patched_dys_gpu,
                                   edge_effect_corrections_gpu, period_size_gpu, cumsum_gpu,
                                   full_sum_gpu, durations_gpu, durations_size_gpu, ootr_gpu,
                                   lowest_residuals_gpu, lc_arr_full_length_gpu, lc_arr_max_len_gpu,
                                   overshoot_gpu, datapoints_gpu, transit_depth_min_gpu,
                                   single_calc_periods, t_size, patched_data_size, single_durations)
    
    return lowest_residuals_gpu

def _execute_refinement_gpu_kernels(module, t_gpu, periods_gpu, phases_gpu, sort_index_gpu,
                                   periods_size_gpu, arrays_dict, patched_datas_gpu, patched_dys_gpu,
                                   y_gpu, dy_gpu, max_duration_gpu, inverse_squared_patched_dys_gpu,
                                   edge_effect_corrections_gpu, period_size_gpu, cumsum_gpu,
                                   full_sum_gpu, durations_gpu, durations_size_gpu, ootr_gpu,
                                   lowest_residuals_gpu, lc_arr_full_length_gpu, lc_arr_max_len_gpu,
                                   overshoot_gpu, datapoints_gpu, transit_depth_min_gpu,
                                   single_calc_periods, t_size, patched_data_size, single_durations):
    """Execute GPU kernels for refinement search with optimized processing."""
    
    # 1. Phase folding
    fast_fold_gpu = module.get_function('foldFast')
    block_size, grid_size_x = calcGridBlockSize(t_size)
    fast_fold_gpu((grid_size_x, single_calc_periods), (block_size,), 
                 (t_gpu, periods_gpu, phases_gpu, periods_size_gpu, arrays_dict['t_size_gpu']))
    
    # 2. Sorting (optimized chunking)
    for i in range(1, MAX_ARGSORT_ITERATIONS + 1):
        start_idx = (i - 1) * single_calc_periods // MAX_ARGSORT_ITERATIONS
        end_idx = i * single_calc_periods // MAX_ARGSORT_ITERATIONS
        sort_index_gpu[start_idx:end_idx] = phases_gpu[start_idx:end_idx].argsort()
    
    # 3. Patch data
    patch_data_gpu = module.get_function('patchData')
    block_size, grid_size_x = calcGridBlockSize(t_size + max_duration_gpu.get()[0])
    patch_data_gpu((grid_size_x, single_calc_periods), (block_size,),
                  (patched_datas_gpu, patched_dys_gpu, arrays_dict['patched_data_size_gpu'],
                   sort_index_gpu, max_duration_gpu, y_gpu, dy_gpu, arrays_dict['t_size_gpu']))
    
    # 4. Calculate inverse squared patched dy
    calc_inverse_squared_patched_dy_gpu = module.get_function('calcInverseSquaredPatchedDy')
    block_size, grid_size_x = calcGridBlockSize(patched_data_size)
    calc_inverse_squared_patched_dy_gpu((grid_size_x, single_calc_periods, 1), (block_size, 1, 1),
                                       (inverse_squared_patched_dys_gpu, patched_dys_gpu, 
                                        arrays_dict['patched_data_size_gpu']))
    
    # 5. Calculate edge effect corrections
    calc_edge_effect_corrections_gpu = module.get_function('calcEdgeEffectCorrections')
    block_size, grid_size_x = calcGridBlockSize(single_calc_periods)
    calc_edge_effect_corrections_gpu((grid_size_x, 1, 1), (block_size, 1, 1),
                                    (edge_effect_corrections_gpu, patched_datas_gpu, 
                                     inverse_squared_patched_dys_gpu, arrays_dict['patched_data_size_gpu'],
                                     max_duration_gpu, period_size_gpu))
    
    # 6. Calculate cumulative sum
    for i in range(single_calc_periods):
        cumsum_gpu[i] = cp.cumsum(patched_datas_gpu[i])
    
    # 7. Calculate full sum
    calc_all_full_sum_gpu = module.get_function('calcAllFullSum')
    block_size, grid_size_x = calcGridBlockSize(len(single_durations))
    calc_all_full_sum_gpu((grid_size_x, single_calc_periods, 1), (block_size, 1, 1),
                         (full_sum_gpu, patched_datas_gpu, inverse_squared_patched_dys_gpu,
                          arrays_dict['patched_data_size_gpu'], durations_gpu, durations_size_gpu, period_size_gpu))
    
    # 8. Calculate out-of-transit residuals - step 1
    calc_ootr_step1_gpu = module.get_function('calcAllOutOfTransitResiduals_step1_2GPU')
    block_size, grid_size_x = calcGridBlockSize(t_size)
    calc_ootr_step1_gpu((grid_size_x, len(single_durations), single_calc_periods), (block_size, 1, 1),
                       (ootr_gpu, patched_datas_gpu, durations_gpu, durations_size_gpu,
                        inverse_squared_patched_dys_gpu, arrays_dict['patched_data_size_gpu'], 
                        arrays_dict['t_size_gpu']))
    
    # 9. Calculate cumulative OOTR
    ootr_gpu = cp.cumsum(ootr_gpu, axis=-1)
    
    # 10. Calculate out-of-transit residuals - step 2
    calc_ootr_step2_gpu = module.get_function('calcAllOutOfTransitResiduals_step2_2GPU')
    calc_ootr_step2_gpu((grid_size_x, len(single_durations), single_calc_periods), (block_size, 1, 1),
                       (ootr_gpu, durations_size_gpu, arrays_dict['patched_data_size_gpu'],
                        durations_gpu, arrays_dict['t_size_gpu'], full_sum_gpu))
    
    # 11. Calculate lowest residuals using refinement kernel (NoSkipTemp)
    calc_lowest_residuals_gpu = module.get_function('calcAllLowestResidualsGPUBNoSkipTemp')
    calc_lowest_residuals_gpu((grid_size_x, len(single_durations), single_calc_periods), (block_size, 1, 1),
                             (lowest_residuals_gpu, arrays_dict['t_size_gpu'], patched_datas_gpu,
                              arrays_dict['patched_data_size_gpu'], durations_gpu, durations_size_gpu,
                              lc_arr_full_length_gpu, lc_arr_max_len_gpu, inverse_squared_patched_dys_gpu,
                              overshoot_gpu, ootr_gpu, full_sum_gpu, edge_effect_corrections_gpu,
                              datapoints_gpu, cumsum_gpu, transit_depth_min_gpu))

def _setup_single_period_gpu_environment(GPUDeviceID):
    """Setup GPU environment for single period search."""
    set_cuda_device(GPUDeviceID)
    gpu_code = GPUFun.getGPUCode()
    module = cp.RawModule(code=gpu_code)
    return module

def _initialize_single_period_arrays(period, t_size, max_duration):
    """Initialize basic GPU arrays for single period search (before duration filtering)."""
    single_calc_periods = 1
    patched_data_size = t_size + max_duration
    
    # Basic arrays
    periods_gpu = cp.asarray([period], dtype=cp.float64)
    durations_max_gpu = cp.empty((single_calc_periods,), dtype=cp.int32)
    durations_min_gpu = cp.empty((single_calc_periods,), dtype=cp.int32)
    
    # Phase folding arrays
    phases_gpu = cp.empty((single_calc_periods, t_size), dtype=cp.float64)
    sort_index_gpu = cp.empty((single_calc_periods, t_size), dtype=cp.int32)
    
    # Size and parameter arrays (will be updated after duration filtering)
    arrays_dict = {
        'periods_size_gpu': cp.asarray([single_calc_periods], dtype=cp.int32),
        't_size_gpu': cp.asarray([t_size], dtype=cp.int32),
        'patched_data_size_gpu': cp.asarray([patched_data_size], dtype=cp.int32),
        'max_duration_gpu': cp.asarray([max_duration], dtype=cp.int32),
        'period_size_gpu': cp.asarray([single_calc_periods], dtype=cp.int32),
    }
    
    return (periods_gpu, durations_max_gpu, durations_min_gpu, phases_gpu, 
            sort_index_gpu, arrays_dict)

def _create_result_arrays_after_filtering(single_durations, t_size):
    """Create result arrays after duration filtering."""
    lowest_residuals_gpu = cp.empty((len(single_durations), t_size), dtype=cp.float32)
    return lowest_residuals_gpu

def _setup_single_period_duration_grid(module, period, t, durations, 
                                       periods_gpu, durations_max_gpu, durations_min_gpu, arrays_dict):
    """Setup duration grid for single period search."""
    t_length_gpu = cp.asarray([max(t) - min(t)], dtype=cp.float32)
    
    # Calculate duration grid
    durations_grid_gpu = module.get_function('durationsGrid')
    block_size, grid_size_x = calcGridBlockSize(1)
    durations_grid_gpu((grid_size_x, 1, 1), (block_size,),
                      (periods_gpu, durations_max_gpu, durations_min_gpu,
                       t_length_gpu, arrays_dict['t_size_gpu'], arrays_dict['periods_size_gpu']))
    
    # Filter durations based on calculated limits
    duration_max = durations_max_gpu.get().item()
    duration_min = durations_min_gpu.get().item()
    durations_bool_list = np.logical_and(durations <= duration_max, durations >= duration_min)
    
    return durations_bool_list

def _execute_single_period_gpu_computation(module, period, t, y, dy, single_durations, 
                                          single_lc_arr, single_lc_cache_overview,
                                          t_size, max_duration, transit_depth_min,
                                          periods_gpu, durations_max_gpu, durations_min_gpu,
                                          phases_gpu, sort_index_gpu, lowest_residuals_gpu, arrays_dict):
    """Execute GPU computations for single period search."""
    
    # Convert input data to GPU arrays
    t_gpu = cp.asarray(t, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float32)
    dy_gpu = cp.asarray(dy, dtype=cp.float32)
    
    # Phase folding
    fast_fold_gpu = module.get_function('foldFast')
    block_size, grid_size_x = calcGridBlockSize(t_size)
    fast_fold_gpu((grid_size_x, 1), (block_size,), 
                 (t_gpu, periods_gpu, phases_gpu, arrays_dict['periods_size_gpu'], arrays_dict['t_size_gpu']))
    
    # Sorting (optimized chunking for single period)
    i_max = MAX_ARGSORT_ITERATIONS
    for i in range(1, i_max + 1):
        start_idx = (i - 1) // i_max
        end_idx = i // i_max
        sort_index_gpu[start_idx:end_idx] = phases_gpu[start_idx:end_idx].argsort()
    
    # Data processing arrays
    patched_datas_gpu = cp.empty((1, t_size + max_duration), dtype=cp.float32)
    patched_dys_gpu = cp.empty((1, t_size + max_duration), dtype=cp.float32)
    
    # Light curve processing
    lc_arr_max_len = np.array([np.max(single_durations)], dtype=np.int32)
    lc_arr_full_length = 1 - np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') 
                                      for x in single_lc_arr])
    
    lc_arr_max_len_gpu = cp.asarray(lc_arr_max_len, dtype=cp.int32)
    lc_arr_full_length_gpu = cp.asarray(lc_arr_full_length, dtype=cp.float32)
    
    # Additional computation arrays
    edge_effect_corrections_gpu = cp.empty(1, dtype=cp.float32)
    inverse_squared_patched_dys_gpu = cp.empty((1, t_size + max_duration), dtype=cp.float32)
    durations_gpu = cp.asarray(single_durations, dtype=cp.int32)
    overshoot_gpu = cp.array(single_lc_cache_overview["overshoot"], dtype=cp.float32)
    datapoints_gpu = cp.array([len(y)], dtype=cp.int32)
    transit_depth_min_gpu = cp.array([transit_depth_min], dtype=cp.float32)
    
    # Main computation arrays
    full_sum_gpu = cp.empty((1, len(single_durations)), dtype=cp.float32)
    cumsum_gpu = cp.empty((1, arrays_dict['patched_data_size_gpu'].get()[0]), dtype=cp.float32)
    ootr_gpu = cp.empty((1, len(single_durations), t_size), dtype=cp.float32)
    
    # Execute GPU kernels for single period
    _execute_single_period_gpu_kernels(module, t_gpu, periods_gpu, phases_gpu, sort_index_gpu,
                                      arrays_dict, patched_datas_gpu, patched_dys_gpu,
                                      y_gpu, dy_gpu, inverse_squared_patched_dys_gpu,
                                      edge_effect_corrections_gpu, cumsum_gpu,
                                      full_sum_gpu, durations_gpu, ootr_gpu,
                                      lowest_residuals_gpu, lc_arr_full_length_gpu, lc_arr_max_len_gpu,
                                      overshoot_gpu, datapoints_gpu, transit_depth_min_gpu,
                                      durations_max_gpu, durations_min_gpu, t_size, max_duration, 
                                      single_durations)
    
    return lowest_residuals_gpu

def _execute_single_period_gpu_kernels(module, t_gpu, periods_gpu, phases_gpu, sort_index_gpu,
                                      arrays_dict, patched_datas_gpu, patched_dys_gpu,
                                      y_gpu, dy_gpu, inverse_squared_patched_dys_gpu,
                                      edge_effect_corrections_gpu, cumsum_gpu,
                                      full_sum_gpu, durations_gpu, ootr_gpu,
                                      lowest_residuals_gpu, lc_arr_full_length_gpu, lc_arr_max_len_gpu,
                                      overshoot_gpu, datapoints_gpu, transit_depth_min_gpu,
                                      durations_max_gpu, durations_min_gpu, t_size, max_duration, 
                                      single_durations):
    """Execute all GPU kernels for single period search."""
    
    # 1. Patch data
    patch_data_gpu = module.get_function('patchData')
    block_size, grid_size_x = calcGridBlockSize(t_size + max_duration)
    patch_data_gpu((grid_size_x, 1), (block_size,),
                  (patched_datas_gpu, patched_dys_gpu, arrays_dict['patched_data_size_gpu'],
                   sort_index_gpu, arrays_dict['max_duration_gpu'], y_gpu, dy_gpu, arrays_dict['t_size_gpu']))
    
    # 2. Calculate inverse squared patched dy
    calc_inverse_squared_patched_dy_gpu = module.get_function('calcInverseSquaredPatchedDy')
    block_size, grid_size_x = calcGridBlockSize(arrays_dict['patched_data_size_gpu'].get()[0])
    calc_inverse_squared_patched_dy_gpu((grid_size_x, 1, 1), (block_size, 1, 1),
                                       (inverse_squared_patched_dys_gpu, patched_dys_gpu, 
                                        arrays_dict['patched_data_size_gpu']))
    
    # 3. Calculate edge effect corrections
    calc_edge_effect_corrections_gpu = module.get_function('calcEdgeEffectCorrections')
    block_size, grid_size_x = calcGridBlockSize(1)
    calc_edge_effect_corrections_gpu((grid_size_x, 1, 1), (block_size, 1, 1),
                                    (edge_effect_corrections_gpu, patched_datas_gpu, 
                                     inverse_squared_patched_dys_gpu, arrays_dict['patched_data_size_gpu'],
                                     arrays_dict['max_duration_gpu'], arrays_dict['period_size_gpu']))
    
    # 4. Calculate cumulative sum
    cumsum_gpu[0] = cp.cumsum(patched_datas_gpu[0])
    
    # 5. Calculate full sum
    calc_all_full_sum_gpu = module.get_function('calcAllFullSum')
    block_size, grid_size_x = calcGridBlockSize(len(single_durations))
    calc_all_full_sum_gpu((grid_size_x, 1, 1), (block_size, 1, 1),
                         (full_sum_gpu, patched_datas_gpu, inverse_squared_patched_dys_gpu,
                          arrays_dict['patched_data_size_gpu'], durations_gpu, arrays_dict['durations_size_gpu'],
                          arrays_dict['period_size_gpu']))
    
    # 6. Calculate out-of-transit residuals - step 1
    calc_ootr_step1_gpu = module.get_function('calcAllOutOfTransitResiduals_step1_2GPU')
    block_size, grid_size_x = calcGridBlockSize(t_size)
    calc_ootr_step1_gpu((grid_size_x, len(single_durations), 1), (block_size, 1, 1),
                       (ootr_gpu, patched_datas_gpu, durations_gpu, arrays_dict['durations_size_gpu'],
                        inverse_squared_patched_dys_gpu, arrays_dict['patched_data_size_gpu'], 
                        arrays_dict['t_size_gpu']))
    
    # 7. Calculate cumulative OOTR
    ootr_gpu = cp.cumsum(ootr_gpu, axis=-1)
    
    # 8. Calculate out-of-transit residuals - step 2
    calc_ootr_step2_gpu = module.get_function('calcAllOutOfTransitResiduals_step2_2GPU')
    calc_ootr_step2_gpu((grid_size_x, len(single_durations), 1), (block_size, 1, 1),
                       (ootr_gpu, arrays_dict['durations_size_gpu'], arrays_dict['patched_data_size_gpu'],
                        durations_gpu, arrays_dict['t_size_gpu'], full_sum_gpu))
    
    # 9. Calculate lowest residuals using NoSkip kernel
    calc_lowest_residuals_gpu = module.get_function('calcAllLowestResidualsGPUBNoSkip')
    calc_lowest_residuals_gpu((grid_size_x, len(single_durations), 1), (block_size, 1, 1),
                             (lowest_residuals_gpu, arrays_dict['t_size_gpu'], patched_datas_gpu,
                              arrays_dict['patched_data_size_gpu'], durations_gpu, arrays_dict['durations_size_gpu'],
                              lc_arr_full_length_gpu, lc_arr_max_len_gpu, inverse_squared_patched_dys_gpu,
                              overshoot_gpu, ootr_gpu, full_sum_gpu, edge_effect_corrections_gpu,
                              datapoints_gpu, cumsum_gpu, durations_max_gpu, durations_min_gpu, 
                              transit_depth_min_gpu))

def _process_single_period_results(lowest_residuals_gpu, period, t, y, dy, single_durations, 
                                  single_lc_cache_overview, t_size, max_duration):
    """Process results from single period search and calculate transit parameters."""
    
    # Find best location
    best_location = lowest_residuals_gpu.argmin().get()
    duration_index = np.floor(best_location / t_size).astype(int)
    duration_points_num = single_durations[duration_index]
    
    # Find corresponding duration in cache
    find = np.where(single_lc_cache_overview["width_in_samples"] == duration_points_num)[0]
    if len(find) > 1:
        find = find[0]
    best_row = find.item()
    raw_duration = single_lc_cache_overview['duration'][best_row]
    
    # Phase fold and process light curve
    best_time, best_flux, best_flux_dy = foldCPU(t, y, dy, period)
    best_flux = np.concatenate((best_flux, best_flux[:max_duration]))
    best_flux_dy = np.concatenate((best_flux_dy, best_flux_dy[:max_duration]))
    
    # Calculate transit parameters
    best_row_t0 = best_location % t_size
    transit_mean = best_flux[best_row_t0:best_row_t0 + duration_points_num].mean()
    
    # Transit depth calculation
    overshoot = single_lc_cache_overview["overshoot"][duration_index]
    transit_depth = ((1 - transit_mean) * overshoot).item()
    
    # Out-of-transit data
    data_out_transit = np.concatenate((best_flux[0:best_row_t0], 
                                      best_flux[best_row_t0 + duration_points_num:]))
    
    # Adjust T0 if needed
    if best_row_t0 > t_size - 1:
        best_row_t0 = best_row_t0 - t_size
    
    return (best_location, duration_index, duration_points_num, best_row, raw_duration,
            best_time, best_flux, best_flux_dy, best_row_t0, transit_depth, data_out_transit)

def _calculate_single_period_statistics(period, t, y, transit_depth, duration_points_num, 
                                       data_out_transit, best_time, best_row_t0, raw_duration):
    """Calculate SNR and timing statistics for single period search."""
    
    # SNR calculations
    snr_fit = (1 - transit_depth) * (duration_points_num ** 0.5) / np.std(data_out_transit)
    
    # Red noise calculation
    data_cumsum = np.cumsum(data_out_transit)
    data_slide_avg = (data_cumsum[duration_points_num:] - data_cumsum[:-duration_points_num]) / duration_points_num
    red_noise = np.std(data_slide_avg)
    
    # Timing calculations
    tx = best_time[best_row_t0]
    t0 = tx - int((tx - min(t)) / period) * period - period
    transit_times = all_transit_times(t0, t, period)
    
    # Pink SNR calculation
    snr_fit_pink = (1 - transit_depth) / ((np.std(data_out_transit)**2 / duration_points_num) + 
                                         (red_noise**2 / len(transit_times)))**0.5
    
    # Transit duration calculation
    transit_duration_in_days = calcDurationDays(t, period, t0, raw_duration)
    
    # Adjust T0 and transit times
    t0 = t0 + transit_duration_in_days / 2
    transit_times = transit_times + transit_duration_in_days / 2
    if t0 < min(t):
        t0 = t0 + period
    
    return t0, transit_times, transit_duration_in_days, snr_fit, snr_fit_pink

def _calculate_comprehensive_snr_statistics(t, y, period, raw_duration, t0, transit_times, transit_duration_in_days):
    """Calculate comprehensive SNR statistics using transit analysis functions."""
    
    # In-transit statistics
    (depth_mean_odd, depth_mean_even, depth_mean_odd_std, depth_mean_even_std, 
     all_flux_intransit_odd, all_flux_intransit_even, per_transit_count, 
     transit_depths, transit_depths_uncertainties) = intransit_stats(t, y, transit_times, transit_duration_in_days)
    
    # SNR per transit calculation
    snr_per_transit, snr_pink_per_transit = snr_stats(
        t=t, y=y, period=period, duration=raw_duration, T0=t0,
        transit_times=transit_times, transit_duration_in_days=transit_duration_in_days,
        per_transit_count=per_transit_count)
    
    # Combined flux analysis
    all_flux_intransit = np.concatenate([all_flux_intransit_odd, all_flux_intransit_even])
    intransit = transit_mask(t, period, 2 * raw_duration, t0)
    flux_ootr = y[~intransit]
    depth_mean = np.mean(all_flux_intransit)
    
    # Final SNR calculations
    snr = ((1 - depth_mean) / np.std(flux_ootr)) * len(all_flux_intransit) ** 0.5
    snr_pink = np.mean(snr_pink_per_transit) * (len(transit_times) ** 0.5)
    
    return snr, snr_pink

# This function is used for refind Duration and T0 since we skip some points in "search_multi_periods"
def search_single_periods(
    period,
    t,
    y,
    dy,
    transit_depth_min,
    lc_arr,
    lc_cache_overview,
    GPUDeviceID=0
):
    """
    Optimized single period search function with improved code structure.
    
    This function is used to refine Duration and T0 since we skip some points in "search_multi_periods".
    
    Args:
        period: Single period to analyze
        t, y, dy: Time series data and uncertainties
        transit_depth_min: Minimum transit depth threshold
        lc_arr: Light curve array
        lc_cache_overview: Cache overview data
        GPUDeviceID: GPU device ID
    
    Returns:
        Tuple containing transit parameters and statistics
    """
    
    # Step 1: Setup GPU environment
    module = _setup_single_period_gpu_environment(GPUDeviceID)
    
    # Step 2: Prepare duration data (reuse helper function)
    durations, lc_arr, lc_cache_overview, max_duration = _prepare_duration_data(lc_cache_overview, lc_arr)
    
    # Step 3: Initialize basic GPU arrays (before duration filtering)
    t_size = len(t)
    (periods_gpu, durations_max_gpu, durations_min_gpu, phases_gpu, 
     sort_index_gpu, arrays_dict) = _initialize_single_period_arrays(
        period, t_size, max_duration)
    
    # Step 4: Setup duration grid and filter durations
    durations_bool_list = _setup_single_period_duration_grid(
        module, period, t, durations, periods_gpu, durations_max_gpu, 
        durations_min_gpu, arrays_dict)
    
    # Filter arrays based on duration limits
    single_durations = durations[durations_bool_list]
    single_lc_arr = lc_arr[durations_bool_list]
    single_lc_cache_overview = lc_cache_overview[durations_bool_list]
    
    # Step 5: Create result arrays after filtering and update arrays_dict
    lowest_residuals_gpu = _create_result_arrays_after_filtering(single_durations, t_size)
    arrays_dict['durations_size_gpu'] = cp.asarray([len(single_durations)], dtype=cp.int32)
    
    # Step 6: Execute GPU computations
    lowest_residuals_gpu = _execute_single_period_gpu_computation(
        module, period, t, y, dy, single_durations, single_lc_arr, single_lc_cache_overview,
        t_size, max_duration, transit_depth_min, periods_gpu, durations_max_gpu, 
        durations_min_gpu, phases_gpu, sort_index_gpu, lowest_residuals_gpu, arrays_dict)
    
    # Step 7: Process results and extract transit parameters
    (best_location, duration_index, duration_points_num, best_row, raw_duration,
     best_time, best_flux, best_flux_dy, best_row_t0, transit_depth, 
     data_out_transit) = _process_single_period_results(
        lowest_residuals_gpu, period, t, y, dy, single_durations, 
        single_lc_cache_overview, t_size, max_duration)
    
    # Step 8: Calculate timing and SNR statistics
    t0, transit_times, transit_duration_in_days, snr_fit, snr_fit_pink = _calculate_single_period_statistics(
        period, t, y, transit_depth, duration_points_num, data_out_transit, 
        best_time, best_row_t0, raw_duration)
    
    # Step 9: Calculate comprehensive SNR statistics
    snr, snr_pink = _calculate_comprehensive_snr_statistics(
        t, y, period, raw_duration, t0, transit_times, transit_duration_in_days)
    
    return (raw_duration, duration_points_num, transit_duration_in_days, transit_depth, 
            t0, transit_times, snr, snr_pink, snr_fit, snr_fit_pink)