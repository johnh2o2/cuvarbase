def getGPUCode():
    GPUCode = """
extern "C"{
    // Configuration constants
    #define SKIP_POINT 8
    
    // Physical constants - optimized with appropriate values
    #define R_STAR_MIN 0.05                    // Minimum stellar radius (solar radii) - updated boundary
    #define R_STAR_MAX 4.0                     // Maximum stellar radius (solar radii) - updated boundary
    #define SECONDS_PER_DAY 86400              // Seconds in a day
    #define R_SUN 695508000                    // Radius of the Sun [m]
    #define R_JUP 69911000                     // Radius of Jupiter [m]
    #define FRACTIONAL_TRANSIT_DURATION_MAX 0.15  // Maximum fractional transit duration - updated value

    // Derived constants for duration calculations - optimized values
    #define PI_GM_MAX 416970                   // Simplified pi*G*M_max for duration calc
    #define PI_GM_MIN 20848                    // Simplified pi*G*M_min for duration calc - updated boundary
    #define RS_MIN (R_SUN * R_STAR_MIN)        // Minimum stellar radius in meters
    #define RS_MAX (R_SUN * R_STAR_MAX)        // Maximum stellar radius in meters

    // Transit fitting constants
    #define SIGNAL_DEPTH 0.5                   // Standard signal depth for fitting
    #define FLOAT_INFINITY 0x7f800000          // IEEE-754 float infinity
    #define SCALE_FACTOR 1000000000000000.0    // Scale factor for duration calculations

    /**
     * Fast phase folding kernel with optimized memory access
     * Calculates phase = (time / period) - floor(time / period) for each time point
     */
    __global__ void foldFast(const double* time, const double* periods, double* phase, 
                             int* periodSize, int* timeSize) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (tid < (*timeSize)) {
            double time_val = time[tid];
            double period = periods[y];
            double phase_raw = time_val / period;
            phase[tid + y * (*timeSize)] = phase_raw - (int)(phase_raw);
        }
    }

    /**
     * Optimized duration grid calculation kernel
     * Based on stellar and planetary parameters with improved performance
     */
    __global__ void durationsGrid(const double* periods, int* durationsMax, int* durationsMin,
                                 const float* tLength, const int* tSize, const int* periodSize) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (tid < (*periodSize)) {
            float length = *tLength;
            int size = *tSize;
            
            // Calculate transit statistics with optimized operations
            double period_days = periods[tid];
            double no_of_transits_naive = length / period_days;
            double correction_factor = (no_of_transits_naive + 1.0) / no_of_transits_naive;

            double period_seconds = period_days * SECONDS_PER_DAY;
            
            // Pre-calculate common factors for efficiency
            double period_factor_min = (4.0 * period_seconds) / (PI_GM_MIN * SCALE_FACTOR);
            double period_factor_max = (4.0 * period_seconds) / (PI_GM_MAX * SCALE_FACTOR);
            
            // Calculate minimum and maximum transit durations
            double T14Min = RS_MIN * pow(period_factor_min, 1.0 / 3.0);
            double T14Max = (RS_MAX + R_JUP * 2.0) * pow(period_factor_max, 1.0 / 3.0);
            
            double durationMin = T14Min / period_seconds;
            double durationMax = T14Max / period_seconds;
            
            // Apply maximum duration constraints efficiently
            durationMin = (durationMin > FRACTIONAL_TRANSIT_DURATION_MAX) ? 
                         FRACTIONAL_TRANSIT_DURATION_MAX : durationMin;
            durationMax = (durationMax > FRACTIONAL_TRANSIT_DURATION_MAX) ? 
                         FRACTIONAL_TRANSIT_DURATION_MAX : durationMax;
            
            // Convert to sample indices with optimized rounding
            int duration_min_in_samples = floor(durationMin * size);
            int duration_max_in_samples = ceil(durationMax * size * correction_factor);
            
            durationsMin[tid] = duration_min_in_samples;
            durationsMax[tid] = duration_max_in_samples;
        }
    }

    /**
     * Optimized duration boolean array generation
     */
    __global__ void durationBool(int* durationsMax, int* durationsMin, const int* durationSize,
                                const int* periodSize, const int* durations, bool* durationBoolArray) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x; // period index
        int y = blockDim.y * blockIdx.y + threadIdx.y;   // duration index

        if (tid < (*periodSize) && y < (*durationSize)) {
            int duration_min = durationsMin[tid];
            int duration_max = durationsMax[tid];
            int current_duration = durations[y];
            
            // Optimized boolean assignment
            durationBoolArray[y + tid * (*durationSize)] = 
                (current_duration >= duration_min && current_duration <= duration_max);
        }
    }

    /**
     * Optimized data patching kernel with improved memory coalescing
     */
    __global__ void patchData(float *in_patchedData, float *in_patchedDys,
                             int *patchedDataSize, int *in_sortIndex, int *maxDuration,
                             float *flux, float *dy, int *tSize) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x; // patchedData index
        int y = blockIdx.y * blockDim.y + threadIdx.y;   // period index

        float *patchedData = in_patchedData + y * (*patchedDataSize);
        float *patchedDys = in_patchedDys + y * (*patchedDataSize);
        int *sortIndex = in_sortIndex + y * (*tSize);

        if (tid < (*tSize)) {
            int src_idx = sortIndex[tid];
            patchedData[tid] = flux[src_idx];
            patchedDys[tid] = dy[src_idx];
        } else if (tid < (*tSize + *maxDuration)) {
            int src_idx = sortIndex[tid - (*tSize)];
            patchedData[tid] = flux[src_idx];
            patchedDys[tid] = dy[src_idx];
        }
    }

    /**
     * Optimized inverse squared calculation
     */
    __global__ void calcInverseSquaredPatchedDy(float *out, float *patched_dys, int *patched_data_size) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (tid < *patched_data_size) {
            float dy_val = patched_dys[tid + y * (*patched_data_size)];
            out[tid + y * (*patched_data_size)] = 1.0f / (dy_val * dy_val);
        }
    }

    /**
     * Optimized edge effect correction calculation
     */
    __global__ void calcEdgeEffectCorrections(float *out, float *patch_data,
                                             float* inverse_squared_patched_dys, int *patched_data_size,
                                             int* maxDuration, int* period_size) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (tid >= *period_size) {
            return;
        }

        float* patched_data = patch_data + tid * (*patched_data_size);
        float* inverse_squared_patched_dy = inverse_squared_patched_dys + tid * (*patched_data_size);

        double edgeEffect = 0.0;
        int start_idx = (*patched_data_size) - (*maxDuration);
        
        for (int j = start_idx; j < (*patched_data_size); j++) {
            double patchDataJ = (double)(patched_data[j]);
            double patchDataDyJ = (double)(inverse_squared_patched_dy[j]);
            edgeEffect += (1.0 + patchDataJ * patchDataJ - 2.0 * patchDataJ) * patchDataDyJ;
        }
        out[tid] = edgeEffect;
    }

    /**
     * Optimized full sum calculation with better memory access patterns
     */
    __global__ void calcAllFullSum(float* fullsums, float *in_patched_data,
                                  float *in_inverse_squared_patched_dy, int *patched_data_size,
                                  int *in_duration, int *duration_size, int *period_size_gpu) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    // duration index
        int y = blockIdx.y * blockDim.y + threadIdx.y;      // period index

        float *patched_data = in_patched_data + y * (*patched_data_size);
        float *inverse_squared_patched_dy = in_inverse_squared_patched_dy + y * (*patched_data_size);

        // Calculate full sum once
        float fullsum = 0.0f;
        for (int i = 0; i < *patched_data_size; i++) {
            float diff = 1.0f - patched_data[i];
            fullsum += diff * diff * inverse_squared_patched_dy[i];
        }

        if (tid < (*duration_size)) {
            int window = in_duration[tid];
            float window_sum = 0.0f;
            
            // Calculate window sum efficiently
            for (int i = 0; i < window; i++) {
                float diff = 1.0f - patched_data[i];
                window_sum += diff * diff * inverse_squared_patched_dy[i];
            }
            
            fullsums[tid + y * (*duration_size)] = fullsum - window_sum;
        }
    }
    
    /**
     * Optimized full sum calculation using precomputed error_prefix_sum
     * This is O(1) per thread instead of O(N) - major performance improvement
     */
    __global__ void calcAllFullSum_v2(float* fullsums, 
                                      float *error_prefix_sum,
                                      int patched_data_size,
                                      int *in_duration, 
                                      int duration_size,
                                      int singleCalcPeriods) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    // duration index
        int period_idx = blockIdx.y;                         // period index

        if (tid >= duration_size || period_idx >= singleCalcPeriods) {
            return;
        }

        // Get pointer to this period's error_prefix_sum row
        float *prefix_sum = error_prefix_sum + period_idx * patched_data_size;
        
        // fullsum = total sum = prefix_sum[last_element]
        float fullsum = prefix_sum[patched_data_size - 1];
        
        // window_sum = prefix_sum[window - 1] (sum of first 'window' elements)
        int window = in_duration[tid];
        float window_sum = (window > 0) ? prefix_sum[window - 1] : 0.0f;
        
        // Result: fullsum - window_sum = sum from window to end
        fullsums[tid + period_idx * duration_size] = fullsum - window_sum;
    }
    
    /**
     * Optimized out-of-transit residuals calculation - Step 1
     */
    __global__ void calcAllOutOfTransitResiduals_step1_2GPU(float *temp_ootr,
                                                            float *in_patched_data, int *in_duration, int *duration_size,
                                                            float *in_inverse_squared_patched_dy, int *patched_data_size,
                                                            int *resultArrayXAxisSize) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    // point index
        int y = blockIdx.y * blockDim.y + threadIdx.y;      // duration index
        int z = blockIdx.z * blockDim.z + threadIdx.z;      // period index

        if (tid >= *resultArrayXAxisSize) {
            return;
        }

        float *patched_data = in_patched_data + z * (*patched_data_size);
        float *inverse_squared_patched = in_inverse_squared_patched_dy + z * (*patched_data_size);
        int window = in_duration[y];
        
        int becomes_visible = tid;
        int becomes_invisible = tid + window;
                
        float diff_visible = 1.0f - patched_data[becomes_visible];
        float diff_invisible = 1.0f - patched_data[becomes_invisible];
        
        float add_visible_left = diff_visible * diff_visible * inverse_squared_patched[becomes_visible];
        float remove_invisible_right = diff_invisible * diff_invisible * inverse_squared_patched[becomes_invisible];
        float weight = add_visible_left - remove_invisible_right;
        
        temp_ootr[tid + y * (*resultArrayXAxisSize) + z * (*resultArrayXAxisSize) * (*duration_size)] = weight;
    }

    /**
     * Optimized out-of-transit residuals calculation - Step 2
     */
    __global__ void calcAllOutOfTransitResiduals_step2_2GPU(float *in_ootr,
                                                            int *duration_size, int *patched_data_size, int *in_duration,
                                                            int *resultArrayXAxisSize, float *in_fullsum) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;      // point index
        int tid = blockIdx.y * blockDim.y + threadIdx.y;    // duration index
        int z = blockIdx.z * blockDim.z + threadIdx.z;      // period index

        if (i >= (*resultArrayXAxisSize)) {
            return;
        }

        float *fullsum = in_fullsum + z * (*duration_size);
        float *ootr = in_ootr + z * (*resultArrayXAxisSize) * (*duration_size);
        
        ootr[i + tid * (*resultArrayXAxisSize)] = fullsum[tid] + ootr[i + tid * (*resultArrayXAxisSize)];
    }

    /**
    * STAGE 1 KERNEL (REPLACEMENT FOR block_scan_and_sum)
    * This kernel pre-computes the prefix sum of the base error term for each period.
    * It's a 2D problem now: (periods, points).
    * It uses the same two-stage scan logic internally but is shown here as a single conceptual function.
    * We will reuse the robust 'block_scan_and_sum' and 'add_offsets_and_finalize' logic for this.
    */
    __global__ void calculate_base_error(
        float* out_base_error, // Shape: (num_periods, patched_data_size)
        const float* in_patched_data,
        const float* in_inverse_squared_patched_dy,
        int patched_data_size,
        int num_periods
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x; // point index
        int z = blockIdx.y; // period index

        if (z >= num_periods || tid >= patched_data_size) {
            return;
        }

        const float* patched_data_period = in_patched_data + z * patched_data_size;
        const float* inverse_squared_patched_period = in_inverse_squared_patched_dy + z * patched_data_size;

        float diff = 1.0f - patched_data_period[tid];
        out_base_error[tid + z * patched_data_size] = diff * diff * inverse_squared_patched_period[tid];
    }
    /**
    * STAGE 2 KERNEL (FINAL, CORRECTED VERSION)
    * Faithfully reproduces the original algorithm's logic, including boundary conditions.
    */
    __global__ void calculate_final_ootr_v3(
        float* out_ootr,
        const float* error_prefix_sum, // PRE-COMPUTED. Shape: (num_periods, patchedDatasSize)
        const float* in_fullsum,
        const int* in_duration,
        int tSize,                 // The analysis length (<= patchedDatasSize)
        int patchedDatasSize,      // The full data length
        int duration_size,
        int num_periods
    ) {
        int p = blockIdx.x * blockDim.x + threadIdx.x; // point index
        int d = blockIdx.y;                           // duration index
        int z = blockIdx.z;                           // period index

        // Only compute for the required analysis length 'tSize'
        if (p >= tSize || d >= duration_size || z >= num_periods) {
            return;
        }

        // Pointer to the current period's prefix sum array (length is patchedDatasSize)
        const float* P_E = error_prefix_sum + (long long)z * patchedDatasSize; 

        int window = in_duration[d];

        // --- Core Telescoping Sum Logic with FAITHFUL BOUNDARY CHECKS ---
        
        // P_E[p] is always in bounds due to the check above
        float p_e_p = P_E[p];

        // Check bounds for P_E[p + window]
        float p_e_p_plus_window = (p + window < patchedDatasSize) ? P_E[p + window] : 0.0f;

        // Check bounds for P_E[window - 1]
        float p_e_window_minus_1 = (window > 0 && window - 1 < patchedDatasSize) ? P_E[window - 1] : 0.0f;

        float cumsum_weight = p_e_p - (p_e_p_plus_window - p_e_window_minus_1);
        
        // --- Finalization Step ---
        float fullsum_val = in_fullsum[(long long)d + (long long)z * duration_size];

        long long final_idx = (long long)p + (long long)d * tSize + (long long)z * tSize * duration_size;
        out_ootr[final_idx] = fullsum_val + cumsum_weight;
    }



    /**
     * Optimized device function for cumsum average calculation
     */
    __device__ float calcAverageFromCumsum(float *inPatchedDataCumsum, int duration, 
                                          int *patched_data_size, int tid, int z) {
        if (tid == 0) {
            return 1.0f - inPatchedDataCumsum[z * (*patched_data_size) + duration - 1] / duration;
        } else {
            float end_val = inPatchedDataCumsum[z * (*patched_data_size) + tid + duration - 1];
            float start_val = inPatchedDataCumsum[z * (*patched_data_size) + tid - 1];
            return 1.0f - (end_val - start_val) / duration;
        }
    }

    /**
    * FIXED VERSION: calcAllLowestResidualsGPUB_SignalTiled_v2
    * REMOVED shared memory tiling to fix __syncthreads() inside conditional branch bug.
    * This was causing NaN values when some threads skipped the conditional block.
    */
    __global__ void calcAllLowestResidualsGPUB_SignalTiled_v2(
        /* ... 参数列表与原版完全相同 ... */
        float *out, int *resultArrayXAxisSize,
        float *in_patched_datas, int *in_patched_datas_size, int *in_duration, int *in_duration_size,
        float *in_signal, int *in_max_signal_x_size, float *in_inverse_squared_patched_dys,
        float *in_overshoot, float *in_ootr, float *in_fullsum,
        float *in_summed_edge_effect_correction, int *in_datapoints, float *cumsumGPU,
        float *in_transit_depth_min
    ) {
        // 线程索引和初始设置
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y;
        int z = blockIdx.z;

        if (tid >= *resultArrayXAxisSize) {
            return;
        }

        int duration = in_duration[y];
        
        int skipPoint = (duration > SKIP_POINT) ? (duration / SKIP_POINT) : 1;
        float transit_depth_min = *in_transit_depth_min;
        int datapoints = *in_datapoints;
        float calc_mean = calcAverageFromCumsum(cumsumGPU, duration, in_patched_datas_size, tid, z);

        float current_stat = (float)datapoints;

        if (calc_mean > transit_depth_min && tid % skipPoint == 0) {
            float ootr = (tid == 0) ? 
                in_fullsum[(long long)z * (*in_duration_size) + y] :
                in_ootr[(long long)y * (*resultArrayXAxisSize) + (long long)z * (*resultArrayXAxisSize) * (*in_duration_size) + tid - 1];
            
            // 指针设置
            float *data = in_patched_datas + (long long)z * (*in_patched_datas_size) + tid;
            float *dy = in_inverse_squared_patched_dys + (long long)z * (*in_patched_datas_size) + tid;
            float *signal = in_signal + (long long)y * (*in_max_signal_x_size);
            
            float reverse_scale = calc_mean * in_overshoot[y] * 2.0f;
            float summed_edge_effect_correction = in_summed_edge_effect_correction[z];
            float intransit_residual = 0.0f;
            int skipSearchPoint = 1;

            // --- 直接全局内存访问 (无 __syncthreads) ---
            for (int i = 0; i < duration; i += skipSearchPoint) {
                float sigi = signal[i] * reverse_scale;
                float loss = data[i] - (1.0f - sigi);
                intransit_residual += loss * loss * dy[i];
            }
            
            // actualLossFraction 计算
            float actualLossFraction = (float)duration / (((duration - 1) / skipSearchPoint) + 1);
            
            current_stat = intransit_residual * actualLossFraction + ootr - summed_edge_effect_correction;
        }

        out[(long long)tid + (long long)y * (*resultArrayXAxisSize) + (long long)z * (*resultArrayXAxisSize) * (*in_duration_size)] = current_stat;
    }

    /**
    * OPTIMIZED VERSION: calcAllLowestResidualsGPUB_SignalTiled_v3
    * Optimizations:
    * 1. Pre-load signal array into shared memory (before conditional branch)
    * 2. Loop unrolling (4-way) for inner loop
    * 3. Use __ldg() for read-only global memory access
    * 4. Precompute (1.0f - signal[i] * reverse_scale) pattern
    */
    __global__ void calcAllLowestResidualsGPUB_SignalTiled_v3(
        float *out, int *resultArrayXAxisSize,
        float *in_patched_datas, int *in_patched_datas_size, int *in_duration, int *in_duration_size,
        float *in_signal, int *in_max_signal_x_size, float *in_inverse_squared_patched_dys,
        float *in_overshoot, float *in_ootr, float *in_fullsum,
        float *in_summed_edge_effect_correction, int *in_datapoints, float *cumsumGPU,
        float *in_transit_depth_min
    ) {
        // Dynamic shared memory for signal array
        extern __shared__ float s_signal[];
        
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y;   // duration index
        int z = blockIdx.z;   // period index
        
        // Load constants once using __ldg for read-only cache
        int resultArraySize = __ldg(resultArrayXAxisSize);
        int duration = __ldg(&in_duration[y]);
        int patched_data_size = __ldg(in_patched_datas_size);
        int duration_size = __ldg(in_duration_size);
        float transit_depth_min = __ldg(in_transit_depth_min);
        int datapoints = __ldg(in_datapoints);
        float overshoot_y = __ldg(&in_overshoot[y]);
        float summed_edge_effect_correction = __ldg(&in_summed_edge_effect_correction[z]);
        
        // === STAGE 1: Cooperatively load signal into shared memory ===
        // All threads participate, regardless of later conditions (safe for __syncthreads)
        float *signal_global = in_signal + (long long)y * __ldg(in_max_signal_x_size);
        for (int i = threadIdx.x; i < duration; i += blockDim.x) {
            s_signal[i] = __ldg(&signal_global[i]);
        }
        __syncthreads();  // Safe: all threads execute this
        
        // === STAGE 2: Early exit for out-of-bounds threads ===
        if (tid >= resultArraySize) {
            return;
        }
        
        // === STAGE 3: Main computation ===
        int skipPoint = (duration > SKIP_POINT) ? (duration / SKIP_POINT) : 1;
        float calc_mean = calcAverageFromCumsum(cumsumGPU, duration, in_patched_datas_size, tid, z);
        
        float current_stat = (float)datapoints;
        
        if (calc_mean > transit_depth_min && tid % skipPoint == 0) {
            // Calculate ootr
            float ootr;
            if (tid == 0) {
                ootr = in_fullsum[(long long)z * duration_size + y];
            } else {
                ootr = in_ootr[(long long)y * resultArraySize + (long long)z * resultArraySize * duration_size + tid - 1];
            }
            
            // Setup pointers with offset
            long long base_offset = (long long)z * patched_data_size + tid;
            const float *data = in_patched_datas + base_offset;
            const float *dy = in_inverse_squared_patched_dys + base_offset;
            
            // Precompute scale factor
            float reverse_scale = calc_mean * overshoot_y * 2.0f;
            
            // === OPTIMIZED INNER LOOP with 4-way unrolling ===
            float intransit_residual = 0.0f;
            int i = 0;
            
            // Process 4 elements at a time
            int duration_unroll = duration & ~3;  // Round down to multiple of 4
            for (; i < duration_unroll; i += 4) {
                // Prefetch signal from shared memory
                float sig0 = s_signal[i];
                float sig1 = s_signal[i + 1];
                float sig2 = s_signal[i + 2];
                float sig3 = s_signal[i + 3];
                
                // Load data and dy using __ldg
                float d0 = __ldg(&data[i]);
                float d1 = __ldg(&data[i + 1]);
                float d2 = __ldg(&data[i + 2]);
                float d3 = __ldg(&data[i + 3]);
                
                float dy0 = __ldg(&dy[i]);
                float dy1 = __ldg(&dy[i + 1]);
                float dy2 = __ldg(&dy[i + 2]);
                float dy3 = __ldg(&dy[i + 3]);
                
                // Compute losses: loss = data - (1 - signal * reverse_scale)
                float loss0 = d0 - (1.0f - sig0 * reverse_scale);
                float loss1 = d1 - (1.0f - sig1 * reverse_scale);
                float loss2 = d2 - (1.0f - sig2 * reverse_scale);
                float loss3 = d3 - (1.0f - sig3 * reverse_scale);
                
                // Accumulate residuals using FMA
                intransit_residual = fmaf(loss0 * loss0, dy0, intransit_residual);
                intransit_residual = fmaf(loss1 * loss1, dy1, intransit_residual);
                intransit_residual = fmaf(loss2 * loss2, dy2, intransit_residual);
                intransit_residual = fmaf(loss3 * loss3, dy3, intransit_residual);
            }
            
            // Handle remaining elements
            for (; i < duration; i++) {
                float sig = s_signal[i];
                float d = __ldg(&data[i]);
                float dy_val = __ldg(&dy[i]);
                float loss = d - (1.0f - sig * reverse_scale);
                intransit_residual = fmaf(loss * loss, dy_val, intransit_residual);
            }
            
            current_stat = intransit_residual + ootr - summed_edge_effect_correction;
        }
        
        // Write output
        long long out_idx = (long long)tid + (long long)y * resultArraySize + (long long)z * resultArraySize * duration_size;
        out[out_idx] = current_stat;
    }

    /**
     * Optimized trapezoid fit kernel
     */
    __global__ void trapezoidFit(float *results, float *inData, float *inInverseSquaredDys,
                                int duration, int inT0Index, float *transitMean, int tidMax) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (tid < tidMax) {
            float *result = results + tid * duration;
            float TrapezoidDepth = ((float)tidMax * (*transitMean) - 0.5f * (float)tid) / 
                                  ((float)tidMax - 0.5f * (float)tid);
            
            float trapezoidQ = ((float)tid / (float)tidMax) * ((float)duration / 2.0f);
            float *data = inData + inT0Index;
            float *dy = inInverseSquaredDys + inT0Index;

            for (int i = 0; i < duration; i++) {
                float signal;
                
                if (i == 0 && tid != 0) {
                    signal = 1.0f - ((1.0f - TrapezoidDepth) / trapezoidQ);
                } else if (i < trapezoidQ) {
                    signal = 1.0f - ((1.0f - TrapezoidDepth) / trapezoidQ) * i;
                } else if (i >= trapezoidQ && i < duration - trapezoidQ) {
                    signal = TrapezoidDepth;
                } else {
                    signal = 1.0f - ((1.0f - TrapezoidDepth) / trapezoidQ) * (duration - i);
                }
                
                float diff = data[i] - signal;
                result[i] = diff * diff * dy[i];
            }
        }
    }

    /**
     * Optimized trapezoid fit generation
     */
    __global__ void generateTrapezoidFit(float *results, int bestFitTid, int duration, 
                                        int trapezoidFitSize, float TrapezoidDepth) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (tid < duration) {
            float trapezoidQ = ((float)bestFitTid / (float)trapezoidFitSize) * ((float)duration / 2.0f);
            float signal;

            if (tid == 0 && bestFitTid != 0) {
                signal = 1.0f - ((1.0f - TrapezoidDepth) / trapezoidQ);
            } else if (tid < trapezoidQ) {
                signal = 1.0f - ((1.0f - TrapezoidDepth) / trapezoidQ) * tid;
            } else if (tid >= trapezoidQ && tid < duration - trapezoidQ) {
                signal = TrapezoidDepth;
            } else {
                signal = 1.0f - ((1.0f - TrapezoidDepth) / trapezoidQ) * (duration - tid);
            }
            
            results[tid] = signal;
        }
    }

    /**
     * Optimized trapezoid SNR loss calculation
     */
    __global__ void trapezoidSNRloss(float *results, int resultSize, float *inData, 
                                     float *inInverseSquaredDys, int duration, float* trapezoidFit) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (tid < resultSize) {
            float *data = inData + tid;
            float *dy = inInverseSquaredDys + tid;
            
            float tempResidual = 0.0f;
            for (int i = 0; i < duration; i++) {
                float diff = data[i] - trapezoidFit[i];
                tempResidual += diff * diff * dy[i];
            }
            results[tid] = tempResidual;
        }
    }

    /**
     * Optimized NoSkip version for high precision calculations
     */
    __global__ void calcAllLowestResidualsGPUBNoSkip(float *out, int *resultArrayXAxisSize,
                                                     float *in_patched_datas, int *in_patched_datas_size, int *in_duration, int *in_duration_size,
                                                     float *in_signal, int *in_max_signal_x_size, float *in_inverse_squared_patched_dys,
                                                     float *in_overshoot, float *in_ootr, float *in_fullsum,
                                                     float *in_summed_edge_effect_correction, int *in_datapoints, float *cumsumGPU,
                                                     int *durationsMax, int *durationsMin, float *in_transit_depth_min) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    // point index
        int y = blockIdx.y * blockDim.y + threadIdx.y;      // duration index

        int durationIndex = y;
        int duration = in_duration[durationIndex];

        if (tid >= *resultArrayXAxisSize) {
            return;
        }

        int durationMax = durationsMax[y];
        int durationMin = durationsMin[y];
        float transit_depth_min = *in_transit_depth_min;
        int datapoints = *in_datapoints;

        float calc_mean = calcAverageFromCumsum(cumsumGPU, duration, in_patched_datas_size, tid, 0);
        float overshoot = in_overshoot[durationIndex];

        if (calc_mean > transit_depth_min) {
            float ootr = (tid == 0) ? 
                in_fullsum[durationIndex] :
                in_ootr[durationIndex * (*resultArrayXAxisSize) + tid - 1];

            float *data = in_patched_datas + tid;
            float *signal = in_signal + durationIndex * (*in_max_signal_x_size);
            float *inverse_squared_patched_dy_arr = in_inverse_squared_patched_dys;
            float summed_edge_effect_correction = in_summed_edge_effect_correction[0];
            float *dy = inverse_squared_patched_dy_arr + tid;
            
            float reverse_scale = calc_mean * overshoot * 2.0f;
            float intransit_residual = 0.0f;

            for (int i = 0; i < duration; i++) {
                float sigi = signal[i] * reverse_scale;
                float loss = data[i] - (1.0f - sigi);
                intransit_residual += loss * loss * dy[i];
            }
            
            float current_stat = intransit_residual + ootr - summed_edge_effect_correction;
            out[tid + durationIndex * (*resultArrayXAxisSize)] = current_stat;
        } else {
            out[tid + durationIndex * (*resultArrayXAxisSize)] = datapoints;
        }
    }

    /**
     * Compatible GPU version with iterator flag support
     */
    __global__ void calcAllLowestResidualsCompatibleGPU(float *out, int *in_mean_size, int *resultArrayXAxisSize,
                                                        float *in_patched_datas, int *in_patched_datas_size, int *in_duration, int *in_duration_size,
                                                        float *in_signal, int *in_max_signal_x_size, float *in_inverse_squared_patched_dys,
                                                        float *in_overshoot, float *in_ootr, float *in_fullsum,
                                                        float *in_summed_edge_effect_correction, int *in_datapoints, float *cumsumGPU,
                                                        int *durationsMax, int *durationsMin, float *in_transit_depth_min,
                                                        int *iter_flag_gpu, int *single_calc_periods_arr_gpu, int *period_size_gpu) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    // point index
        int y = blockIdx.y * blockDim.y + threadIdx.y;      // period index

        int y_input = y + (*iter_flag_gpu) * (*single_calc_periods_arr_gpu);
        float transit_depth_min = *in_transit_depth_min;
        int datapoints = *in_datapoints;
        
        if (y_input < (*period_size_gpu)) {
            int durationMax = durationsMax[y_input];
            int durationMin = durationsMin[y_input];

            for (int durationIndex = 0; durationIndex < *in_duration_size; durationIndex++) {
                int mean_size = in_mean_size[durationIndex];
                int duration = in_duration[durationIndex];

                int skipGap = (duration > 100) ? (duration / 100) : 1;
                
                if (duration >= durationMin && duration <= durationMax && tid % skipGap == 0) {
                    float calc_mean = calcAverageFromCumsum(cumsumGPU, duration, in_patched_datas_size, tid, y);
                    float overshoot = in_overshoot[durationIndex];
                    
                    if (tid < *resultArrayXAxisSize) {
                        out[tid + durationIndex * (*resultArrayXAxisSize) + y * (*resultArrayXAxisSize) * (*in_duration_size)] = datapoints;
                    }
                    
                    if (tid < mean_size && calc_mean > transit_depth_min) {
                        float ootr = (tid == 0) ? 
                            in_fullsum[y * (*in_duration_size) + durationIndex] :
                            in_ootr[durationIndex * (*resultArrayXAxisSize) + y * (*resultArrayXAxisSize) * (*in_duration_size) + tid - 1];

                        float *data = in_patched_datas + y_input * (*in_patched_datas_size) + tid;
                        float *signal = in_signal + durationIndex * (*in_max_signal_x_size);
                        float *inverse_squared_patched_dy_arr = in_inverse_squared_patched_dys + y_input * (*in_patched_datas_size);
                        float summed_edge_effect_correction = in_summed_edge_effect_correction[y_input];
                        float *dy = inverse_squared_patched_dy_arr + tid;
                        
                        float reverse_scale = calc_mean * overshoot * 2.0f;
                        float intransit_residual = 0.0f;
                        
                        for (int i = 0; i < duration; i++) {
                            float sigi = signal[i] * reverse_scale;
                            float loss = data[i] - (1.0f - sigi);
                            intransit_residual += loss * loss * dy[i];
                        }

                        float current_stat = intransit_residual + ootr - summed_edge_effect_correction;
                        out[tid + durationIndex * (*resultArrayXAxisSize) + y * (*resultArrayXAxisSize) * (*in_duration_size)] = current_stat;
                    }
                } else {
                    if (tid < *resultArrayXAxisSize) {
                        out[tid + durationIndex * (*resultArrayXAxisSize) + y * (*resultArrayXAxisSize) * (*in_duration_size)] = FLOAT_INFINITY;
                    }
                }
            }
        }
    }

    /**
     * Optimized trapezoid fit with atomic operations
     */
    __global__ void trapezoidFitAtom(float *results, float *inData, float *inInverseSquaredDys,
                                    int duration, int inT0Index, float *transitMean, int trapezoidFitSize) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x; // point index in trapezoid
        int y = blockIdx.y * blockDim.y + threadIdx.y;   // trapezoid type index

        if (y < trapezoidFitSize && tid < duration) {
            float *result = results + y * duration + tid;
            float TrapezoidDepth = ((float)trapezoidFitSize * (*transitMean) - 0.5f * (float)y) / 
                                  ((float)trapezoidFitSize - 0.5f * (float)y);
            
            float trapezoidQ = ((float)y / (float)trapezoidFitSize) * ((float)duration / 2.0f);
            float *data = inData + inT0Index + tid;
            float *dy = inInverseSquaredDys + inT0Index + tid;

            float signal;
            if (tid == 0 && y != 0) {
                signal = 1.0f - ((1.0f - TrapezoidDepth) / trapezoidQ);
            } else if (tid < trapezoidQ) {
                signal = 1.0f - ((1.0f - TrapezoidDepth) / trapezoidQ) * tid;
            } else if (tid >= trapezoidQ && tid < duration - trapezoidQ) {
                signal = TrapezoidDepth;
            } else {
                signal = 1.0f - ((1.0f - TrapezoidDepth) / trapezoidQ) * (duration - tid);
            }
            
            float diff = (*data) - signal;
            *result = diff * diff;
        }
    }

    /**
     * Optimized trapezoid SNR loss with atomic operations
     */
    __global__ void trapezoidSNRlossAtom(float *results, int resultSize, float *inData, 
                                        float *inInverseSquaredDys, int duration, float* trapezoidFit) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x; // point in duration
        int y = blockIdx.y * blockDim.y + threadIdx.y;   // first point index

        if (y < resultSize && tid < duration) {
            float *data = inData + y + tid;
            float *dy = inInverseSquaredDys + y + tid;
            float *result = results + y * duration + tid;

            float diff = (*data) - trapezoidFit[tid];
            *result = diff * diff * (*dy);
        }
    }


    /**
     * Optimized NoSkipTemp version - high precision calculations without skipping
     */
    __global__ void calcAllLowestResidualsGPUBNoSkipTemp(float *out, int *resultArrayXAxisSize,
                                                         float *in_patched_datas, int *in_patched_datas_size, int *in_duration, int *in_duration_size,
                                                         float *in_signal, int *in_max_signal_x_size, float *in_inverse_squared_patched_dys,
                                                         float *in_overshoot, float *in_ootr, float *in_fullsum,
                                                         float *in_summed_edge_effect_correction, int *in_datapoints, float *cumsumGPU,
                                                         float *in_transit_depth_min) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;    // point index
        int y = blockIdx.y * blockDim.y + threadIdx.y;      // duration index
        int z = blockIdx.z * blockDim.z + threadIdx.z;      // period index

        int durationIndex = y;
        int periodIndex = z;
        int duration = in_duration[durationIndex];

        if (tid >= *resultArrayXAxisSize) {
            return;
        }

        float transit_depth_min = *in_transit_depth_min;
        int datapoints = *in_datapoints;

        float calc_mean = calcAverageFromCumsum(cumsumGPU, duration, in_patched_datas_size, tid, periodIndex);
        float overshoot = in_overshoot[durationIndex];

        if (calc_mean > transit_depth_min) {
            // Optimized OOTR calculation
            float ootr = (tid == 0) ? 
                in_fullsum[periodIndex * (*in_duration_size) + durationIndex] :
                in_ootr[durationIndex * (*resultArrayXAxisSize) + periodIndex * (*resultArrayXAxisSize) * (*in_duration_size) + tid - 1];

            // Pre-calculate pointers for better memory access
            float *data = in_patched_datas + periodIndex * (*in_patched_datas_size) + tid;
            float *signal = in_signal + durationIndex * (*in_max_signal_x_size);
            float *inverse_squared_patched_dy_arr = in_inverse_squared_patched_dys + periodIndex * (*in_patched_datas_size);
            float summed_edge_effect_correction = in_summed_edge_effect_correction[periodIndex];
            float *dy = inverse_squared_patched_dy_arr + tid;

            // Optimized scale calculation with proper float type
            float reverse_scale = calc_mean * overshoot * 2.0f;  // 2.0f for SIGNAL_DEPTH = 0.5

            float intransit_residual = 0.0f;

            // Optimized inner loop with reduced variable declarations
            for (int i = 0; i < duration; i++) {
                float sigi = signal[i] * reverse_scale;
                float loss = data[i] - (1.0f - sigi);
                intransit_residual += loss * loss * dy[i];
            }

            float current_stat = intransit_residual + ootr - summed_edge_effect_correction;
            out[tid + durationIndex * (*resultArrayXAxisSize) + periodIndex * (*resultArrayXAxisSize) * (*in_duration_size)] = current_stat;
        } else {
            out[tid + durationIndex * (*resultArrayXAxisSize) + periodIndex * (*resultArrayXAxisSize) * (*in_duration_size)] = datapoints;
        }
    }

}
"""
    return GPUCode
