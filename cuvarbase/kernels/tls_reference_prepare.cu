/*
Adapted from GTLS 74e449c325792a763dde4fbffab98039c5e8c111 GPUFun.cu.
MIT License

Copyright (c) 2018 Michael Hippke 2023 Quanquan Hu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
extern "C" {
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

__global__ void calcInverseSquaredPatchedDy(float *out, float *patched_dys, int *patched_data_size) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (tid < *patched_data_size) {
            float dy_val = patched_dys[tid + y * (*patched_data_size)];
            out[tid + y * (*patched_data_size)] = 1.0f / (dy_val * dy_val);
        }
    }

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
}
