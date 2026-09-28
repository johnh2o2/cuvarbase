/*
 * Fused observation-rank TLS search, preserving public GTLS search math.
 *
 * Adapted from GTLS src/gputls/GPUFun.py:getGPUCode(), specifically
 * calcAverageFromCumsum, calcAllFullSum_v2, calculate_final_ootr_v3,
 * calcAllLowestResidualsGPUB_SignalTiled_v2, and edge-effect correction.
 * Source snapshot: benchmarks/results/tls_profile_2026-09-08/sources/gtls-head.tar
 * SHA256 of the returned CUDA string:
 * 25570532816bd94b390c10cd6a1b0477de7873c52715191de7c422c5b8d3eb9c
 *
 * MIT License
 * Copyright (c) 2018 Michael Hippke 2023 Quanquan Hu
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * This implementation retains the supplied GTLS templates, observation-rank
 * windows, unweighted window-mean depth, epoch skip schedule, chi2 sentinel,
 * and float32 residual accumulation. It does not replace them with an
 * analytic-depth or physical-phase objective. In particular, template rows
 * must contain 1 minus the original zero-padded flux cache; a padded entry
 * therefore has deficit 1, not 0, exactly as in the pinned host code.
 *
 * Required inputs are the SAME patched sorted arrays, prefix sums and width
 * union as the reference. The host controls width coverage and sort/prefix
 * semantics. Native GTLS unions width masks over its memory-dependent period
 * chunk, so changing chunks can change coverage unless the host fixes it.
 *
 * All observations remain in global memory. Only block reductions use shared
 * memory. Work tiles contain only the native evaluated start positions; the
 * first omitted position is inserted as a sentinel candidate when needed.
 * Fusing out-of-transit residuals and reducing each tile removes the native
 * [period, duration, observation] residual and OOTR tensors.
 * The full refinement stage instead retains its native OOTR scan order in a
 * bounded selected-period tensor, while still reducing residuals in tiles.
 */

#ifndef TLS_REFERENCE_BLOCK_SIZE
#define TLS_REFERENCE_BLOCK_SIZE 256
#endif

typedef unsigned long long tls_ref_key;

__device__ __forceinline__ tls_ref_key tls_ref_empty_key() {
    return ~((tls_ref_key)0);
}

/* Match first-index argmin, including NaN propagation. Finite cleaned inputs
 * should not produce NaN; retaining its ordering also makes debug comparisons
 * explicit rather than silently replacing a malformed score by a finite one. */
__device__ __forceinline__ bool tls_ref_better(
    float candidate, tls_ref_key candidate_key,
    float incumbent, tls_ref_key incumbent_key)
{
    const bool candidate_nan = isnan(candidate);
    const bool incumbent_nan = isnan(incumbent);
    if (candidate_nan != incumbent_nan) return candidate_nan;
    if (candidate_nan) return candidate_key < incumbent_key;
    return candidate < incumbent ||
        (candidate == incumbent && candidate_key < incumbent_key);
}

__device__ __forceinline__ float tls_ref_mean_depth(
    const float* flux_prefix, int width, int start)
{
    if (start == 0) {
        return 1.0f - flux_prefix[width - 1] / width;
    } else {
        const float end_val = flux_prefix[start + width - 1];
        const float start_val = flux_prefix[start - 1];
        return 1.0f - (end_val - start_val) / width;
    }
}

/* Arithmetic order follows the three reference kernels. Explicit rounded
 * additions/subtractions retain the original float32 intermediate writes
 * even though those intermediates now stay in registers. */
__device__ __forceinline__ float tls_ref_ootr(
    const float* error_prefix, int stride, int width, int start)
{
    const float window_prefix = error_prefix[width - 1];
    const float fullsum = __fsub_rn(error_prefix[stride - 1], window_prefix);
    if (start == 0) return fullsum;
    const int p = start - 1;
    const float p_e_p = error_prefix[p];
    const float p_e_p_plus_window =
        p + width < stride ? error_prefix[p + width] : 0.0f;
    const float cumsum_weight = __fsub_rn(
        p_e_p, __fsub_rn(p_e_p_plus_window, window_prefix));
    return __fadd_rn(fullsum, cumsum_weight);
}

__device__ __forceinline__ float tls_ref_window(
    const float* data, const float* invvar,
    const float* flux_prefix, const float* error_prefix,
    const float* signal, float overshoot, float edge_correction,
    int ndata, int stride, int width, int start, int skip_factor,
    float transit_depth_min, float* fitted_depth)
{
    const int skip = width > skip_factor ? width / skip_factor : 1;
    const float calc_mean = tls_ref_mean_depth(flux_prefix, width, start);
    float current_stat = (float)ndata;
    *fitted_depth = 0.0f;
    if (calc_mean > transit_depth_min && start % skip == 0) {
        const float ootr = tls_ref_ootr(error_prefix, stride, width, start);
        const float reverse_scale = calc_mean * overshoot * 2.0f;
        float intransit_residual = 0.0f;
        for (int i = 0; i < width; i++) {
            const float sigi = signal[i] * reverse_scale;
            const float loss = data[start + i] - (1.0f - sigi);
            intransit_residual += loss * loss * invvar[start + i];
        }
        const int skip_search_point = 1;
        const float actual_loss_fraction = (float)width /
            (((width - 1) / skip_search_point) + 1);
        current_stat = intransit_residual * actual_loss_fraction + ootr
            - edge_correction;
        *fitted_depth = calc_mean * overshoot;
    }
    return current_stat;
}

/* Search one packed tile of one cached width for each period row.
 *
 * All matrix inputs are row-major [nrows, stride], where
 * stride = ndata + largest cached width, rounded as in native GTLS.
 * Widths and templates have already been selected by the host's width union.
 * template_deficits is [nwidths, template_stride].
 *
 * Host tiles, for each width d:
 *   skip = max(width[d] // skip_factor, 1)
 *   evaluated_count = ceil(ndata / skip)
 *   for first in range(0, evaluated_count, TLS_REFERENCE_BLOCK_SIZE):
 *       tile_duration.append(d); tile_first_trial.append(first)
 *
 * Grid=(ntiles,nrows,1), block=(TLS_REFERENCE_BLOCK_SIZE,1,1).
 * Partial arrays are [nrows,ntiles]. No phase or ndata accuracy cap.
 */
extern "C" __global__ void tls_reference_search(
    const float* __restrict__ patched_flux,
    const float* __restrict__ inverse_variance,
    const float* __restrict__ flux_prefix,
    const float* __restrict__ error_prefix,
    const float* __restrict__ edge_correction,
    const int* __restrict__ widths,
    const float* __restrict__ template_deficits,
    const float* __restrict__ overshoot,
    const int* __restrict__ tile_duration,
    const int* __restrict__ tile_first_trial,
    int nrows, int ndata, int stride, int nwidths, int template_stride,
    int ntiles, int skip_factor, float transit_depth_min,
    float* __restrict__ partial_chi2,
    tls_ref_key* __restrict__ partial_key,
    float* __restrict__ partial_depth)
{
    const int tile = blockIdx.x;
    const int row = blockIdx.y;
    if (tile >= ntiles || row >= nrows) return;
    const int d = tile_duration[tile];
    const int width = widths[d];
    const int skip = width > skip_factor ? width / skip_factor : 1;
    const long long trial = (long long)tile_first_trial[tile] + threadIdx.x;
    const long long start_long = trial * skip;
    const long long row_offset = (long long)row * stride;
    float value = __int_as_float(0x7f800000);
    tls_ref_key key = tls_ref_empty_key();
    float depth = 0.0f;
    if (start_long < ndata) {
        const int start = (int)start_long;
        key = (tls_ref_key)d * ndata + start;
        value = tls_ref_window(
            patched_flux + row_offset, inverse_variance + row_offset,
            flux_prefix + row_offset, error_prefix + row_offset,
            template_deficits + (long long)d * template_stride,
            overshoot[d], edge_correction[row], ndata, stride, width,
            start, skip_factor, transit_depth_min, &depth);
    }
    /* All skipped starts have the same residual ndata. Retain their first
     * logical index, rather than clamping all results to ndata: if no starts
     * are skipped and every fitted residual exceeds ndata, native argmin
     * must still return that larger residual. */
    if (threadIdx.x == 0 && tile_first_trial[tile] == 0 && skip > 1 && ndata > 1) {
        const tls_ref_key skipped_key = (tls_ref_key)d * ndata + 1;
        if (tls_ref_better((float)ndata, skipped_key, value, key)) {
            value = (float)ndata;
            key = skipped_key;
            depth = 0.0f;
        }
    }
    __shared__ float values[TLS_REFERENCE_BLOCK_SIZE];
    __shared__ tls_ref_key keys[TLS_REFERENCE_BLOCK_SIZE];
    __shared__ float depths[TLS_REFERENCE_BLOCK_SIZE];
    values[threadIdx.x] = value;
    keys[threadIdx.x] = key;
    depths[threadIdx.x] = depth;
    __syncthreads();
    for (int step = TLS_REFERENCE_BLOCK_SIZE / 2; step > 0; step /= 2) {
        if (threadIdx.x < step && tls_ref_better(
                values[threadIdx.x + step], keys[threadIdx.x + step],
                values[threadIdx.x], keys[threadIdx.x])) {
            values[threadIdx.x] = values[threadIdx.x + step];
            keys[threadIdx.x] = keys[threadIdx.x + step];
            depths[threadIdx.x] = depths[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const long long output = (long long)row * ntiles + tile;
        partial_chi2[output] = values[0];
        partial_key[output] = keys[0];
        partial_depth[output] = depths[0];
    }
}

/* Grid=(nrows,1,1), block=(TLS_REFERENCE_BLOCK_SIZE,1,1). */
extern "C" __global__ void tls_reference_reduce(
    const float* __restrict__ partial_chi2,
    const tls_ref_key* __restrict__ partial_key,
    const float* __restrict__ partial_depth,
    const int* __restrict__ widths,
    int nrows, int ndata, int ntiles,
    float* __restrict__ minimum_chi2,
    int* __restrict__ best_start,
    int* __restrict__ best_width_index,
    int* __restrict__ best_width,
    float* __restrict__ best_depth)
{
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const long long row_offset = (long long)row * ntiles;
    float value = __int_as_float(0x7f800000);
    tls_ref_key key = tls_ref_empty_key();
    float depth = 0.0f;
    for (int tile = threadIdx.x; tile < ntiles; tile += blockDim.x) {
        const long long k = row_offset + tile;
        if (tls_ref_better(partial_chi2[k], partial_key[k], value, key)) {
            value = partial_chi2[k];
            key = partial_key[k];
            depth = partial_depth[k];
        }
    }
    __shared__ float values[TLS_REFERENCE_BLOCK_SIZE];
    __shared__ tls_ref_key keys[TLS_REFERENCE_BLOCK_SIZE];
    __shared__ float depths[TLS_REFERENCE_BLOCK_SIZE];
    values[threadIdx.x] = value;
    keys[threadIdx.x] = key;
    depths[threadIdx.x] = depth;
    __syncthreads();
    for (int step = TLS_REFERENCE_BLOCK_SIZE / 2; step > 0; step /= 2) {
        if (threadIdx.x < step && tls_ref_better(
                values[threadIdx.x + step], keys[threadIdx.x + step],
                values[threadIdx.x], keys[threadIdx.x])) {
            values[threadIdx.x] = values[threadIdx.x + step];
            keys[threadIdx.x] = keys[threadIdx.x + step];
            depths[threadIdx.x] = depths[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        minimum_chi2[row] = values[0];
        const bool has_key = keys[0] != tls_ref_empty_key();
        const int d = has_key ? (int)(keys[0] / ndata) : -1;
        best_start[row] = has_key ? (int)(keys[0] % ndata) : -1;
        best_width_index[row] = d;
        best_width[row] = has_key ? widths[d] : 0;
        best_depth[row] = depths[0];
    }
}

/* Diagnostic only: materialize the native logical tensor on small fixtures.
 * Grid=(ceil(ndata/blocksize),nwidths,nrows). The production search uses the
 * packed tiles above and never allocates this tensor. */
extern "C" __global__ void tls_reference_window_values(
    const float* __restrict__ patched_flux,
    const float* __restrict__ inverse_variance,
    const float* __restrict__ flux_prefix,
    const float* __restrict__ error_prefix,
    const float* __restrict__ edge_correction,
    const int* __restrict__ widths,
    const float* __restrict__ template_deficits,
    const float* __restrict__ overshoot,
    int nrows, int ndata, int stride, int nwidths, int template_stride,
    int skip_factor, float transit_depth_min,
    float* __restrict__ values)
{
    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    const int d = blockIdx.y;
    const int row = blockIdx.z;
    if (start >= ndata || d >= nwidths || row >= nrows) return;
    const long long offset = (long long)row * stride;
    float depth;
    values[((long long)row * nwidths + d) * ndata + start] = tls_ref_window(
        patched_flux + offset, inverse_variance + offset,
        flux_prefix + offset, error_prefix + offset,
        template_deficits + (long long)d * template_stride, overshoot[d],
        edge_correction[row], ndata, stride, widths[d], start, skip_factor,
        transit_depth_min, &depth);
}

/* Legacy full-mode sum. Native calcAllFullSum repeats the same sequential
 * full-data sum for every width; reuse that sum and its sequential window
 * prefixes. Widths must be positive and strictly increasing. Both sequences
 * retain the native float32 multiply/add order, with no parallel reduction. */
extern "C" __global__ void tls_reference_fullsum_legacy(
    const float* __restrict__ patched_flux,
    const float* __restrict__ inverse_variance,
    const int* __restrict__ widths,
    int nrows, int stride, int nwidths,
    float* __restrict__ fullsum_out)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    const long long offset = (long long)row * stride;
    float fullsum = 0.0f;
    for (int i = 0; i < stride; i++) {
        const float diff = 1.0f - patched_flux[offset + i];
        fullsum += diff * diff * inverse_variance[offset + i];
    }
    float window_sum = 0.0f;
    int i = 0;
    for (int d = 0; d < nwidths; d++) {
        const int width = widths[d];
        while (i < width) {
            const float diff = 1.0f - patched_flux[offset + i];
            window_sum += diff * diff * inverse_variance[offset + i];
            i++;
        }
        fullsum_out[(long long)row * nwidths + d] = fullsum - window_sum;
    }
}

/* Native full-stage OOTR preparation, before the host applies
 * cp.cumsum(delta, axis=-1) with the same shape as the reference.
 * Grid=(ceil(ndata/blocksize),nwidths,nrows). */
extern "C" __global__ void tls_reference_ootr_delta(
    const float* __restrict__ patched_flux,
    const float* __restrict__ inverse_variance,
    const int* __restrict__ widths,
    int nrows, int ndata, int stride, int nwidths,
    float* __restrict__ delta)
{
    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    const int d = blockIdx.y;
    const int row = blockIdx.z;
    if (start >= ndata || d >= nwidths || row >= nrows) return;
    const long long offset = (long long)row * stride;
    const int width = widths[d];
    const float visible = 1.0f - patched_flux[offset + start];
    const float invisible = 1.0f - patched_flux[offset + start + width];
    const float add_visible = visible * visible * inverse_variance[offset + start];
    const float remove_invisible = invisible * invisible *
        inverse_variance[offset + start + width];
    delta[((long long)row * nwidths + d) * ndata + start] =
        add_visible - remove_invisible;
}

/* Complete native full-stage OOTR after the host's prefix scan. */
extern "C" __global__ void tls_reference_ootr_add(
    float* __restrict__ ootr,
    const float* __restrict__ fullsum,
    int nrows, int ndata, int nwidths)
{
    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    const int d = blockIdx.y;
    const int row = blockIdx.z;
    if (start >= ndata || d >= nwidths || row >= nrows) return;
    const long long k = ((long long)row * nwidths + d) * ndata + start;
    ootr[k] = fullsum[(long long)row * nwidths + d] + ootr[k];
}

/* Full refinement, corresponding to both native NoSkipTemp (many rows) and
 * NoSkip (the final single row). Unlike the fast stage, fullsum and OOTR
 * arrive from the native sequential/full-difference-scan preparation above.
 * The host supplies every start position: tile_first_trial=0,256,512,... for
 * each width. Search and reduction outputs share the fast-stage layout.
 * Grid=(ntiles,nrows), block=(TLS_REFERENCE_BLOCK_SIZE,1,1).
 */
__device__ __forceinline__ float tls_ref_full_window(
    const float* data, const float* invvar, const float* flux_prefix,
    const float* signal, float fullsum, const float* ootr,
    float overshoot, float edge_correction, int ndata, int width, int start,
    float transit_depth_min, float* fitted_depth)
{
    const float mean = tls_ref_mean_depth(flux_prefix, width, start);
    *fitted_depth = 0.0f;
    if (!(mean > transit_depth_min)) return (float)ndata;
    const float outside = start == 0 ? fullsum : ootr[start - 1];
    const float reverse_scale = mean * overshoot * 2.0f;
    float residual = 0.0f;
    for (int i = 0; i < width; i++) {
        const float sigi = signal[i] * reverse_scale;
        const float loss = data[start + i] - (1.0f - sigi);
        residual += loss * loss * invvar[start + i];
    }
    *fitted_depth = mean * overshoot;
    return residual + outside - edge_correction;
}

extern "C" __global__ void tls_reference_full_search(
    const float* __restrict__ patched_flux,
    const float* __restrict__ inverse_variance,
    const float* __restrict__ flux_prefix,
    const float* __restrict__ fullsum,
    const float* __restrict__ ootr,
    const float* __restrict__ edge_correction,
    const int* __restrict__ widths,
    const float* __restrict__ template_deficits,
    const float* __restrict__ overshoot,
    const int* __restrict__ tile_duration,
    const int* __restrict__ tile_first_trial,
    int nrows, int ndata, int stride, int nwidths, int template_stride,
    int ntiles, float transit_depth_min,
    float* __restrict__ partial_chi2,
    tls_ref_key* __restrict__ partial_key,
    float* __restrict__ partial_depth)
{
    const int tile = blockIdx.x;
    const int row = blockIdx.y;
    if (tile >= ntiles || row >= nrows) return;
    const int d = tile_duration[tile];
    const int width = widths[d];
    const long long start_long = (long long)tile_first_trial[tile] + threadIdx.x;
    const long long offset = (long long)row * stride;
    float value = __int_as_float(0x7f800000);
    tls_ref_key key = tls_ref_empty_key();
    float depth = 0.0f;
    if (start_long < ndata) {
        const int start = (int)start_long;
        key = (tls_ref_key)d * ndata + start;
        value = tls_ref_full_window(patched_flux + offset,
            inverse_variance + offset, flux_prefix + offset,
            template_deficits + (long long)d * template_stride,
            fullsum[(long long)row * nwidths + d],
            ootr + ((long long)row * nwidths + d) * ndata,
            overshoot[d], edge_correction[row], ndata, width, start,
            transit_depth_min, &depth);
    }
    __shared__ float values[TLS_REFERENCE_BLOCK_SIZE];
    __shared__ tls_ref_key keys[TLS_REFERENCE_BLOCK_SIZE];
    __shared__ float depths[TLS_REFERENCE_BLOCK_SIZE];
    values[threadIdx.x] = value;
    keys[threadIdx.x] = key;
    depths[threadIdx.x] = depth;
    __syncthreads();
    for (int step = TLS_REFERENCE_BLOCK_SIZE / 2; step > 0; step /= 2) {
        if (threadIdx.x < step && tls_ref_better(
                values[threadIdx.x + step], keys[threadIdx.x + step],
                values[threadIdx.x], keys[threadIdx.x])) {
            values[threadIdx.x] = values[threadIdx.x + step];
            keys[threadIdx.x] = keys[threadIdx.x + step];
            depths[threadIdx.x] = depths[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const long long output = (long long)row * ntiles + tile;
        partial_chi2[output] = values[0];
        partial_key[output] = keys[0];
        partial_depth[output] = depths[0];
    }
}

/* Diagnostic only, full-stage logical window residuals. */
extern "C" __global__ void tls_reference_full_window_values(
    const float* __restrict__ patched_flux,
    const float* __restrict__ inverse_variance,
    const float* __restrict__ flux_prefix,
    const float* __restrict__ fullsum,
    const float* __restrict__ ootr,
    const float* __restrict__ edge_correction,
    const int* __restrict__ widths,
    const float* __restrict__ template_deficits,
    const float* __restrict__ overshoot,
    int nrows, int ndata, int stride, int nwidths, int template_stride,
    float transit_depth_min, float* __restrict__ values)
{
    const int start = blockIdx.x * blockDim.x + threadIdx.x;
    const int d = blockIdx.y;
    const int row = blockIdx.z;
    if (start >= ndata || d >= nwidths || row >= nrows) return;
    const long long offset = (long long)row * stride;
    float depth;
    values[((long long)row * nwidths + d) * ndata + start] = tls_ref_full_window(
        patched_flux + offset, inverse_variance + offset, flux_prefix + offset,
        template_deficits + (long long)d * template_stride,
        fullsum[(long long)row * nwidths + d],
        ootr + ((long long)row * nwidths + d) * ndata,
        overshoot[d], edge_correction[row], ndata, widths[d], start,
        transit_depth_min, &depth);
}
