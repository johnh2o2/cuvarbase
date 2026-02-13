/*
 * FFA-BLS (Fast Folding Algorithm for Box-Least Squares)
 *
 * Implements the Fast Folding Algorithm (Staelin 1969) for BLS transit search
 * as described in Shahaf et al. (2022, MNRAS 513, 2732).
 *
 * Kernels:
 *   ffa_bin_sections    - Bin observations into per-section phase profiles
 *   ffa_combine_pairs   - Combine adjacent section pairs with 0/1 shift (Level 0)
 *   ffa_butterfly       - One butterfly level (combine groups with shift vector)
 *   ffa_score           - Box scan on folded profiles -> max SR per period
 */

#include <stdio.h>
//{CPP_DEFS}

__device__ int ffa_mod(int a, int b){
    int r = a % b;
    return (r < 0) ? r + b : r;
}

__device__ float ffa_mod1(float a){
    return a - floorf(a);
}

__device__ float ffa_bls_value(float yw_sum, float w_sum,
                                unsigned int ignore_negative_delta_sols){
    float bls = (w_sum > 1e-10f && w_sum < 1.f - 1e-10f)
                ? yw_sum * yw_sum / (w_sum * (1.f - w_sum))
                : 0.f;
    return ((ignore_negative_delta_sols == 1) & (yw_sum > 0.f)) ? 0.f : bls;
}


/*
 * ffa_bin_sections: Bin observations into per-section phase profiles.
 *
 * Grid:  N_p blocks (one per section)
 * Block: BLOCK_SIZE threads
 *
 * Each block processes observations in section[blockIdx.x].
 * Uses shared memory atomics for the m-bin histogram, then writes to global.
 *
 * Parameters:
 *   t, yw, w         - observation times, weighted residuals, weights (sorted by time)
 *   section_starts   - start index in t[] for each section
 *   section_ends     - end index in t[] for each section
 *   section_yw       - output: [N_p, m] per-section binned yw
 *   section_w        - output: [N_p, m] per-section binned w
 *   P0               - base period for this octave (= m * dt)
 *   m                - number of phase bins
 *   N_p              - number of sections
 */
__global__ void ffa_bin_sections(
        const float * __restrict__ t,
        const float * __restrict__ yw,
        const float * __restrict__ w,
        const unsigned int * __restrict__ section_starts,
        const unsigned int * __restrict__ section_ends,
        float * __restrict__ section_yw,
        float * __restrict__ section_w,
        float P0,
        unsigned int m,
        unsigned int N_p)
{
    unsigned int sec = blockIdx.x;
    if (sec >= N_p) return;

    extern __shared__ float sh[];
    float *sh_yw = sh;
    float *sh_w  = &sh[m];

    // Initialize shared memory bins
    for (unsigned int i = threadIdx.x; i < m; i += blockDim.x){
        sh_yw[i] = 0.f;
        sh_w[i]  = 0.f;
    }
    __syncthreads();

    // Bin observations for this section
    unsigned int start = section_starts[sec];
    unsigned int end   = section_ends[sec];

    for (unsigned int k = start + threadIdx.x; k < end; k += blockDim.x){
        float phase = ffa_mod1(t[k] / P0);
        int bin = (int) floorf(m * phase);
        if (bin >= (int)m) bin = m - 1;  // safety clamp

        atomicAdd(&sh_yw[bin], yw[k]);
        atomicAdd(&sh_w[bin],  w[k]);
    }
    __syncthreads();

    // Write shared memory to global memory
    unsigned int offset = sec * m;
    for (unsigned int i = threadIdx.x; i < m; i += blockDim.x){
        section_yw[offset + i] = sh_yw[i];
        section_w[offset + i]  = sh_w[i];
    }
}


/*
 * ffa_combine_pairs: Combine adjacent section pairs with 0 and 1 bin shift.
 *
 * This is Level 0 of the FFA butterfly.
 * For each pair p = (section[2p], section[2p+1]):
 *   fold[p][drift=0][b] = section[2p][b] + section[2p+1][b]
 *   fold[p][drift=1][b] = section[2p][b] + section[2p+1][(b-1) % m]
 *
 * Grid:  N_p/2 blocks (one per pair)
 * Block: BLOCK_SIZE threads
 *
 * Output layout: folds[fold_index * m + bin]
 * where fold_index = pair * 2 + drift, total N_p folds.
 *
 * Parameters:
 *   section_yw, section_w  - [N_p, m] per-section binned profiles
 *   folds_yw, folds_w      - output [N_p, m] pair-folds (2 per pair)
 *   m                      - number of phase bins
 *   N_p                    - number of sections (must be even)
 */
__global__ void ffa_combine_pairs(
        const float * __restrict__ section_yw,
        const float * __restrict__ section_w,
        float * __restrict__ folds_yw,
        float * __restrict__ folds_w,
        unsigned int m,
        unsigned int N_p)
{
    unsigned int pair = blockIdx.x;
    unsigned int n_pairs = N_p / 2;
    if (pair >= n_pairs) return;

    unsigned int left  = 2 * pair;
    unsigned int right = 2 * pair + 1;

    unsigned int left_off  = left  * m;
    unsigned int right_off = right * m;

    // Output: drift=0 is fold index (pair*2), drift=1 is fold index (pair*2+1)
    unsigned int out_d0 = (pair * 2)     * m;
    unsigned int out_d1 = (pair * 2 + 1) * m;

    for (unsigned int b = threadIdx.x; b < m; b += blockDim.x){
        float l_yw = section_yw[left_off + b];
        float l_w  = section_w[left_off + b];

        // drift=0: right section unshifted
        float r_yw_d0 = section_yw[right_off + b];
        float r_w_d0  = section_w[right_off + b];

        folds_yw[out_d0 + b] = l_yw + r_yw_d0;
        folds_w[out_d0 + b]  = l_w  + r_w_d0;

        // drift=1: right section shifted by 1 bin
        // We want right[(b-1) % m], which means the right section's
        // bin (b-1) maps to output bin b.
        unsigned int b_shifted = (b == 0) ? m - 1 : b - 1;
        float r_yw_d1 = section_yw[right_off + b_shifted];
        float r_w_d1  = section_w[right_off + b_shifted];

        folds_yw[out_d1 + b] = l_yw + r_yw_d1;
        folds_w[out_d1 + b]  = l_w  + r_w_d1;
    }
}


/*
 * ffa_butterfly: One level of the FFA butterfly (Staelin 1969).
 *
 * Classic FFA binary decomposition: at level l, shift = -(2^l)
 * if bit l of d is set, else 0. Both left and right sub-groups
 * use the same sub-fold index d_sub = d_local % 2^l.
 *
 * At level l, we have N_groups = N_p / 2^(l+1) groups.
 * Each group produces 2^(l+1) output folds from two half-groups
 * of 2^l folds each.
 *
 * Grid:  Total output folds = N_p (one block per output fold)
 * Block: BLOCK_SIZE threads (process bins in parallel)
 *
 * Parameters:
 *   in_yw, in_w        - input folds [N_p, m]
 *   out_yw, out_w      - output folds [N_p, m]
 *   shift_array        - [N_p] per-fold classic FFA shifts
 *   m                  - number of phase bins
 *   N_p                - total number of folds
 *   folds_per_group    - 2^(l+1) = number of folds per output group
 *   half_folds         - 2^l = number of folds per half-group
 */
__global__ void ffa_butterfly(
        const float * __restrict__ in_yw,
        const float * __restrict__ in_w,
        float * __restrict__ out_yw,
        float * __restrict__ out_w,
        const int * __restrict__ shift_array,
        unsigned int m,
        unsigned int N_p,
        unsigned int folds_per_group,
        unsigned int half_folds)
{
    unsigned int fold_idx = blockIdx.x;
    if (fold_idx >= N_p) return;

    // Which group and which fold within the group
    unsigned int group = fold_idx / folds_per_group;
    unsigned int s     = fold_idx % folds_per_group;

    // Same sub-fold index for both left and right (Shahaf 2022)
    unsigned int d_sub = s % half_folds;

    // Input fold indices (in the flat [N_p, m] array)
    unsigned int left_group  = 2 * group;
    unsigned int right_group = 2 * group + 1;

    unsigned int left_fold_idx  = left_group  * half_folds + d_sub;
    unsigned int right_fold_idx = right_group * half_folds + d_sub;

    // Per-fold analytical shift
    int shift = shift_array[fold_idx];

    unsigned int left_off  = left_fold_idx  * m;
    unsigned int right_off = right_fold_idx * m;
    unsigned int out_off   = fold_idx * m;

    for (unsigned int b = threadIdx.x; b < m; b += blockDim.x){
        float l_yw = in_yw[left_off + b];
        float l_w  = in_w[left_off + b];

        // Apply circular shift to right fold
        int b_shifted = ffa_mod((int)b - shift, (int)m);
        float r_yw = in_yw[right_off + b_shifted];
        float r_w  = in_w[right_off + b_shifted];

        out_yw[out_off + b] = l_yw + r_yw;
        out_w[out_off + b]  = l_w  + r_w;
    }
}


/*
 * ffa_score: Box scan on folded profiles to find max BLS SR per period.
 *
 * Each block processes one folded profile (= one trial period).
 * Scans all (bin_start, bin_width) combinations as in standard BLS.
 *
 * Grid:  N_p blocks (one per trial period)
 * Block: BLOCK_SIZE threads
 *
 * Shared memory: 2*m floats (yw_bins, w_bins) + BLOCK_SIZE floats (reduction)
 *
 * Parameters:
 *   folds_yw, folds_w  - [N_p, m] folded profiles
 *   sr_out             - [N_p] output max SR per period
 *   m                  - number of phase bins
 *   N_p                - number of trial periods (folds)
 *   nbins0             - minimum bin width (= 1/qmax in bins, = ceil(m * qmin))
 *   nbinsf             - maximum bin width (= floor(m * qmax))
 *   dlogq              - logarithmic spacing of trial widths
 *   ignore_negative_delta_sols - flag
 */
__global__ void ffa_score(
        const float * __restrict__ folds_yw,
        const float * __restrict__ folds_w,
        float * __restrict__ sr_out,
        unsigned int m,
        unsigned int N_p,
        unsigned int nbins0,
        unsigned int nbinsf,
        float dlogq,
        unsigned int ignore_negative_delta_sols)
{
    unsigned int fold_idx = blockIdx.x;
    if (fold_idx >= N_p) return;

    extern __shared__ float sh[];
    float *sh_yw    = sh;
    float *sh_w     = &sh[m];
    float *best_bls = &sh[2 * m];

    // Load fold into shared memory
    unsigned int offset = fold_idx * m;
    for (unsigned int i = threadIdx.x; i < m; i += blockDim.x){
        sh_yw[i] = folds_yw[offset + i];
        sh_w[i]  = folds_w[offset + i];
    }
    __syncthreads();

    // Box scan: same pattern as full_bls_no_sol scoring loop
    float thread_max_bls = 0.f;

    // For each starting bin (distributed across threads)
    for (unsigned int n = threadIdx.x; n < m; n += blockDim.x){
        float acc_yw = 0.f;
        float acc_w  = 0.f;
        unsigned int width = 0;
        unsigned int next_check = nbins0;

        // Accumulate bins from n to n+width, checking at log-spaced widths
        for (unsigned int k = 0; k < nbinsf; k++){
            unsigned int bin_idx = (n + k) % m;
            acc_yw += sh_yw[bin_idx];
            acc_w  += sh_w[bin_idx];
            width++;

            if (width >= next_check){
                float bls1 = ffa_bls_value(acc_yw, acc_w,
                                            ignore_negative_delta_sols);
                if (bls1 > thread_max_bls)
                    thread_max_bls = bls1;

                // Advance to next trial width
                if (dlogq > 0.f){
                    unsigned int step = (unsigned int) floorf(dlogq * next_check);
                    if (step < 1) step = 1;
                    next_check += step;
                } else {
                    next_check++;
                }

                if (next_check > nbinsf) break;
            }
        }
    }

    // Store per-thread max to shared memory for reduction
    best_bls[threadIdx.x] = thread_max_bls;
    __syncthreads();

    // Tree reduction down to warp level
    for (unsigned int k = (blockDim.x / 2); k >= 32; k /= 2){
        if (threadIdx.x < k){
            float a = best_bls[threadIdx.x];
            float b = best_bls[threadIdx.x + k];
            best_bls[threadIdx.x] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    // Final warp reduction using shuffle
    if (threadIdx.x < 32){
        float val = best_bls[threadIdx.x];
        for (int off = 16; off > 0; off /= 2){
            float other = __shfl_down_sync(0xffffffff, val, off);
            val = (val > other) ? val : other;
        }
        if (threadIdx.x == 0)
            best_bls[0] = val;
    }

    // Store result
    if (threadIdx.x == 0)
        sr_out[fold_idx] = best_bls[0];
}


/* ===================================================================
 * Phase 2: Batch kernels — process multiple octaves sharing the same
 * N_p (number of sections / folds) in a single kernel launch.
 *
 * Memory layout: octaves are stored contiguously in flat buffers,
 * indexed via per-octave offsets.  Parameter arrays (one element per
 * octave) are read from global memory; the fold data is at:
 *     buf[fold_offsets[oct_idx] + fold_within_oct * m_oct + bin]
 *
 * Grid convention:
 *     blockIdx.x  — section / fold index within the octave  (0..N_p-1)
 *     blockIdx.y  — octave index within the group            (0..n_oct-1)
 * =================================================================== */


/*
 * ffa_bin_sections_batch: Bin observations for all octaves in a N_p group.
 *
 * Grid:  (N_p, n_oct)
 * Block: BLOCK_SIZE threads
 * Shared memory: 2 * max_m * sizeof(float)
 *
 * Parameters (per-octave arrays indexed by blockIdx.y):
 *   P0_arr          [n_oct]       — base period per octave
 *   m_oct_arr       [n_oct]       — phase bins per octave
 *   fold_offsets    [n_oct]       — starting float offset into flat buffers
 *   starts_all      [n_oct * N_p] — section start indices
 *   ends_all        [n_oct * N_p] — section end indices
 *   max_m           scalar        — max(m_oct) in group (for shared-mem sizing)
 */
__global__ void ffa_bin_sections_batch(
        const float * __restrict__ t,
        const float * __restrict__ yw,
        const float * __restrict__ w,
        const float * __restrict__ P0_arr,
        const unsigned int * __restrict__ m_oct_arr,
        const unsigned int * __restrict__ fold_offsets,
        const unsigned int * __restrict__ starts_all,
        const unsigned int * __restrict__ ends_all,
        float * __restrict__ section_yw,
        float * __restrict__ section_w,
        unsigned int N_p,
        unsigned int max_m)
{
    unsigned int sec     = blockIdx.x;
    unsigned int oct_idx = blockIdx.y;
    if (sec >= N_p) return;

    float         P0   = P0_arr[oct_idx];
    unsigned int  m    = m_oct_arr[oct_idx];
    unsigned int  foff = fold_offsets[oct_idx];

    extern __shared__ float sh[];
    float *sh_yw = sh;
    float *sh_w  = &sh[max_m];

    // Zero shared memory (up to max_m — only first m bins are used)
    for (unsigned int i = threadIdx.x; i < m; i += blockDim.x){
        sh_yw[i] = 0.f;
        sh_w[i]  = 0.f;
    }
    __syncthreads();

    // Section boundaries for this octave stored at starts_all[oct_idx * N_p + sec]
    unsigned int se_off = oct_idx * N_p + sec;
    unsigned int start  = starts_all[se_off];
    unsigned int end    = ends_all[se_off];

    for (unsigned int k = start + threadIdx.x; k < end; k += blockDim.x){
        float phase = ffa_mod1(t[k] / P0);
        int bin = (int) floorf(m * phase);
        if (bin >= (int)m) bin = m - 1;

        atomicAdd(&sh_yw[bin], yw[k]);
        atomicAdd(&sh_w[bin],  w[k]);
    }
    __syncthreads();

    // Write to global flat buffer
    unsigned int out_off = foff + sec * m;
    for (unsigned int i = threadIdx.x; i < m; i += blockDim.x){
        section_yw[out_off + i] = sh_yw[i];
        section_w[out_off + i]  = sh_w[i];
    }
}


/*
 * ffa_butterfly_batch: One butterfly level for all octaves in a N_p group.
 *
 * On-device shift computation — no shift arrays transferred from host.
 *
 * Grid:  (N_p, n_oct)
 * Block: BLOCK_SIZE threads
 *
 * Parameters:
 *   level           scalar        — current butterfly level (0-indexed)
 *   m_oct_arr       [n_oct]       — phase bins per octave
 *   fold_offsets    [n_oct]       — flat-buffer offsets
 *   N_p             scalar
 */
__global__ void ffa_butterfly_batch(
        const float * __restrict__ in_yw,
        const float * __restrict__ in_w,
        float * __restrict__ out_yw,
        float * __restrict__ out_w,
        const unsigned int * __restrict__ m_oct_arr,
        const unsigned int * __restrict__ fold_offsets,
        unsigned int N_p,
        unsigned int level)
{
    unsigned int fold_idx = blockIdx.x;
    unsigned int oct_idx  = blockIdx.y;
    if (fold_idx >= N_p) return;

    unsigned int m    = m_oct_arr[oct_idx];
    unsigned int foff = fold_offsets[oct_idx];

    unsigned int folds_per_group = 1u << (level + 1);
    unsigned int half_folds      = 1u << level;

    unsigned int group  = fold_idx / folds_per_group;
    unsigned int s      = fold_idx % folds_per_group;
    unsigned int d_sub  = s % half_folds;

    unsigned int left_fold_idx  = (2 * group)     * half_folds + d_sub;
    unsigned int right_fold_idx = (2 * group + 1) * half_folds + d_sub;

    /* On-device shift: at level l, shift = -(2^l) if bit l of fold_idx
       is set, else 0.  Computed mod m. */
    int shift_mag = (1u << level) % m;
    int shift = ((fold_idx >> level) & 1)
                ? ((int)m - shift_mag) % (int)m
                : 0;

    unsigned int left_off  = foff + left_fold_idx  * m;
    unsigned int right_off = foff + right_fold_idx * m;
    unsigned int out_off   = foff + fold_idx * m;

    for (unsigned int b = threadIdx.x; b < m; b += blockDim.x){
        float l_yw = in_yw[left_off + b];
        float l_w  = in_w[left_off + b];

        int b_shifted = ffa_mod((int)b - shift, (int)m);
        float r_yw = in_yw[right_off + b_shifted];
        float r_w  = in_w[right_off + b_shifted];

        out_yw[out_off + b] = l_yw + r_yw;
        out_w[out_off + b]  = l_w  + r_w;
    }
}


/*
 * ffa_score_batch: BLS box scan for all octaves in a N_p group.
 *
 * Grid:  (N_p, n_oct)
 * Block: BLOCK_SIZE threads
 * Shared memory: (2 * max_m + BLOCK_SIZE) * sizeof(float)
 *
 * Parameters:
 *   m_oct_arr       [n_oct]
 *   fold_offsets    [n_oct]
 *   nbins0_arr      [n_oct]
 *   nbinsf_arr      [n_oct]
 *   sr_offsets      [n_oct]  — offset into sr_out for this octave
 */
__global__ void ffa_score_batch(
        const float * __restrict__ folds_yw,
        const float * __restrict__ folds_w,
        float * __restrict__ sr_out,
        const unsigned int * __restrict__ m_oct_arr,
        const unsigned int * __restrict__ fold_offsets,
        const unsigned int * __restrict__ nbins0_arr,
        const unsigned int * __restrict__ nbinsf_arr,
        const unsigned int * __restrict__ sr_offsets,
        unsigned int N_p,
        unsigned int max_m,
        float dlogq,
        unsigned int ignore_negative_delta_sols)
{
    unsigned int fold_idx = blockIdx.x;
    unsigned int oct_idx  = blockIdx.y;
    if (fold_idx >= N_p) return;

    unsigned int m      = m_oct_arr[oct_idx];
    unsigned int foff   = fold_offsets[oct_idx];
    unsigned int nbins0 = nbins0_arr[oct_idx];
    unsigned int nbinsf = nbinsf_arr[oct_idx];
    unsigned int sr_off = sr_offsets[oct_idx];

    extern __shared__ float sh[];
    float *sh_yw    = sh;
    float *sh_w     = &sh[max_m];
    float *best_bls = &sh[2 * max_m];

    // Load fold into shared memory
    unsigned int data_off = foff + fold_idx * m;
    for (unsigned int i = threadIdx.x; i < m; i += blockDim.x){
        sh_yw[i] = folds_yw[data_off + i];
        sh_w[i]  = folds_w[data_off + i];
    }
    __syncthreads();

    // Box scan
    float thread_max_bls = 0.f;

    for (unsigned int n = threadIdx.x; n < m; n += blockDim.x){
        float acc_yw = 0.f;
        float acc_w  = 0.f;
        unsigned int width = 0;
        unsigned int next_check = nbins0;

        for (unsigned int k = 0; k < nbinsf; k++){
            unsigned int bin_idx = (n + k) % m;
            acc_yw += sh_yw[bin_idx];
            acc_w  += sh_w[bin_idx];
            width++;

            if (width >= next_check){
                float bls1 = ffa_bls_value(acc_yw, acc_w,
                                            ignore_negative_delta_sols);
                if (bls1 > thread_max_bls)
                    thread_max_bls = bls1;

                if (dlogq > 0.f){
                    unsigned int step = (unsigned int) floorf(dlogq * next_check);
                    if (step < 1) step = 1;
                    next_check += step;
                } else {
                    next_check++;
                }
                if (next_check > nbinsf) break;
            }
        }
    }

    best_bls[threadIdx.x] = thread_max_bls;
    __syncthreads();

    // Tree reduction
    for (unsigned int k = (blockDim.x / 2); k >= 32; k /= 2){
        if (threadIdx.x < k){
            float a = best_bls[threadIdx.x];
            float b = best_bls[threadIdx.x + k];
            best_bls[threadIdx.x] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    // Warp shuffle reduction
    if (threadIdx.x < 32){
        float val = best_bls[threadIdx.x];
        for (int off = 16; off > 0; off /= 2){
            float other = __shfl_down_sync(0xffffffff, val, off);
            val = (val > other) ? val : other;
        }
        if (threadIdx.x == 0)
            best_bls[0] = val;
    }

    if (threadIdx.x == 0)
        sr_out[sr_off + fold_idx] = best_bls[0];
}
