/*
 * Fast Transit Least Squares (TLS) GPU kernel — batch-native.
 *
 * Algorithmic differences from tls.cu (the reference kernel):
 *
 * 1. Closed-form chi2. For the weighted least-squares transit fit with
 *    template T and depth d, chi2(d) = chi2_0 - 2 d num + d^2 den with
 *      num  = sum_i (1 - y_i) T_i / sigma_i^2
 *      den  = sum_i T_i^2 / sigma_i^2
 *      chi2_0 = sum_i (y_i - 1)^2 / sigma_i^2   (per-lightcurve constant)
 *    At the optimal depth d* = num/den, chi2 = chi2_0 - num^2/den, so a
 *    single accumulation pass yields both the depth and the chi2 — the
 *    reference kernel's second full-data chi2 pass is redundant.
 *    Minimizing chi2 over trials is exactly maximizing num^2/den, so the
 *    per-period argmin never suffers cancellation against chi2_0.
 *
 * 2. Phase-binned evaluation. Each block folds its lightcurve at its
 *    period ONCE into NBINS phase bins (A_k = sum (1-y)/sigma^2,
 *    B_k = sum 1/sigma^2), then every (duration, t0) trial integrates
 *    only the ~q*NBINS bins inside the transit window instead of
 *    scanning all ndata points. This removes both the O(ndata) factor
 *    from the trial loop and the shared-memory cap on ndata (raw data
 *    stay in global memory and are read exactly once per period).
 *
 * 3. Integrated template tables. S1(x) = int_{-1}^{x} T dx and
 *    S2(x) = int_{-1}^{x} T^2 dx are precomputed on the CPU
 *    (tls_models.generate_template_integrals). The bin-averaged
 *    template over a bin's transit-coordinate span [c0, c1] is
 *    (S1(c1)-S1(c0))/(c1-c0): area sampling rather than point
 *    sampling. This reduces quadrature error; phase compression still
 *    loses within-bin information and can reduce sensitivity.
 *
 * 4. Batch-native. Grid is (nperiods, nlc); per-lightcurve data are
 *    concatenated with offset/length arrays. A whole survey chunk is a
 *    single kernel launch sharing one period grid and one template.
 *
 * The (duration, t0) trial grid is IDENTICAL to tls.cu: n_durations
 * log-spaced durations in [qmin, qmax], t0 = j/n_t0 with
 * n_t0 = clamp(ceil(T0_OVERSAMPLE/q), MIN_N_T0, MAX_N_T0), and the
 * same validity gate 0 < depth < 0.5.
 *
 * References:
 * [1] Hippke & Heller (2019), A&A 623, A39
 * [2] Kovacs et al. (2002), A&A 391, 369
 */

#include <stdio.h>

//{CPP_DEFS}

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

/* Number of phase bins; must be a power of two (wrap uses a mask). */
#ifndef NBINS
#define NBINS 2048
#endif

/* Number of intervals in the integrated template tables (tables have
 * NTEMPLATE+1 entries). Must match the Python-side table length. */
#ifndef NTEMPLATE
#define NTEMPLATE 1024
#endif

/* Maximum n_durations supported by the per-duration shared staging. */
#ifndef MAX_DURATIONS
#define MAX_DURATIONS 64
#endif

/* Number of local durations scanned by the refinement kernel (odd). */
#ifndef REFINE_ND
#define REFINE_ND 3
#endif

#ifndef T0_OVERSAMPLE
#define T0_OVERSAMPLE 3.0f
#endif
#ifndef MIN_N_T0
#define MIN_N_T0 30
#endif
#ifndef MAX_N_T0
#define MAX_N_T0 20000
#endif

#define WARP_SIZE 32

/* Keep the original dense loop available for numerical/performance
 * comparisons. This changes only how zero-weight bins are visited,
 * never the template, histogram resolution, or trial grid. */
#ifndef TLS_SKIP_EMPTY_BINS
#define TLS_SKIP_EMPTY_BINS 1
#endif

/* The reduction workspace is unused until the trial scan finishes.
 * Reuse it for an occupancy bitmap, next-nonempty-word links, and the
 * population count. Small block-size overrides that cannot hold this
 * workspace simply retain the dense loop. */
#if TLS_SKIP_EMPTY_BINS && NBINS >= 1024 && \
        (2 * NBINS / WARP_SIZE + 1 <= 4 * BLOCK_SIZE)
#define TLS_SPARSE_SCAN_AVAILABLE 1
#else
#define TLS_SPARSE_SCAN_AVAILABLE 0
#endif

__device__ inline float mod1f(float x) {
    return x - floorf(x);
}

__device__ inline int t0_grid_size(float duration_phase) {
    int n_t0 = (int)ceilf(T0_OVERSAMPLE / duration_phase);
    if (n_t0 < MIN_N_T0) n_t0 = MIN_N_T0;
    if (n_t0 > MAX_N_T0) n_t0 = MAX_N_T0;
    return n_t0;
}

/*
 * Evaluate an integrated table S (NTEMPLATE+1 entries spanning
 * x in [-1, 1]) at x, with linear interpolation. Outside [-1, 1] the
 * template is zero, so S saturates at its endpoint values.
 */
__device__ inline float lookup_integral(const float* __restrict__ S, float x)
{
    float idx_f = (x + 1.0f) * (0.5f * (float)NTEMPLATE);
    idx_f = fminf(fmaxf(idx_f, 0.0f), (float)NTEMPLATE);
    int i0 = (int)idx_f;
    if (i0 >= NTEMPLATE) i0 = NTEMPLATE - 1;
    float frac = idx_f - (float)i0;
    return S[i0] + (S[i0 + 1] - S[i0]) * frac;
}

/*
 * Fast TLS search kernel (batch-native, Keplerian duration constraints).
 *
 * Grid: (nperiods, nlc, 1); Block: (BLOCK_SIZE, 1, 1)
 *
 * Inputs (global memory):
 *   t_hi_all, t_lo_all, a_all, b_all : concatenated per-point arrays;
 *       t is stored as an epoch-subtracted float-float pair
 *       (t = t_hi + t_lo to float64 precision), and for point i,
 *       a = (1 - y)/sigma^2 and b = 1/sigma^2
 *       (sigma^2 includes the +1e-10 regularizer, matching tls.cu)
 *   lc_off, lc_len : per-lightcurve offset/length into the above
 *   periods[nperiods_band], qmin[...], qmax[...] : the trial grid FOR
 *       THIS LAUNCH. The host may split the full grid into bands that
 *       compile with different NBINS (narrow durations need finer
 *       bins; the scan cost is proportional to NBINS, so coarse bands
 *       should not pay the finest band's price).
 *   period_map[nperiods_band] : global period index of each band entry
 *       (identity when the grid is not banded)
 *   S1, S2 : integrated template tables (NTEMPLATE+1 entries each)
 *
 * Outputs, laid out as [lc * nperiods_total + period_map[band idx]]:
 *   score_out (num^2/den = chi2_0 - chi2; <= 0 marks a failed period;
 *   the host reconstructs chi2 in float64), best_t0_out,
 *   best_duration_out, best_depth_out
 *
 * Shared memory layout (floats):
 *   A[NBINS] | B[NBINS] | S1[NTEMPLATE+1] | S2[NTEMPLATE+1] |
 *   red_score[BLOCK_SIZE] | red_t0[BLOCK_SIZE] | red_dur[BLOCK_SIZE] |
 *   red_depth[BLOCK_SIZE] | dur_q[MAX_DURATIONS] | dur_cum[MAX_DURATIONS+1]
 */
extern "C" __global__ void tls_fast_search_kernel(
    const float* __restrict__ t_hi_all,
    const float* __restrict__ t_lo_all,
    const float* __restrict__ a_all,
    const float* __restrict__ b_all,
    const int*   __restrict__ lc_off,
    const int*   __restrict__ lc_len,
    const float* __restrict__ periods,
    const float* __restrict__ qmin,
    const float* __restrict__ qmax,
    const int*   __restrict__ period_map,
    const float* __restrict__ S1_g,
    const float* __restrict__ S2_g,
    const int nperiods_band,
    const int nperiods_total,
    const int n_durations,
    float* __restrict__ score_out,
    float* __restrict__ best_t0_out,
    float* __restrict__ best_duration_out,
    float* __restrict__ best_depth_out)
{
    extern __shared__ float shared_mem[];
    float* A         = shared_mem;
    float* B         = &A[NBINS];
    float* S1        = &B[NBINS];
    float* S2        = &S1[NTEMPLATE + 1];
    float* red_score = &S2[NTEMPLATE + 1];
    float* red_t0    = &red_score[BLOCK_SIZE];
    float* red_dur   = &red_t0[BLOCK_SIZE];
    float* red_depth = &red_dur[BLOCK_SIZE];
    float* dur_q     = &red_depth[BLOCK_SIZE];
    /* trial-index prefix sums per duration (stored as float-cast ints
     * would lose precision above 2^24; keep a separate int view) */
    int*   dur_cum   = (int*)&dur_q[MAX_DURATIONS];

    const int period_idx = blockIdx.x;
    const int lc_idx     = blockIdx.y;
    if (period_idx >= nperiods_band) return;

    const int   off    = lc_off[lc_idx];
    const int   nd     = lc_len[lc_idx];
    const float period = periods[period_idx];

    /* --- Stage integrated template tables and zero the bins --- */
    for (int i = threadIdx.x; i < NTEMPLATE + 1; i += blockDim.x) {
        S1[i] = S1_g[i];
        S2[i] = S2_g[i];
    }
    for (int i = threadIdx.x; i < NBINS; i += blockDim.x) {
        A[i] = 0.0f;
        B[i] = 0.0f;
    }

    /* --- Per-duration trial bookkeeping (one thread; tiny) --- */
    if (threadIdx.x == 0) {
        float lqmin = logf(qmin[period_idx]);
        float lqmax = logf(qmax[period_idx]);
        int cum = 0;
        for (int d = 0; d < n_durations; d++) {
            float lq = (n_durations > 1)
                ? lqmin + (lqmax - lqmin) * d / (n_durations - 1)
                : lqmin;
            float q = expf(lq);
            dur_q[d] = q;
            dur_cum[d] = cum;
            cum += t0_grid_size(q);
        }
        dur_cum[n_durations] = cum;
    }
    __syncthreads();

    /* --- Fold and bin the lightcurve at this period ---
     * Float-float ("double-single") fold: at plain float32, t/P for a
     * 1,400-day baseline and a short period carries a phase error of
     * ~1e-4 — the size of a whole bin. Times are stored as a hi/lo
     * float32 pair (t = t_hi + t_lo exactly to float64 precision) and
     * the period reciprocal is split the same way, so the fractional
     * phase is recovered to ~1e-7 with pure FP32 FMAs. This avoids
     * double-precision math, which runs at 1/64 rate on consumer GPUs
     * and would otherwise dominate the whole kernel at large ndata. */
    const double inv_period_d = 1.0 / (double)period;
    const float inv_hi = (float)inv_period_d;
    const float inv_lo = (float)(inv_period_d - (double)inv_hi);
    for (int i = threadIdx.x; i < nd; i += blockDim.x) {
        const float th = t_hi_all[off + i];
        const float tl = t_lo_all[off + i];
        float u = th * inv_hi;
        float e = fmaf(th, inv_hi, -u);   /* exact product residual */
        float c = fmaf(th, inv_lo, fmaf(tl, inv_hi, e));
        float phi = (u - floorf(u)) + c;
        phi -= floorf(phi);
        int k = (int)(phi * (float)NBINS);
        k &= (NBINS - 1);
        atomicAdd(&A[k], a_all[off + i]);
        atomicAdd(&B[k], b_all[off + i]);
    }
    __syncthreads();

#if TLS_SPARSE_SCAN_AVAILABLE
    /* At fine resolutions sparse lightcurves leave most phase bins
     * empty. Build a forward link across each empty run, using the
     * empty B entries themselves; all nonzero A/B entries retain
     * their original values. This has no extra shared-memory cost.
     * A negative B value is metadata, never a statistical weight.
     *
     * Restrict this path to lightcurves with fewer than NBINS/4
     * observations, guaranteeing that at least 75% of bins are empty.
     * At intermediate occupancy, preserving the dense coordinate
     * arithmetic across every gap can cost more than the saved
     * integral lookups. Dense data avoid the preparation altogether.
     */
    bool sparse_scan = false;
    if (nd < NBINS / 4) {
        const int n_words = NBINS / WARP_SIZE;
        unsigned int* occupied = (unsigned int*)red_score;
        unsigned int* next_word = &occupied[n_words + 1];
        const int lane = threadIdx.x & (WARP_SIZE - 1);
        for (int word = threadIdx.x / WARP_SIZE; word < n_words;
                word += blockDim.x / WARP_SIZE) {
            const int k = word * WARP_SIZE + lane;
            const unsigned int mask = __ballot_sync(
                0xffffffff, A[k] != 0.0f || B[k] != 0.0f);
            if (lane == 0) occupied[word] = mask;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int count = 0;
            for (int word = 0; word < n_words; word++)
                count += __popc(occupied[word]);
            occupied[n_words] = count;
        }
        __syncthreads();
        const unsigned int count = occupied[n_words];
        sparse_scan = count > 0 && count < NBINS / 2;
        /* All threads must read the decision before any thread can
         * reuse this workspace for the final block reduction. */
        __syncthreads();
        if (sparse_scan) {
            /* Search empty runs at the word level once, not once
             * for every bin in the run. This also bounds setup work
             * for pathological lightcurves concentrated in a few
             * phase bins. */
            for (int word = threadIdx.x; word < n_words;
                    word += blockDim.x) {
                int next = word;
                do {
                    next = (next + 1) & (n_words - 1);
                } while (!occupied[next]);
                next_word[word] = next;
            }
            __syncthreads();
            for (int k = threadIdx.x; k < NBINS; k += blockDim.x) {
                if (A[k] == 0.0f && B[k] == 0.0f) {
                    const int lane_k = k & (WARP_SIZE - 1);
                    int word = k / WARP_SIZE;
                    /* Strictly later bits; the expression is also
                     * well defined for lane_k == 31 (result zero). */
                    unsigned int mask = occupied[word]
                        & (0xfffffffeu << lane_k);
                    if (!mask) {
                        word = next_word[word];
                        mask = occupied[word];
                    }
                    const int next = word * WARP_SIZE + __ffs(mask) - 1;
                    const int jump = (next - k) & (NBINS - 1);
                    B[k] = -(float)jump;
                }
            }
            __syncthreads();
        }
    }
#endif

    const int total_trials = dur_cum[n_durations];

    /* --- Scan all (duration, t0) trials, flattened across threads --- */
    float best_score = -1.0f;   /* score = num^2/den = chi2_0 - chi2 */
    float best_t0    = 0.0f;
    float best_dur   = 0.0f;
    float best_depth = 0.0f;

    int d_idx = 0;
    for (int trial = threadIdx.x; trial < total_trials; trial += blockDim.x) {
        /* locate the duration bucket (monotonically increasing) */
        while (dur_cum[d_idx + 1] <= trial) d_idx++;

        const float q      = dur_q[d_idx];
        const float hd     = 0.5f * q;
        const float inv_hd = 1.0f / hd;
        const int   n_t0   = dur_cum[d_idx + 1] - dur_cum[d_idx];
        const int   t0_idx = trial - dur_cum[d_idx];
        const float t0     = (float)t0_idx / (float)n_t0;

        /* bins overlapping the window [t0 - hd, t0 + hd] */
        const float invNB = 1.0f / (float)NBINS;
        int k0 = (int)floorf((t0 - hd) * (float)NBINS);
        int k1 = (int)ceilf((t0 + hd) * (float)NBINS) - 1;
        /* q >= 1 is rejected host-side; belt-and-braces so a rogue
         * window can never visit a bin twice */
        if (k1 - k0 >= NBINS) k1 = k0 + NBINS - 1;

        /* transit coordinate of bin kk's left edge, and per-bin span */
        const float dc = invNB * inv_hd;

        float num = 0.0f;
        float den = 0.0f;
        float c0 = ((float)k0 * invNB - t0) * inv_hd;
#if TLS_SPARSE_SCAN_AVAILABLE
        if (sparse_scan) {
            float s1_prev = lookup_integral(S1, c0);
            float s2_prev = lookup_integral(S2, c0);
            int kk = k0;
            while (kk <= k1) {
                int k = kk & (NBINS - 1);
                if (B[k] < 0.0f) {
                    const int jump = (int)-B[k];
                    kk += jump;
                    if (kk > k1) break;
                    /* Preserve the dense loop's float32 coordinates.
                     * Replacing these additions with dc*jump changes
                     * rounding; subtracting nearly equal cumulative
                     * template integrals at a transit edge can amplify
                     * that tiny shift into a material score change.
                     * Empty bins still need no table or weight loads. */
                    for (int skipped = 0; skipped < jump; skipped++)
                        c0 += dc;
                    k = kk & (NBINS - 1);
                    s1_prev = lookup_integral(S1, c0);
                    s2_prev = lookup_integral(S2, c0);
                }
                const float c1 = c0 + dc;
                const float s1_next = lookup_integral(S1, c1);
                const float s2_next = lookup_integral(S2, c1);
                num += A[k] * (s1_next - s1_prev);
                den += B[k] * (s2_next - s2_prev);
                s1_prev = s1_next;
                s2_prev = s2_next;
                c0 = c1;
                kk++;
            }
        } else
#endif
        {
            float s1_prev = lookup_integral(S1, c0);
            float s2_prev = lookup_integral(S2, c0);
            for (int kk = k0; kk <= k1; kk++) {
                int k = kk & (NBINS - 1);
                float c1 = c0 + dc;
                float s1_next = lookup_integral(S1, c1);
                float s2_next = lookup_integral(S2, c1);
                num += A[k] * (s1_next - s1_prev);
                den += B[k] * (s2_next - s2_prev);
                s1_prev = s1_next;
                s2_prev = s2_next;
                c0 = c1;
            }
        }
        /* bin-average scale: 1/(c1-c0) = hd*NBINS applied once */
        const float scale = hd * (float)NBINS;
        num *= scale;
        den *= scale;

        if (den > 1e-10f && num > 0.0f) {
            float depth = num / den;
            if (depth < 0.5f) {
                float score = num * depth;   /* num^2/den */
                if (score > best_score) {
                    best_score = score;
                    best_t0    = t0;
                    best_dur   = q * period;
                    best_depth = depth;
                }
            }
        }
    }

    /* --- Block reduction (max score) --- */
    red_score[threadIdx.x] = best_score;
    red_t0[threadIdx.x]    = best_t0;
    red_dur[threadIdx.x]   = best_dur;
    red_depth[threadIdx.x] = best_depth;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride /= 2) {
        if (threadIdx.x < stride) {
            if (red_score[threadIdx.x + stride] > red_score[threadIdx.x]) {
                red_score[threadIdx.x] = red_score[threadIdx.x + stride];
                red_t0[threadIdx.x]    = red_t0[threadIdx.x + stride];
                red_dur[threadIdx.x]   = red_dur[threadIdx.x + stride];
                red_depth[threadIdx.x] = red_depth[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < WARP_SIZE) {
        float v_score = red_score[threadIdx.x];
        float v_t0    = red_t0[threadIdx.x];
        float v_dur   = red_dur[threadIdx.x];
        float v_dep   = red_depth[threadIdx.x];

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float o_score = __shfl_down_sync(0xffffffff, v_score, offset);
            float o_t0    = __shfl_down_sync(0xffffffff, v_t0, offset);
            float o_dur   = __shfl_down_sync(0xffffffff, v_dur, offset);
            float o_dep   = __shfl_down_sync(0xffffffff, v_dep, offset);
            if (o_score > v_score) {
                v_score = o_score;
                v_t0    = o_t0;
                v_dur   = o_dur;
                v_dep   = o_dep;
            }
        }

        if (threadIdx.x == 0) {
            const size_t out_idx = (size_t)lc_idx * nperiods_total
                                   + period_map[period_idx];
            /* Write the SCORE (delta-chi2 = num^2/den = chi2_0 - chi2),
             * not chi2 itself: subtracting from the large per-LC
             * constant in float32 would quantize the spectrum by
             * ulp(chi2_0) ~ 6e-8 * ndata. The host reconstructs
             * chi2 = chi2_0 - score in float64. score <= 0 marks a
             * period with no valid trial. */
            if (v_score > 0.0f) {
                score_out[out_idx]         = v_score;
                best_t0_out[out_idx]       = v_t0;
                best_duration_out[out_idx] = v_dur;
                best_depth_out[out_idx]    = v_dep;
            } else {
                score_out[out_idx]         = -1.0f;
                best_t0_out[out_idx]       = 0.0f;
                best_duration_out[out_idx] = 0.0f;
                best_depth_out[out_idx]    = 0.0f;
            }
        }
    }
}

/*
 * Exact refinement kernel.
 *
 * The binned scan uses a coarse epoch grid and smears each point's
 * template weight over its bin. This kernel re-evaluates the best
 * candidate periods per lightcurve EXACTLY (per-point template lookup,
 * no binning) on a fine local (duration, t0) grid centered on the
 * coarse solution.
 *
 * Results go to separate compact per-candidate outputs — the coarse
 * per-period spectrum is left untouched. Detection statistics (SDE)
 * must be computed from a UNIFORM-fidelity spectrum: a finer trial
 * grid digs deeper chi2 minima everywhere (noise included), so mixing
 * refined values into the coarse spectrum — or refining everything —
 * shifts the SR distribution and deflates the SDE scale that the
 * legacy kernel and its calibrated thresholds established. Refinement
 * therefore only sharpens the best-fit parameters (period choice among
 * the candidates, t0, duration, depth, chi2_min).
 *
 * Grid: (n_candidates, nlc, 1); Block: (BLOCK_SIZE, 1, 1)
 * cand_period_idx[lc * n_candidates + c] gives the period index to
 * refine (a value < 0 disables that slot).
 *
 * Trial layout per candidate: REFINE_ND durations log-spaced within
 * [q0/dur_span, q0*dur_span] (bracketing one coarse duration-grid
 * step), each with n_t0_local epochs spanning +/- t0_halfwidth around
 * the coarse t0 at stride q/refine_oversample.
 *
 * Each trial is owned by one warp-group slice of the block: trials are
 * distributed round-robin over (blockDim/WARP_SIZE) warps; a warp
 * accumulates num/den over all points with lane-strided reads and
 * reduces with shuffles. Points stream from global memory (coalesced);
 * the point template T is staged in shared memory.
 *
 * Shared memory layout (floats):
 *   T[NTEMPLATE + 1] | warp_best[4 * (BLOCK_SIZE/WARP_SIZE)]
 */
extern "C" __global__ void tls_refine_kernel(
    const float* __restrict__ t_hi_all,
    const float* __restrict__ t_lo_all,
    const float* __restrict__ a_all,
    const float* __restrict__ b_all,
    const int*   __restrict__ lc_off,
    const int*   __restrict__ lc_len,
    const float* __restrict__ periods,
    const int*   __restrict__ cand_period_idx,
    const float* __restrict__ T_g,
    const int nperiods,
    const int n_candidates,
    const float dur_span,          /* e.g. one coarse log-step, ~1.10 */
    const float t0_halfwidth_frac, /* halfwidth in units of duration */
    const float refine_oversample, /* t0 stride = q / refine_oversample */
    const float* __restrict__ coarse_t0_in,
    const float* __restrict__ coarse_duration_in,
    float* __restrict__ refined_score_out,     /* [lc * n_candidates + c] */
    float* __restrict__ refined_t0_out,
    float* __restrict__ refined_duration_out,
    float* __restrict__ refined_depth_out)
{
    extern __shared__ float shared_mem[];
    float* T_sh      = shared_mem;
    float* warp_best = &T_sh[NTEMPLATE + 1];   /* 4 floats per warp */

    const int cand_idx = blockIdx.x;
    const int lc_idx   = blockIdx.y;
    if (cand_idx >= n_candidates) return;

    const size_t slot = (size_t)lc_idx * n_candidates + cand_idx;
    const int period_idx = cand_period_idx[slot];

    /* disabled or sentinel candidates still need a sentinel output */
    float coarse_dur = 0.0f, coarse_t0 = 0.0f, period = 1.0f;
    if (period_idx >= 0) {
        const size_t in_idx = (size_t)lc_idx * nperiods + period_idx;
        coarse_dur = coarse_duration_in[in_idx];
        coarse_t0  = coarse_t0_in[in_idx];
        period     = periods[period_idx];
    }
    if (period_idx < 0 || coarse_dur <= 0.0f) {
        if (threadIdx.x == 0) {
            refined_score_out[slot]    = -1.0f;
            refined_t0_out[slot]       = 0.0f;
            refined_duration_out[slot] = 0.0f;
            refined_depth_out[slot]    = 0.0f;
        }
        return;
    }

    const int   off = lc_off[lc_idx];
    const int   nd  = lc_len[lc_idx];
    const double inv_period_d = 1.0 / (double)period;
    const float inv_hi = (float)inv_period_d;
    const float inv_lo = (float)(inv_period_d - (double)inv_hi);
    const float q0 = coarse_dur / period;

    for (int i = threadIdx.x; i < NTEMPLATE + 1; i += blockDim.x) {
        T_sh[i] = T_g[i];
    }
    __syncthreads();

    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int n_warps  = blockDim.x / WARP_SIZE;

    /* local trial grid */
    const float lq0   = logf(q0);
    const float ldspan = logf(dur_span);
    const int n_dur_local = REFINE_ND;   /* compile-time, odd, e.g. 5 */

    float w_best_score = -1.0f;
    float w_best_t0 = 0.0f, w_best_dur = 0.0f, w_best_depth = 0.0f;

    /* count t0 trials for the central duration to fix the grid size
     * (same count reused for all durations so trial indexing is flat) */
    const float t0_hw = t0_halfwidth_frac * q0;
    const float dt0   = q0 / refine_oversample;
    int n_t0_local = 2 * (int)ceilf(t0_hw / dt0) + 1;

    const int total_trials = n_dur_local * n_t0_local;

    for (int trial = warp_id; trial < total_trials; trial += n_warps) {
        const int d_i  = trial / n_t0_local;
        const int t0_i = trial % n_t0_local;

        const float lq = lq0 + ldspan * (2.0f * d_i / (n_dur_local - 1) - 1.0f);
        const float q  = expf(lq);
        const float hd = 0.5f * q;
        const float inv_hd = 1.0f / hd;
        float t0 = coarse_t0 + dt0 * (float)(t0_i - n_t0_local / 2);
        t0 = t0 - floorf(t0);   /* wrap to [0, 1) */

        float num = 0.0f;
        float den = 0.0f;
        for (int i = lane; i < nd; i += WARP_SIZE) {
            const float th = t_hi_all[off + i];
            const float tl = t_lo_all[off + i];
            float u = th * inv_hi;
            float e = fmaf(th, inv_hi, -u);
            float cc = fmaf(th, inv_lo, fmaf(tl, inv_hi, e));
            float phi = (u - floorf(u)) + cc;
            phi -= floorf(phi);
            float rel = phi - t0;
            rel -= rintf(rel);            /* wrap to [-0.5, 0.5] */
            float c = rel * inv_hd;
            if (fabsf(c) < 1.0f) {
                /* point template lookup (linear interpolation) */
                float idx_f = (c + 1.0f) * (0.5f * (float)NTEMPLATE);
                int i0 = (int)idx_f;
                if (i0 >= NTEMPLATE) i0 = NTEMPLATE - 1;
                float frac = idx_f - (float)i0;
                float Tv = T_sh[i0] + (T_sh[i0 + 1] - T_sh[i0]) * frac;
                num += a_all[off + i] * Tv;
                den += b_all[off + i] * Tv * Tv;
            }
        }
        /* warp reduction of the two partial sums */
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            num += __shfl_down_sync(0xffffffff, num, offset);
            den += __shfl_down_sync(0xffffffff, den, offset);
        }

        if (lane == 0 && den > 1e-10f && num > 0.0f) {
            float depth = num / den;
            if (depth < 0.5f) {
                float score = num * depth;
                if (score > w_best_score) {
                    w_best_score = score;
                    w_best_t0    = t0;
                    w_best_dur   = q * period;
                    w_best_depth = depth;
                }
            }
        }
    }

    /* combine warp winners via shared memory (few warps; lane 0 only) */
    if (lane == 0) {
        warp_best[4 * warp_id + 0] = w_best_score;
        warp_best[4 * warp_id + 1] = w_best_t0;
        warp_best[4 * warp_id + 2] = w_best_dur;
        warp_best[4 * warp_id + 3] = w_best_depth;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float b_score = -1.0f, b_t0 = 0.0f, b_dur = 0.0f, b_dep = 0.0f;
        for (int w = 0; w < n_warps; w++) {
            if (warp_best[4 * w] > b_score) {
                b_score = warp_best[4 * w];
                b_t0    = warp_best[4 * w + 1];
                b_dur   = warp_best[4 * w + 2];
                b_dep   = warp_best[4 * w + 3];
            }
        }
        if (b_score > 0.0f) {
            refined_score_out[slot]    = b_score;
            refined_t0_out[slot]       = b_t0;
            refined_duration_out[slot] = b_dur;
            refined_depth_out[slot]    = b_dep;
        } else {
            refined_score_out[slot]    = -1.0f;
            refined_t0_out[slot]       = 0.0f;
            refined_duration_out[slot] = 0.0f;
            refined_depth_out[slot]    = 0.0f;
        }
    }
}
