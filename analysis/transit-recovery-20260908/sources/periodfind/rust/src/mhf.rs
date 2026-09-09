/// Multi-Harmonic Fourier (MHF) periodogram — CPU implementation.
///
/// For each trial period, fits models with k = 0..max_harmonics Fourier terms:
///   y(t) = offset + slope*(t - tmin) + Σ_{n=1}^{k} [An*cos(n*φ) + Bn*sin(n*φ)]
/// where φ = 2π * phase(t, period, period_dt).
///
/// The score at each period is ΔBIC = BIC_flat - BIC_best, so higher = more periodic.
/// Uses BIC model selection to automatically choose the optimal number of harmonics.
use rayon::prelude::*;

use crate::fold;
use crate::fourier;

/// Maximum harmonics supported.
const MAX_K: usize = 5;
/// Maximum number of parameters: 2 (offset, slope) + 2*MAX_K (cos/sin pairs).
const MAX_PARAMS: usize = 2 + 2 * MAX_K; // 12

/// Scratch buffers reused across period iterations to avoid allocation.
struct MhfScratch {
    /// Column-major design matrix columns, each of length n_data.
    x_cols: Vec<Vec<f64>>,
    /// Normal equation matrix (MAX_PARAMS x MAX_PARAMS).
    ata: Vec<f64>,
    /// Normal equation RHS (MAX_PARAMS).
    atb: Vec<f64>,
    /// Chi-squared for each model (MAX_K + 1 models).
    chi2: [f64; MAX_K + 1],
    /// Whether each model solved successfully.
    model_ok: [bool; MAX_K + 1],
    /// Solution vector for current sub-system.
    sol: Vec<f64>,
    /// Sub-system matrix for Cholesky solve.
    sub_a: Vec<f64>,
}

impl MhfScratch {
    fn new(n_data: usize) -> Self {
        let mut x_cols = Vec::with_capacity(MAX_PARAMS);
        for _ in 0..MAX_PARAMS {
            x_cols.push(vec![0.0_f64; n_data]);
        }
        MhfScratch {
            x_cols,
            ata: vec![0.0_f64; MAX_PARAMS * MAX_PARAMS],
            atb: vec![0.0_f64; MAX_PARAMS],
            chi2: [0.0; MAX_K + 1],
            model_ok: [false; MAX_K + 1],
            sol: vec![0.0_f64; MAX_PARAMS],
            sub_a: vec![0.0_f64; MAX_PARAMS * MAX_PARAMS],
        }
    }
}

/// Result of evaluating MHF at a single period.
struct MhfPerPeriodResult {
    /// ΔBIC for each harmonic level k=0..max_k (BIC_flat - BIC_k).
    delta_bic_per_k: [f64; MAX_K + 1],
    /// Best (maximum) ΔBIC across all k.
    best_delta_bic: f64,
    /// The k that achieved the best ΔBIC.
    best_k: usize,
}

/// Evaluate MHF at a single (period, period_dt) for one light curve.
///
/// This is the shared inner computation used by both `calc_mhf` (grid search)
/// and `calc_mhf_per_k` (single-period per-K extraction).
fn eval_mhf_single_period(
    times: &[f32],
    y: &[f64],
    w: &[f64],
    ln_n: f64,
    period: f32,
    period_dt: f32,
    max_k: usize,
    scratch: &mut MhfScratch,
) -> MhfPerPeriodResult {
    let n_data = times.len();
    let mut result = MhfPerPeriodResult {
        delta_bic_per_k: [0.0; MAX_K + 1],
        best_delta_bic: 0.0,
        best_k: 0,
    };

    let pdt_corr = fold::pdt_correction(period, period_dt);

    if period <= 0.0 || !period.is_finite() {
        return result;
    }

    // Find tmin for slope term
    let tmin = times.iter().copied().fold(f32::INFINITY, f32::min);

    // Build design matrix columns using trig recurrence.
    let np_full = 2 + 2 * max_k;

    for i in 0..n_data {
        let phase = fold::fold_time(times[i], period, pdt_corr);
        let phi = 2.0 * std::f64::consts::PI * phase as f64;

        scratch.x_cols[0][i] = 1.0;
        scratch.x_cols[1][i] = (times[i] - tmin) as f64;

        if max_k >= 1 {
            let (s1, c1) = phi.sin_cos();
            scratch.x_cols[2][i] = c1;
            scratch.x_cols[3][i] = s1;

            let mut c_prev2 = 1.0_f64;
            let mut s_prev2 = 0.0_f64;
            let mut c_prev1 = c1;
            let mut s_prev1 = s1;
            let two_c1 = 2.0 * c1;

            for harm in 2..=max_k {
                let c_k = two_c1 * c_prev1 - c_prev2;
                let s_k = two_c1 * s_prev1 - s_prev2;
                scratch.x_cols[2 * harm][i] = c_k;
                scratch.x_cols[2 * harm + 1][i] = s_k;

                c_prev2 = c_prev1;
                s_prev2 = s_prev1;
                c_prev1 = c_k;
                s_prev1 = s_k;
            }
        }
    }

    // Build full normal equations: A = X^T W X, b = X^T W y
    for col_j in 0..np_full {
        for col_i in 0..=col_j {
            let mut s = 0.0_f64;
            for d in 0..n_data {
                s += w[d] * scratch.x_cols[col_i][d] * scratch.x_cols[col_j][d];
            }
            scratch.ata[col_j * np_full + col_i] = s;
            scratch.ata[col_i * np_full + col_j] = s;
        }
        let mut s = 0.0_f64;
        for d in 0..n_data {
            s += w[d] * scratch.x_cols[col_j][d] * y[d];
        }
        scratch.atb[col_j] = s;
    }

    // Try each model k = 0..max_k
    scratch.chi2 = [f64::INFINITY; MAX_K + 1];
    scratch.model_ok = [false; MAX_K + 1];

    for k in 0..=max_k {
        let np_k = 2 + 2 * k;

        if n_data <= np_k {
            continue;
        }

        for row in 0..np_k {
            for col in 0..np_k {
                scratch.sub_a[row * np_k + col] = scratch.ata[row * np_full + col];
            }
            scratch.sol[row] = scratch.atb[row];
        }

        if !fourier::cholesky_solve(
            &mut scratch.sub_a[..np_k * np_k],
            &mut scratch.sol[..np_k],
            np_k,
        ) {
            continue;
        }

        scratch.model_ok[k] = true;

        let mut c2 = 0.0_f64;
        for d in 0..n_data {
            let mut pred = 0.0_f64;
            for j in 0..np_k {
                pred += scratch.sol[j] * scratch.x_cols[j][d];
            }
            let resid = y[d] - pred;
            c2 += w[d] * resid * resid;
        }
        scratch.chi2[k] = c2;
    }

    // If even the constant model failed, return zeros
    if !scratch.model_ok[0] {
        return result;
    }

    // BIC for the flat (k=0) model
    let bic_flat = scratch.chi2[0] + ln_n * 2.0;

    // Compute per-k ΔBIC and find best
    let mut best_delta = 0.0_f64;
    let mut best_k = 0_usize;

    for k in 0..=max_k {
        if !scratch.model_ok[k] {
            result.delta_bic_per_k[k] = 0.0;
            continue;
        }
        let np_k = (2 + 2 * k) as f64;
        let bic_k = scratch.chi2[k] + ln_n * np_k;
        let delta = (bic_flat - bic_k).max(0.0);
        result.delta_bic_per_k[k] = delta;
        if delta > best_delta {
            best_delta = delta;
            best_k = k;
        }
    }

    result.best_delta_bic = best_delta;
    result.best_k = best_k;
    result
}

/// Compute the MHF periodogram for a single light curve over a grid of
/// (period, period_dt) pairs.
///
/// Returns a flat `Vec<f32>` of length `n_periods * n_pdts`.
/// Each element is ΔBIC = BIC_flat - BIC_best (higher = more periodic).
pub fn calc_mhf(
    times: &[f32],
    mags: &[f32],
    errs: &[f32],
    periods: &[f32],
    period_dts: &[f32],
    max_harmonics: usize,
) -> Vec<f32> {
    let n_periods = periods.len();
    let n_pdts = period_dts.len();
    let n_data = times.len();
    let max_k = max_harmonics.min(MAX_K);

    // Need at least 3 points for any meaningful fit
    if n_data < 3 {
        return vec![0.0_f32; n_periods * n_pdts];
    }

    // Precompute in f64 for numerical stability
    let y: Vec<f64> = mags.iter().map(|m| *m as f64).collect();
    let w: Vec<f64> = errs
        .iter()
        .map(|e| {
            let e64 = *e as f64;
            if e64 > 0.0 {
                1.0 / (e64 * e64)
            } else {
                0.0
            }
        })
        .collect();

    let n_f64 = n_data as f64;
    let ln_n = n_f64.ln();

    let total = n_periods * n_pdts;
    let results: Vec<f32> = (0..total)
        .into_par_iter()
        .map_init(
            || MhfScratch::new(n_data),
            |scratch, flat_idx| {
                let period_idx = flat_idx / n_pdts;
                let pdt_idx = flat_idx % n_pdts;

                let r = eval_mhf_single_period(
                    times,
                    &y,
                    &w,
                    ln_n,
                    periods[period_idx],
                    period_dts[pdt_idx],
                    max_k,
                    scratch,
                );
                r.best_delta_bic as f32
            },
        )
        .collect();

    results
}

/// Evaluate MHF per-K ΔBIC at a single period for a single light curve.
///
/// Returns a `Vec<f32>` of length `max_harmonics + 2`:
///   `[ΔBIC_k0, ΔBIC_k1, ..., ΔBIC_kN, best_k]`
///
/// This is designed for morphology discrimination at a known period:
/// - ΔBIC(K=3) >> ΔBIC(K=1) → non-sinusoidal (sawtooth, eclipsing)
/// - ΔBIC(K=3) ≈ ΔBIC(K=1) → sinusoidal (rotation, simple pulsation)
pub fn calc_mhf_per_k(
    times: &[f32],
    mags: &[f32],
    errs: &[f32],
    period: f32,
    period_dt: f32,
    max_harmonics: usize,
) -> Vec<f32> {
    let n_data = times.len();
    let max_k = max_harmonics.min(MAX_K);
    let out_len = max_k + 2; // ΔBIC for k=0..max_k, plus best_k

    if n_data < 3 {
        return vec![0.0_f32; out_len];
    }

    let y: Vec<f64> = mags.iter().map(|m| *m as f64).collect();
    let w: Vec<f64> = errs
        .iter()
        .map(|e| {
            let e64 = *e as f64;
            if e64 > 0.0 {
                1.0 / (e64 * e64)
            } else {
                0.0
            }
        })
        .collect();

    let ln_n = (n_data as f64).ln();
    let mut scratch = MhfScratch::new(n_data);

    let r = eval_mhf_single_period(times, &y, &w, ln_n, period, period_dt, max_k, &mut scratch);

    let mut out = Vec::with_capacity(out_len);
    for k in 0..=max_k {
        out.push(r.delta_bic_per_k[k] as f32);
    }
    out.push(r.best_k as f32);
    out
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a pure sinusoid with noise.
    fn make_sinusoid(n: usize, period: f32, amp: f32, offset: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let times: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
        let mags: Vec<f32> = times
            .iter()
            .map(|t| {
                let phi = 2.0 * std::f32::consts::PI * t / period;
                offset + amp * phi.cos()
            })
            .collect();
        let errs = vec![0.01_f32; n];
        (times, mags, errs)
    }

    /// Generate a sawtooth-like signal using 3 harmonics.
    fn make_sawtooth(n: usize, period: f32, amp: f32, offset: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let times: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
        let mags: Vec<f32> = times
            .iter()
            .map(|t| {
                let phi = 2.0 * std::f32::consts::PI * t / period;
                // Sawtooth approximation: sum of sin(nφ)/n
                offset
                    + amp * (phi.sin()
                        - 0.5 * (2.0 * phi).sin()
                        + (1.0 / 3.0) * (3.0 * phi).sin())
            })
            .collect();
        let errs = vec![0.01_f32; n];
        (times, mags, errs)
    }

    #[test]
    fn test_sinusoid_detection() {
        let (times, mags, errs) = make_sinusoid(200, 3.0, 1.0, 15.0);
        let periods = vec![3.0_f32];
        let period_dts = vec![0.0_f32];

        let result = calc_mhf(&times, &mags, &errs, &periods, &period_dts, 5);
        assert_eq!(result.len(), 1);
        // Should detect strong periodicity
        assert!(
            result[0] > 10.0,
            "sinusoid ΔBIC = {} (expected > 10)",
            result[0]
        );
    }

    #[test]
    fn test_sawtooth_detection() {
        let (times, mags, errs) = make_sawtooth(200, 3.0, 1.0, 15.0);
        let periods = vec![3.0_f32];
        let period_dts = vec![0.0_f32];

        let result = calc_mhf(&times, &mags, &errs, &periods, &period_dts, 5);
        assert_eq!(result.len(), 1);
        // Should detect strong periodicity for sawtooth too
        assert!(
            result[0] > 10.0,
            "sawtooth ΔBIC = {} (expected > 10)",
            result[0]
        );
    }

    #[test]
    fn test_flat_signal_low_score() {
        let n = 200;
        let times: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
        let mags = vec![15.0_f32; n];
        let errs = vec![0.01_f32; n];

        let periods = vec![3.0_f32];
        let period_dts = vec![0.0_f32];

        let result = calc_mhf(&times, &mags, &errs, &periods, &period_dts, 5);
        assert_eq!(result.len(), 1);
        // Flat signal should have near-zero score (clamped to 0)
        assert!(
            result[0] < 1.0,
            "flat signal ΔBIC = {} (expected ~0)",
            result[0]
        );
    }

    #[test]
    fn test_score_non_negative() {
        let n = 50;
        let times: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mags = vec![10.0_f32; n];
        let errs = vec![1.0_f32; n];

        let periods = vec![1.0_f32, 2.0, 5.0];
        let period_dts = vec![0.0_f32];

        let result = calc_mhf(&times, &mags, &errs, &periods, &period_dts, 5);
        for (i, &v) in result.iter().enumerate() {
            assert!(v >= 0.0, "score[{}] = {} (expected >= 0)", i, v);
        }
    }

    #[test]
    fn test_k5_beats_k1_for_sawtooth() {
        let (times, mags, errs) = make_sawtooth(200, 3.0, 1.0, 15.0);
        let periods = vec![3.0_f32];
        let period_dts = vec![0.0_f32];

        let score_k1 = calc_mhf(&times, &mags, &errs, &periods, &period_dts, 1);
        let score_k5 = calc_mhf(&times, &mags, &errs, &periods, &period_dts, 5);

        // K=5 should give higher or equal ΔBIC than K=1 for a sawtooth
        assert!(
            score_k5[0] >= score_k1[0],
            "K=5 ({}) should beat K=1 ({}) for sawtooth",
            score_k5[0],
            score_k1[0]
        );
    }

    #[test]
    fn test_too_few_points() {
        let times = vec![1.0_f32, 2.0];
        let mags = vec![10.0_f32, 11.0];
        let errs = vec![0.1_f32, 0.1];

        let periods = vec![1.0_f32];
        let period_dts = vec![0.0_f32];

        let result = calc_mhf(&times, &mags, &errs, &periods, &period_dts, 5);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn test_multiple_periods() {
        let (times, mags, errs) = make_sinusoid(200, 3.0, 1.0, 15.0);
        // Test with the true period and some wrong periods
        let periods = vec![1.5_f32, 3.0, 6.0, 10.0];
        let period_dts = vec![0.0_f32];

        let result = calc_mhf(&times, &mags, &errs, &periods, &period_dts, 5);
        assert_eq!(result.len(), 4);

        // The true period (index 1) should have the highest score
        let max_idx = result
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            max_idx, 1,
            "true period (idx 1) should have max score, but idx {} does: {:?}",
            max_idx, result
        );
    }

    #[test]
    fn test_period_dt_grid() {
        let (times, mags, errs) = make_sinusoid(100, 2.0, 0.5, 15.0);
        let periods = vec![2.0_f32, 4.0];
        let period_dts = vec![-0.001_f32, 0.0, 0.001];

        let result = calc_mhf(&times, &mags, &errs, &periods, &period_dts, 3);
        // Should be n_periods * n_pdts = 2 * 3 = 6
        assert_eq!(result.len(), 6);

        // All values should be non-negative
        for &v in &result {
            assert!(v >= 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // Per-K tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_per_k_sinusoid_k1_dominates() {
        // Pure sinusoid: K=1 should capture nearly all the signal.
        // Higher K should not add much ΔBIC.
        let (times, mags, errs) = make_sinusoid(200, 3.0, 1.0, 15.0);
        let result = calc_mhf_per_k(&times, &mags, &errs, 3.0, 0.0, 3);

        // result = [dbic_k0, dbic_k1, dbic_k2, dbic_k3, best_k]
        assert_eq!(result.len(), 5);
        let dbic_k1 = result[1];
        let dbic_k3 = result[3];
        let best_k = result[4] as usize;

        assert!(dbic_k1 > 10.0, "sinusoid dbic_k1={} should be large", dbic_k1);
        // K=1 should be optimal or very close to best
        assert!(best_k <= 1, "sinusoid best_k={} (expected 1)", best_k);
        // K=3 should not be much bigger than K=1 for a pure sinusoid
        // (BIC penalty means K=3 may even be slightly less)
        let ratio = if dbic_k1 > 0.0 { dbic_k3 / dbic_k1 } else { 0.0 };
        assert!(
            ratio < 1.5,
            "sinusoid k3/k1 ratio={} (expected <1.5 for sinusoidal)",
            ratio
        );
    }

    #[test]
    fn test_per_k_sawtooth_higher_k_wins() {
        // Sawtooth: higher harmonics should contribute significantly.
        let (times, mags, errs) = make_sawtooth(200, 3.0, 1.0, 15.0);
        let result = calc_mhf_per_k(&times, &mags, &errs, 3.0, 0.0, 3);

        assert_eq!(result.len(), 5);
        let dbic_k1 = result[1];
        let dbic_k3 = result[3];
        let best_k = result[4] as usize;

        assert!(dbic_k1 > 10.0, "sawtooth dbic_k1={}", dbic_k1);
        assert!(dbic_k3 > dbic_k1, "sawtooth dbic_k3={} should exceed dbic_k1={}", dbic_k3, dbic_k1);
        assert!(best_k >= 2, "sawtooth best_k={} (expected >=2)", best_k);
    }

    #[test]
    fn test_per_k_flat_all_zero() {
        let n = 200;
        let times: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
        let mags = vec![15.0_f32; n];
        let errs = vec![0.01_f32; n];

        let result = calc_mhf_per_k(&times, &mags, &errs, 3.0, 0.0, 3);
        assert_eq!(result.len(), 5);

        // All ΔBIC values should be ~0 for flat signal
        for k in 0..=3 {
            assert!(
                result[k] < 1.0,
                "flat signal dbic_k{}={} (expected ~0)",
                k,
                result[k]
            );
        }
        // best_k should be 0 (flat model wins)
        assert_eq!(result[4] as usize, 0, "flat best_k should be 0");
    }

    #[test]
    fn test_per_k_consistency_with_calc_mhf() {
        // The best_delta_bic from per_k should match calc_mhf at the same period.
        let (times, mags, errs) = make_sawtooth(200, 3.0, 1.0, 15.0);

        let grid_result = calc_mhf(&times, &mags, &errs, &[3.0_f32], &[0.0_f32], 3);
        let per_k_result = calc_mhf_per_k(&times, &mags, &errs, 3.0, 0.0, 3);

        // The max of per_k ΔBIC values should equal the grid result
        let max_per_k = per_k_result[0..=3]
            .iter()
            .copied()
            .fold(0.0_f32, f32::max);

        let diff = (grid_result[0] - max_per_k).abs();
        assert!(
            diff < 0.01,
            "grid={} vs per_k_max={} differ by {}",
            grid_result[0],
            max_per_k,
            diff
        );
    }

    #[test]
    fn test_per_k_too_few_points() {
        let times = vec![1.0_f32, 2.0];
        let mags = vec![10.0_f32, 11.0];
        let errs = vec![0.1_f32, 0.1];

        let result = calc_mhf_per_k(&times, &mags, &errs, 1.0, 0.0, 3);
        assert_eq!(result.len(), 5);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }
}
