/// Fourier decomposition via weighted linear least-squares with BIC model selection.
///
/// For a fixed period, the model
///   y(t) = offset + slope*(t - tmin) + Σ [An*cos(n*2π*t/p) + Bn*sin(n*2π*t/p)]
/// is linear in parameters.  We solve the normal equations directly via
/// Cholesky decomposition (max 12×12), avoiding any iterative optimizer.
///
/// Returns 14 features per curve:
///   [power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]
use rayon::prelude::*;

/// Maximum number of Fourier harmonics to try.
const MAX_HARMONICS: usize = 5;

/// Total number of output features per curve.
pub const NUM_FEATURES: usize = 14; // 2 (power, BIC) + 2 (offset, slope) + 2*MAX_HARMONICS

/// Solve a symmetric positive-definite system Ax = b in place via Cholesky.
///
/// `a` is the lower triangle of the matrix stored row-major in a flat array
/// of length `n*n`.  `b` has length `n`.  On success, `b` contains the
/// solution.  Returns `false` if the matrix is not positive-definite.
pub(crate) fn cholesky_solve(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    // Cholesky factorisation: A = L L^T  (in-place, lower triangle)
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return false;
                }
                a[i * n + j] = s.sqrt();
            } else {
                a[i * n + j] = s / a[j * n + j];
            }
        }
    }

    // Forward substitution: L z = b
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }

    // Back substitution: L^T x = z
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= a[k * n + i] * b[k];
        }
        b[i] = s / a[i * n + i];
    }

    true
}

/// Compute Fourier decomposition for a single light curve.
///
/// Returns an array of [`NUM_FEATURES`] f32 values.  If the input is
/// degenerate (< 3 points, zero/negative period, singular matrix) all
/// elements are NaN.
pub fn calc_fourier_single(
    times: &[f32],
    mags: &[f32],
    errs: &[f32],
    period: f32,
) -> [f32; NUM_FEATURES] {
    let nan_result = [f32::NAN; NUM_FEATURES];
    let n = times.len();

    if n < 3 || period <= 0.0 || !period.is_finite() {
        return nan_result;
    }

    // Find tmin for the slope term
    let tmin = times.iter().copied().fold(f32::INFINITY, f32::min);

    // Precompute weights and shifted times (in f64 for numerical stability)
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
    let y: Vec<f64> = mags.iter().map(|m| *m as f64).collect();
    let dt: Vec<f64> = times.iter().map(|t| (*t - tmin) as f64).collect();
    let phase_base: Vec<f64> = times
        .iter()
        .map(|t| 2.0 * std::f64::consts::PI * (*t as f64) / (period as f64))
        .collect();

    let n_f64 = n as f64;
    let ln_n = n_f64.ln();

    // We try models with k = 0..=MAX_HARMONICS harmonics.
    // Model k has n_params = 2 + 2*k parameters: [offset, slope, A1, B1, ..., Ak, Bk]
    // We store chi2 and the full 12-element parameter vector for each.
    let max_params = 2 + 2 * MAX_HARMONICS; // 12
    let num_models = MAX_HARMONICS + 1; // 6

    let mut chi2 = [0.0_f64; 6];
    let mut all_params = [[0.0_f64; 12]; 6];
    let mut model_ok = [false; 6];

    // Build the full design matrix columns for the biggest model.
    // Columns: [1, dt, cos(φ), sin(φ), cos(2φ), sin(2φ), ..., cos(5φ), sin(5φ)]
    // We store them column-major for convenience.
    let mut x_cols: Vec<Vec<f64>> = Vec::with_capacity(max_params);
    // Column 0: constant
    x_cols.push(vec![1.0; n]);
    // Column 1: slope
    x_cols.push(dt.clone());
    // Columns 2..max_params: Fourier terms
    for harm in 1..=MAX_HARMONICS {
        let cos_col: Vec<f64> = phase_base
            .iter()
            .map(|phi| (harm as f64 * phi).cos())
            .collect();
        let sin_col: Vec<f64> = phase_base
            .iter()
            .map(|phi| (harm as f64 * phi).sin())
            .collect();
        x_cols.push(cos_col);
        x_cols.push(sin_col);
    }

    for k in 0..num_models {
        let np_k = 2 + 2 * k; // number of parameters

        // Build normal equations: A = X^T W X, b = X^T W y
        let mut a = vec![0.0_f64; np_k * np_k];
        let mut b = vec![0.0_f64; np_k];

        for col_j in 0..np_k {
            for col_i in 0..=col_j {
                let mut s = 0.0_f64;
                for d in 0..n {
                    s += w[d] * x_cols[col_i][d] * x_cols[col_j][d];
                }
                a[col_j * np_k + col_i] = s;
                a[col_i * np_k + col_j] = s;
            }
            let mut s = 0.0_f64;
            for d in 0..n {
                s += w[d] * x_cols[col_j][d] * y[d];
            }
            b[col_j] = s;
        }

        // Solve via Cholesky
        if !cholesky_solve(&mut a, &mut b, np_k) {
            // Singular matrix — skip this model
            chi2[k] = f64::INFINITY;
            continue;
        }

        model_ok[k] = true;

        // Copy parameters
        for j in 0..np_k {
            all_params[k][j] = b[j];
        }

        // Compute chi2 = Σ ((y - X*beta) / err)^2
        let mut c2 = 0.0_f64;
        for d in 0..n {
            let mut pred = 0.0_f64;
            for j in 0..np_k {
                pred += b[j] * x_cols[j][d];
            }
            let resid = y[d] - pred;
            c2 += w[d] * resid * resid;
        }
        chi2[k] = c2;
    }

    // If even the constant model (k=0) failed, return NaN
    if !model_ok[0] {
        return nan_result;
    }

    // BIC model selection
    let mut best = 0;
    let mut best_bic = f64::INFINITY;
    for k in 0..num_models {
        if !model_ok[k] {
            continue;
        }
        let np_k = (2 + 2 * k) as f64;
        let bic = chi2[k] + ln_n * np_k;
        if bic < best_bic {
            best_bic = bic;
            best = k;
        }
    }

    let power = if chi2[0] > 0.0 {
        (chi2[0] - chi2[best]) / chi2[0]
    } else {
        0.0
    };

    let mut result = [0.0_f32; NUM_FEATURES];
    result[0] = power as f32;
    result[1] = best_bic as f32;
    // Copy the best model's parameters into slots 2..14
    // Pad with zeros for unused harmonics (already zeroed by initialization)
    for j in 0..max_params {
        result[2 + j] = all_params[best][j] as f32;
    }

    result
}

/// Compute Fourier decomposition for a batch of light curves in parallel.
///
/// Each curve gets its own period.  Returns a flat Vec of length
/// `n_curves * NUM_FEATURES`.
pub fn calc_fourier_batch(
    times_list: &[&[f32]],
    mags_list: &[&[f32]],
    errs_list: &[&[f32]],
    periods: &[f32],
) -> Vec<f32> {
    let n_curves = times_list.len();
    assert_eq!(n_curves, mags_list.len());
    assert_eq!(n_curves, errs_list.len());
    assert_eq!(n_curves, periods.len());

    let results: Vec<[f32; NUM_FEATURES]> = (0..n_curves)
        .into_par_iter()
        .map(|i| calc_fourier_single(times_list[i], mags_list[i], errs_list[i], periods[i]))
        .collect();

    // Flatten into a single contiguous Vec
    let mut flat = Vec::with_capacity(n_curves * NUM_FEATURES);
    for r in &results {
        flat.extend_from_slice(r);
    }
    flat
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: absolute error.
    fn abs_err(a: f32, b: f32) -> f32 {
        (a - b).abs()
    }

    #[test]
    fn test_cholesky_solver_identity() {
        // 2x2 identity — solution should be the rhs itself.
        let mut a = vec![1.0, 0.0, 0.0, 1.0];
        let mut b = vec![3.0, 7.0];
        assert!(cholesky_solve(&mut a, &mut b, 2));
        assert!((b[0] - 3.0).abs() < 1e-12);
        assert!((b[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_cholesky_solver_3x3() {
        // A = [[4,2,1],[2,5,3],[1,3,6]], b = [1,2,3]
        // numpy.linalg.solve gives: [0.08955224, 0.10447761, 0.43283582]
        let mut a = vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0];
        let mut b = vec![1.0, 2.0, 3.0];
        assert!(cholesky_solve(&mut a, &mut b, 3));
        assert!((b[0] - 0.08955224).abs() < 1e-6);
        assert!((b[1] - 0.10447761).abs() < 1e-6);
        assert!((b[2] - 0.43283582).abs() < 1e-6);
    }

    #[test]
    fn test_cholesky_singular() {
        // Singular matrix should fail.
        let mut a = vec![1.0, 1.0, 1.0, 1.0];
        let mut b = vec![1.0, 2.0];
        assert!(!cholesky_solve(&mut a, &mut b, 2));
    }

    #[test]
    fn test_constant_signal() {
        // Constant signal: power should be ~0, offset should match.
        let n = 50;
        let period = 1.0_f32;
        let mag_val = 18.5_f32;
        let err_val = 0.01_f32;

        let times: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mags: Vec<f32> = vec![mag_val; n];
        let errs: Vec<f32> = vec![err_val; n];

        let result = calc_fourier_single(&times, &mags, &errs, period);

        // Power should be essentially zero
        assert!(result[0].abs() < 0.01, "power = {}", result[0]);
        // Offset should be close to the mag value
        assert!(abs_err(result[2], mag_val) < 0.01, "offset = {}", result[2]);
        // Slope should be ~0
        assert!(result[3].abs() < 0.01, "slope = {}", result[3]);
    }

    #[test]
    fn test_pure_sine_recovery() {
        // y = 10.0 + 2.0 * cos(2πt/p) + 1.5 * sin(2πt/p)
        // Should recover: offset≈10, A1≈2, B1≈1.5, power > 0
        let n = 200;
        let period = 3.0_f32;
        let times: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
        let mags: Vec<f32> = times
            .iter()
            .map(|t| {
                let phi = 2.0 * std::f32::consts::PI * t / period;
                10.0 + 2.0 * phi.cos() + 1.5 * phi.sin()
            })
            .collect();
        let errs: Vec<f32> = vec![0.01; n];

        let result = calc_fourier_single(&times, &mags, &errs, period);

        // Power should be large (close to 1)
        assert!(result[0] > 0.9, "power = {}", result[0]);
        // Offset ≈ 10
        assert!(abs_err(result[2], 10.0) < 0.1, "offset = {}", result[2]);
        // Slope ≈ 0
        assert!(result[3].abs() < 0.01, "slope = {}", result[3]);
        // A1 ≈ 2.0
        assert!(abs_err(result[4], 2.0) < 0.1, "A1 = {}", result[4]);
        // B1 ≈ 1.5
        assert!(abs_err(result[5], 1.5) < 0.1, "B1 = {}", result[5]);
    }

    #[test]
    fn test_edge_cases() {
        // Empty input
        let r = calc_fourier_single(&[], &[], &[], 1.0);
        assert!(r[0].is_nan());

        // Single point
        let r = calc_fourier_single(&[1.0], &[2.0], &[0.1], 1.0);
        assert!(r[0].is_nan());

        // Two points
        let r = calc_fourier_single(&[1.0, 2.0], &[2.0, 3.0], &[0.1, 0.1], 1.0);
        assert!(r[0].is_nan());

        // Zero period
        let r = calc_fourier_single(&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0], &[0.1, 0.1, 0.1], 0.0);
        assert!(r[0].is_nan());

        // Negative period
        let r = calc_fourier_single(&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0], &[0.1, 0.1, 0.1], -1.0);
        assert!(r[0].is_nan());
    }

    #[test]
    fn test_batch_processing() {
        let n = 50;
        let period = 2.0_f32;
        let times: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mags1: Vec<f32> = vec![15.0; n];
        let mags2: Vec<f32> = times
            .iter()
            .map(|t| {
                let phi = 2.0 * std::f32::consts::PI * t / period;
                15.0 + 1.0 * phi.cos()
            })
            .collect();
        let errs: Vec<f32> = vec![0.01; n];

        let times_list: Vec<&[f32]> = vec![&times, &times];
        let mags_list: Vec<&[f32]> = vec![&mags1, &mags2];
        let errs_list: Vec<&[f32]> = vec![&errs, &errs];
        let periods = vec![period, period];

        let flat = calc_fourier_batch(&times_list, &mags_list, &errs_list, &periods);
        assert_eq!(flat.len(), 2 * NUM_FEATURES);

        // First curve (constant): low power
        assert!(flat[0].abs() < 0.01, "curve 0 power = {}", flat[0]);
        // Second curve (sinusoidal): high power
        assert!(
            flat[NUM_FEATURES] > 0.5,
            "curve 1 power = {}",
            flat[NUM_FEATURES]
        );
    }
}
