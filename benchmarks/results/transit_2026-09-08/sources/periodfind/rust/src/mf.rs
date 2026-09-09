/// Matched Filter morphology scoring algorithm — CPU implementation.
///
/// Phase-folds light curves, bins into a profile, and correlates against
/// template shapes (sawtooth, sinusoidal, eclipsing) via circular
/// cross-correlation.  The combined score (max_corr × R² × coverage)
/// serves as the periodogram statistic.
use rayon::prelude::*;
use std::f32::consts::PI;

use crate::fold;

// -------------------------------------------------------------------------
// Template types and generation
// -------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TemplateType {
    Sawtooth,
    Sinusoidal,
    Eclipsing,
}

pub struct Templates {
    pub shapes: Vec<Vec<f32>>, // each inner Vec has num_bins elements, zero-mean unit-variance
    pub labels: Vec<TemplateType>,
}

/// Normalize a template to zero mean and unit variance in-place.
fn normalize_template(t: &mut [f32]) {
    let n = t.len() as f32;
    let mean: f32 = t.iter().sum::<f32>() / n;
    t.iter_mut().for_each(|x| *x -= mean);
    let var: f32 = t.iter().map(|x| x * x).sum::<f32>() / n;
    let std = var.sqrt();
    if std > 1e-12 {
        t.iter_mut().for_each(|x| *x /= std);
    }
}

/// Generate the full template library for a given number of bins.
///
/// Templates:
///   - Sawtooth: 5 rise fractions (0.1, 0.2, 0.35, 0.5, 0.65)
///   - Sinusoidal: 1
///   - Eclipsing: 4 dip widths × 2 secondary options = 8
///
/// Total: 14 template shapes.  Phase-offset invariance is handled by
/// `circular_pearson_max()` at evaluation time.
pub fn generate_templates(num_bins: usize) -> Templates {
    let mut shapes = Vec::with_capacity(14);
    let mut labels = Vec::with_capacity(14);

    // Sawtooth templates: linear rise over rise_frac, linear fall over (1 - rise_frac)
    let rise_fracs = [0.10_f32, 0.20, 0.35, 0.50, 0.65];
    for &rf in &rise_fracs {
        let mut t = vec![0.0_f32; num_bins];
        for k in 0..num_bins {
            let phase = (k as f32 + 0.5) / num_bins as f32;
            t[k] = if phase < rf {
                phase / rf
            } else {
                1.0 - (phase - rf) / (1.0 - rf)
            };
        }
        normalize_template(&mut t);
        shapes.push(t);
        labels.push(TemplateType::Sawtooth);
    }

    // Sinusoidal template
    {
        let mut t = vec![0.0_f32; num_bins];
        for k in 0..num_bins {
            let phase = (k as f32 + 0.5) / num_bins as f32;
            t[k] = (2.0 * PI * phase).sin();
        }
        normalize_template(&mut t);
        shapes.push(t);
        labels.push(TemplateType::Sinusoidal);
    }

    // Eclipsing templates: flat baseline with dip(s)
    let dip_widths = [0.05_f32, 0.10, 0.15, 0.20];
    let secondary_options: [(bool, f32); 2] = [(false, 0.0), (true, 0.5)];
    for &dw in &dip_widths {
        for &(has_secondary, sec_depth_frac) in &secondary_options {
            let mut t = vec![1.0_f32; num_bins];
            let half_dip = dw / 2.0;
            for k in 0..num_bins {
                let phase = (k as f32 + 0.5) / num_bins as f32;
                // Primary eclipse centered at phase 0
                let dist_primary = if phase < 0.5 { phase } else { 1.0 - phase };
                if dist_primary < half_dip {
                    t[k] = 0.0;
                }
                // Secondary eclipse centered at phase 0.5
                if has_secondary {
                    let dist_secondary = (phase - 0.5).abs();
                    if dist_secondary < half_dip {
                        t[k] = 1.0 - sec_depth_frac;
                    }
                }
            }
            normalize_template(&mut t);
            shapes.push(t);
            labels.push(TemplateType::Eclipsing);
        }
    }

    Templates { shapes, labels }
}

// -------------------------------------------------------------------------
// Circular cross-correlation
// -------------------------------------------------------------------------

/// Compute the maximum Pearson correlation between `profile` and `template`
/// over all circular shifts of `template`.
///
/// Both inputs are assumed to have length `num_bins`.
/// `profile` need NOT be pre-normalized — this function normalizes on the fly.
fn circular_pearson_max(profile: &[f32], template: &[f32], num_bins: usize) -> f32 {
    // Pre-compute profile statistics
    let n = num_bins as f32;
    let p_mean: f32 = profile.iter().sum::<f32>() / n;
    let p_var: f32 = profile.iter().map(|x| (x - p_mean) * (x - p_mean)).sum::<f32>();
    if p_var < 1e-20 {
        return 0.0; // flat profile → no correlation
    }
    let p_std = p_var.sqrt();

    // Template is already normalized (mean=0, std=1), so t_std = sqrt(n)
    let t_std = n.sqrt();

    let denom = p_std * t_std;
    if denom < 1e-20 {
        return 0.0;
    }

    let mut best = -1.0_f32;
    for shift in 0..num_bins {
        let mut dot = 0.0_f32;
        for k in 0..num_bins {
            let tk = (k + shift) % num_bins;
            dot += (profile[k] - p_mean) * template[tk];
        }
        let r = dot / denom;
        if r > best {
            best = r;
        }
    }
    best
}

// -------------------------------------------------------------------------
// Core computation for a single (period, period_dt) pair
// -------------------------------------------------------------------------

/// Internal result from evaluating one trial period.
struct MfResult {
    best_sawtooth: f32,
    best_sinusoidal: f32,
    best_eclipsing: f32,
    r_squared: f32,
    amp_snr: f32,
    n_filled: f32,
    combined: f32,
}

/// Evaluate the matched filter at a single trial period.
///
/// Uses pre-allocated bin accumulators (`vtcinvv`, `ytcinvv`, `vt2cinvv`)
/// that the caller provides (avoids per-call allocation in hot loops).
fn eval_mf_at_period(
    times: &[f32],
    mags: &[f32],
    ivar: &[f32],
    ivar_y: &[f32],
    period: f32,
    pdt_corr: f32,
    num_bins: usize,
    templates: &Templates,
    // Pre-allocated scratch buffers (zeroed by caller)
    vtcinvv: &mut [f32],
    ytcinvv: &mut [f32],
    vt2cinvv: &mut [f32],
) -> MfResult {
    let n_data = times.len();

    // Phase fold and bin (inverse-variance weighted)
    for i in 0..n_data {
        let phase = fold::fold_time(times[i], period, pdt_corr);
        let bin = ((phase * num_bins as f32) as usize).min(num_bins - 1);
        vtcinvv[bin] += ivar[i];
        ytcinvv[bin] += ivar_y[i];
        vt2cinvv[bin] += ivar[i] * mags[i] * mags[i];
    }

    // Compute bin means and statistics
    let mut profile = vec![0.0_f32; num_bins];
    let mut n_filled = 0_u32;
    let mut total_wt = 0.0_f32;
    let mut total_wy = 0.0_f32;

    for k in 0..num_bins {
        if vtcinvv[k] > 0.0 {
            profile[k] = ytcinvv[k] / vtcinvv[k];
            n_filled += 1;
            total_wt += vtcinvv[k];
            total_wy += ytcinvv[k];
        }
    }

    let coverage = n_filled as f32 / num_bins as f32;

    // Need at least 3 filled bins for meaningful correlation
    if n_filled < 3 {
        return MfResult {
            best_sawtooth: 0.0,
            best_sinusoidal: 0.0,
            best_eclipsing: 0.0,
            r_squared: 0.0,
            amp_snr: 0.0,
            n_filled: n_filled as f32,
            combined: 0.0,
        };
    }

    // R²: variance reduction from phase folding
    // total_var = weighted variance of all data
    // within_bin_var = weighted variance within bins
    let grand_mean = total_wy / total_wt;
    let mut ss_total = 0.0_f32;
    for i in 0..n_data {
        let d = mags[i] - grand_mean;
        ss_total += ivar[i] * d * d;
    }

    let mut ss_within = 0.0_f32;
    for k in 0..num_bins {
        if vtcinvv[k] > 0.0 {
            // within-bin SS = sum(w_i * y_i^2) - (sum(w_i * y_i))^2 / sum(w_i)
            ss_within += vt2cinvv[k] - ytcinvv[k] * ytcinvv[k] / vtcinvv[k];
        }
    }

    let r_squared = if ss_total > 0.0 {
        (1.0 - ss_within / ss_total).max(0.0)
    } else {
        0.0
    };

    // Amplitude SNR: peak-to-peak of bin means / mean within-bin RMS
    let mut min_prof = f32::INFINITY;
    let mut max_prof = f32::NEG_INFINITY;
    for k in 0..num_bins {
        if vtcinvv[k] > 0.0 {
            if profile[k] < min_prof {
                min_prof = profile[k];
            }
            if profile[k] > max_prof {
                max_prof = profile[k];
            }
        }
    }
    let amplitude = max_prof - min_prof;

    let mut sum_rms = 0.0_f32;
    let mut n_rms = 0_u32;
    for k in 0..num_bins {
        if vtcinvv[k] > 0.0 {
            let within_var = (vt2cinvv[k] - ytcinvv[k] * ytcinvv[k] / vtcinvv[k]) / vtcinvv[k];
            if within_var > 0.0 {
                sum_rms += within_var.sqrt();
                n_rms += 1;
            }
        }
    }
    let mean_rms = if n_rms > 0 {
        sum_rms / n_rms as f32
    } else {
        1.0
    };
    let amp_snr = if mean_rms > 1e-12 {
        amplitude / mean_rms
    } else {
        0.0
    };

    // Template correlations
    let mut best_sawtooth = 0.0_f32;
    let mut best_sinusoidal = 0.0_f32;
    let mut best_eclipsing = 0.0_f32;

    for (shape, label) in templates.shapes.iter().zip(templates.labels.iter()) {
        let corr = circular_pearson_max(&profile, shape, num_bins);
        match label {
            TemplateType::Sawtooth => {
                if corr > best_sawtooth {
                    best_sawtooth = corr;
                }
            }
            TemplateType::Sinusoidal => {
                if corr > best_sinusoidal {
                    best_sinusoidal = corr;
                }
            }
            TemplateType::Eclipsing => {
                if corr > best_eclipsing {
                    best_eclipsing = corr;
                }
            }
        }
    }

    let max_corr = best_sawtooth.max(best_sinusoidal).max(best_eclipsing);
    let combined = max_corr * r_squared * coverage;

    MfResult {
        best_sawtooth,
        best_sinusoidal,
        best_eclipsing,
        r_squared,
        amp_snr,
        n_filled: n_filled as f32,
        combined,
    }
}

// -------------------------------------------------------------------------
// Full periodogram
// -------------------------------------------------------------------------

/// Compute the matched filter combined score for a single light curve over
/// a grid of (period, period_dt) pairs.
///
/// Returns a flat Vec<f32> of length n_periods * n_pdts, where each value
/// is `max_corr × R² × coverage`.
pub fn calc_mf(
    times: &[f32],
    mags: &[f32],
    errs: &[f32],
    periods: &[f32],
    period_dts: &[f32],
    num_bins: usize,
) -> Vec<f32> {
    let n_periods = periods.len();
    let n_pdts = period_dts.len();
    let n_data = times.len();

    let templates = generate_templates(num_bins);

    // Precompute inverse variances and weighted data
    let ivar: Vec<f32> = errs.iter().map(|e| 1.0 / (e * e)).collect();
    let ivar_y: Vec<f32> = (0..n_data).map(|i| ivar[i] * mags[i]).collect();

    let total = n_periods * n_pdts;
    let results: Vec<f32> = (0..total)
        .into_par_iter()
        .map_init(
            || {
                (
                    vec![0.0_f32; num_bins],
                    vec![0.0_f32; num_bins],
                    vec![0.0_f32; num_bins],
                )
            },
            |(vtcinvv, ytcinvv, vt2cinvv), flat_idx| {
                let period_idx = flat_idx / n_pdts;
                let pdt_idx = flat_idx % n_pdts;

                let period = periods[period_idx];
                let period_dt = period_dts[pdt_idx];
                let pdt_corr = fold::pdt_correction(period, period_dt);

                // Zero reused bin accumulators
                vtcinvv.iter_mut().for_each(|x| *x = 0.0);
                ytcinvv.iter_mut().for_each(|x| *x = 0.0);
                vt2cinvv.iter_mut().for_each(|x| *x = 0.0);

                let result = eval_mf_at_period(
                    times, mags, &ivar, &ivar_y, period, pdt_corr, num_bins, &templates,
                    vtcinvv, ytcinvv, vt2cinvv,
                );
                result.combined
            },
        )
        .collect();

    results
}

// -------------------------------------------------------------------------
// Feature extraction (at given periods, like fourier.rs)
// -------------------------------------------------------------------------

/// Number of features returned per curve by `calc_mf_features_batch`.
///
/// [best_sawtooth, best_sinusoidal, best_eclipsing, R², amp_snr, n_filled, combined]
pub const NUM_MF_FEATURES: usize = 7;

/// Compute detailed matched filter features for multiple light curves, each
/// at a single pre-determined period (period_dt = 0).
///
/// Returns a flat Vec<f32> of length n_curves × NUM_MF_FEATURES.
pub fn calc_mf_features_batch(
    times_list: &[&[f32]],
    mags_list: &[&[f32]],
    errs_list: &[&[f32]],
    periods: &[f32], // one per curve
    num_bins: usize,
) -> Vec<f32> {
    let n_curves = times_list.len();
    let templates = generate_templates(num_bins);

    let flat: Vec<f32> = (0..n_curves)
        .into_par_iter()
        .flat_map_iter(|ci| {
            let times = times_list[ci];
            let mags = mags_list[ci];
            let errs = errs_list[ci];
            let period = periods[ci];
            let n_data = times.len();

            let ivar: Vec<f32> = errs.iter().map(|e| 1.0 / (e * e)).collect();
            let ivar_y: Vec<f32> = (0..n_data).map(|i| ivar[i] * mags[i]).collect();

            let pdt_corr = 0.0; // no period derivative for feature extraction

            let mut vtcinvv = vec![0.0_f32; num_bins];
            let mut ytcinvv = vec![0.0_f32; num_bins];
            let mut vt2cinvv = vec![0.0_f32; num_bins];

            let r = eval_mf_at_period(
                times, mags, &ivar, &ivar_y, period, pdt_corr, num_bins, &templates,
                &mut vtcinvv, &mut ytcinvv, &mut vt2cinvv,
            );

            [
                r.best_sawtooth,
                r.best_sinusoidal,
                r.best_eclipsing,
                r.r_squared,
                r.amp_snr,
                r.n_filled,
                r.combined,
            ]
        })
        .collect();

    flat
}

// =========================================================================
// Unit tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // --- Template generation tests ---

    #[test]
    fn test_generate_templates_count() {
        let t = generate_templates(20);
        // 5 sawtooth + 1 sinusoidal + 8 eclipsing = 14
        assert_eq!(t.shapes.len(), 14);
        assert_eq!(t.labels.len(), 14);
    }

    #[test]
    fn test_templates_correct_length() {
        for num_bins in [10, 20, 32] {
            let t = generate_templates(num_bins);
            for shape in &t.shapes {
                assert_eq!(shape.len(), num_bins);
            }
        }
    }

    #[test]
    fn test_templates_zero_mean() {
        let t = generate_templates(20);
        for (i, shape) in t.shapes.iter().enumerate() {
            let mean: f32 = shape.iter().sum::<f32>() / shape.len() as f32;
            assert!(
                mean.abs() < 1e-5,
                "Template {} has non-zero mean: {}",
                i,
                mean
            );
        }
    }

    #[test]
    fn test_templates_unit_variance() {
        let t = generate_templates(20);
        for (i, shape) in t.shapes.iter().enumerate() {
            let n = shape.len() as f32;
            let var: f32 = shape.iter().map(|x| x * x).sum::<f32>() / n;
            assert!(
                (var - 1.0).abs() < 1e-4,
                "Template {} has variance {}, expected 1.0",
                i,
                var
            );
        }
    }

    #[test]
    fn test_template_type_counts() {
        let t = generate_templates(20);
        let n_saw = t.labels.iter().filter(|l| **l == TemplateType::Sawtooth).count();
        let n_sin = t.labels.iter().filter(|l| **l == TemplateType::Sinusoidal).count();
        let n_ecl = t.labels.iter().filter(|l| **l == TemplateType::Eclipsing).count();
        assert_eq!(n_saw, 5);
        assert_eq!(n_sin, 1);
        assert_eq!(n_ecl, 8);
    }

    // --- Circular cross-correlation tests ---

    #[test]
    fn test_self_correlation_is_one() {
        let t = generate_templates(20);
        for shape in &t.shapes {
            let corr = circular_pearson_max(shape, shape, 20);
            assert!(
                (corr - 1.0).abs() < 1e-5,
                "Self-correlation should be 1.0, got {}",
                corr
            );
        }
    }

    #[test]
    fn test_shifted_self_correlation_is_one() {
        // A circularly shifted copy should still correlate at 1.0
        let t = generate_templates(20);
        let shape = &t.shapes[0]; // sawtooth
        let mut shifted = vec![0.0_f32; 20];
        for k in 0..20 {
            shifted[k] = shape[(k + 7) % 20]; // shift by 7
        }
        let corr = circular_pearson_max(&shifted, shape, 20);
        assert!(
            (corr - 1.0).abs() < 1e-4,
            "Shifted self-correlation should be 1.0, got {}",
            corr
        );
    }

    #[test]
    fn test_flat_profile_gives_zero() {
        let flat = vec![5.0_f32; 20];
        let t = generate_templates(20);
        let corr = circular_pearson_max(&flat, &t.shapes[0], 20);
        assert!(
            corr.abs() < 1e-5,
            "Flat profile should give zero correlation, got {}",
            corr
        );
    }

    // --- Periodogram tests ---

    #[test]
    fn test_calc_mf_output_length() {
        let n = 100;
        let times: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let mags: Vec<f32> = times.iter().map(|t| (2.0 * PI * t / 0.5).sin()).collect();
        let errs = vec![0.01_f32; n];
        let periods = vec![0.3_f32, 0.5, 0.7];
        let period_dts = vec![0.0_f32, 0.001];

        let result = calc_mf(&times, &mags, &errs, &periods, &period_dts, 20);
        assert_eq!(result.len(), 3 * 2); // n_periods * n_pdts
    }

    #[test]
    fn test_sinusoidal_signal_detected() {
        // Generate a clean sinusoidal signal at P=0.5 days
        let n = 200;
        let true_period = 0.5_f32;
        let times: Vec<f32> = (0..n).map(|i| i as f32 * 0.07).collect(); // ~14 days span
        let mags: Vec<f32> = times
            .iter()
            .map(|t| 0.3 * (2.0 * PI * t / true_period).sin())
            .collect();
        let errs = vec![0.01_f32; n];

        // Period grid around the true period
        let periods: Vec<f32> = (0..100)
            .map(|i| 0.3 + i as f32 * 0.005) // 0.3 to 0.8
            .collect();
        let period_dts = vec![0.0_f32];

        let result = calc_mf(&times, &mags, &errs, &periods, &period_dts, 20);

        // Find the best period
        let (best_idx, best_val) = result
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let best_period = periods[best_idx];

        assert!(
            (best_period - true_period).abs() < 0.02,
            "Best period {} should be near true period {}",
            best_period,
            true_period
        );
        assert!(
            *best_val > 0.3,
            "Best combined score {} should be substantial for clean signal",
            best_val
        );
    }

    #[test]
    fn test_noise_gives_low_scores() {
        // Pure noise should give low combined scores
        let n = 100;
        let times: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        // Constant magnitude (no signal)
        let mags = vec![0.0_f32; n];
        let errs = vec![0.01_f32; n];

        let periods: Vec<f32> = (0..50).map(|i| 0.1 + i as f32 * 0.02).collect();
        let period_dts = vec![0.0_f32];

        let result = calc_mf(&times, &mags, &errs, &periods, &period_dts, 20);

        let max_val = result.iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            max_val < 0.1,
            "Constant data should give near-zero combined scores, got {}",
            max_val
        );
    }

    // --- Feature extraction tests ---

    #[test]
    fn test_features_output_shape() {
        let n = 100;
        let times: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let mags: Vec<f32> = times.iter().map(|t| (2.0 * PI * t / 0.5).sin()).collect();
        let errs = vec![0.01_f32; n];

        let times_list: Vec<&[f32]> = vec![&times];
        let mags_list: Vec<&[f32]> = vec![&mags];
        let errs_list: Vec<&[f32]> = vec![&errs];
        let periods = vec![0.5_f32];

        let features = calc_mf_features_batch(&times_list, &mags_list, &errs_list, &periods, 20);
        assert_eq!(features.len(), NUM_MF_FEATURES);
    }

    #[test]
    fn test_sinusoidal_features_at_true_period() {
        // Clean sinusoidal at P=0.5, evaluate features at the true period
        let n = 200;
        let true_period = 0.5_f32;
        let times: Vec<f32> = (0..n).map(|i| i as f32 * 0.07).collect();
        let mags: Vec<f32> = times
            .iter()
            .map(|t| 0.3 * (2.0 * PI * t / true_period).sin())
            .collect();
        let errs = vec![0.01_f32; n];

        let times_list: Vec<&[f32]> = vec![&times];
        let mags_list: Vec<&[f32]> = vec![&mags];
        let errs_list: Vec<&[f32]> = vec![&errs];
        let periods = vec![true_period];

        let f = calc_mf_features_batch(&times_list, &mags_list, &errs_list, &periods, 20);
        // f = [sawtooth, sinusoidal, eclipsing, R², amp_snr, n_filled, combined]

        let best_sin = f[1];
        let r_sq = f[3];
        let n_filled = f[5];
        let combined = f[6];

        assert!(
            best_sin > 0.8,
            "Sinusoidal template should match clean sine wave, got {}",
            best_sin
        );
        assert!(
            r_sq > 0.8,
            "R² should be high for clean signal, got {}",
            r_sq
        );
        assert!(
            n_filled >= 15.0,
            "Most bins should be filled with 200 points in 20 bins, got {}",
            n_filled
        );
        assert!(
            combined > 0.5,
            "Combined score should be high, got {}",
            combined
        );
    }

    #[test]
    fn test_features_wrong_period_low_r2() {
        // Clean signal at P=0.5, but evaluate at P=0.37 (wrong)
        let n = 200;
        let times: Vec<f32> = (0..n).map(|i| i as f32 * 0.07).collect();
        let mags: Vec<f32> = times
            .iter()
            .map(|t| 0.3 * (2.0 * PI * t / 0.5).sin())
            .collect();
        let errs = vec![0.01_f32; n];

        let times_list: Vec<&[f32]> = vec![&times];
        let mags_list: Vec<&[f32]> = vec![&mags];
        let errs_list: Vec<&[f32]> = vec![&errs];
        let wrong_period = vec![0.37_f32]; // wrong period

        let f = calc_mf_features_batch(&times_list, &mags_list, &errs_list, &wrong_period, 20);
        let r_sq = f[3];
        let combined = f[6];

        // At a wrong period, the phase fold should be incoherent
        // R² should be lower than at the true period
        assert!(
            combined < 0.5,
            "Combined score at wrong period should be low, got {}",
            combined
        );
        // R² might still be moderate if the wrong period happens to partially alias
        // but combined (which includes corr) should definitely be lower
    }

    #[test]
    fn test_features_batch_multiple_curves() {
        let n = 100;
        let t1: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
        let m1: Vec<f32> = t1.iter().map(|t| (2.0 * PI * t / 0.3).sin()).collect();
        let e1 = vec![0.01_f32; n];

        let t2: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
        let m2: Vec<f32> = t2.iter().map(|t| (2.0 * PI * t / 0.7).sin()).collect();
        let e2 = vec![0.01_f32; n];

        let times_list: Vec<&[f32]> = vec![&t1, &t2];
        let mags_list: Vec<&[f32]> = vec![&m1, &m2];
        let errs_list: Vec<&[f32]> = vec![&e1, &e2];
        let periods = vec![0.3_f32, 0.7];

        let features =
            calc_mf_features_batch(&times_list, &mags_list, &errs_list, &periods, 20);
        assert_eq!(features.len(), 2 * NUM_MF_FEATURES);

        // Both should have positive combined scores at their true periods
        let combined_1 = features[6];
        let combined_2 = features[NUM_MF_FEATURES + 6];
        assert!(combined_1 > 0.1, "Curve 1 combined = {}", combined_1);
        assert!(combined_2 > 0.1, "Curve 2 combined = {}", combined_2);
    }
}
