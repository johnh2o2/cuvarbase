use rayon::prelude::*;

pub const NUM_BASIC_STATS: usize = 22;

/// Compute the linear-interpolation percentile matching numpy's default method.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = q / 100.0 * (n as f64 - 1.0);
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    let frac = idx - lo as f64;
    if hi >= n {
        sorted[n - 1]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Weighted mean and weighted standard deviation.
fn weighted_mean_std(mag: &[f64], w: &[f64]) -> (f64, f64) {
    let sum_w: f64 = w.iter().sum();
    if sum_w == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let wmean: f64 = mag
        .iter()
        .zip(w.iter())
        .map(|(&m, &wi)| m * wi)
        .sum::<f64>()
        / sum_w;
    let var: f64 = mag
        .iter()
        .zip(w.iter())
        .map(|(&m, &wi)| wi * (m - wmean).powi(2))
        .sum::<f64>()
        / sum_w;
    (wmean, var.sqrt())
}

/// Normalised peak-to-peak amplitude.
fn calc_norm_peak_to_peak_amp(mag: &[f64], err: &[f64]) -> f64 {
    let max_me: f64 = mag
        .iter()
        .zip(err.iter())
        .map(|(&m, &e)| m - e)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_me: f64 = mag
        .iter()
        .zip(err.iter())
        .map(|(&m, &e)| m + e)
        .fold(f64::INFINITY, f64::min);
    let denom = max_me + min_me;
    if denom == 0.0 {
        f64::NAN
    } else {
        (max_me - min_me) / denom
    }
}

/// Normalised excess variance.
fn calc_norm_excess_var(mag: &[f64], err: &[f64], n: usize, wmean: f64) -> f64 {
    let denom = n as f64 * wmean * wmean;
    if denom == 0.0 {
        return f64::NAN;
    }
    let stat: f64 = mag
        .iter()
        .zip(err.iter())
        .map(|(&m, &e)| (m - wmean).powi(2) - e * e)
        .sum::<f64>();
    stat / denom
}

/// Small kurtosis.
fn calc_smallkurt(mag: &[f64], err: &[f64], n: usize, wmean: f64) -> f64 {
    let nf = n as f64;
    let coeff = nf * (nf + 1.0) / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0));
    let sum4: f64 = mag
        .iter()
        .zip(err.iter())
        .map(|(&m, &e)| {
            let e_safe = e.max(1e-30);
            ((m - wmean) / e_safe).powi(4)
        })
        .sum::<f64>();
    coeff * sum4 - 3.0 * (nf - 1.0).powi(2) / ((nf - 2.0) * (nf - 3.0))
}

/// Time-weighted inverse Von Neumann statistic.
fn calc_inv_neumann(times: &[f64], mags: &[f64], wstd: f64) -> f64 {
    if wstd == 0.0 || times.len() < 2 {
        return f64::NAN;
    }
    let mut num = 0.0;
    let mut sum_w = 0.0;
    for i in 0..(times.len() - 1) {
        let dt = times[i + 1] - times[i];
        let dm = mags[i + 1] - mags[i];
        if dt == 0.0 {
            continue;
        }
        let w = 1.0 / (dt * dt);
        num += w * dm * dm;
        sum_w += w;
    }
    if sum_w == 0.0 || num == 0.0 {
        return f64::NAN;
    }
    let eta = num / (sum_w * wstd * wstd);
    1.0 / eta
}

/// Welch/Stetson I, Stetson J, and Stetson K.
fn calc_stetson(mag: &[f64], err: &[f64], n: usize, wmean: f64) -> (f64, f64, f64) {
    let nf = n as f64;
    let scale = (nf / (nf - 1.0)).sqrt();

    // d_i = scale * (mag_i - wmean) / err_i
    let d: Vec<f64> = mag
        .iter()
        .zip(err.iter())
        .map(|(&m, &e)| {
            let e_safe = e.max(1e-30);
            scale * (m - wmean) / e_safe
        })
        .collect();

    // P_i = d_i * d_{i+1}
    let p: Vec<f64> = d.windows(2).map(|w| w[0] * w[1]).collect();

    // Stetson I = sum(P)
    let stetson_i: f64 = p.iter().sum();

    // Stetson J = sum(sign(P) * sqrt(|P|))
    let stetson_j: f64 = p.iter().map(|&pi| pi.signum() * pi.abs().sqrt()).sum();

    // Stetson K = sum(|d|)/N / sqrt(sum(d^2)/N)
    let sum_abs_d: f64 = d.iter().map(|&di| di.abs()).sum();
    let sum_d2: f64 = d.iter().map(|&di| di * di).sum();
    let denom = (sum_d2 / nf).sqrt();
    let stetson_k = if denom == 0.0 {
        f64::NAN
    } else {
        (sum_abs_d / nf) / denom
    };

    (stetson_i, stetson_j, stetson_k)
}

/// Anderson-Darling test statistic via the `normality` crate.
fn anderson_darling(data: &[f64]) -> f64 {
    if data.len() < 4 {
        return f64::NAN;
    }
    match normality::anderson_darling(data.iter().copied()) {
        Ok(result) => result.statistic,
        Err(_) => f64::NAN,
    }
}

/// Shapiro-Wilk test statistic via the `normality` crate.
fn shapiro_wilk(data: &[f64]) -> f64 {
    if data.len() < 4 || data.len() > 5000 {
        return f64::NAN;
    }
    match normality::shapiro_wilk(data.iter().copied()) {
        Ok(result) => result.statistic,
        Err(_) => f64::NAN,
    }
}

/// Compute 22 basic light curve statistics for a single source.
///
/// All intermediate math uses f64 for stability; the final result is f32.
///
/// Returns NaN array if N < 4.
///
/// Statistics order:
///   0: N, 1: median, 2: wmean, 3: chi2red, 4: RoMS, 5: wstd,
///   6: NormPeaktoPeakamp, 7: NormExcessVar, 8: medianAbsDev, 9: iqr,
///   10: i60r, 11: i70r, 12: i80r, 13: i90r,
///   14: skew, 15: smallkurt, 16: invNeumann,
///   17: WelchI, 18: StetsonJ, 19: StetsonK, 20: AD, 21: SW
pub fn calc_basic_stats_single(
    times: &[f32],
    mags: &[f32],
    errs: &[f32],
) -> [f32; NUM_BASIC_STATS] {
    let n = mags.len();

    if n < 4 {
        return [f32::NAN; NUM_BASIC_STATS];
    }

    // Convert to f64
    let t64: Vec<f64> = times.iter().map(|&v| v as f64).collect();
    let m64: Vec<f64> = mags.iter().map(|&v| v as f64).collect();
    let e64: Vec<f64> = errs.iter().map(|&v| (v as f64).max(1e-30)).collect();

    // Weights
    let w: Vec<f64> = e64.iter().map(|&e| 1.0 / (e * e)).collect();
    let (wmean, wstd) = weighted_mean_std(&m64, &w);

    // Sorted magnitudes for percentile calculations
    let mut sorted_mag = m64.clone();
    sorted_mag.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let nf = n as f64;
    let median = percentile(&sorted_mag, 50.0);

    // chi2red
    let chi2red: f64 = m64
        .iter()
        .zip(w.iter())
        .map(|(&m, &wi)| (wmean - m).powi(2) * wi)
        .sum::<f64>()
        / (nf - 1.0);

    // RoMS
    let roms: f64 = m64
        .iter()
        .zip(e64.iter())
        .map(|(&m, &e)| (m - median).abs() / e)
        .sum::<f64>()
        / (nf - 1.0);

    // Deviation from median
    let norm_peak_to_peak = calc_norm_peak_to_peak_amp(&m64, &e64);
    let norm_excess_var = calc_norm_excess_var(&m64, &e64, n, wmean);

    let mut abs_dev: Vec<f64> = m64.iter().map(|&m| (m - median).abs()).collect();
    abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_abs_dev = percentile(&abs_dev, 50.0);

    let iqr = percentile(&sorted_mag, 75.0) - percentile(&sorted_mag, 25.0);
    let i60r = percentile(&sorted_mag, 80.0) - percentile(&sorted_mag, 20.0);
    let i70r = percentile(&sorted_mag, 85.0) - percentile(&sorted_mag, 15.0);
    let i80r = percentile(&sorted_mag, 90.0) - percentile(&sorted_mag, 10.0);
    let i90r = percentile(&sorted_mag, 95.0) - percentile(&sorted_mag, 5.0);

    // Skew
    let skew: f64 = {
        let coeff = nf / ((nf - 1.0) * (nf - 2.0));
        let sum3: f64 = m64
            .iter()
            .zip(e64.iter())
            .map(|(&m, &e)| ((m - wmean) / e).powi(3))
            .sum::<f64>();
        coeff * sum3
    };

    let smallkurt = calc_smallkurt(&m64, &e64, n, wmean);
    let inv_neumann = calc_inv_neumann(&t64, &m64, wstd);
    let (welch_i, stetson_j, stetson_k) = calc_stetson(&m64, &e64, n, wmean);

    // Anderson-Darling and Shapiro-Wilk on mag/err
    let mag_over_err: Vec<f64> = m64.iter().zip(e64.iter()).map(|(&m, &e)| m / e).collect();
    let ad = anderson_darling(&mag_over_err);
    let sw = shapiro_wilk(&mag_over_err);

    [
        nf as f32,
        median as f32,
        wmean as f32,
        chi2red as f32,
        roms as f32,
        wstd as f32,
        norm_peak_to_peak as f32,
        norm_excess_var as f32,
        median_abs_dev as f32,
        iqr as f32,
        i60r as f32,
        i70r as f32,
        i80r as f32,
        i90r as f32,
        skew as f32,
        smallkurt as f32,
        inv_neumann as f32,
        welch_i as f32,
        stetson_j as f32,
        stetson_k as f32,
        ad as f32,
        sw as f32,
    ]
}

/// Batch basic statistics computation across multiple light curves using Rayon.
///
/// Returns a flat `Vec<f32>` of length `n_curves * NUM_BASIC_STATS`.
pub fn calc_basic_stats_batch(
    times_list: &[&[f32]],
    mags_list: &[&[f32]],
    errs_list: &[&[f32]],
) -> Vec<f32> {
    let results: Vec<[f32; NUM_BASIC_STATS]> = (0..times_list.len())
        .into_par_iter()
        .map(|i| calc_basic_stats_single(times_list[i], mags_list[i], errs_list[i]))
        .collect();

    let mut flat = Vec::with_capacity(results.len() * NUM_BASIC_STATS);
    for r in results {
        flat.extend_from_slice(&r);
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 0.0) - 1.0).abs() < 1e-10);
        assert!((percentile(&data, 50.0) - 3.0).abs() < 1e-10);
        assert!((percentile(&data, 100.0) - 5.0).abs() < 1e-10);
        assert!((percentile(&data, 25.0) - 2.0).abs() < 1e-10);
        assert!((percentile(&data, 75.0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_too_few_points() {
        let t = vec![1.0f32, 2.0, 3.0];
        let m = vec![10.0, 11.0, 12.0];
        let e = vec![0.1, 0.1, 0.1];
        let result = calc_basic_stats_single(&t, &m, &e);
        for &v in &result {
            assert!(v.is_nan());
        }
    }

    #[test]
    fn test_basic_stats_known() {
        // Simple test: 5 points with known values
        let t = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let m = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let e = vec![0.1, 0.1, 0.1, 0.1, 0.1];

        let result = calc_basic_stats_single(&t, &m, &e);

        // N = 5
        assert!((result[0] - 5.0).abs() < 1e-5);
        // median = 12.0
        assert!((result[1] - 12.0).abs() < 1e-3);
        // All values should be finite (except AD/SW which may be NaN for tiny samples)
        for i in 0..NUM_BASIC_STATS {
            if i == 20 || i == 21 {
                // AD and SW may return NaN for very small samples via normality crate
                continue;
            }
            assert!(
                result[i].is_finite(),
                "stat[{}] is not finite: {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_constant_magnitude() {
        let t = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let m = vec![15.0, 15.0, 15.0, 15.0, 15.0];
        let e = vec![0.1, 0.1, 0.1, 0.1, 0.1];

        let result = calc_basic_stats_single(&t, &m, &e);

        // N = 5
        assert!((result[0] - 5.0).abs() < 1e-5);
        // median = 15.0
        assert!((result[1] - 15.0).abs() < 1e-5);
        // wmean = 15.0
        assert!((result[2] - 15.0).abs() < 1e-5);
        // chi2red = 0 (no scatter)
        assert!(result[3].abs() < 1e-5);
        // wstd = 0
        assert!(result[5].abs() < 1e-5);
        // iqr = 0
        assert!(result[9].abs() < 1e-5);
    }

    #[test]
    fn test_batch_basic_stats() {
        let t1 = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let m1 = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let e1 = vec![0.1, 0.1, 0.1, 0.1, 0.1];

        let t2 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let m2 = vec![20.0, 20.5, 21.0, 20.0, 20.5, 21.0];
        let e2 = vec![0.2, 0.2, 0.2, 0.2, 0.2, 0.2];

        let flat = calc_basic_stats_batch(&[&t1, &t2], &[&m1, &m2], &[&e1, &e2]);

        assert_eq!(flat.len(), 2 * NUM_BASIC_STATS);
        // First curve N = 5
        assert!((flat[0] - 5.0).abs() < 1e-5);
        // Second curve N = 6
        assert!((flat[NUM_BASIC_STATS] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_anderson_darling_normal_data() {
        // Generate approximate normal data (0, 1, 2, ... pattern)
        let data: Vec<f64> = (0..50).map(|i| -2.0 + 4.0 * (i as f64) / 49.0).collect();
        let ad = anderson_darling(&data);
        assert!(ad.is_finite());
        // For uniform-ish data the statistic should be moderate
        assert!(ad > 0.0);
    }

    #[test]
    fn test_shapiro_wilk_returns_finite() {
        let data: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let sw = shapiro_wilk(&data);
        assert!(sw.is_finite());
        assert!(sw > 0.0 && sw <= 1.0);
    }

    #[test]
    fn test_anderson_darling_and_shapiro_wilk_via_normality() {
        // Verify the normality crate wrappers work for our use case
        let data: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let ad = anderson_darling(&data);
        let sw = shapiro_wilk(&data);
        assert!(ad.is_finite(), "AD should be finite");
        assert!(sw.is_finite(), "SW should be finite");
        assert!(sw > 0.0 && sw <= 1.0, "SW should be in (0, 1], got {}", sw);
    }
}
