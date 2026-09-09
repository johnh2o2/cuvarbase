/// Lomb-Scargle algorithm — CPU implementation matching CUDA kernel.
///
/// Translates LombScargleKernel from ls.cu.
use rayon::prelude::*;
use std::f32::consts::PI;

use crate::fold;

const TWO_PI: f32 = 2.0 * PI;

/// Compute Lomb-Scargle periodogram for a single light curve over a grid of
/// (period, period_dt) pairs.
///
/// Returns a flat Vec<f32> of length n_periods * n_pdts.
pub fn calc_ls(times: &[f32], mags: &[f32], periods: &[f32], period_dts: &[f32]) -> Vec<f32> {
    let n_periods = periods.len();
    let n_pdts = period_dts.len();
    let length = times.len();

    let total = n_periods * n_pdts;
    let results: Vec<f32> = (0..total)
        .into_par_iter()
        .map(|flat_idx| {
            let period_idx = flat_idx / n_pdts;
            let pdt_idx = flat_idx % n_pdts;

            let period = periods[period_idx];
            let period_dt = period_dts[pdt_idx];
            let pdt_corr = fold::pdt_correction(period, period_dt);

            let mut mag_cos = 0.0_f32;
            let mut mag_sin = 0.0_f32;
            let mut cos_cos = 0.0_f32;
            let mut cos_sin = 0.0_f32;

            for i in 0..length {
                let folded = fold::fold_time(times[i], period, pdt_corr);
                let angle = TWO_PI * folded;
                let (sin_val, cos_val) = angle.sin_cos();

                mag_cos += mags[i] * cos_val;
                mag_sin += mags[i] * sin_val;
                cos_cos += cos_val * cos_val;
                cos_sin += cos_val * sin_val;
            }

            let sin_sin = length as f32 - cos_cos;

            // Compute tau
            let tau_angle = 0.5 * (2.0 * cos_sin).atan2(cos_cos - sin_sin);
            let (sin_tau, cos_tau) = tau_angle.sin_cos();

            // Numerators
            let num_l = {
                let v = cos_tau * mag_cos + sin_tau * mag_sin;
                v * v
            };
            let num_r = {
                let v = cos_tau * mag_sin - sin_tau * mag_cos;
                v * v
            };

            // Denominators
            let den_l = cos_tau * cos_tau * cos_cos
                + 2.0 * cos_tau * sin_tau * cos_sin
                + sin_tau * sin_tau * sin_sin;
            let den_r = cos_tau * cos_tau * sin_sin - 2.0 * cos_tau * sin_tau * cos_sin
                + sin_tau * sin_tau * cos_cos;

            if den_l == 0.0 || den_r == 0.0 {
                0.0
            } else {
                0.5 * (num_l / den_l + num_r / den_r)
            }
        })
        .collect();

    results
}
