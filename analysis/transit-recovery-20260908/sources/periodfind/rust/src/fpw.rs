/// Fast Phase-folding Weighted (FPW) algorithm — CPU implementation.
///
/// Implements the FPW statistic from Finkbeiner et al. 2025, adapted to the
/// periodfind period/period_dt grid convention.
use rayon::prelude::*;

use crate::fold;

/// Compute the FPW statistic for a single light curve over a grid of
/// (period, period_dt) pairs.
///
/// Returns a flat Vec<f32> of length n_periods * n_pdts.
pub fn calc_fpw(
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

    // Precompute inverse variances and weighted data
    let ivar: Vec<f32> = errs.iter().map(|e| 1.0 / (e * e)).collect();
    let ivar_y: Vec<f32> = (0..n_data).map(|i| ivar[i] * mags[i]).collect();

    let total = n_periods * n_pdts;
    let results: Vec<f32> = (0..total)
        .into_par_iter()
        .map_init(
            || (vec![0.0_f32; num_bins], vec![0.0_f32; num_bins]),
            |(vtcinvv, ytcinvv), flat_idx| {
            let period_idx = flat_idx / n_pdts;
            let pdt_idx = flat_idx % n_pdts;

            let period = periods[period_idx];
            let period_dt = period_dts[pdt_idx];
            let pdt_corr = fold::pdt_correction(period, period_dt);

            // Zero reused bin accumulators
            vtcinvv.iter_mut().for_each(|x| *x = 0.0);
            ytcinvv.iter_mut().for_each(|x| *x = 0.0);

            for i in 0..n_data {
                let phase = fold::fold_time(times[i], period, pdt_corr);
                let bin = ((phase * num_bins as f32) as usize).min(num_bins - 1);

                vtcinvv[bin] += ivar[i];
                ytcinvv[bin] += ivar_y[i];
            }

            // Compute FPW statistic: S = sum_k (ytcinvv[k])^2 / (2 * vtcinvv[k])
            let mut delta_chi: f32 = 0.0;
            for k in 0..num_bins {
                if vtcinvv[k] > 0.0 {
                    delta_chi += ytcinvv[k] * ytcinvv[k] / (2.0 * vtcinvv[k]);
                }
            }

            delta_chi
        },)
        .collect();

    results
}
