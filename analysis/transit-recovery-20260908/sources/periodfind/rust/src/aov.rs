/// Analysis of Variance algorithm — CPU implementation matching CUDA kernels.
///
/// Translates FoldBinKernel + AOVKernel from aov.cu.
use rayon::prelude::*;

use crate::fold;

/// Compute AOV F-statistic for a single light curve over a grid of
/// (period, period_dt) pairs.
///
/// Returns a flat Vec<f32> of length n_periods * n_pdts.
pub fn calc_aov(
    times: &[f32],
    mags: &[f32],
    periods: &[f32],
    period_dts: &[f32],
    num_bins: usize,
    num_overlap: usize,
) -> Vec<f32> {
    let n_periods = periods.len();
    let n_pdts = period_dts.len();
    let length = times.len();

    let bin_size = 1.0_f32 / num_bins as f32;

    // Compute overall mean magnitude
    let avg: f32 = mags.iter().copied().sum::<f32>() / length as f32;

    let total = n_periods * n_pdts;
    let results: Vec<f32> = (0..total)
        .into_par_iter()
        .map_init(
            || {
                (
                    vec![0u32; num_bins],
                    vec![0.0_f32; num_bins],
                    vec![0.0_f32; num_bins],
                )
            },
            |(count, sums, sq_sums), flat_idx| {
            let period_idx = flat_idx / n_pdts;
            let pdt_idx = flat_idx % n_pdts;

            let period = periods[period_idx];
            let period_dt = period_dts[pdt_idx];
            let pdt_corr = fold::pdt_correction(period, period_dt);

            // Zero reused bin accumulators
            count.iter_mut().for_each(|x| *x = 0);
            sums.iter_mut().for_each(|x| *x = 0.0);
            sq_sums.iter_mut().for_each(|x| *x = 0.0);

            for i in 0..length {
                let folded = fold::fold_time(times[i], period, pdt_corr);
                let bin = (folded / bin_size) as usize;
                let mag = mags[i];

                for o in 0..num_overlap {
                    let b = (bin + o) % num_bins;
                    count[b] += 1;
                    sums[b] += mag;
                    sq_sums[b] += mag * mag;
                }
            }

            // Compute F-statistic
            let mut s1 = 0.0_f32;
            let mut s2 = 0.0_f32;

            for b in 0..num_bins {
                let n = count[b] as f32;
                if n != 0.0 {
                    let aux = sums[b] / n;
                    let residual = aux - avg;
                    s1 += n * residual * residual;
                    s2 += sq_sums[b] - n * aux * aux;
                }
            }

            if s2 == 0.0 {
                0.0
            } else {
                ((length as f32 - num_bins as f32) / (num_bins as f32 - 1.0)) * (s1 / s2)
            }
        },)
        .collect();

    results
}
