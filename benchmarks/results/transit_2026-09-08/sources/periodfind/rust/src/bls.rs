/// Box Least Squares (BLS) transit-detection algorithm — CPU implementation.
///
/// Searches for periodic box-shaped (flat-bottom) dips in time-series data,
/// as described by Kovács, Zucker & Mazeh (2002).  Adapted to the periodfind
/// period/period_dt grid convention.
use rayon::prelude::*;

use crate::fold;

/// Compute the BLS statistic for a single light curve over a grid of
/// (period, period_dt) pairs.
///
/// For each trial period the data is phase-folded and binned.  The algorithm
/// then searches over all contiguous bin windows whose width corresponds to
/// transit durations between `qmin` and `qmax` (as fractions of the period).
/// The returned value for each grid point is the maximum BLS power found over
/// all (duration, phase-offset) combinations.
///
/// Returns a flat Vec<f32> of length n_periods * n_pdts.
pub fn calc_bls(
    times: &[f32],
    mags: &[f32],
    errs: &[f32],
    periods: &[f32],
    period_dts: &[f32],
    num_bins: usize,
    qmin: f32,
    qmax: f32,
) -> Vec<f32> {
    let n_periods = periods.len();
    let n_pdts = period_dts.len();
    let n_data = times.len();

    // Precompute inverse variances and weighted data
    let ivar: Vec<f32> = errs.iter().map(|e| 1.0 / (e * e)).collect();
    let total_w: f32 = ivar.iter().sum();
    let mean_y: f32 = if total_w > 0.0 {
        ivar.iter()
            .zip(mags.iter())
            .map(|(w, y)| w * y)
            .sum::<f32>()
            / total_w
    } else {
        0.0
    };
    // Weighted, centered data
    let yw: Vec<f32> = (0..n_data).map(|i| ivar[i] * (mags[i] - mean_y)).collect();

    // Bin-width range (in number of bins)
    let nb_min = ((qmin * num_bins as f32).floor() as usize).max(1);
    let nb_max = ((qmax * num_bins as f32).ceil() as usize).min(num_bins - 1);

    let total = n_periods * n_pdts;
    let results: Vec<f32> = (0..total)
        .into_par_iter()
        .map_init(
            || {
                (
                    vec![0.0_f32; num_bins],
                    vec![0.0_f32; num_bins],
                    vec![0.0_f32; num_bins + 1],
                    vec![0.0_f32; num_bins + 1],
                )
            },
            |(w_bin, yw_bin, w_prefix, yw_prefix), flat_idx| {
            let period_idx = flat_idx / n_pdts;
            let pdt_idx = flat_idx % n_pdts;

            let period = periods[period_idx];
            let period_dt = period_dts[pdt_idx];
            let pdt_corr = fold::pdt_correction(period, period_dt);

            // Zero reused bin accumulators
            w_bin.iter_mut().for_each(|x| *x = 0.0);
            yw_bin.iter_mut().for_each(|x| *x = 0.0);

            for i in 0..n_data {
                let phase = fold::fold_time(times[i], period, pdt_corr);
                let bin = ((phase * num_bins as f32) as usize).min(num_bins - 1);
                w_bin[bin] += ivar[i];
                yw_bin[bin] += yw[i];
            }

            // Prefix sums for O(1) range queries (circular)
            w_prefix[0] = 0.0;
            yw_prefix[0] = 0.0;
            for k in 0..num_bins {
                w_prefix[k + 1] = w_prefix[k] + w_bin[k];
                yw_prefix[k + 1] = yw_prefix[k] + yw_bin[k];
            }
            let w_total = w_prefix[num_bins];

            if w_total <= 0.0 {
                return 0.0;
            }

            let mut best_bls: f32 = 0.0;

            // Search over transit durations
            for nb in nb_min..=nb_max {
                // Search over phase offsets
                for phi in 0..num_bins {
                    let end = phi + nb;
                    let (r, s) = if end <= num_bins {
                        // No wrap-around
                        (
                            w_prefix[end] - w_prefix[phi],
                            yw_prefix[end] - yw_prefix[phi],
                        )
                    } else {
                        // Wrap-around: [phi..num_bins] + [0..end-num_bins]
                        let wrap = end - num_bins;
                        (
                            (w_prefix[num_bins] - w_prefix[phi]) + w_prefix[wrap],
                            (yw_prefix[num_bins] - yw_prefix[phi]) + yw_prefix[wrap],
                        )
                    };

                    let r_frac = r / w_total;
                    if r_frac > 0.0 && r_frac < 1.0 {
                        let bls = (s * s) / (r * (w_total - r));
                        if bls > best_bls {
                            best_bls = bls;
                        }
                    }
                }
            }

            best_bls
        },)
        .collect();

    results
}
