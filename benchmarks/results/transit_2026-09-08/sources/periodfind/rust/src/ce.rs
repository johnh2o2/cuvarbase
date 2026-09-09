/// Conditional Entropy algorithm — CPU implementation matching CUDA kernels.
///
/// Translates FoldBinKernel + ConditionalEntropyKernel from ce.cu.
use rayon::prelude::*;

use crate::fold;

/// Compute Conditional Entropy for a single light curve over a grid of
/// (period, period_dt) pairs.
///
/// Returns a flat Vec<f32> of length n_periods * n_pdts, laid out as
/// [period_idx * n_pdts + pdt_idx].
pub fn calc_ce(
    times: &[f32],
    mags: &[f32],
    periods: &[f32],
    period_dts: &[f32],
    num_phase: usize,
    num_mag: usize,
    phase_overlap: usize,
    mag_overlap: usize,
) -> Vec<f32> {
    let n_periods = periods.len();
    let n_pdts = period_dts.len();
    let length = times.len();
    let num_bins = num_phase * num_mag;

    let phase_bin_size = 1.0_f32 / num_phase as f32;
    let mag_bin_size = 1.0_f32 / num_mag as f32;

    let norm_divisor = (length * phase_overlap * mag_overlap) as f32;

    // Build index pairs for parallelization
    let total = n_periods * n_pdts;
    let results: Vec<f32> = (0..total)
        .into_par_iter()
        .map_init(
            || vec![0u32; num_bins],
            |hist, flat_idx| {
            let period_idx = flat_idx / n_pdts;
            let pdt_idx = flat_idx % n_pdts;

            let period = periods[period_idx];
            let period_dt = period_dts[pdt_idx];
            let pdt_corr = fold::pdt_correction(period, period_dt);

            // Zero reused histogram
            hist.iter_mut().for_each(|x| *x = 0);

            for i in 0..length {
                let folded = fold::fold_time(times[i], period, pdt_corr);
                let phase_bin = (folded / phase_bin_size) as usize;
                let mag_bin = (mags[i] / mag_bin_size) as usize;

                // Overlap: add to adjacent bins with wrapping
                for po in 0..phase_overlap {
                    for mo in 0..mag_overlap {
                        let pb = (phase_bin + po) % num_phase;
                        let mb = (mag_bin + mo) % num_mag;
                        hist[pb * num_mag + mb] += 1;
                    }
                }
            }

            // Normalize histogram to probabilities
            let hist_f: Vec<f32> = hist.iter().map(|&c| c as f32 / norm_divisor).collect();

            // Compute conditional entropy
            // CE = sum over phase bins j: sum over mag bins i: p_ij * ln(p_j / p_ij)
            let mut ce = 0.0_f32;
            for j in 0..num_phase {
                // p_j = sum over mag bins
                let mut p_j = 0.0_f32;
                for i in 0..num_mag {
                    p_j += hist_f[j * num_mag + i];
                }

                for i in 0..num_mag {
                    let p_ij = hist_f[j * num_mag + i];
                    if p_ij != 0.0 {
                        ce += p_ij * (p_j / p_ij).ln();
                    }
                }
            }

            ce
        },)
        .collect();

    results
}
