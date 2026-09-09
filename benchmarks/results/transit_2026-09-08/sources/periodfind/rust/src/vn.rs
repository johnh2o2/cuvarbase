/// Viterbi Narrowband (VN) period-finding score — CPU implementation.
///
/// Builds the same 2D phase-magnitude histogram as Conditional Entropy, then
/// runs a circular Viterbi algorithm to find the most likely narrow path
/// through phase-mag space.  The score is the fraction of histogram mass
/// concentrated within `margin` bins of the optimal path.
use rayon::prelude::*;

use crate::fold;

/// Run circular Viterbi on a normalised histogram and return the
/// concentration ratio.
///
/// `hist` is laid out as `[phase_bin * n_mag + mag_bin]`.
///
/// Returns a value in [0, 1]; higher means more mass along the path.
fn circular_viterbi_concentration(
    hist: &[f32],
    n_phase: usize,
    n_mag: usize,
    bandwidth: usize,
    margin: usize,
) -> f32 {
    let eps: f32 = 1e-10;
    let neg_inf = f32::NEG_INFINITY;

    // Pre-compute log-emission: emit[phi][m] = log(hist[phi*n_mag + m] + eps)
    let mut emit = vec![0.0f32; n_phase * n_mag];
    for phi in 0..n_phase {
        for m in 0..n_mag {
            emit[phi * n_mag + m] = (hist[phi * n_mag + m] + eps).ln();
        }
    }

    let mut best_total_score = neg_inf;
    let mut best_path = vec![0usize; n_phase];

    // Try each starting state for the circular constraint
    for start_state in 0..n_mag {
        // V_prev[m] = Viterbi score ending in mag bin m
        let mut v_prev = vec![neg_inf; n_mag];
        let mut v_curr = vec![neg_inf; n_mag];

        // Backpointer: back[phi][m] = previous mag bin
        let mut back = vec![vec![0usize; n_mag]; n_phase];

        // Initialise: only start_state is valid at phi=0
        v_prev[start_state] = emit[start_state]; // emit(0, start_state)

        for phi in 1..n_phase {
            for m in 0..n_mag {
                let mut best_prev = neg_inf;
                let mut best_prev_m = 0usize;

                // Transition: previous mag can be within [m-bw, m+bw]
                let lo = if m >= bandwidth { m - bandwidth } else { 0 };
                let hi = if m + bandwidth < n_mag {
                    m + bandwidth
                } else {
                    n_mag - 1
                };

                for mp in lo..=hi {
                    if v_prev[mp] > best_prev {
                        best_prev = v_prev[mp];
                        best_prev_m = mp;
                    }
                }

                v_curr[m] = emit[phi * n_mag + m] + best_prev;
                back[phi][m] = best_prev_m;
            }

            // Swap
            std::mem::swap(&mut v_prev, &mut v_curr);
            v_curr.iter_mut().for_each(|x| *x = neg_inf);
        }

        // Circular constraint: end state must equal start_state
        let total_score = v_prev[start_state];
        if total_score > best_total_score {
            best_total_score = total_score;

            // Backtrace
            best_path[n_phase - 1] = start_state;
            for phi in (1..n_phase).rev() {
                best_path[phi - 1] = back[phi][best_path[phi]];
            }
        }
    }

    // Score: geometric mean emission probability along the Viterbi path.
    // At the true period, the path traverses high-probability bins (points
    // cluster into a narrow band).  At wrong periods, mass is spread
    // uniformly so per-bin probabilities are low.
    //
    // score = exp( best_total_score / n_phase )
    //       = (∏ P(path_bin))^{1/n_phase}
    //
    // Falls in (0, 1]; peaks sharply at the true period.

    if best_total_score == neg_inf {
        return 0.0;
    }

    // Per-column concentration along the Viterbi path (+/- margin).
    //
    // For each phase bin, measure what fraction of that column's mass
    // falls within `margin` bins of the path.  Average over non-empty
    // columns and normalise against the uniform expectation.
    //
    // This avoids saturation: with uniform data, concentration ≈
    // (2*margin+1)/n_mag.  With a real signal, it's close to 1.0.
    let mut sum_frac = 0.0f32;
    let mut n_cols = 0u32;

    for phi in 0..n_phase {
        // Column total
        let mut col_mass = 0.0f32;
        for m in 0..n_mag {
            col_mass += hist[phi * n_mag + m];
        }
        if col_mass <= 0.0 {
            continue;
        }

        // Mass within margin of the path
        let path_m = best_path[phi];
        let lo = if path_m >= margin { path_m - margin } else { 0 };
        let hi = if path_m + margin < n_mag { path_m + margin } else { n_mag - 1 };
        let mut path_mass = 0.0f32;
        for m in lo..=hi {
            path_mass += hist[phi * n_mag + m];
        }

        sum_frac += path_mass / col_mass;
        n_cols += 1;
    }

    if n_cols == 0 {
        return 0.0;
    }

    let avg_conc = sum_frac / n_cols as f32;

    // Normalise: uniform baseline = (2*margin+1)/n_mag, ceiling = 1.0
    let band = (2 * margin + 1).min(n_mag) as f32;
    let baseline = band / n_mag as f32;
    ((avg_conc - baseline) / (1.0 - baseline)).max(0.0).min(1.0)
}

/// Compute Viterbi Narrowband scores for a single light curve over a grid of
/// (period, period_dt) pairs.
///
/// Returns a flat Vec<f32> of length n_periods * n_pdts, laid out as
/// [period_idx * n_pdts + pdt_idx].
pub fn calc_vn(
    times: &[f32],
    mags: &[f32],
    periods: &[f32],
    period_dts: &[f32],
    num_phase: usize,
    num_mag: usize,
    phase_overlap: usize,
    mag_overlap: usize,
    bandwidth: usize,
    margin: usize,
) -> Vec<f32> {
    let n_periods = periods.len();
    let n_pdts = period_dts.len();
    let length = times.len();
    let num_bins = num_phase * num_mag;

    let phase_bin_size = 1.0_f32 / num_phase as f32;
    let mag_bin_size = 1.0_f32 / num_mag as f32;

    let norm_divisor = (length * phase_overlap * mag_overlap) as f32;

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

                    for po in 0..phase_overlap {
                        for mo in 0..mag_overlap {
                            let pb = (phase_bin + po) % num_phase;
                            let mb = (mag_bin + mo) % num_mag;
                            hist[pb * num_mag + mb] += 1;
                        }
                    }
                }

                // Normalize histogram to probabilities
                let hist_f: Vec<f32> =
                    hist.iter().map(|&c| c as f32 / norm_divisor).collect();

                // Run circular Viterbi and return concentration ratio
                circular_viterbi_concentration(&hist_f, num_phase, num_mag, bandwidth, margin)
            },
        )
        .collect();

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Generate a clean sinusoidal light curve with given period.
    fn make_sinusoid(n: usize, true_period: f32) -> (Vec<f32>, Vec<f32>) {
        let mut times = Vec::with_capacity(n);
        let mut mags = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 * 0.1;
            times.push(t);
            // Sinusoid normalised to (0, 1) range
            let phase = 2.0 * PI * t / true_period;
            let m = 0.5 + 0.45 * phase.sin();
            // Clamp to (0, 1)
            let m = m.max(0.001).min(0.999);
            mags.push(m);
        }
        (times, mags)
    }

    #[test]
    fn sinusoid_detection() {
        let true_period = 3.7;
        let (times, mags) = make_sinusoid(500, true_period);

        // Trial periods: include the true one plus distractors
        let periods: Vec<f32> = (0..50)
            .map(|i| 1.0 + i as f32 * 0.1)
            .collect();
        let period_dts = vec![0.0f32];

        let scores = calc_vn(
            &times, &mags, &periods, &period_dts,
            20, 20, 1, 1, 2, 1,
        );

        // The best score should be near the true period
        let best_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let best_period = periods[best_idx];
        assert!(
            (best_period - true_period).abs() < 0.5,
            "Expected period near {}, got {} (idx {})",
            true_period, best_period, best_idx
        );
    }

    #[test]
    fn noise_gives_low_score() {
        // Pure noise should give lower concentration than a clean signal.
        let n = 200;
        let mut times = Vec::with_capacity(n);
        let mut mags = Vec::with_capacity(n);
        // Simple deterministic "noise"
        for i in 0..n {
            times.push(i as f32 * 0.2);
            let pseudo_random = ((i as f32 * 7.3 + 2.1).sin() * 10000.0).fract().abs();
            mags.push(pseudo_random * 0.998 + 0.001);
        }

        let periods = vec![2.0f32, 3.0, 5.0, 7.0];
        let period_dts = vec![0.0f32];

        let scores = calc_vn(
            &times, &mags, &periods, &period_dts,
            20, 20, 1, 1, 2, 1,
        );

        // All noise scores should be modest
        for &s in &scores {
            assert!(
                s < 0.5,
                "Noise score {} is unexpectedly high",
                s
            );
        }
    }

    #[test]
    fn score_bounds() {
        // Scores should always be in [0, 1]
        let (times, mags) = make_sinusoid(100, 2.5);
        let periods = vec![1.0f32, 2.5, 5.0];
        let period_dts = vec![0.0f32];

        let scores = calc_vn(
            &times, &mags, &periods, &period_dts,
            20, 20, 1, 1, 2, 1,
        );

        for &s in &scores {
            assert!(s >= 0.0 && s <= 1.0, "Score {} out of bounds", s);
        }
    }

    #[test]
    fn clean_vs_noise_discrimination() {
        // A clean sinusoid at the true period should score higher than noise.
        let true_period = 4.0;
        let (times, mags) = make_sinusoid(300, true_period);

        let periods = vec![true_period];
        let period_dts = vec![0.0f32];

        let clean_scores = calc_vn(
            &times, &mags, &periods, &period_dts,
            20, 20, 1, 1, 2, 1,
        );

        // Generate noise
        let n = 300;
        let mut noise_mags = Vec::with_capacity(n);
        for i in 0..n {
            let pseudo_random = ((i as f32 * 13.7 + 0.3).sin() * 10000.0).fract().abs();
            noise_mags.push(pseudo_random * 0.998 + 0.001);
        }

        let noise_scores = calc_vn(
            &times, &noise_mags, &periods, &period_dts,
            20, 20, 1, 1, 2, 1,
        );

        assert!(
            clean_scores[0] > noise_scores[0],
            "Clean signal score {} should exceed noise score {}",
            clean_scores[0], noise_scores[0]
        );
    }
}
