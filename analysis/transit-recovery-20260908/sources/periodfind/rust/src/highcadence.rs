use rayon::prelude::*;

/// Return indices of points to keep after removing high-cadence duplicates.
///
/// Keeps index 0, then each subsequent index whose time is at least
/// `cadence_days` after the last kept point.
pub fn remove_high_cadence_indices(times: &[f32], cadence_days: f32) -> Vec<usize> {
    if times.is_empty() {
        return Vec::new();
    }
    let mut kept = Vec::with_capacity(times.len());
    kept.push(0);
    let mut last_t = times[0];
    for i in 1..times.len() {
        if times[i] - last_t >= cadence_days {
            kept.push(i);
            last_t = times[i];
        }
    }
    kept
}

/// Batch high-cadence removal across multiple light curves using Rayon.
///
/// Returns one `(times, mags, errs)` triple per input curve, filtered to
/// keep only points separated by at least `cadence_days`.
pub fn remove_high_cadence_batch(
    times_list: &[&[f32]],
    mags_list: &[&[f32]],
    errs_list: &[&[f32]],
    cadence_days: f32,
) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    (0..times_list.len())
        .into_par_iter()
        .map(|i| {
            let idx = remove_high_cadence_indices(times_list[i], cadence_days);
            let t: Vec<f32> = idx.iter().map(|&j| times_list[i][j]).collect();
            let m: Vec<f32> = idx.iter().map(|&j| mags_list[i][j]).collect();
            let e: Vec<f32> = idx.iter().map(|&j| errs_list[i][j]).collect();
            (t, m, e)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let idx = remove_high_cadence_indices(&[], 0.5);
        assert!(idx.is_empty());
    }

    #[test]
    fn test_single_point() {
        let idx = remove_high_cadence_indices(&[1.0], 0.5);
        assert_eq!(idx, vec![0]);
    }

    #[test]
    fn test_all_within_cadence() {
        // Points 0.1 apart, cadence 1.0 day => keep only first
        let times: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let idx = remove_high_cadence_indices(&times, 1.0);
        assert_eq!(idx, vec![0]);
    }

    #[test]
    fn test_all_beyond_cadence() {
        // Points 2.0 apart, cadence 1.0 day => keep all
        let times: Vec<f32> = (0..5).map(|i| i as f32 * 2.0).collect();
        let idx = remove_high_cadence_indices(&times, 1.0);
        assert_eq!(idx, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_hand_calculated() {
        // cadence = 30 minutes = 30/1440 days ≈ 0.02083
        let cadence_days = 30.0 / 1440.0;
        let times = vec![0.0, 0.01, 0.02, 0.025, 0.05, 0.06, 0.10];
        // idx 0: kept (first)
        // idx 1: 0.01 - 0.0 = 0.01 < 0.02083 => skip
        // idx 2: 0.02 - 0.0 = 0.02 < 0.02083 => skip
        // idx 3: 0.025 - 0.0 = 0.025 >= 0.02083 => keep
        // idx 4: 0.05 - 0.025 = 0.025 >= 0.02083 => keep
        // idx 5: 0.06 - 0.05 = 0.01 < 0.02083 => skip
        // idx 6: 0.10 - 0.05 = 0.05 >= 0.02083 => keep
        let idx = remove_high_cadence_indices(&times, cadence_days);
        assert_eq!(idx, vec![0, 3, 4, 6]);
    }

    #[test]
    fn test_batch() {
        let t1 = vec![0.0, 1.0, 2.0, 3.0];
        let m1 = vec![10.0, 11.0, 12.0, 13.0];
        let e1 = vec![0.1, 0.1, 0.1, 0.1];

        let t2 = vec![0.0, 0.001, 0.002, 5.0];
        let m2 = vec![20.0, 21.0, 22.0, 23.0];
        let e2 = vec![0.2, 0.2, 0.2, 0.2];

        let result = remove_high_cadence_batch(&[&t1, &t2], &[&m1, &m2], &[&e1, &e2], 0.5);

        assert_eq!(result.len(), 2);
        // First curve: all points >= 0.5 apart
        assert_eq!(result[0].0, vec![0.0, 1.0, 2.0, 3.0]);
        // Second curve: only first and last
        assert_eq!(result[1].0, vec![0.0, 5.0]);
        assert_eq!(result[1].1, vec![20.0, 23.0]);
        assert_eq!(result[1].2, vec![0.2, 0.2]);
    }
}
