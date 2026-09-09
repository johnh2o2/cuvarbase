use rayon::prelude::*;

/// Find the bin index for `value` given sorted `edges`.
///
/// Uses `partition_point` (binary search).  Returns `None` if the value
/// falls outside the range `[edges[0], edges[last]]`.
/// Last-bin-inclusive: values equal to `edges[last]` map to the final bin.
#[inline]
fn find_bin(edges: &[f32], value: f32) -> Option<usize> {
    let n_bins = edges.len() - 1;
    if n_bins == 0 {
        return None;
    }
    if value < edges[0] || value > edges[n_bins] {
        return None;
    }
    // Last-bin-inclusive: clamp to final bin
    if value == edges[n_bins] {
        return Some(n_bins - 1);
    }
    // partition_point returns first index where edges[idx] > value
    let idx = edges.partition_point(|&e| e <= value);
    if idx == 0 {
        return None;
    }
    Some(idx - 1)
}

/// Compute the dm-dt histogram for a single light curve.
///
/// Fuses pairwise differences and histogram accumulation in one pass
/// (never materialises the O(N^2) diff arrays).
///
/// Returns a flat `Vec<f32>` of length `n_dm_bins * n_dt_bins`, stored in
/// row-major order with dm on the outer axis (matching the Python
/// `hh.T` convention).  The result is L2-normalised.
pub fn compute_dmdt_single(
    times: &[f32],
    mags: &[f32],
    dt_edges: &[f32],
    dm_edges: &[f32],
) -> Vec<f32> {
    let n_dt_bins = if dt_edges.len() > 1 {
        dt_edges.len() - 1
    } else {
        0
    };
    let n_dm_bins = if dm_edges.len() > 1 {
        dm_edges.len() - 1
    } else {
        0
    };
    let total = n_dm_bins * n_dt_bins;

    if total == 0 || times.len() < 2 {
        return vec![0.0; total];
    }

    let mut hist = vec![0.0f32; total];

    let n = times.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let dt = times[j] - times[i];
            let dm = mags[j] - mags[i];

            if let (Some(dt_bin), Some(dm_bin)) = (find_bin(dt_edges, dt), find_bin(dm_edges, dm)) {
                // Transposed storage: dm_bin * n_dt_bins + dt_bin
                hist[dm_bin * n_dt_bins + dt_bin] += 1.0;
            }
        }
    }

    // L2 normalise
    let norm: f32 = hist.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in hist.iter_mut() {
            *v /= norm;
        }
    }

    hist
}

/// Batch dm-dt computation across multiple light curves using Rayon.
///
/// Returns a flat `Vec<f32>` of length `n_curves * n_dm_bins * n_dt_bins`,
/// with each curve's histogram laid out contiguously.
pub fn compute_dmdt_batch(
    times_list: &[&[f32]],
    mags_list: &[&[f32]],
    dt_edges: &[f32],
    dm_edges: &[f32],
) -> Vec<f32> {
    let n_dt_bins = if dt_edges.len() > 1 {
        dt_edges.len() - 1
    } else {
        0
    };
    let n_dm_bins = if dm_edges.len() > 1 {
        dm_edges.len() - 1
    } else {
        0
    };
    let hist_size = n_dm_bins * n_dt_bins;

    let results: Vec<Vec<f32>> = (0..times_list.len())
        .into_par_iter()
        .map(|i| compute_dmdt_single(times_list[i], mags_list[i], dt_edges, dm_edges))
        .collect();

    let mut flat = Vec::with_capacity(results.len() * hist_size);
    for r in results {
        flat.extend_from_slice(&r);
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_bin_within() {
        let edges = vec![0.0, 1.0, 2.0, 3.0];
        assert_eq!(find_bin(&edges, 0.5), Some(0));
        assert_eq!(find_bin(&edges, 1.0), Some(1));
        assert_eq!(find_bin(&edges, 1.5), Some(1));
        assert_eq!(find_bin(&edges, 2.5), Some(2));
    }

    #[test]
    fn test_find_bin_edges() {
        let edges = vec![0.0, 1.0, 2.0, 3.0];
        // First edge
        assert_eq!(find_bin(&edges, 0.0), Some(0));
        // Last edge (inclusive)
        assert_eq!(find_bin(&edges, 3.0), Some(2));
        // Outside
        assert_eq!(find_bin(&edges, -0.1), None);
        assert_eq!(find_bin(&edges, 3.1), None);
    }

    #[test]
    fn test_empty_and_single_point() {
        let dt_edges = vec![0.0, 1.0];
        let dm_edges = vec![-1.0, 1.0];

        let h = compute_dmdt_single(&[], &[], &dt_edges, &dm_edges);
        assert_eq!(h, vec![0.0]);

        let h = compute_dmdt_single(&[1.0], &[5.0], &dt_edges, &dm_edges);
        assert_eq!(h, vec![0.0]);
    }

    #[test]
    fn test_small_known_example() {
        // 3 points => 3 pairs
        let times = vec![0.0, 1.0, 3.0];
        let mags = vec![10.0, 11.0, 10.5];
        let dt_edges = vec![0.0, 2.0, 4.0];
        let dm_edges = vec![-1.0, 0.0, 2.0];

        // Pairs:
        // (0,1): dt=1.0 -> bin 0, dm=1.0 -> bin 1
        // (0,2): dt=3.0 -> bin 1, dm=0.5 -> bin 1
        // (1,2): dt=2.0 -> bin 1, dm=-0.5 -> bin 0

        // Histogram (dm_bin, dt_bin):
        // [0,0]=0  [0,1]=1  (dm<0 row)
        // [1,0]=1  [1,1]=1  (dm>=0 row)
        let h = compute_dmdt_single(&times, &mags, &dt_edges, &dm_edges);
        assert_eq!(h.len(), 4);

        // Before normalization: [0, 1, 1, 1]
        // L2 norm = sqrt(0 + 1 + 1 + 1) = sqrt(3)
        let norm = 3.0f32.sqrt();
        let expected = vec![0.0 / norm, 1.0 / norm, 1.0 / norm, 1.0 / norm];
        for (a, b) in h.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {}, expected {}", a, b);
        }
    }

    #[test]
    fn test_normalization() {
        let times = vec![0.0, 1.0];
        let mags = vec![10.0, 11.0];
        let dt_edges = vec![0.0, 2.0];
        let dm_edges = vec![0.0, 2.0];

        let h = compute_dmdt_single(&times, &mags, &dt_edges, &dm_edges);
        // Single pair lands in the one bin => value is 1/1 = 1.0
        assert_eq!(h.len(), 1);
        assert!((h[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch() {
        let t1 = vec![0.0, 1.0, 2.0];
        let m1 = vec![10.0, 11.0, 12.0];
        let t2 = vec![0.0, 5.0];
        let m2 = vec![10.0, 10.0];

        let dt_edges = vec![0.0, 3.0, 6.0];
        let dm_edges = vec![-3.0, 0.0, 3.0];

        let flat = compute_dmdt_batch(&[&t1, &t2], &[&m1, &m2], &dt_edges, &dm_edges);

        // 2 dm bins * 2 dt bins = 4 per curve, 2 curves = 8
        assert_eq!(flat.len(), 8);
    }
}
