/// Chunked greedy peak finder inspired by ZVAR-Period's cudaFindPeaksChunk.
///
/// Finds the top N peaks in a 1D signal with a minimum distance constraint.
/// Processes data in a single streaming pass, maintaining state across chunks
/// so that full periodograms never need to be materialized.
///
/// The algorithm:
/// 1. Scan linearly, detecting local extrema (value vs. both neighbours).
/// 2. When an extremum exceeds the current dynamic threshold, set it as the
///    *candidate*. If a better extremum appears before the distance window
///    expires, replace the candidate.
/// 3. Once `min_distance` samples have passed since the candidate was set,
///    promote it into the sorted top-N list.
/// 4. The dynamic threshold equals the worst value currently in the top-N,
///    so weak candidates are filtered out as the list fills.

/// Sorted top-N peak list (best first).
pub struct PeakState {
    /// Global indices of accepted peaks, sorted best-first.
    pub indices: Vec<i64>,
    /// Corresponding values, sorted best-first.
    pub values: Vec<f32>,
    /// How many slots are occupied (≤ n_peaks).
    pub count: usize,

    n_peaks: usize,
    min_distance: usize,
    use_max: bool,

    // -- streaming state --
    candidate_idx: i64,
    candidate_val: f32,
    has_candidate: bool,
    dist_since_candidate: usize,

    // boundary context: last two values from the previous chunk
    prev_val_0: f32, // second-to-last
    prev_val_1: f32, // last
    has_prev: bool,
}

impl PeakState {
    pub fn new(n_peaks: usize, min_distance: usize, use_max: bool) -> Self {
        let worst = if use_max {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        Self {
            indices: vec![0i64; n_peaks],
            values: vec![worst; n_peaks],
            count: 0,
            n_peaks,
            min_distance,
            use_max,
            candidate_idx: -1,
            candidate_val: worst,
            has_candidate: false,
            dist_since_candidate: 0,
            prev_val_0: 0.0,
            prev_val_1: 0.0,
            has_prev: false,
        }
    }

    /// Current dynamic threshold: the worst value in the top-N.
    /// Before the list is full this returns the initial worst sentinel so
    /// that every peak is accepted.
    #[inline]
    fn threshold(&self) -> f32 {
        if self.count < self.n_peaks {
            if self.use_max {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            // worst entry sits at the end
            self.values[self.count - 1]
        }
    }

    /// Is `a` better than `b`?
    #[inline]
    fn is_better(&self, a: f32, b: f32) -> bool {
        if self.use_max {
            a > b
        } else {
            a < b
        }
    }

    /// Is the given value a local peak relative to its neighbours?
    #[inline]
    fn is_peak(&self, left: f32, val: f32, right: f32) -> bool {
        if self.use_max {
            val > left && val > right
        } else {
            val < left && val < right
        }
    }

    /// Insert a confirmed peak into the sorted list.
    fn insert(&mut self, idx: i64, val: f32) {
        if self.count < self.n_peaks {
            // List not full — append and bubble into sorted position.
            let pos = self.count;
            self.indices[pos] = idx;
            self.values[pos] = val;
            self.count += 1;
            // Bubble toward the front (best-first).
            let mut i = pos;
            while i > 0 && self.is_better(self.values[i], self.values[i - 1]) {
                self.values.swap(i, i - 1);
                self.indices.swap(i, i - 1);
                i -= 1;
            }
        } else if self.is_better(val, self.values[self.count - 1]) {
            // Replace the worst (last) entry.
            self.values[self.count - 1] = val;
            self.indices[self.count - 1] = idx;
            // Bubble toward the front.
            let mut i = self.count - 1;
            while i > 0 && self.is_better(self.values[i], self.values[i - 1]) {
                self.values.swap(i, i - 1);
                self.indices.swap(i, i - 1);
                i -= 1;
            }
        }
    }

    /// Try to promote the current candidate.
    #[inline]
    fn try_promote_candidate(&mut self) {
        if self.has_candidate && self.is_better(self.candidate_val, self.threshold()) {
            self.insert(self.candidate_idx, self.candidate_val);
        }
        self.has_candidate = false;
        self.dist_since_candidate = 0;
    }

    /// Feed a single value at global index `global_idx`.
    /// `left` and `right` are the neighbouring values (caller provides).
    #[inline]
    fn feed(&mut self, val: f32, left: f32, right: f32, global_idx: i64) {
        if self.is_peak(left, val, right) && self.is_better(val, self.threshold()) {
            if !self.has_candidate || self.is_better(val, self.candidate_val) {
                self.candidate_idx = global_idx;
                self.candidate_val = val;
                self.has_candidate = true;
                self.dist_since_candidate = 0;
            }
        }

        if self.has_candidate {
            self.dist_since_candidate += 1;
            if self.dist_since_candidate >= self.min_distance {
                self.try_promote_candidate();
            }
        }
    }

    /// Process a contiguous chunk of data.
    ///
    /// `chunk_start` is the global index of `data[0]`.
    pub fn process_chunk(&mut self, data: &[f32], chunk_start: i64) {
        let n = data.len();
        if n == 0 {
            return;
        }

        // --- Handle boundary between previous chunk and this chunk ----------
        if self.has_prev && n >= 1 {
            // Check if the last sample of the previous chunk is a peak.
            // left = prev_val_0, centre = prev_val_1, right = data[0]
            self.feed(self.prev_val_1, self.prev_val_0, data[0], chunk_start - 1);

            if n >= 2 {
                // Check if data[0] is a peak (left = prev_val_1).
                self.feed(data[0], self.prev_val_1, data[1], chunk_start);
            }
        }

        // --- Main loop: data[1] .. data[n-2] --------------------------------
        let start = if self.has_prev { 1 } else { 0 };
        for i in start.max(1)..n.saturating_sub(1) {
            self.feed(data[i], data[i - 1], data[i + 1], chunk_start + i as i64);
        }

        // --- Save boundary context for next chunk ----------------------------
        if n >= 2 {
            self.prev_val_0 = data[n - 2];
            self.prev_val_1 = data[n - 1];
        } else {
            // n == 1
            self.prev_val_0 = if self.has_prev {
                self.prev_val_1
            } else {
                data[0]
            };
            self.prev_val_1 = data[0];
        }
        self.has_prev = true;
    }

    /// Flush any remaining candidate after all data has been fed.
    pub fn finalize(&mut self) {
        self.try_promote_candidate();
    }

    /// Return (indices, values) trimmed to actual count.
    pub fn results(&self) -> (Vec<i64>, Vec<f32>) {
        (
            self.indices[..self.count].to_vec(),
            self.values[..self.count].to_vec(),
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone: find top peaks on a pre-computed flat array
// ---------------------------------------------------------------------------

/// Find top-N peaks in a pre-computed 1D periodogram slice.
///
/// `data` is a flat row-major array (e.g. n_periods * n_pdts).
/// Returns (indices, values) for up to `n_peaks` peaks, sorted best-first.
pub fn find_top_peaks(
    data: &[f32],
    n_peaks: usize,
    min_distance: usize,
    use_max: bool,
) -> (Vec<i64>, Vec<f32>) {
    let mut state = PeakState::new(n_peaks, min_distance, use_max);
    state.process_chunk(data, 0);
    state.finalize();
    state.results()
}

// ---------------------------------------------------------------------------
// Fused: compute a periodogram value-by-value without materialising the array
// ---------------------------------------------------------------------------

/// Generic interface for computing a single periodogram value.
/// Implementations call the algorithm-specific kernel (CE, AOV, LS, FPW).
pub trait PeriodogramEval {
    /// Compute the statistic for one (period, period_dt) pair.
    fn eval(&self, period_idx: usize, pdt_idx: usize) -> f32;
}

/// Compute only the top-N peaks of a periodogram without allocating the full
/// output array.
///
/// `n_periods` × `n_pdts` values are computed on-the-fly and streamed through
/// `PeakState`.  Memory usage is O(n_peaks) instead of O(n_periods × n_pdts).
pub fn find_peaks_fused(
    evaluator: &dyn PeriodogramEval,
    n_periods: usize,
    n_pdts: usize,
    n_peaks: usize,
    min_distance: usize,
    use_max: bool,
) -> (Vec<i64>, Vec<f32>) {
    let total = n_periods * n_pdts;
    let chunk_size = 2048; // values per chunk — tuned for L1 cache
    let mut state = PeakState::new(n_peaks, min_distance, use_max);
    let mut buf = Vec::with_capacity(chunk_size);

    let mut pos = 0usize;
    while pos < total {
        buf.clear();
        let end = (pos + chunk_size).min(total);
        for flat_idx in pos..end {
            let period_idx = flat_idx / n_pdts;
            let pdt_idx = flat_idx % n_pdts;
            buf.push(evaluator.eval(period_idx, pdt_idx));
        }
        state.process_chunk(&buf, pos as i64);
        pos = end;
    }

    state.finalize();
    state.results()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_peak() {
        // Clear single maximum in the middle.
        let data = vec![0.0, 1.0, 3.0, 1.0, 0.0];
        let (idx, val) = find_top_peaks(&data, 32, 1, true);
        assert_eq!(idx.len(), 1);
        assert_eq!(idx[0], 2);
        assert!((val[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn single_minimum() {
        let data = vec![5.0, 3.0, 1.0, 3.0, 5.0];
        let (idx, val) = find_top_peaks(&data, 32, 1, false);
        assert_eq!(idx.len(), 1);
        assert_eq!(idx[0], 2);
        assert!((val[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn distance_constraint() {
        // Two peaks close together — only the better one should survive.
        let data = vec![0.0, 5.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let (idx, val) = find_top_peaks(&data, 32, 4, true);
        // Peak at 1 (val=5) and peak at 3 (val=4) are 2 apart — within
        // distance=4 window, so 5 wins. Peak at 8 (val=3) is far enough.
        assert_eq!(idx.len(), 2);
        assert_eq!(idx[0], 1);
        assert_eq!(idx[1], 8);
        assert!((val[0] - 5.0).abs() < 1e-6);
        assert!((val[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn respects_n_peaks_limit() {
        // Many peaks — should keep only top n_peaks.
        let mut data = vec![0.0f32; 200];
        for i in (1..200).step_by(10) {
            data[i] = (i as f32) / 10.0;
        }
        let (idx, val) = find_top_peaks(&data, 5, 1, true);
        assert_eq!(idx.len(), 5);
        // Best peak should be at index 191 (value 19.1).
        assert_eq!(idx[0], 191);
        assert!(val[0] > val[1]); // sorted best-first
    }

    #[test]
    fn chunked_matches_single_pass() {
        // Process in two chunks — should give same result as single pass.
        let data: Vec<f32> = (0..100).map(|i| ((i as f32) * 0.3).sin()).collect();

        let (idx_single, val_single) = find_top_peaks(&data, 8, 3, true);

        let mut state = PeakState::new(8, 3, true);
        state.process_chunk(&data[..50], 0);
        state.process_chunk(&data[50..], 50);
        state.finalize();
        let (idx_chunked, val_chunked) = state.results();

        assert_eq!(idx_single.len(), idx_chunked.len());
        for i in 0..idx_single.len() {
            assert_eq!(idx_single[i], idx_chunked[i]);
            assert!((val_single[i] - val_chunked[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn empty_data() {
        let (idx, val) = find_top_peaks(&[], 32, 1, true);
        assert_eq!(idx.len(), 0);
        assert_eq!(val.len(), 0);
    }

    #[test]
    fn monotonic_no_peaks() {
        // Monotonically increasing — no local maxima.
        let data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let (idx, _val) = find_top_peaks(&data, 32, 1, true);
        assert_eq!(idx.len(), 0);
    }
}
