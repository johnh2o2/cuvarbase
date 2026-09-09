/// Shared time-folding logic used by all algorithms.
///
/// Phase computation is promoted to f64 internally to avoid
/// precision loss when baseline/period > ~10^5 cycles (e.g. LISA
/// verification binaries in ZTF with ~2000-day baselines and
/// sub-hour periods).
///
/// CUDA implementation note: the corresponding CUDA kernels should
/// also be updated to use double-precision for the phase calculation.

/// Compute the period derivative correction factor.
#[inline(always)]
pub fn pdt_correction(period: f32, period_dt: f32) -> f32 {
    (period_dt / period) / 2.0
}

/// Fold a time value into the [0, 1) phase range with period derivative correction.
///
/// Internally promotes to f64 for the division and fract operation to
/// maintain phase accuracy over long baselines.
#[inline(always)]
pub fn fold_time(t: f32, period: f32, pdt_corr: f32) -> f32 {
    let t_corr = (t as f64) - (pdt_corr as f64) * (t as f64) * (t as f64);
    let ratio = t_corr / (period as f64);
    (ratio - ratio.floor()).abs() as f32
}
