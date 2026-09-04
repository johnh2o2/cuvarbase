"""
Example usage of the NUFFT-based Likelihood Ratio Test for transit detection.

Demonstrates ``NUFFTLRTAsyncProcess`` on gappy ground-based sampling with
absolute (BJD-scale) timestamps and a transit injected at a random epoch:
the default ``epochs=None`` scans an automatic epoch grid per (period,
duration) cell and returns the maximum statistic together with the epoch
that attains it. Note the statistic is a whitened correlation, not an
N(0, 1) SNR -- a detection threshold has to be calibrated on signal-free
data (sketch at the end); see docs/NUFFT_LRT_README.md.

The period grid matters: a box of duration ``d`` at period ``P`` drifts
by ``T * dP / P`` over a baseline ``T`` when the trial period is off by
``dP``, so the grid step must be ``dP <~ d * P / (2 T)`` or the true
period falls between grid points and a harmonic alias (P/2, 2P) that
happens to sit on the grid wins. That makes a blind search over a wide
period range expensive (one NFFT per template); the intended use is a
focused search around candidate periods, as here.
"""
import numpy as np
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess


def ground_based_times(rng, baseline=90.0, n=600):
    """Nightly visibility windows with weather losses."""
    nights = np.arange(int(baseline))
    nights = nights[rng.rand(len(nights)) > 0.35]
    per_night = max(1, int(round(n / max(len(nights), 1))))
    t = (nights[:, None] + 0.25 * rng.rand(len(nights), per_night)).ravel()
    return np.sort(t[:n])


def generate_transit_lightcurve(rng, t, period, epoch, duration, depth,
                                noise_level=0.003):
    """Relative flux with a box transit and white noise."""
    phase = np.fmod(t - epoch, period) / period
    phase[phase < 0] += 1.0
    phase[phase > 0.5] -= 1.0
    y = np.ones_like(t)
    y[np.abs(phase) <= duration / (2.0 * period)] -= depth
    return y + noise_level * rng.randn(len(t))


def example_basic_usage():
    """Focused period search with the automatic epoch grid"""
    print("=" * 60)
    print("NUFFT LRT Example: focused search with the automatic epoch grid")
    print("=" * 60)

    rng = np.random.RandomState(42)
    bjd0 = 2457000.0                       # absolute timestamps are fine
    t = bjd0 + ground_based_times(rng)
    baseline = t.max() - t.min()

    true_period = 5.3
    true_duration = 0.22
    true_epoch = bjd0 + rng.uniform(0, true_period)   # random phase
    depth = 0.01                           # 1% transit depth

    y = generate_transit_lightcurve(rng, t, true_period, true_epoch,
                                    true_duration, depth)

    print(f"\n{len(t)} observations over {baseline:.0f} d, "
          f"BJD {t.min():.1f} .. {t.max():.1f}")
    print(f"True period: {true_period:.2f} d, duration: {true_duration:.2f} d, "
          f"depth: {depth:.4f}, epoch: BJD {true_epoch:.3f}")

    proc = NUFFTLRTAsyncProcess()

    # Period step from the drift criterion dP <~ d P / (2 T) for the
    # shortest duration searched; a candidate near 5.3 d is assumed
    # (e.g. from a BLS pass) and refined over +-5%.
    durations = np.array([0.12, 0.25])
    p_lo, p_hi = 0.95 * true_period, 1.05 * true_period
    dp = durations.min() * p_lo / (2.0 * baseline)
    periods = np.arange(p_lo, p_hi, dp)

    print(f"\nSearching {len(periods)} periods in [{p_lo:.2f}, {p_hi:.2f}] d "
          f"(step {dp:.4f} d) x {len(durations)} durations")
    print("(automatic epoch grid: up to 96 epochs per cell) ...")
    snr, best_epoch = proc.run(t, y, periods, durations=durations)

    i, j = np.unravel_index(np.argmax(snr), snr.shape)
    found_epoch = best_epoch[i, j]
    dphase = ((found_epoch - true_epoch) / true_period + 0.5) % 1.0 - 0.5
    i_true = int(np.argmin(np.abs(periods - true_period)))
    j_true = int(np.argmin(np.abs(durations - true_duration)))

    print("\nBest cell:")
    print(f"  Period:    {periods[i]:.4f} d (true: {true_period:.4f}; "
          f"nearest grid point {periods[i_true]:.4f})")
    print(f"  Duration:  {durations[j]:.2f} d (true: {true_duration:.2f})")
    print(f"  Epoch:     BJD {found_epoch:.3f} "
          f"(true, mod P: {dphase * true_period:+.3f} d away)")
    print(f"  Statistic: {snr[i, j]:.2f}  (whitened correlation, not an "
          f"N(0,1) SNR -- calibrate a threshold, see below)")
    print(f"  Statistic at the true cell: {snr[i_true, j_true]:.2f}")

    # Threshold calibration sketch: the 95th percentile of the search
    # maximum over signal-free light curves is the 5% per-search
    # false-alarm threshold for THIS sampling, grid and PSD estimator.
    # (A real calibration uses >= 60 draws; 3 here keep the example short.)
    null_max = []
    for k in range(3):
        y_null = 1.0 + 0.003 * rng.randn(len(t))
        s_null, _ = proc.run(t, y_null, periods, durations=durations)
        null_max.append(s_null.max())
    print(f"\nSearch maximum on 3 signal-free draws: "
          f"{np.round(null_max, 2)} (detection: {snr[i, j]:.2f})")

    print("\nExample completed successfully!")


if __name__ == '__main__':
    print("\nNUFFT-based Likelihood Ratio Test for Transit Detection")
    print("========================================================\n")
    print("Whitened matched filter for box transits in correlated noise")
    print("(Taaki, Kamalabadi & Kemball 2020; Taaki, Kemball & Kamalabadi 2025).")
    print("Reference implementation:")
    print("https://github.com/star-skelly/code_nova_exoghosts/blob/main/nufft_detector.py\n")

    example_basic_usage()
