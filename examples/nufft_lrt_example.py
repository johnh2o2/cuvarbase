"""
Example usage of NUFFT-based Likelihood Ratio Test for transit detection.

This example demonstrates how to use the NUFFTLRTAsyncProcess class to detect
transits in lightcurve data with gappy sampling.
"""
import numpy as np
import matplotlib.pyplot as plt
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess


def generate_transit_lightcurve(t, period, epoch, duration, depth, noise_level=0.1):
    """
    Generate a simple transit lightcurve.
    
    Parameters
    ----------
    t : array-like
        Time values
    period : float
        Orbital period
    epoch : float
        Time of first transit
    duration : float
        Transit duration
    depth : float
        Transit depth
    noise_level : float, optional
        Standard deviation of Gaussian noise
        
    Returns
    -------
    y : np.ndarray
        Lightcurve with transits and noise
    """
    # Phase fold
    phase = np.fmod(t - epoch, period) / period
    phase[phase < 0] += 1.0
    phase[phase > 0.5] -= 1.0
    
    # Generate transit signal
    signal = np.zeros_like(t)
    phase_width = duration / (2.0 * period)
    in_transit = np.abs(phase) <= phase_width
    signal[in_transit] = -depth
    
    # Add noise
    noise = noise_level * np.random.randn(len(t))
    
    return signal + noise


def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("NUFFT LRT Example: Basic Usage")
    print("=" * 60)
    
    # Generate gappy time series
    np.random.seed(42)
    n_points = 200
    t = np.sort(np.random.uniform(0, 20, n_points))
    
    # True transit parameters
    true_period = 3.5
    true_duration = 0.3
    true_epoch = 0.5
    depth = 0.02  # 2% transit depth
    
    # Generate lightcurve
    y = generate_transit_lightcurve(
        t, true_period, true_epoch, true_duration, depth, noise_level=0.01
    )
    
    print(f"\nGenerated lightcurve with {len(t)} observations")
    print(f"True period: {true_period:.2f} days")
    print(f"True duration: {true_duration:.2f} days")
    print(f"True depth: {depth:.4f}")
    
    # Initialize NUFFT LRT processor
    proc = NUFFTLRTAsyncProcess()
    
    # Search over periods and durations
    periods = np.linspace(2.0, 5.0, 50)
    durations = np.linspace(0.1, 0.5, 10)
    
    print(f"\nSearching {len(periods)} periods × {len(durations)} durations...")
    snr = proc.run(t, y, periods, durations=durations)
    
    # Find best match
    best_idx = np.unravel_index(np.argmax(snr), snr.shape)
    best_period = periods[best_idx[0]]
    best_duration = durations[best_idx[1]]
    best_snr = snr[best_idx]
    
    print(f"\nBest match:")
    print(f"  Period: {best_period:.2f} days (true: {true_period:.2f})")
    print(f"  Duration: {best_duration:.2f} days (true: {true_duration:.2f})")
    print(f"  SNR: {best_snr:.2f}")
    
    print("\nExample completed successfully!")


if __name__ == '__main__':
    print("\nNUFFT-based Likelihood Ratio Test for Transit Detection")
    print("========================================================\n")
    print("This implementation is based on the matched filter approach")
    print("described in the IEEE paper on detection of known (up to parameters)")
    print("signals in unknown correlated Gaussian noise.\n")
    print("Reference implementation:")
    print("https://github.com/star-skelly/code_nova_exoghosts/blob/main/nufft_detector.py\n")
    
    example_basic_usage()
