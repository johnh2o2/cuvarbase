#!/usr/bin/env python3
"""
Example: GPU-Accelerated Transit Least Squares

This script demonstrates how to use cuvarbase's GPU-accelerated TLS
implementation to detect planetary transits in photometric time series.

Requirements:
- PyCUDA
- NumPy
- batman-package (optional, for generating synthetic transits)
"""

import numpy as np
import matplotlib.pyplot as plt

# Check if we can import TLS modules
try:
    from cuvarbase import tls_grids, tls_models, tls
    TLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import TLS modules: {e}")
    TLS_AVAILABLE = False

# Check if batman is available for generating synthetic data
try:
    import batman
    BATMAN_AVAILABLE = True
except ImportError:
    BATMAN_AVAILABLE = False
    print("batman-package not available. Using simple synthetic transit.")


def generate_synthetic_transit(period=10.0, depth=0.01, duration=0.1,
                               t0=0.0, ndata=1000, noise_level=0.001,
                               T_span=100.0):
    """
    Generate synthetic light curve with transit.

    Parameters
    ----------
    period : float
        Orbital period (days)
    depth : float
        Transit depth (fractional)
    duration : float
        Transit duration (days)
    t0 : float
        Mid-transit time (days)
    ndata : int
        Number of data points
    noise_level : float
        Gaussian noise level
    T_span : float
        Total observation span (days)

    Returns
    -------
    t, y, dy : ndarray
        Time, flux, and uncertainties
    """
    # Generate time series
    t = np.sort(np.random.uniform(0, T_span, ndata))

    # Start with flat light curve
    y = np.ones(ndata)

    if BATMAN_AVAILABLE:
        # Use Batman for realistic transit
        params = batman.TransitParams()
        params.t0 = t0
        params.per = period
        params.rp = np.sqrt(depth)  # Radius ratio
        params.a = 15.0  # Semi-major axis
        params.inc = 90.0  # Edge-on
        params.ecc = 0.0
        params.w = 90.0
        params.limb_dark = "quadratic"
        params.u = [0.4804, 0.1867]

        m = batman.TransitModel(params, t)
        y = m.light_curve(params)
    else:
        # Simple box transit
        phases = (t % period) / period
        duration_phase = duration / period

        # Transit at phase 0
        in_transit = (phases < duration_phase / 2) | (phases > 1 - duration_phase / 2)
        y[in_transit] -= depth

    # Add noise
    noise = np.random.normal(0, noise_level, ndata)
    y += noise

    # Uncertainties
    dy = np.ones(ndata) * noise_level

    return t, y, dy


def run_tls_example(use_gpu=True):
    """
    Run TLS example on synthetic data.

    Parameters
    ----------
    use_gpu : bool
        Use GPU implementation (default: True)
    """
    if not TLS_AVAILABLE:
        print("TLS modules not available. Cannot run example.")
        return

    print("=" * 60)
    print("GPU-Accelerated Transit Least Squares Example")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic transit...")
    period_true = 12.5  # days
    depth_true = 0.008  # 0.8% depth
    duration_true = 0.12  # days

    t, y, dy = generate_synthetic_transit(
        period=period_true,
        depth=depth_true,
        duration=duration_true,
        ndata=800,
        noise_level=0.0005,
        T_span=100.0
    )

    print(f"   Data points: {len(t)}")
    print(f"   Time span: {np.max(t) - np.min(t):.1f} days")
    print(f"   True period: {period_true:.2f} days")
    print(f"   True depth: {depth_true:.4f} ({depth_true*1e6:.0f} ppm)")
    print(f"   True duration: {duration_true:.3f} days")

    # Generate period grid
    print("\n2. Generating period grid...")
    periods = tls_grids.period_grid_ofir(
        t, R_star=1.0, M_star=1.0,
        oversampling_factor=3,
        period_min=8.0,
        period_max=20.0
    )
    print(f"   Testing {len(periods)} periods from {periods[0]:.2f} to {periods[-1]:.2f} days")

    # Run TLS search
    print("\n3. Running TLS search...")
    if use_gpu:
        try:
            results = tls.tls_search_gpu(
                t, y, dy,
                periods=periods,
                R_star=1.0,
                M_star=1.0,
                use_simple=True  # Use simple kernel for this dataset size
            )
            print("   ✓ GPU search completed")
        except Exception as e:
            print(f"   ✗ GPU search failed: {e}")
            print("   Tip: Make sure you have a CUDA-capable GPU and PyCUDA installed")
            return
    else:
        print("   CPU implementation not yet available")
        return

    # Display results
    print("\n4. Results:")
    print(f"   Best period: {results['period']:.4f} ± {results['period_uncertainty']:.4f} days")
    print(f"   Best depth: {results['depth']:.6f} ({results['depth']*1e6:.1f} ppm)")
    print(f"   Best duration: {results['duration']:.4f} days")
    print(f"   Best T0: {results['T0']:.4f} (phase)")
    print(f"   Number of transits: {results['n_transits']}")
    print(f"\n   Statistics:")
    print(f"   SDE: {results['SDE']:.2f}")
    print(f"   SNR: {results['SNR']:.2f}")
    print(f"   FAP: {results['FAP']:.2e}")

    # Compare to truth
    period_error = np.abs(results['period'] - period_true)
    depth_error = np.abs(results['depth'] - depth_true)
    duration_error = np.abs(results['duration'] - duration_true)

    print(f"\n   Recovery accuracy:")
    print(f"   Period error: {period_error:.4f} days ({period_error/period_true*100:.1f}%)")
    print(f"   Depth error: {depth_error:.6f} ({depth_error/depth_true*100:.1f}%)")
    print(f"   Duration error: {duration_error:.4f} days ({duration_error/duration_true*100:.1f}%)")

    # Plot results
    print("\n5. Creating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Periodogram
    ax = axes[0, 0]
    ax.plot(results['periods'], results['power'], 'b-', linewidth=0.5)
    ax.axvline(period_true, color='r', linestyle='--', label='True period')
    ax.axvline(results['period'], color='g', linestyle='--', label='Best period')
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Power (detrended SR)')
    ax.set_title('TLS Periodogram')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Chi-squared
    ax = axes[0, 1]
    ax.plot(results['periods'], results['chi2'], 'b-', linewidth=0.5)
    ax.axvline(period_true, color='r', linestyle='--', label='True period')
    ax.axvline(results['period'], color='g', linestyle='--', label='Best period')
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Chi-squared')
    ax.set_title('Chi-squared vs Period')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Phase-folded light curve at best period
    ax = axes[1, 0]
    phases = (t % results['period']) / results['period']
    ax.plot(phases, y, 'k.', alpha=0.3, markersize=2)
    # Plot best-fit model
    model_phases = np.linspace(0, 1, 1000)
    model_flux = np.ones(1000)
    duration_phase = results['duration'] / results['period']
    t0_phase = results['T0']
    in_transit = np.abs((model_phases - t0_phase + 0.5) % 1.0 - 0.5) < duration_phase / 2
    model_flux[in_transit] = 1 - results['depth']
    ax.plot(model_phases, model_flux, 'r-', linewidth=2, label='Best-fit model')
    ax.set_xlabel('Phase')
    ax.set_ylabel('Relative Flux')
    ax.set_title(f'Phase-Folded at P={results["period"]:.4f} days')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Raw light curve
    ax = axes[1, 1]
    ax.plot(t, y, 'k.', alpha=0.5, markersize=1)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Relative Flux')
    ax.set_title('Raw Light Curve')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tls_example_results.png', dpi=150, bbox_inches='tight')
    print("   ✓ Plot saved to 'tls_example_results.png'")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == '__main__':
    import sys

    # Check for --no-gpu flag
    use_gpu = '--no-gpu' not in sys.argv

    if use_gpu and not TLS_AVAILABLE:
        print("Error: TLS modules not available.")
        print("Make sure you're in the cuvarbase directory or have installed it.")
        sys.exit(1)

    try:
        run_tls_example(use_gpu=use_gpu)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
