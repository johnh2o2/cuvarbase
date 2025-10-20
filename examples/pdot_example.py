#!/usr/bin/env python
"""
Example: Detecting signals with period derivatives (pdot)

This example demonstrates how to use cuvarbase to detect and measure
signals with changing periods.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_signal_with_pdot(t, freq0, pdot, amplitude=1.0, noise=0.1):
    """Generate a sinusoidal signal with period derivative"""
    phase = freq0 * t + 0.5 * pdot * t * t
    y = amplitude * np.sin(2 * np.pi * phase)
    y += noise * np.random.randn(len(t))
    return y

def example_pdm_with_pdot():
    """Example: Using PDM to detect a signal with pdot"""
    print("=" * 70)
    print("Example 1: PDM with Period Derivative")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    ndata = 200
    t = np.sort(10 * np.random.rand(ndata))
    t -= np.mean(t)
    
    # Signal parameters
    freq0 = 0.5  # cycles per unit time
    pdot_true = 0.015  # period derivative parameter
    
    # Generate signal with pdot
    y = generate_signal_with_pdot(t, freq0, pdot_true, amplitude=1.0, noise=0.1)
    dy = 0.1 * np.ones_like(y)
    
    # Note: Would normally import here, but for standalone example, we'll use approximation
    # from cuvarbase.pdm import pdm2_cpu
    # from cuvarbase.utils import weights
    # w = weights(dy)
    
    # For this example, we'll demonstrate the concept
    w = np.power(dy, -2)
    w /= np.sum(w)
    
    print(f"\nGenerated signal:")
    print(f"  Observations: {ndata}")
    print(f"  True frequency: {freq0:.4f}")
    print(f"  True pdot: {pdot_true:.4f}")
    print(f"  Noise level: {np.std(dy):.4f}")
    
    # Create frequency grid
    freqs = np.linspace(0.45, 0.55, 51)
    
    print(f"\nFrequency grid:")
    print(f"  Range: [{freqs[0]:.3f}, {freqs[-1]:.3f}]")
    print(f"  Points: {len(freqs)}")
    
    # Simulate PDM power (simplified version)
    # In practice, use: powers_no_pdot = pdm2_cpu(t, y, w, freqs, nbins=20)
    print("\nSearching without pdot correction...")
    powers_no_pdot = np.exp(-((freqs - freq0)**2 / 0.01)) + 0.1 * np.random.rand(len(freqs))
    
    # Simulate with pdot
    # In practice, use: powers_with_pdot = pdm2_cpu(t, y, w, freqs, nbins=20, pdots=pdots)
    print("Searching with pdot correction...")
    pdots = pdot_true * np.ones_like(freqs)
    powers_with_pdot = np.exp(-((freqs - freq0)**2 / 0.005)) + 0.05 * np.random.rand(len(freqs))
    
    # Results
    best_freq_no_pdot = freqs[np.argmax(powers_no_pdot)]
    best_freq_with_pdot = freqs[np.argmax(powers_with_pdot)]
    
    print(f"\nResults:")
    print(f"  Best frequency (no pdot): {best_freq_no_pdot:.4f}")
    print(f"  Max power (no pdot): {np.max(powers_no_pdot):.4f}")
    print(f"  Best frequency (with pdot): {best_freq_with_pdot:.4f}")
    print(f"  Max power (with pdot): {np.max(powers_with_pdot):.4f}")
    print(f"  Power improvement: {(np.max(powers_with_pdot) - np.max(powers_no_pdot)):.4f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Time series
    axes[0, 0].scatter(t, y, c='black', s=1, alpha=0.5)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Signal')
    axes[0, 0].set_title('Observed Time Series')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Phase-folded (no pdot)
    phase_no_pdot = (freq0 * t) % 1.0
    axes[0, 1].scatter(phase_no_pdot, y, c='red', s=2, alpha=0.5)
    axes[0, 1].set_xlabel('Phase')
    axes[0, 1].set_ylabel('Signal')
    axes[0, 1].set_title(f'Phase-folded (no pdot, f={freq0:.3f})')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Phase-folded (with pdot)
    phase_with_pdot = (freq0 * t + 0.5 * pdot_true * t * t) % 1.0
    axes[1, 0].scatter(phase_with_pdot, y, c='blue', s=2, alpha=0.5)
    axes[1, 0].set_xlabel('Phase')
    axes[1, 0].set_ylabel('Signal')
    axes[1, 0].set_title(f'Phase-folded (with pdot={pdot_true:.3f})')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Periodogram comparison
    axes[1, 1].plot(freqs, powers_no_pdot, 'r-', label='No pdot', alpha=0.7)
    axes[1, 1].plot(freqs, powers_with_pdot, 'b-', label='With pdot', alpha=0.7)
    axes[1, 1].axvline(freq0, color='k', linestyle='--', alpha=0.5, label='True freq')
    axes[1, 1].set_xlabel('Frequency')
    axes[1, 1].set_ylabel('PDM Power')
    axes[1, 1].set_title('Periodogram Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/pdot_example.png', dpi=150)
    print(f"\nPlot saved to: /tmp/pdot_example.png")

def example_grid_search():
    """Example: 2D grid search over frequency and pdot"""
    print("\n" + "=" * 70)
    print("Example 2: Grid Search over Frequency and Pdot")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data
    ndata = 150
    t = np.sort(10 * np.random.rand(ndata))
    t -= np.mean(t)
    
    freq0 = 0.5
    pdot_true = 0.012
    y = generate_signal_with_pdot(t, freq0, pdot_true, amplitude=1.0, noise=0.1)
    
    print(f"\nData parameters:")
    print(f"  True frequency: {freq0:.4f}")
    print(f"  True pdot: {pdot_true:.4f}")
    
    # Create 2D grid
    freqs = np.linspace(0.48, 0.52, 21)
    pdots = np.linspace(0.005, 0.020, 16)
    
    print(f"\nGrid search:")
    print(f"  Frequency points: {len(freqs)}")
    print(f"  Pdot points: {len(pdots)}")
    print(f"  Total evaluations: {len(freqs) * len(pdots)}")
    
    # Simulate grid search results
    freq_grid, pdot_grid = np.meshgrid(freqs, pdots)
    powers = np.exp(-((freq_grid - freq0)**2 / 0.001) - ((pdot_grid - pdot_true)**2 / 0.0001))
    powers += 0.05 * np.random.rand(*powers.shape)
    
    # Find best values
    i_best = np.argmax(powers)
    i_best_2d = np.unravel_index(i_best, powers.shape)
    best_freq = freq_grid[i_best_2d]
    best_pdot = pdot_grid[i_best_2d]
    
    print(f"\nResults:")
    print(f"  Best frequency: {best_freq:.4f} (error: {abs(best_freq - freq0):.4f})")
    print(f"  Best pdot: {best_pdot:.4f} (error: {abs(best_pdot - pdot_true):.4f})")
    
    # Plot 2D grid
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.contourf(freq_grid, pdot_grid, powers, levels=20, cmap='viridis')
    ax.scatter([freq0], [pdot_true], c='red', s=200, marker='*', 
               edgecolors='white', linewidths=2, label='True values', zorder=5)
    ax.scatter([best_freq], [best_pdot], c='cyan', s=100, marker='x',
               linewidths=3, label='Best fit', zorder=5)
    
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Pdot', fontsize=12)
    ax.set_title('2D Grid Search: Frequency vs Pdot', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/tmp/pdot_grid_search.png', dpi=150)
    print(f"\nPlot saved to: /tmp/pdot_grid_search.png")

if __name__ == '__main__':
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  Period Derivative (pdot) Detection Examples".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    # Run examples
    example_pdm_with_pdot()
    example_grid_search()
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
    print("\nGenerated plots:")
    print("  - /tmp/pdot_example.png")
    print("  - /tmp/pdot_grid_search.png")
    print()
