#!/usr/bin/env python3
"""Test TLS with Keplerian duration constraints"""
import numpy as np
from cuvarbase import tls_grids

# Test parameters
ndata = 500
baseline = 50.0
period_true = 10.0
depth_true = 0.01

# Generate synthetic data
np.random.seed(42)
t = np.sort(np.random.uniform(0, baseline, ndata)).astype(np.float32)
y = np.ones(ndata, dtype=np.float32)

# Add transit
phase = (t % period_true) / period_true
in_transit = (phase < 0.01) | (phase > 0.99)
y[in_transit] -= depth_true
y += np.random.normal(0, 0.001, ndata).astype(np.float32)
dy = np.ones(ndata, dtype=np.float32) * 0.001

print("Data: {} points, transit at {:.1f} days with depth {:.3f}".format(
    len(t), period_true, depth_true))

# Generate period grid
periods = tls_grids.period_grid_ofir(
    t, R_star=1.0, M_star=1.0,
    period_min=5.0,
    period_max=20.0
).astype(np.float32)

print(f"Period grid: {len(periods)} periods from {periods[0]:.2f} to {periods[-1]:.2f}")

# Test 1: Original duration grid (fixed range for all periods)
print("\n=== Original Duration Grid (Fixed Range) ===")
# Fixed 0.5% to 15% of period
q_fixed_min = 0.005
q_fixed_max = 0.15
n_dur = 15

for i, period in enumerate(periods[:3]):  # Show first 3
    dur_min = q_fixed_min * period
    dur_max = q_fixed_max * period
    print(f"Period {period:6.2f} days: duration range {dur_min:7.4f} - {dur_max:6.4f} days "
          f"(q = {q_fixed_min:.4f} - {q_fixed_max:.4f})")

# Test 2: Keplerian duration grid (scales with stellar parameters)
print("\n=== Keplerian Duration Grid (Stellar-Parameter Aware) ===")
qmin_fac = 0.5  # Search 0.5x to 2.0x Keplerian value
qmax_fac = 2.0
R_planet = 1.0  # Earth-size planet

# Calculate Keplerian q for each period
q_kep = tls_grids.q_transit(periods, R_star=1.0, M_star=1.0, R_planet=R_planet)

for i in range(min(3, len(periods))):  # Show first 3
    period = periods[i]
    q_k = q_kep[i]
    q_min = q_k * qmin_fac
    q_max = q_k * qmax_fac
    dur_min = q_min * period
    dur_max = q_max * period
    print(f"Period {period:6.2f} days: q_keplerian = {q_k:.5f}, "
          f"search q = {q_min:.5f} - {q_max:.5f}, "
          f"durations {dur_min:7.4f} - {dur_max:6.4f} days")

# Test 3: Generate full Keplerian duration grid
print("\n=== Full Keplerian Duration Grid ===")
durations, dur_counts, q_values = tls_grids.duration_grid_keplerian(
    periods, R_star=1.0, M_star=1.0, R_planet=1.0,
    qmin_fac=0.5, qmax_fac=2.0, n_durations=15
)

print(f"Generated {len(durations)} duration arrays (one per period)")
print(f"Duration counts: min={np.min(dur_counts)}, max={np.max(dur_counts)}, "
      f"mean={np.mean(dur_counts):.1f}")

# Show examples
print("\nExample duration arrays:")
for i in [0, len(periods)//2, -1]:
    period = periods[i]
    durs = durations[i]
    print(f"  Period {period:6.2f} days: {len(durs)} durations, "
          f"range {durs[0]:7.4f} - {durs[-1]:7.4f} days "
          f"(q = {durs[0]/period:.5f} - {durs[-1]/period:.5f})")

# Test 4: Compare efficiency
print("\n=== Efficiency Comparison ===")

# Original approach: search same q range for all periods
# At short periods (5 days), q=0.005-0.15 may be too wide
# At long periods (20 days), q=0.005-0.15 may miss wide transits

period_short = 5.0
period_long = 20.0

# For Earth around Sun-like star
q_kep_short = tls_grids.q_transit(period_short, 1.0, 1.0, 1.0)
q_kep_long = tls_grids.q_transit(period_long, 1.0, 1.0, 1.0)

print(f"\nFor Earth-size planet around Sun-like star:")
print(f"  At P={period_short:4.1f} days: q_keplerian = {q_kep_short:.5f}")
print(f"    Fixed search: q = 0.00500 - 0.15000 (way too wide!)")
print(f"    Keplerian:   q = {q_kep_short*qmin_fac:.5f} - {q_kep_short*qmax_fac:.5f} (focused)")
print(f"\n  At P={period_long:4.1f} days: q_keplerian = {q_kep_long:.5f}")
print(f"    Fixed search: q = 0.00500 - 0.15000 (wastes time on impossible durations)")
print(f"    Keplerian:   q = {q_kep_long*qmin_fac:.5f} - {q_kep_long*qmax_fac:.5f} (focused)")

print("\n✓ Keplerian approach focuses search on physically plausible durations!")
print("✓ This is the same strategy BLS uses for efficient transit searches.")
