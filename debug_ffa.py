#!/usr/bin/env python3
"""Diagnostic script to verify corrected FFA butterfly algorithm."""
import sys
import os
import types
import importlib.util

import numpy as np

# Stub pycuda so we can import ffa_bls without GPU
pycuda_stub = types.ModuleType('pycuda')
pycuda_stub.autoprimaryctx = types.ModuleType('pycuda.autoprimaryctx')
pycuda_stub.driver = types.ModuleType('pycuda.driver')
pycuda_stub.gpuarray = types.ModuleType('pycuda.gpuarray')
pycuda_compiler = types.ModuleType('pycuda.compiler')
pycuda_compiler.SourceModule = None

for k, v in {
    'pycuda': pycuda_stub,
    'pycuda.autoprimaryctx': pycuda_stub.autoprimaryctx,
    'pycuda.driver': pycuda_stub.driver,
    'pycuda.gpuarray': pycuda_stub.gpuarray,
    'pycuda.compiler': pycuda_compiler,
}.items():
    sys.modules[k] = v

# Load utils
utils_path = os.path.join(os.path.dirname(__file__), 'cuvarbase', 'utils.py')
utils_spec = importlib.util.spec_from_file_location('cuvarbase.utils', utils_path)
utils_mod = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(utils_mod)
sys.modules['cuvarbase.utils'] = utils_mod
sys.modules['cuvarbase'] = types.ModuleType('cuvarbase')
sys.modules['cuvarbase'].utils = utils_mod

# Load ffa_bls
ffa_path = os.path.join(os.path.dirname(__file__), 'cuvarbase', 'ffa_bls.py')
spec = importlib.util.spec_from_file_location('cuvarbase.ffa_bls', ffa_path)
ffa = importlib.util.module_from_spec(spec)
ffa.__package__ = 'cuvarbase'
spec.loader.exec_module(ffa)


def direct_fold_at_period(t, yw, w, period, m_bins):
    """Direct BLS fold: bin all observations at the given period."""
    phases = (t / period) % 1.0
    bins = np.clip(np.floor(m_bins * phases).astype(int), 0, m_bins - 1)
    yw_bins = np.zeros(m_bins, dtype=np.float64)
    w_bins = np.zeros(m_bins, dtype=np.float64)
    for k in range(len(t)):
        yw_bins[bins[k]] += yw[k]
        w_bins[bins[k]] += w[k]
    return yw_bins, w_bins


def score_profile(yw_bins, w_bins, m_bins, qmin=0.01, qmax=0.15, dlogq=0.2):
    """BLS box scan on a folded profile."""
    nbins0 = max(1, int(np.ceil(m_bins * qmin)))
    nbinsf = min(m_bins, max(nbins0, int(np.floor(m_bins * qmax))))

    max_bls = 0.0
    for start_bin in range(m_bins):
        acc_yw = 0.0
        acc_w = 0.0
        width = 0
        next_check = nbins0

        for k in range(nbinsf):
            bin_idx = (start_bin + k) % m_bins
            acc_yw += yw_bins[bin_idx]
            acc_w += w_bins[bin_idx]
            width += 1

            if width >= next_check:
                if acc_w > 1e-10 and acc_w < 1.0 - 1e-10:
                    bls_val = acc_yw ** 2 / (acc_w * (1.0 - acc_w))
                else:
                    bls_val = 0.0
                if acc_yw > 0:
                    bls_val = 0.0
                if bls_val > max_bls:
                    max_bls = bls_val

                if dlogq > 0:
                    step = max(1, int(np.floor(dlogq * next_check)))
                    next_check += step
                else:
                    next_check += 1
                if next_check > nbinsf:
                    break
    return max_bls


# ============================================================
# Test 1: Verify drift patterns for N_p=8 with analytical shifts
# ============================================================
print("=" * 70)
print("TEST 1: Verify FFA drift patterns (N_p=8, m=16) - analytical shifts")
print("=" * 70)

N_p_test = 8
m_test = 16
n_levels_test = 3
dt_test = 1.0
P0_test = m_test * dt_test  # P0 = 16

# Create sections where section s has all signal at bin 0
test_sections = np.zeros((N_p_test, m_test), dtype=np.float32)
for s in range(N_p_test):
    test_sections[s, 0] = 1.0

fold = test_sections.copy()

# Apply butterfly with analytical shifts
for level in range(n_levels_test):
    half = 1 << level
    group_size = 1 << (level + 1)
    n_groups = N_p_test // group_size
    shifts = ffa._compute_butterfly_shifts(m_test, N_p_test, level)

    new_fold = np.zeros_like(fold)
    for g in range(n_groups):
        for d_local in range(group_size):
            d_sub = d_local % half
            d_global = g * group_size + d_local
            shift = int(shifts[d_global])
            left_idx = (2 * g) * half + d_sub
            right_idx = (2 * g + 1) * half + d_sub
            new_fold[d_global] = fold[left_idx] + np.roll(fold[right_idx], shift)
    fold = new_fold

print("\nFFA output fold profiles (each section had signal at bin 0):")
for d in range(N_p_test):
    P = (m_test + d / max(1, N_p_test - 1)) * dt_test
    nonzero_bins = {int(b): int(fold[d, b]) for b in range(m_test) if fold[d, b] > 0.1}
    print(f"  d={d} (P={P:.3f}): shift bins→count = {nonzero_bins}")

# Check that fold d=0 sums all sections without shift
direct_sum = np.sum(test_sections, axis=0)
print(f"\nFold d=0 matches direct sum: {np.allclose(fold[0], direct_sum)}")

# Check fold d=N_p-1 (maximum drift)
print(f"Fold d={N_p_test-1} nonzero bins: {np.where(fold[N_p_test-1] > 0.1)[0].tolist()}")
expected_max = list(range(N_p_test))
print(f"Expected (one per bin 0..{N_p_test-1}): bins 0-{N_p_test-1}")

# Verify each fold against brute-force
print("\nBrute-force verification (max 1-bin error expected):")
all_ok = True
for d in range(N_p_test):
    P = (m_test + d / max(1, N_p_test - 1)) * dt_test
    bf_fold = np.zeros(m_test, dtype=np.float32)
    for s in range(N_p_test):
        ideal_shift = round(m_test * ((s * P0_test / P) % 1.0)) % m_test
        bf_fold += np.roll(test_sections[s], ideal_shift)

    max_diff = np.max(np.abs(fold[d] - bf_fold))
    if max_diff > 1.5:
        print(f"  d={d}: LARGE MISMATCH (max_diff={max_diff:.1f})")
        all_ok = False
    else:
        print(f"  d={d}: OK (max_diff={max_diff:.1f})")

if all_ok:
    print("\nAll folds within acceptable tolerance!")


# ============================================================
# Test 2: Transit injection/recovery
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Transit injection/recovery (multiple periods)")
print("=" * 70)

rng = np.random.RandomState(42)
ndata = 500
baseline = 365.0
sigma = 0.1

for freq_true in [0.5, 1.0, 2.0, 0.1]:
    P_true = 1.0 / freq_true
    q = 0.05
    snr = 30
    delta = snr * sigma / np.sqrt(ndata * q * (1 - q))

    t = baseline * np.sort(rng.rand(ndata))
    phases_true = (t * freq_true) % 1.0
    y = np.zeros(ndata)
    y[phases_true < q] -= delta
    y += sigma * rng.randn(ndata)
    dy = sigma * np.ones(ndata)

    pmin = P_true * 0.8
    pmax = P_true * 1.2
    m_bins = 100

    periods, power = ffa.eebls_ffa_cpu(
        t, y, dy, pmin, pmax, m_bins=m_bins,
        qmin=0.01, qmax=0.15, dlogq=0.2,
        ignore_negative_delta_sols=True)

    if len(periods) == 0:
        print(f"  P={P_true:.1f}d: NO PERIODS GENERATED")
        continue

    best_idx = np.argmax(power)
    best_period = periods[best_idx]
    rel_err = abs(best_period - P_true) / P_true

    # Also get direct BLS at the true period for power comparison
    t_pp, yw, w_f32, yy = ffa._preprocess(t, y, dy)
    direct_yw, direct_w = direct_fold_at_period(
        t_pp, yw.astype(np.float64), w_f32.astype(np.float64), P_true, m_bins)
    direct_score = score_profile(direct_yw, direct_w, m_bins) / yy

    # FFA power at closest period to true
    closest_idx = np.argmin(np.abs(periods - P_true))
    ffa_at_true = power[closest_idx]

    status = "PASS" if rel_err < 0.05 else "FAIL"
    print(f"  P={P_true:6.1f}d: best={best_period:.4f} (err={rel_err:.4f}) "
          f"FFA_power={power[best_idx]:.4f} direct={direct_score:.4f} "
          f"ratio={ffa_at_true/direct_score:.3f} [{status}]")


# ============================================================
# Test 3: FFA vs direct BLS power comparison at specific octave
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: FFA fold d=0 vs direct BLS at P0")
print("=" * 70)

rng = np.random.RandomState(42)
ndata = 500
t = baseline * np.sort(rng.rand(ndata))
P_true = 2.0
freq_true = 1.0 / P_true
q = 0.05
delta = 30 * 0.1 / np.sqrt(ndata * q * (1 - q))
phases_true = (t * freq_true) % 1.0
y = np.zeros(ndata)
y[phases_true < q] -= delta
y += 0.1 * rng.randn(ndata)
dy = 0.1 * np.ones(ndata)

t_pp, yw, w_f32, yy = ffa._preprocess(t, y, dy)
m_bins = 100
dt = 1.6 / m_bins
m_oct = int(round(P_true / dt))
P0 = m_oct * dt
N_p = ffa._next_power_of_2(max(1, int(np.round(baseline / P0))))

sr = ffa._ffa_single_octave_cpu(
    t_pp, yw, w_f32, P0, m_oct, N_p,
    qmin=0.01, qmax=0.15, dlogq=0.2,
    ignore_negative_delta_sols=True)
sr /= yy

periods = (m_oct + np.arange(N_p, dtype=np.float64) / max(1, N_p - 1)) * dt

# Direct BLS
direct_yw, direct_w = direct_fold_at_period(
    t_pp, yw.astype(np.float64), w_f32.astype(np.float64), P0, m_oct)
direct_score = score_profile(direct_yw, direct_w, m_oct) / yy

closest_idx = np.argmin(np.abs(periods - P_true))
best_idx = np.argmax(sr)

print(f"  P0={P0:.6f}, m_oct={m_oct}, N_p={N_p}")
print(f"  FFA fold[0] power: {sr[0]:.6f}")
print(f"  Direct BLS at P0:  {direct_score:.6f}")
print(f"  Match: {abs(sr[0] - direct_score) / max(1e-10, direct_score) < 0.01}")
print(f"  FFA best: P={periods[best_idx]:.4f}, power={sr[best_idx]:.6f}")
print(f"  FFA at Ptrue: P={periods[closest_idx]:.4f}, power={sr[closest_idx]:.6f}")
