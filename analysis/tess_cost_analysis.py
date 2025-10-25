#!/usr/bin/env python3
"""
Cost-effectiveness analysis for running BLS on entire TESS catalog.

Compares CPU vs different GPU options to find the most economical solution
for large-scale transit searches.
"""

import numpy as np
from typing import Dict, List, Tuple
import json

# ============================================================================
# TESS Catalog Parameters
# ============================================================================

TESS_CATALOG = {
    'total_lightcurves': 1_000_000,  # ~1M targets with 2-min cadence
    'typical_ndata': 20_000,  # ~27 days * 720 points/day (2-min cadence)
    'nfreq_per_lightcurve': 1_000,  # Typical frequency search for BLS
    'batch_size_cpu': 1,  # CPU processes one at a time
    'batch_size_gpu': 100,  # GPU can batch efficiently
}

# From our benchmark: ndata=1000, nbatch=1
# Scaling to TESS: ndata=20000 is 20x larger → 400x slower (O(N²))
BENCHMARK_REFERENCE = {
    'ndata': 1000,
    'nfreq': 100,
    'nbatch': 1,
    'cpu_time': 447.89,  # seconds
    'gpu_time': 1.42,  # seconds (RTX 4000 Ada)
}


# ============================================================================
# Hardware Configurations
# ============================================================================

HARDWARE_OPTIONS = {
    # CPU-based solutions
    'aws_c7i_24xlarge': {
        'name': 'AWS c7i.24xlarge (96 vCPU)',
        'type': 'cpu',
        'cores': 96,
        'cpu_speedup': 96 * 0.8,  # 80% parallel efficiency
        'cost_per_hour': 4.08,  # On-demand pricing
        'spot_available': True,
        'spot_discount': 0.70,  # Typical 70% discount
    },
    'aws_c7i_48xlarge': {
        'name': 'AWS c7i.48xlarge (192 vCPU)',
        'type': 'cpu',
        'cores': 192,
        'cpu_speedup': 192 * 0.75,  # Slightly worse efficiency at scale
        'cost_per_hour': 8.16,
        'spot_available': True,
        'spot_discount': 0.70,
    },
    'hetzner_ccx63': {
        'name': 'Hetzner CCX63 (48 vCPU)',
        'type': 'cpu',
        'cores': 48,
        'cpu_speedup': 48 * 0.85,  # Good for dedicated
        'cost_per_hour': 0.82,  # Much cheaper than AWS!
        'spot_available': False,
        'spot_discount': 1.0,
    },

    # GPU-based solutions
    'runpod_rtx4000': {
        'name': 'RunPod RTX 4000 Ada',
        'type': 'gpu',
        'gpu_speedup': 315,  # Our measured result!
        'batch_multiplier': 100,  # Can process 100 lightcurves at once
        'cost_per_hour': 0.29,  # Community cloud
        'spot_available': True,
        'spot_discount': 0.80,  # Lower discount than CPU
    },
    'runpod_rtx_a5000': {
        'name': 'RunPod RTX A5000',
        'type': 'gpu',
        'gpu_speedup': 315,  # Similar to RTX 4000
        'batch_multiplier': 100,
        'cost_per_hour': 0.34,
        'spot_available': True,
        'spot_discount': 0.80,
    },
    'runpod_l40': {
        'name': 'RunPod L40',
        'type': 'gpu',
        'gpu_speedup': 315 * 1.5,  # ~1.5x faster than RTX 4000
        'batch_multiplier': 120,  # More VRAM = bigger batches
        'cost_per_hour': 0.49,
        'spot_available': True,
        'spot_discount': 0.80,
    },
    'runpod_a100_40gb': {
        'name': 'RunPod A100 40GB',
        'type': 'gpu',
        'gpu_speedup': 315 * 2.0,  # ~2x faster (bandwidth)
        'batch_multiplier': 150,
        'cost_per_hour': 0.89,
        'spot_available': True,
        'spot_discount': 0.85,
    },
    'runpod_h100': {
        'name': 'RunPod H100',
        'type': 'gpu',
        'gpu_speedup': 315 * 3.5,  # ~3.5x faster
        'batch_multiplier': 200,
        'cost_per_hour': 1.99,
        'spot_available': True,
        'spot_discount': 0.85,
    },
    'aws_p4d_24xlarge': {
        'name': 'AWS p4d.24xlarge (8x A100 80GB)',
        'type': 'gpu',
        'gpu_count': 8,
        'gpu_speedup': 315 * 2.5,  # 80GB version slightly better
        'batch_multiplier': 200,
        'cost_per_hour': 32.77,
        'spot_available': True,
        'spot_discount': 0.70,
    },
}


# ============================================================================
# Cost Calculation Functions
# ============================================================================

def scale_benchmark_time(ndata_target: int, nfreq_target: int,
                        base_time: float, base_ndata: int, base_nfreq: int) -> float:
    """
    Scale benchmark time using O(N²×Nfreq) complexity.

    Parameters
    ----------
    ndata_target, nfreq_target : int
        Target problem size
    base_time : float
        Reference time in seconds
    base_ndata, base_nfreq : int
        Reference problem size

    Returns
    -------
    scaled_time : float
        Estimated time in seconds
    """
    scale_ndata = (ndata_target / base_ndata) ** 2  # O(N²)
    scale_nfreq = nfreq_target / base_nfreq  # O(Nfreq)
    return base_time * scale_ndata * scale_nfreq


def calculate_cost(hardware: Dict, catalog: Dict, use_spot: bool = True) -> Dict:
    """
    Calculate total cost and time to process TESS catalog.

    Returns
    -------
    result : dict
        Contains total_hours, total_cost, cost_per_lightcurve, etc.
    """
    # Scale benchmark to TESS lightcurve size
    base_cpu_time = scale_benchmark_time(
        catalog['typical_ndata'], catalog['nfreq_per_lightcurve'],
        BENCHMARK_REFERENCE['cpu_time'],
        BENCHMARK_REFERENCE['ndata'], BENCHMARK_REFERENCE['nfreq']
    )

    base_gpu_time = scale_benchmark_time(
        catalog['typical_ndata'], catalog['nfreq_per_lightcurve'],
        BENCHMARK_REFERENCE['gpu_time'],
        BENCHMARK_REFERENCE['ndata'], BENCHMARK_REFERENCE['nfreq']
    )

    total_lightcurves = catalog['total_lightcurves']

    if hardware['type'] == 'cpu':
        # CPU: parallel processing across cores
        time_per_lc = base_cpu_time / hardware['cpu_speedup']
        total_seconds = time_per_lc * total_lightcurves

    else:  # GPU
        # GPU: speedup from GPU acceleration
        time_per_lc_single = base_cpu_time / hardware['gpu_speedup']

        # Batching: GPU can process multiple lightcurves simultaneously
        # This reduces overhead and improves efficiency
        batch_size = hardware['batch_multiplier']
        num_batches = (total_lightcurves + batch_size - 1) // batch_size

        # Time per batch (assuming linear scaling with batch size)
        time_per_batch = time_per_lc_single * batch_size

        # For multi-GPU systems
        gpu_count = hardware.get('gpu_count', 1)
        time_per_batch = time_per_batch / gpu_count

        total_seconds = time_per_batch * num_batches

    total_hours = total_seconds / 3600

    # Calculate cost
    cost_per_hour = hardware['cost_per_hour']
    if use_spot and hardware['spot_available']:
        cost_per_hour *= hardware['spot_discount']

    total_cost = total_hours * cost_per_hour
    cost_per_lightcurve = total_cost / total_lightcurves

    return {
        'hardware': hardware['name'],
        'type': hardware['type'],
        'using_spot': use_spot and hardware['spot_available'],
        'total_hours': total_hours,
        'total_days': total_hours / 24,
        'total_cost': total_cost,
        'cost_per_lightcurve': cost_per_lightcurve * 1000,  # Convert to millicents
        'cost_per_hour': cost_per_hour,
        'time_per_lightcurve': total_seconds / total_lightcurves,  # seconds
    }


# ============================================================================
# Analysis and Visualization
# ============================================================================

def run_cost_analysis(catalog: Dict = TESS_CATALOG) -> List[Dict]:
    """Run cost analysis for all hardware options."""
    results = []

    for hw_id, hardware in HARDWARE_OPTIONS.items():
        # On-demand pricing
        result_ondemand = calculate_cost(hardware, catalog, use_spot=False)
        result_ondemand['pricing'] = 'on-demand'
        result_ondemand['hw_id'] = hw_id
        results.append(result_ondemand)

        # Spot/preemptible pricing if available
        if hardware['spot_available']:
            result_spot = calculate_cost(hardware, catalog, use_spot=True)
            result_spot['pricing'] = 'spot'
            result_spot['hw_id'] = hw_id
            results.append(result_spot)

    return results


def print_analysis(results: List[Dict]):
    """Print formatted cost analysis."""
    print("=" * 100)
    print("COST ANALYSIS: TESS CATALOG BLS SEARCH (SINGLE GPU/SERVER)")
    print("=" * 100)
    print(f"\nCatalog: {TESS_CATALOG['total_lightcurves']:,} lightcurves")
    print(f"Typical size: {TESS_CATALOG['typical_ndata']:,} observations")
    print(f"Frequency grid: {TESS_CATALOG['nfreq_per_lightcurve']:,} points")
    print(f"\n⚠️  NOTE: Times shown are for a SINGLE GPU/server instance.")
    print(f"⚠️  To complete in reasonable time, use MULTIPLE GPUs in parallel!")
    print()

    # Sort by total cost
    results_sorted = sorted(results, key=lambda x: x['total_cost'])

    print(f"{'Rank':<5} {'Hardware':<40} {'Pricing':<10} {'Time':<15} {'Total Cost':<15} {'$/1k LC':<12}")
    print("-" * 100)

    for i, r in enumerate(results_sorted, 1):
        time_str = f"{r['total_days']:.1f} days" if r['total_days'] < 30 else f"{r['total_days']/30:.1f} months"
        cost_str = f"${r['total_cost']:,.2f}"
        cost_per_1k = f"${r['cost_per_lightcurve']:.2f}"

        print(f"{i:<5} {r['hardware']:<40} {r['pricing']:<10} {time_str:<15} {cost_str:<15} {cost_per_1k:<12}")

    # Highlight top 3
    print("\n" + "=" * 100)
    print("TOP 3 MOST COST-EFFECTIVE SOLUTIONS:")
    print("=" * 100)

    for i, r in enumerate(results_sorted[:3], 1):
        print(f"\n#{i}: {r['hardware']} ({r['pricing']})")
        print(f"  Total Cost: ${r['total_cost']:,.2f}")
        print(f"  Total Time: {r['total_days']:.1f} days ({r['total_hours']:.1f} hours)")
        print(f"  Cost per 1000 LC: ${r['cost_per_lightcurve']:.2f}")
        print(f"  Time per LC: {r['time_per_lightcurve']:.2f} seconds")

        # Calculate savings vs worst option
        worst_cost = results_sorted[-1]['total_cost']
        savings = worst_cost - r['total_cost']
        savings_pct = (savings / worst_cost) * 100
        print(f"  Savings vs worst: ${savings:,.2f} ({savings_pct:.1f}%)")

    # Analysis insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS:")
    print("=" * 100)

    best = results_sorted[0]
    best_cpu = [r for r in results_sorted if r['type'] == 'cpu'][0]
    best_gpu = [r for r in results_sorted if r['type'] == 'gpu'][0]

    print(f"\n1. OVERALL WINNER: {best['hardware']}")
    print(f"   Cost: ${best['total_cost']:,.2f}, Time: {best['total_days']:.1f} days")

    print(f"\n2. BEST CPU SOLUTION: {best_cpu['hardware']}")
    print(f"   Cost: ${best_cpu['total_cost']:,.2f}, Time: {best_cpu['total_days']:.1f} days")

    print(f"\n3. BEST GPU SOLUTION: {best_gpu['hardware']}")
    print(f"   Cost: ${best_gpu['total_cost']:,.2f}, Time: {best_gpu['total_days']:.1f} days")

    cost_ratio = best_cpu['total_cost'] / best_gpu['total_cost']
    time_ratio = best_cpu['total_hours'] / best_gpu['total_hours']

    print(f"\n4. CPU vs GPU COMPARISON:")
    print(f"   GPU is {cost_ratio:.1f}x MORE cost-effective")
    print(f"   GPU is {time_ratio:.1f}x FASTER")

    # Practical recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS:")
    print("=" * 100)

    if best['type'] == 'gpu':
        print(f"\n✓ USE GPU: {best['hardware']}")
        print(f"  - Most cost-effective for large-scale BLS searches")
        print(f"  - ${best['total_cost']:,.0f} total cost")
        print(f"  - {best['total_days']:.0f} days to completion")
        if best['using_spot']:
            print(f"  - Using spot instances (check interruption rates)")
            print(f"  - Consider checkpointing every {min(100, int(best['total_hours']/10))} hours")

    # Risk analysis
    print(f"\n⚠ RISK CONSIDERATIONS:")
    if best['using_spot']:
        print(f"  - Spot instances can be interrupted")
        print(f"  - Implement checkpointing/resumption")
        print(f"  - Monitor spot price volatility")

    print(f"  - Validate results on subset before full run")
    print(f"  - Budget buffer: add 10-20% for failures/retries")

    # Parallel GPU analysis
    print(f"\n🚀 PARALLEL GPU DEPLOYMENT:")
    print(f"  Single {best['hardware']}: {best['total_days']:.0f} days (${best['total_cost']:,.0f})")
    print()
    for target_days in [30, 90, 365]:
        num_gpus = int(best['total_days'] / target_days) + 1
        parallel_cost = best['total_cost']  # Same total cost regardless of parallelization
        cost_per_gpu = parallel_cost / num_gpus
        print(f"  To finish in {target_days} days ({target_days/30:.0f} months):")
        print(f"    - GPUs needed: {num_gpus:,}")
        print(f"    - Total cost: ${parallel_cost:,.0f} (same)")
        print(f"    - Cost per GPU: ${cost_per_gpu:,.0f}")
        print(f"    - Throughput: {TESS_CATALOG['total_lightcurves']/target_days:,.0f} LC/day")
        print()

    # Scaling analysis
    print(f"📈 SCALING TO LARGER CATALOGS:")
    print(f"  For 2x more lightcurves:")
    print(f"    - Cost: ${best['total_cost']*2:,.0f}")
    print(f"    - Time (single GPU): {best['total_days']*2:.0f} days")
    print(f"  For 10x more lightcurves:")
    print(f"    - Cost: ${best['total_cost']*10:,.0f}")
    print(f"    - Time (single GPU): {best['total_days']*10:.0f} days")


def sensitivity_analysis():
    """Analyze how results change with different assumptions."""
    print("\n" + "=" * 100)
    print("SENSITIVITY ANALYSIS")
    print("=" * 100)

    scenarios = {
        'base': {'total_lightcurves': 1_000_000, 'typical_ndata': 20_000, 'nfreq_per_lightcurve': 1_000},
        'fine_grid': {'total_lightcurves': 1_000_000, 'typical_ndata': 20_000, 'nfreq_per_lightcurve': 5_000},
        'multi_sector': {'total_lightcurves': 1_000_000, 'typical_ndata': 60_000, 'nfreq_per_lightcurve': 1_000},
        'full_tess_multi': {'total_lightcurves': 2_000_000, 'typical_ndata': 60_000, 'nfreq_per_lightcurve': 2_000},
    }

    for scenario_name, params in scenarios.items():
        catalog = TESS_CATALOG.copy()
        catalog.update(params)

        results = run_cost_analysis(catalog)
        best = sorted(results, key=lambda x: x['total_cost'])[0]

        print(f"\n{scenario_name.upper().replace('_', ' ')}:")
        print(f"  Lightcurves: {catalog['total_lightcurves']:,}")
        print(f"  Observations: {catalog['typical_ndata']:,}")
        print(f"  Best solution: {best['hardware']} ({best['pricing']})")
        print(f"  Cost: ${best['total_cost']:,.2f}")
        print(f"  Time: {best['total_days']:.1f} days")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run complete cost analysis."""
    results = run_cost_analysis()
    print_analysis(results)
    sensitivity_analysis()

    # Save results
    with open('tess_cost_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: tess_cost_analysis.json")


if __name__ == '__main__':
    main()
