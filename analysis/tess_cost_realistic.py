#!/usr/bin/env python3
"""
Realistic cost-effectiveness analysis for running BLS on entire TESS catalog.

This analysis:
1. Uses realistic TESS parameters (10k-30k datapoints, 5-7M objects)
2. Compares against astropy BoxLeastSquares as CPU baseline
3. Accounts for GPU batching efficiency
4. Considers both sparse BLS and traditional (Keplerian) BLS
5. Analyzes parallel GPU deployment strategies
"""

import numpy as np
from typing import Dict, List, Tuple
import json

# ============================================================================
# TESS Catalog - Realistic Parameters
# ============================================================================

TESS_SCENARIOS = {
    'single_sector': {
        'description': 'Single 27-day sector, 2-min cadence',
        'total_lightcurves': 5_000_000,  # ~5M targets from TESS
        'typical_ndata': 19_440,  # 27 days * 720 obs/day
        'nfreq_per_lightcurve': 1_000,  # Typical BLS frequency grid
    },
    'multi_sector_3x': {
        'description': '3 sectors (81 days)',
        'total_lightcurves': 2_000_000,  # Fewer have 3+ sectors
        'typical_ndata': 58_320,  # 3 * 19,440
        'nfreq_per_lightcurve': 1_500,  # Slightly finer for longer baseline
    },
    'single_sector_conservative': {
        'description': 'Single sector, conservative frequency grid',
        'total_lightcurves': 5_000_000,
        'typical_ndata': 20_000,
        'nfreq_per_lightcurve': 500,  # Coarser but faster
    },
}

# ============================================================================
# Benchmark Reference Data
# ============================================================================

# From actual benchmarks on RTX 4000 Ada Generation
# ndata=1000, nfreq=100
BENCHMARK_SPARSE_BLS = {
    'ndata': 1000,
    'nfreq': 100,
    'nbatch': 1,
    'cpu_time': 447.89,  # cuvarbase sparse_bls_cpu
    'gpu_time_nbatch1': 1.42,  # Single lightcurve
    'gpu_time_nbatch10': 13.42,  # 10 lightcurves batched
}

# Estimated performance for astropy BoxLeastSquares
# Astropy uses binned BLS which is O(N log N) for sorting + O(N * Nfreq) for search
# This is MUCH faster than sparse BLS for large ndata
BENCHMARK_ASTROPY_BLS = {
    'description': 'Estimated from astropy BoxLeastSquares',
    'ndata': 1000,
    'nfreq': 100,
    'nbatch': 1,
    'cpu_time': 5.0,  # Estimate: ~100x faster than sparse BLS
    'complexity_ndata': 1.2,  # O(N log N) ≈ N^1.2 for practical purposes
    'complexity_nfreq': 1.0,  # O(Nfreq)
}

# Keplerian assumption BLS (only tests transit-like durations)
# Even faster than binned BLS
BENCHMARK_KEPLERIAN_BLS = {
    'description': 'BLS with Keplerian duration assumption',
    'ndata': 1000,
    'nfreq': 100,
    'nbatch': 1,
    'cpu_time': 1.0,  # Estimate: ~5x faster than astropy
    'complexity_ndata': 1.2,  # Similar to binned BLS
    'complexity_nfreq': 1.0,
}

# ============================================================================
# Hardware Configurations
# ============================================================================

HARDWARE_OPTIONS = {
    # GPU options - focusing on cost-effective choices
    'runpod_rtx4000': {
        'name': 'RunPod RTX 4000 Ada',
        'type': 'gpu',
        'gpu_speedup_single': 315,  # For nbatch=1
        'gpu_speedup_batch10': 33,  # For nbatch=10 (measured)
        'batch_efficiency': 0.94,  # 13.42s for 10x work vs 1.42s = 9.4x throughput
        'optimal_batch_size': 10,
        'cost_per_hour': 0.29,
        'spot_available': True,
        'spot_discount': 0.80,
    },
    'runpod_l40': {
        'name': 'RunPod L40',
        'type': 'gpu',
        'gpu_speedup_single': 315 * 1.5,  # Estimated 1.5x faster
        'gpu_speedup_batch10': 33 * 1.5,
        'batch_efficiency': 0.94,
        'optimal_batch_size': 12,
        'cost_per_hour': 0.49,
        'spot_available': True,
        'spot_discount': 0.80,
    },
    'runpod_a100': {
        'name': 'RunPod A100 40GB',
        'type': 'gpu',
        'gpu_speedup_single': 315 * 2.0,  # ~2x faster bandwidth
        'gpu_speedup_batch10': 33 * 2.0,
        'batch_efficiency': 0.94,
        'optimal_batch_size': 15,
        'cost_per_hour': 0.89,
        'spot_available': True,
        'spot_discount': 0.85,
    },

    # CPU options
    'hetzner_ccx63': {
        'name': 'Hetzner CCX63 (48 vCPU)',
        'type': 'cpu',
        'cores': 48,
        'parallel_efficiency': 0.85,  # 85% efficiency
        'cost_per_hour': 0.82,
        'spot_available': False,
    },
    'aws_c7i_24xl': {
        'name': 'AWS c7i.24xlarge (96 vCPU)',
        'type': 'cpu',
        'cores': 96,
        'parallel_efficiency': 0.80,
        'cost_per_hour': 4.08,
        'spot_available': True,
        'spot_discount': 0.70,
    },
}

# ============================================================================
# Cost Calculation Functions
# ============================================================================

def scale_benchmark_time(ndata_target: int, nfreq_target: int,
                        base_time: float, base_ndata: int, base_nfreq: int,
                        complexity_ndata: float = 2.0, complexity_nfreq: float = 1.0) -> float:
    """
    Scale benchmark time using algorithm complexity.

    Parameters
    ----------
    complexity_ndata : float
        Exponent for ndata scaling (2.0 for sparse BLS, 1.2 for binned BLS)
    complexity_nfreq : float
        Exponent for nfreq scaling (1.0 for all BLS variants)
    """
    scale_ndata = (ndata_target / base_ndata) ** complexity_ndata
    scale_nfreq = (nfreq_target / base_nfreq) ** complexity_nfreq
    return base_time * scale_ndata * scale_nfreq


def calculate_cost_sparse_bls_gpu(hardware: Dict, catalog: Dict, use_spot: bool = True) -> Dict:
    """Calculate cost for sparse BLS on GPU."""
    # Scale to TESS lightcurve size
    time_per_lc = scale_benchmark_time(
        catalog['typical_ndata'], catalog['nfreq_per_lightcurve'],
        BENCHMARK_SPARSE_BLS['gpu_time_nbatch1'],
        BENCHMARK_SPARSE_BLS['ndata'], BENCHMARK_SPARSE_BLS['nfreq'],
        complexity_ndata=2.0, complexity_nfreq=1.0
    )

    # Account for batching efficiency
    batch_size = hardware.get('optimal_batch_size', 10)
    batch_efficiency = hardware.get('batch_efficiency', 0.94)
    effective_time_per_lc = time_per_lc / (batch_size * batch_efficiency)

    total_lightcurves = catalog['total_lightcurves']
    total_seconds = effective_time_per_lc * total_lightcurves
    total_hours = total_seconds / 3600

    # Calculate cost
    cost_per_hour = hardware['cost_per_hour']
    if use_spot and hardware.get('spot_available', False):
        cost_per_hour *= hardware['spot_discount']

    total_cost = total_hours * cost_per_hour

    return {
        'hardware': hardware['name'],
        'algorithm': 'sparse_bls',
        'type': 'gpu',
        'using_spot': use_spot and hardware.get('spot_available', False),
        'total_hours': total_hours,
        'total_days': total_hours / 24,
        'total_cost': total_cost,
        'cost_per_lightcurve': total_cost / total_lightcurves,
        'time_per_lightcurve': total_seconds / total_lightcurves,
        'batch_size': batch_size,
        'cost_per_hour': cost_per_hour,
    }


def calculate_cost_cpu(hardware: Dict, catalog: Dict, benchmark: Dict,
                       algorithm: str, use_spot: bool = False) -> Dict:
    """Calculate cost for CPU-based BLS."""
    # Scale to TESS lightcurve size
    time_per_lc = scale_benchmark_time(
        catalog['typical_ndata'], catalog['nfreq_per_lightcurve'],
        benchmark['cpu_time'],
        benchmark['ndata'], benchmark['nfreq'],
        complexity_ndata=benchmark.get('complexity_ndata', 2.0),
        complexity_nfreq=benchmark.get('complexity_nfreq', 1.0)
    )

    # Parallel processing across cores
    cores = hardware['cores']
    parallel_efficiency = hardware['parallel_efficiency']
    effective_speedup = cores * parallel_efficiency

    time_per_lc_parallel = time_per_lc / effective_speedup

    total_lightcurves = catalog['total_lightcurves']
    total_seconds = time_per_lc_parallel * total_lightcurves
    total_hours = total_seconds / 3600

    # Calculate cost
    cost_per_hour = hardware['cost_per_hour']
    if use_spot and hardware.get('spot_available', False):
        cost_per_hour *= hardware['spot_discount']

    total_cost = total_hours * cost_per_hour

    return {
        'hardware': hardware['name'],
        'algorithm': algorithm,
        'type': 'cpu',
        'using_spot': use_spot and hardware.get('spot_available', False),
        'total_hours': total_hours,
        'total_days': total_hours / 24,
        'total_cost': total_cost,
        'cost_per_lightcurve': total_cost / total_lightcurves,
        'time_per_lightcurve': total_seconds / total_lightcurves,
        'cores': cores,
        'cost_per_hour': cost_per_hour,
    }


def run_comprehensive_analysis(catalog_name: str = 'single_sector'):
    """Run comprehensive cost analysis for a TESS catalog scenario."""
    catalog = TESS_SCENARIOS[catalog_name]

    results = []

    # GPU: sparse BLS
    for hw_id in ['runpod_rtx4000', 'runpod_l40', 'runpod_a100']:
        hardware = HARDWARE_OPTIONS[hw_id]

        # Spot pricing
        result = calculate_cost_sparse_bls_gpu(hardware, catalog, use_spot=True)
        result['hw_id'] = hw_id
        result['pricing'] = 'spot'
        results.append(result)

        # On-demand
        result = calculate_cost_sparse_bls_gpu(hardware, catalog, use_spot=False)
        result['hw_id'] = hw_id
        result['pricing'] = 'on-demand'
        results.append(result)

    # CPU: sparse BLS (cuvarbase baseline)
    for hw_id in ['hetzner_ccx63', 'aws_c7i_24xl']:
        hardware = HARDWARE_OPTIONS[hw_id]

        result = calculate_cost_cpu(hardware, catalog, BENCHMARK_SPARSE_BLS,
                                    'sparse_bls_cpu', use_spot=False)
        result['hw_id'] = hw_id
        result['pricing'] = 'on-demand' if not hardware['spot_available'] else 'spot'
        results.append(result)

        if hardware['spot_available']:
            result = calculate_cost_cpu(hardware, catalog, BENCHMARK_SPARSE_BLS,
                                       'sparse_bls_cpu', use_spot=True)
            result['hw_id'] = hw_id
            result['pricing'] = 'spot'
            results.append(result)

    # CPU: astropy BLS (more realistic baseline)
    for hw_id in ['hetzner_ccx63', 'aws_c7i_24xl']:
        hardware = HARDWARE_OPTIONS[hw_id]

        result = calculate_cost_cpu(hardware, catalog, BENCHMARK_ASTROPY_BLS,
                                   'astropy_bls', use_spot=False)
        result['hw_id'] = hw_id
        result['pricing'] = 'on-demand' if not hardware['spot_available'] else 'spot'
        results.append(result)

        if hardware['spot_available']:
            result = calculate_cost_cpu(hardware, catalog, BENCHMARK_ASTROPY_BLS,
                                       'astropy_bls', use_spot=True)
            result['hw_id'] = hw_id
            result['pricing'] = 'spot'
            results.append(result)

    # CPU: Keplerian BLS (fastest CPU option)
    for hw_id in ['hetzner_ccx63', 'aws_c7i_24xl']:
        hardware = HARDWARE_OPTIONS[hw_id]

        result = calculate_cost_cpu(hardware, catalog, BENCHMARK_KEPLERIAN_BLS,
                                   'keplerian_bls', use_spot=False)
        result['hw_id'] = hw_id
        result['pricing'] = 'on-demand' if not hardware['spot_available'] else 'spot'
        results.append(result)

        if hardware['spot_available']:
            result = calculate_cost_cpu(hardware, catalog, BENCHMARK_KEPLERIAN_BLS,
                                       'keplerian_bls', use_spot=True)
            result['hw_id'] = hw_id
            result['pricing'] = 'spot'
            results.append(result)

    return catalog, results


def print_analysis(catalog: Dict, results: List[Dict]):
    """Print formatted analysis."""
    print("=" * 120)
    print("REALISTIC TESS CATALOG BLS COST ANALYSIS")
    print("=" * 120)
    print(f"\nScenario: {catalog['description']}")
    print(f"Total lightcurves: {catalog['total_lightcurves']:,}")
    print(f"Observations per LC: {catalog['typical_ndata']:,}")
    print(f"Frequency grid points: {catalog['nfreq_per_lightcurve']:,}")
    print(f"\n⚠️  Times shown are for SINGLE instance. Use parallel deployment for faster completion.")
    print()

    # Sort by cost
    results_sorted = sorted(results, key=lambda x: x['total_cost'])

    # Print table
    print(f"{'Rank':<5} {'Hardware':<35} {'Algorithm':<18} {'Pricing':<10} {'Days':<12} {'Cost':<15} {'$/LC'}")
    print("-" * 120)

    for i, r in enumerate(results_sorted[:20], 1):  # Top 20
        days_str = f"{r['total_days']:.1f}"
        cost_str = f"${r['total_cost']:,.0f}"
        cost_per_lc = f"${r['cost_per_lightcurve']:.4f}"

        print(f"{i:<5} {r['hardware']:<35} {r['algorithm']:<18} {r['pricing']:<10} {days_str:<12} {cost_str:<15} {cost_per_lc}")

    # Analysis
    print("\n" + "=" * 120)
    print("KEY FINDINGS:")
    print("=" * 120)

    best_overall = results_sorted[0]
    best_gpu = [r for r in results_sorted if r['type'] == 'gpu'][0]
    best_cpu = [r for r in results_sorted if r['type'] == 'cpu'][0]
    best_astropy = [r for r in results_sorted if r['algorithm'] == 'astropy_bls'][0]
    best_keplerian = [r for r in results_sorted if r['algorithm'] == 'keplerian_bls'][0]

    print(f"\n1. BEST OVERALL: {best_overall['hardware']} ({best_overall['algorithm']})")
    print(f"   Cost: ${best_overall['total_cost']:,.0f}")
    print(f"   Time: {best_overall['total_days']:.0f} days on single instance")
    print(f"   Cost per LC: ${best_overall['cost_per_lightcurve']:.4f}")

    print(f"\n2. BEST GPU: {best_gpu['hardware']}")
    print(f"   Cost: ${best_gpu['total_cost']:,.0f}")
    print(f"   Time: {best_gpu['total_days']:.0f} days")
    print(f"   Batch size: {best_gpu.get('batch_size', 'N/A')}")

    print(f"\n3. BEST CPU (sparse BLS): {best_cpu['hardware']}")
    print(f"   Cost: ${best_cpu['total_cost']:,.0f}")
    print(f"   Time: {best_cpu['total_days']:.0f} days")

    print(f"\n4. BEST CPU (astropy BLS): {best_astropy['hardware']}")
    print(f"   Cost: ${best_astropy['total_cost']:,.0f}")
    print(f"   Time: {best_astropy['total_days']:.0f} days")
    print(f"   Speedup vs sparse BLS: {best_cpu['total_cost']/best_astropy['total_cost']:.1f}x cheaper")

    print(f"\n5. BEST CPU (Keplerian BLS): {best_keplerian['hardware']}")
    print(f"   Cost: ${best_keplerian['total_cost']:,.0f}")
    print(f"   Time: {best_keplerian['total_days']:.0f} days")
    print(f"   Speedup vs sparse BLS: {best_cpu['total_cost']/best_keplerian['total_cost']:.1f}x cheaper")

    # Parallel deployment
    print("\n" + "=" * 120)
    print("PARALLEL DEPLOYMENT (using best option):")
    print("=" * 120)

    best = best_overall
    print(f"\nUsing: {best['hardware']} ({best['algorithm']}, {best['pricing']})")
    print(f"Single instance: {best['total_days']:.0f} days, ${best['total_cost']:,.0f} total cost")
    print()

    for target_days in [30, 90, 180, 365]:
        num_instances = int(np.ceil(best['total_days'] / target_days))
        cost_per_instance = best['total_cost'] / num_instances  # Cost amortized
        throughput = catalog['total_lightcurves'] / target_days

        print(f"  Complete in {target_days} days ({target_days/30:.1f} months):")
        print(f"    - Instances needed: {num_instances:,}")
        print(f"    - Total cost: ${best['total_cost']:,.0f} (same, amortized)")
        print(f"    - Cost per instance: ${cost_per_instance:,.0f}")
        print(f"    - Throughput: {throughput:,.0f} LC/day")
        print()


def main():
    """Run analysis for all scenarios."""
    for scenario_name in ['single_sector', 'multi_sector_3x', 'single_sector_conservative']:
        catalog, results = run_comprehensive_analysis(scenario_name)
        print_analysis(catalog, results)
        print("\n\n")

        # Save results
        output_file = f'tess_cost_{scenario_name}.json'
        with open(output_file, 'w') as f:
            json.dump({
                'catalog': catalog,
                'results': results
            }, f, indent=2)
        print(f"Results saved to: {output_file}\n")


if __name__ == '__main__':
    main()
