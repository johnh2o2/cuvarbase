# Benchmarks and validation

The [transit benchmark report](../docs/TRANSIT_BENCHMARKS.md) is the source for the README's performance claims. It compares v1 BLS with PyPI 0.2.5 and tested CPU/GPU alternatives, and v1 TLS with public GTLS, on observed TESS and ZTF cadences with synthetic transits and noise.

| Directory | Purpose |
|---|---|
| [tls_reference/](tls_reference/README.md) | Standard observation-level TLS: numerical parity, independent injections/nulls, full public-call and component timings |
| [results/tls_reference_2026-09-10/](results/tls_reference_2026-09-10/README.md) | Current default TLS versus full GTLS, including thin-transit regimes |
| [transit/](transit/README.md) | Timing figure, recovery analysis and transit benchmark workers |
| [results/transit_2026-09-08/](results/transit_2026-09-08/README.md) | BLS competitor benchmark and initial TLS experiment |
| [tls_sensitivity/](tls_sensitivity/README.md) | Independent TLS recovery, exclusive timing and numerical-resolution tools |
| [results/tls_sensitivity_2026-09-09/](results/tls_sensitivity_2026-09-09/README.md) | Historical binned TLS sensitivity study, resolution tradeoffs and secondary BLS control |
| [tls_accuracy/](tls_accuracy/README.md) | TLS approximation diagnostics, kernel parity/timing and focused high-impact recovery tools |
| [results/tls_accuracy_2026-09-09/](results/tls_accuracy_2026-09-09/README.md) | Narrow-transit accuracy limits and validation of sparse-bin traversal |
| [tls_profile/](tls_profile/README.md) | Supplementary TLS profiling and CPU failure diagnostics |
| [results/tls_profile_2026-09-08/](results/tls_profile_2026-09-08/README.md) | TLS component measurements and numerical comparisons |
| [nufft_lrt/](nufft_lrt/README.md) | Validation tools for the experimental NUFFT-LRT detector |
| [results/nufft_lrt_validation_2026-09-06/](results/nufft_lrt_validation_2026-09-06/README.md) | Independent validation supporting the NUFFT-LRT documentation |

Benchmarks are separate from the [release correctness checks](../docs/validation/README.md). Superseded timing claims and their provenance are documented in the [historical-claim audit](../docs/BENCHMARK_PROVENANCE.md); the original campaigns remain in Git history.
