# Benchmarks and validation

The [transit benchmark report](../docs/TRANSIT_BENCHMARKS.md) is the source for the README's performance claims. It compares v1 BLS with PyPI 0.2.5 and tested CPU/GPU alternatives, and v1 TLS with public GTLS, on observed TESS and ZTF cadences with synthetic transits and noise.

| Directory | Purpose |
|---|---|
| [transit/](transit/README.md) | Timing figure, recovery analysis and transit benchmark workers |
| [results/transit_2026-09-08/](results/transit_2026-09-08/README.md) | BLS competitor benchmark and initial TLS experiment |
| [tls_sensitivity/](tls_sensitivity/README.md) | Independent TLS recovery, exclusive timing and numerical-resolution tools |
| [results/tls_sensitivity_2026-09-09/](results/tls_sensitivity_2026-09-09/README.md) | Current TLS evidence, resolution tradeoffs and secondary BLS control |
| [tls_profile/](tls_profile/README.md) | Supplementary TLS profiling and CPU failure diagnostics |
| [results/tls_profile_2026-09-08/](results/tls_profile_2026-09-08/README.md) | TLS component measurements and numerical comparisons |
| [nufft_lrt/](nufft_lrt/README.md) | Validation tools for the experimental NUFFT-LRT detector |
| [results/nufft_lrt_validation_2026-09-06/](results/nufft_lrt_validation_2026-09-06/README.md) | Independent validation supporting the NUFFT-LRT documentation |

Benchmarks are separate from the [release correctness checks](../docs/validation/README.md). Superseded timing claims and their provenance are documented in the [historical-claim audit](../docs/BENCHMARK_PROVENANCE.md); the original campaigns remain in Git history.
