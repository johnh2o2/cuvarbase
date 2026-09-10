# v1.0 release validation

The new observation-level TLS engine has its own [10 September validation](tls-default-20260910/README.md): **265 TLS tests passed on an A40**, plus installed-wheel checks. The [independent numerical study](../../benchmarks/results/tls_reference_2026-09-10/README.md) validates its search outputs and supplies the timing comparison.

The checks below ran on 6 September 2026 against frozen source `1032caf029570dc4841db1c594a2cbb1654e8fd8`. They establish correctness and packaging checks for that source, separately from the [performance benchmark](../TRANSIT_BENCHMARKS.md). Later implementation and documentation changes are not covered by these original full-suite counts.

| Check | Outcome | Evidence |
|---|---|---|
| Full source suite | 1,785 passed, 1 expected failure; no skips or failures | [Source test log](v1.0.0/suite_full.log) |
| Additional release checks | 14/14 passed | [Release gate log](v1.0.0/release_gate.log) |
| Clean GPU Sphinx build | Passed with warnings treated as errors | [Build log](v1.0.0/docs_build.log), [figure log](v1.0.0/docs_figures.log) |
| Wheel and source distribution | Build and strict metadata checks passed | [Build log](v1.0.0/build.log), [Twine log](v1.0.0/twine_check.log) |
| Installed wheel suite | 1,773 passed, 11 source-only skips; no failures | [Wheel test log](v1.0.0/wheel_pyargs.log) |
| Installed source-distribution suite | 1,773 passed, 11 source-only skips; no failures | [Source-distribution test log](v1.0.0/sdist_pyargs.log) |
| Import without PyCUDA | Wheel and source distribution passed | [Wheel smoke log](v1.0.0/wheel_smoke.log), [source-distribution smoke log](v1.0.0/sdist_smoke.log) |

The expected failure covers the PDM notebook's known non-raw TeX label strings. [Environment details](v1.0.0/env_record.txt) and [source provenance](v1.0.0/source_provenance_final.log) accompany the logs. The full original execution record, including release orchestration, is available in [Git history](https://github.com/johnh2o2/cuvarbase/tree/f0dc981/analysis/v1.0-release-gate-20260906).

To run current checks, see [developer tools](../../tools/README.md).
