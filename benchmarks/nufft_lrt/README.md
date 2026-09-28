# NUFFT-LRT validation

`validate.py` generates seeded null lightcurves and transit injections, searches the selected detector configurations, and records recovery and timings. It requires a CUDA environment with the cuvarbase test dependencies. `summarize.py` derives the tables used in the experimental detector documentation. Both accept `--help`; the validation tool can split work by configuration/arm and merge the outputs.

The TLS comparator is the earlier **binned** engine, retained today as `method='binned'`. The dated NUFFT-LRT tables do not compare against the new observation-level TLS default. Use the [current transit report](../../docs/TRANSIT_BENCHMARKS.md) for standard TLS validation and speed claims.

The [September 2026 validation record](../results/nufft_lrt_validation_2026-09-06/README.md) specifies the frozen source, protocol, process split and measured results. Its launch script is a historical execution record; use the maintained entry points here for a new run.
