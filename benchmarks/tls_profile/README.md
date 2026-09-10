# TLS component diagnostics (2026-09-08)

`profile_tls.py` measures synchronized GTLS and earlier **binned** cuvarbase API phases and compares two diagnostic GTLS host-loop changes. These profiles do not measure the new observation-level TLS default. `diagnose_cpu.py` records where the pinned reference CPU TLS API fails on retained inputs. Run either with `--help` for its arguments; searches require the corresponding backend environment and a CUDA device for GPU methods.

The [component report](../results/tls_profile_2026-09-08/README.md) contains the measured timings, numerical differences and failure stages. These two diagnostic inputs are supplementary evidence; the [current transit benchmark](../../docs/TRANSIT_BENCHMARKS.md) provides the release comparison.
