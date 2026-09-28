# Full TLS search timing

The declared timing scope covers **single-source latency and a batch of 16
distinct noise-only light curves** for TESS, gapped TESS and ZTF. It compares
cuvarbase's standard TLS engine with pinned GTLS's full search: five single
calls, three batches, and five separate component calls per regime. Native
throughput is tested with persistent pools of 1, 2 and 4 workers. All calls
must pass the recorded output and process-ownership gates before a speed
ratio is eligible. The earlier single-only budget fallback stopped before
warmup; the restored full scope was declared before any valid measurement.

These tools do not provision resources or change installed source files.
Numerical validation must pass before timing begins. Original independent
results and reproduced results retain separate labels; timing does not
establish detection sensitivity by itself.

The published campaign retained a failed four-worker GTLS warmup on gapped
TESS. Its original all-configuration acceptance remains failed. The
[results report](../../results/tls_reference_2026-09-10/timing/README.md)
uses a separately labeled post hoc assessment of complete configurations,
with all attempts disclosed. `report_completed.py` enforces that reporting
scope without changing the original driver or numerical gates. Its
[CPU replay instructions](../../results/tls_reference_2026-09-10/sources/timing/README.md)
reproduce the published figure data from the retained evidence.

## Inputs and measured calls

The single source is the predeclared `null_0000` of each regime. If it fails
either public API, the established selection rule uses the earliest
manifest-order paired-success null in the supporting cohort. The receipt
records the failure and fallback. Selection never uses elapsed time, measured
SNR or signal recovery. A new timing failure invalidates its configuration;
the failed elapsed time is retained and never enters a speed ratio.

The supporting input bank contains 16 distinct nulls per regime: eight from
the main 160-case confirmation and eight from the separately sealed 24-case
supplement. Their original manifests, seals, acceptance receipts and result
directories remain distinct. Production and native-reference sources and
per-regime search settings must agree. Only the chosen single source is
executed in `--measurement-scope single`.

`benchmark.py` measures five complete single-source public calls and three
complete batches in full scope. Input arrays and period
grids are loaded before clocks start. Constructor work, input validation,
template-cache construction, the full search, candidate and harmonic
refinement, final fitting and returned arrays are included. Imports, CUDA
context setup and one complete warmup are reported separately. A persistent
worker uses one CPU numerical-library thread; the parent stops the clock
after the public return and CUDA synchronization. Output hashing follows the
timing barrier.

Each implementation must reproduce its own validated complete period, power,
chi-squared, selected-period and SDE identities. This permits the separately
disclosed correction to an undefined native trial without silently claiming
literal cross-implementation equality. The native package remains unchanged;
`corrected_reference.py` temporarily applies the host correction in memory.
When the supporting cohort lacks a complete no-op proof, an additional
one-worker corrected-native measurement records its effect separately.

Read-only GPU/process telemetry is sampled at 1 Hz. Each worker starts on an
empty GPU and retains a synchronized CUDA allocation before any search. When
the container hides its outer PIDs, exactly N newly visible NVML PIDs can be
bound as a set to N distinct live workers. The harness never invents an
individual mapping for hidden IDs. Every call requires that same complete
PID set and no other compute process; acceptance also requires all contexts
to disappear after clean worker exits. Component measurements use the same parent/child check,
outside their timing clocks. CPU quota, affinity,
thread settings, dependencies, input/source hashes, startup, warmups, all
repetitions, errors and validation time are retained. Run the campaign
exclusively; snapshots cannot exclude a job entirely between samples.

## Search and reporting components

`components.py` is separate from the uninstrumented public timings. It makes
a literal public warmup followed by five instrumented calls. Every returned
field must remain identical to the literal call, whose search arrays must
match the validated study. These component times never replace the public-call
denominator.

The native common-search endpoint is immediately after the pinned final
single-period statement `bestLocation = lowestResidualsGPU.argmin().get()`.
The cuvarbase endpoint is immediately after `engine.search_full` returns.
Both include constructor, validation, cache construction, full search and
final window selection, and exclude subsequent physical-parameter reporting.
The boundaries represent the same search stage, with a small implementation
difference: cuvarbase transfers compact winner fields and returns its engine
result before stamping, while GTLS stamps after the final argmin transfer.

GTLS additionally computes CPU per-transit SNR and pink-noise diagnostics.
Their inclusive durations and nested pink-noise durations remain separate;
nested component durations must not be added together. Full-public and
common-search ratios answer different questions and are reported separately.

## Reproduce the published timing cohort

From the repository root, first restore and validate the main population with
the [parent README](../README.md#reproduce-the-numerical-comparison). Use the
recorded CUDA environment, a pinned GTLS installation, and new output
directories. CUDA compilation requires `nvcc` on `PATH`. Then restore and
validate the supplementary null population and merge both accepted origins:

```sh
python benchmarks/tls_reference/inputs.py restore \
  --bank benchmarks/results/tls_reference_2026-09-10/inputs --study supplement \
  --manifest benchmarks/results/tls_reference_2026-09-10/supplement/input_manifest.json \
  --out reproduced-supplement-inputs
python benchmarks/tls_reference/reproduce.py --repo-root . \
  --manifest reproduced-supplement-inputs/manifest.json \
  --seal benchmarks/results/tls_reference_2026-09-10/supplement/seal.json \
  --out reproduced-supplement-results
python benchmarks/tls_reference/timing/merge.py \
  --study main reproduced-inputs/manifest.json reproduced-results \
  --study supplement reproduced-supplement-inputs/manifest.json reproduced-supplement-results \
  --out reproduced-timing-inputs
python benchmarks/tls_reference/timing/benchmark.py \
  --manifest reproduced-timing-inputs/manifest.json --paired-results reproduced-results \
  --measurement-scope full \
  --correction-adapter benchmarks/tls_reference/corrected_reference.py \
  --output reproduced-public-timings
python benchmarks/tls_reference/timing/components.py \
  --manifest reproduced-timing-inputs/manifest.json --paired-results reproduced-results \
  --measurement-scope full --backend gtls --output reproduced-gtls-components
python benchmarks/tls_reference/timing/components.py \
  --manifest reproduced-timing-inputs/manifest.json --paired-results reproduced-results \
  --measurement-scope full --backend candidate --output reproduced-cuvarbase-components
python benchmarks/tls_reference/timing/components.py \
  --manifest reproduced-timing-inputs/manifest.json --paired-results reproduced-results \
  --measurement-scope full --backend gtls_corrected \
  --correction-adapter benchmarks/tls_reference/corrected_reference.py \
  --output reproduced-corrected-components
python benchmarks/tls_reference/timing/summarize.py reproduced-public-timings \
  --components-gtls reproduced-gtls-components \
  --components-candidate reproduced-cuvarbase-components \
  --components-corrected reproduced-corrected-components \
  --output reproduced-timing-checks.json
python benchmarks/tls_reference/analyze_timing.py \
  --checks reproduced-timing-checks.json --manifest reproduced-timing-inputs/manifest.json \
  --acceptance reproduced-results/acceptance.json --output reproduced-timing-summary.json
python -m pytest -q benchmarks/tls_reference
```

`merge.py` copies the 48 accepted null inputs and uses relative paths to both
original manifests and result trees. Keep those directories together when
moving the experiment. It verifies the original population, actual validator
and adapter, production hashes and passing comparisons. Its receipt uses
`reproduction_gate`; the timing summary reports
`numerical_evidence_kind: reproduction` and `measurement_scope: single_and_batch`.
Neither step changes an original seal or turns a rerun into independent
sensitivity evidence. The results archive separately preserves the exact
original scientific and timing source snapshots.

Expected outputs use complete per-array hashes, dtypes, shapes and masks,
plus exact scalar period/SDE hashes. Removing a large NPZ after a successful
comparison does not remove these identities. `verify_study_hashes.py` can
check conversion of a retained study record against its array archive
without executing another search.

## Batch scope and competitor selection

`--measurement-scope full` runs three batch calls on the 16 distinct
nulls in each regime. Cuvarbase uses its public `tls_search_batch` API; native
GTLS uses persistent pools of 1, 2 and 4 workers with fixed round-robin source
assignment. The clock includes dispatch and completion of every source.

A native pool is eligible only when every repetition preserves all complete
search-output identities against the literal one-worker run and each method
matches its own validated output. The fastest eligible median is selected;
all pool results and failures remain visible. These finite batches measure
throughput for the recorded workload and hardware. They do not by themselves
establish performance across an entire survey or all CPU configurations.
`--measurement-scope single` remains available for a smaller experiment with
no batch or strongest-pool claim. The optional `--row-ab` prefix-dispatch
attribution experiment is separate from the declared campaign.
