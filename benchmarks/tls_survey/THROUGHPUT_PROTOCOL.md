# Sustained TLS throughput protocol

This protocol is declared before the first tuning pilot. The scientific
calibration and recovery study has its own frozen protocol. Timing cannot
establish detection sensitivity.

## Workload and competitors

Use the first eight manifest-order development inputs in each of `tess_solar`,
`tess_gap_long`, and `ztf_solar`: 24 distinct sources with dense, separated-sector,
and sparse sampling and different observation/grid sizes. Run complete blind
searches on the supplied period arrays. Do not insert truth periods, shorten
grids, or use approximate screening. The executed campaign compares branch
baseline, optimized default TLS, public pinned GTLS, and selected BLS. Corrected
GTLS remains supported by the reproducer but is not included in this timing
campaign; its numerical role is covered by the archived corrected-reference
study. That adapter changes only the documented invalid-candidate host mask.
Include GPU BLS using the science seal's chosen method and ranker for each
regime; require unchanged scientific and production source hashes. Run the same
GPU BLS search and chosen ranking operations. Only qualifying calls compute
complete power/period arrays, mask hashes and all development rankers; timed calls
compute the selected ranker alone. CPU parity checks compare this wrapper's full
and compact results with the frozen science runner for all three rankers on the
same supplied powers. GPU checks require exact selected-period/score repeatability.
Tune BLS's pool and task batch size independently under the same
five-configuration rule. A BLS task calls its single-lightcurve API serially;
batching bundles calls for dispatch. BLS throughput does not imply that its
separately calibrated recovery equals TLS recovery.

Each method runs by itself on the same single GPU and container CPU/memory
allocation. Numerical libraries use one CPU thread per worker. Record GPU UUID,
CPU quota, memory limit, source/input hashes and worker-context ownership.
Concurrent CPU calibration/generation on the rental must finish before timing;
input generation on another machine may overlap.

## Predeclared tuning rule

Tune each backend independently. First compare persistent pools of **one, two
and four workers at batch size one**. Select the fastest numerically eligible
worker count. At that count, compare **batch sizes four and eight**, retaining
the batch-one result. This tests five configurations per backend. It explores a
conditional parameter space and does not establish a global optimum. The optional
`--exhaustive` reproduction flag tests all nine combinations; it is not part of
the declared executed campaign.

Each pilot completes at least 24 sources and at least 30 seconds, in complete
cohort cycles, with one repetition. Select the highest completed-source rate
among eligible tested configurations; exact ties choose fewer workers and then
smaller batches. Record actual API batch sizes. Batches never mix different
period arrays or search options. Both current observation-level APIs process
the sources within a worker's task serially; this batch control bundles public
calls and changes dispatch/load balance, rather than introducing an approximate
multi-source kernel.

All failures and attempted settings remain in `campaign.json`. A failure is
never a successful timing denominator. If the one-worker reference fails, that
backend has no qualifying pool. The selected operating configuration is frozen
before independent timing outcomes are opened. Per-cadence final panels use
that configuration; they are not separately tuned cadence-specific optima.

## Numerical qualification

For TLS and GTLS, first freeze each backend's successful one-worker complete
period, chi-squared and mask arrays and selected period/SDE. Before and after each queue,
every distinct input is searched with full arrays on every worker; all those
outputs must equal the backend's one-worker reference exactly. This prevents a
native memory-dependent duration-group change from qualifying itself.
The optimized
implementation must also match the branch baseline's complete search-output
fingerprints on the actual timing cohort.

**BLS amendment, accepted before scientific freeze or any tuning pilot
(2026-09-11 UTC).** BLS requires exact period arrays and finite masks, and exact
period and score of the science-selected ranker, against its own one-worker
reference before/after queues and against the compact output during every queue.
The complete BLS power arrays and all ranker outputs are retained outside timed
queues; report nonwinning power changes, maximum absolute/relative differences,
and each ranker's selected-period/score variation separately. Nonwinning power
or unused-ranker variation alone does not reject BLS. Any actual selected-period
or selected-score mismatch still disqualifies that pool or panel. Preserve the
numeric values and all failures; do not relax this endpoint gate after tuning or
held-out outcomes. This is an operational BLS repeatability gate, not a claim of
full BLS spectrum equivalence or equal BLS/TLS sensitivity.

The amendment follows the retained development-only `smoke-harness-v3` failure
and a six-repeat diagnostic on each combination of ordinary TESS/ZTF inputs and
`bls_finest`/`bls_strong` (24 calls, 66.78 seconds). Full science calls changed
nonwinning powers by at most 9.31e-9 (maximum 10 ULP), with identical finite
masks. Within each combination the raw/likelihood selected periods and scores
were exact; unused detrended scores varied by up to 1.37e-5. This is consistent
with the existing [BLS reproducibility documentation](../../docs/source/bls.rst)
and unordered float32 atomic accumulation in both fused and multipass kernels.
No supported deterministic setting exists for this fast backend. The bounded
probe does not guarantee selected endpoint stability on the final population.
Old smoke protocols, failed receipts, complete repeat arrays and the diagnostic
script remain archived; the TLS numerical gates are unchanged.

Measured tasks check selected period/SDE and source membership while keeping
only compact results. Complete-array validation surrounds the queue; it does
not establish the identity of every unreturned intermediate array in every
repeated call. Long shared float32 scans can be nondeterministic; a failed strict
TLS/GTLS gate remains a failure and requires separate diagnosis. No numerical
tolerance is expanded after a tuning or held-out failure.

## Independent sustained measurement

Use the first **16 manifest-order independent nulls per cadence**, selected by
identity and successful input generation, never by timing or detection outcome.
These are distinct from development inputs. Search each cadence separately.
Replace the final balanced-mixture panel with **96 distinct derived nulls**:
the first 32 independent nulls per cadence, with deterministic retained fractions
`0.8 + 0.2*i/31`, for manifest positions `i=0..31`. Round retained counts with
NumPy `rint`. Always retain the first/last observation and draw the remaining
interior indices without replacement, using `default_rng` seeded from SHA256
of `tls-survey-throughput-varied-v1`, a NUL byte, and the original filename.
Slice times, flux and errors together and preserve the original period grid.
Keep original input hashes and retained indices in the derived-input manifest.
These modified nulls are for throughput only and must not enter recovery or
false-positive inference. They exercise many observation-array lengths and
repeated cache-plan construction throughout the queue. For each backend's frozen setting, run **three
queues**, each completing **at least 96 light curves and at least 120 seconds**.
Finish whole cohort cycles and keep at most one task per worker in flight.
Repeated cycles measure execution on fixed inputs, not additional independent
astrophysical trials. The varied queue has 96 distinct inputs before any repeat;
the per-cadence panels reuse their 16 distinct original nulls in complete cycles.

The elapsed queue clock includes dispatch, input validation, template preparation,
transfers, full coarse search, candidate/harmonic refinement, output construction,
scalar checking and completion. cuvarbase uses its normal compact survey output
(`return_arrays=False`); public GTLS always returns spectra and additional noise
diagnostics. The public-call comparison includes that output-policy difference.
BLS's public GPU API downloads its power array, which is required by the chosen
ranker; timed calls omit unused rankers and complete-spectrum hashing.
Profiles are separate, instrumented diagnostics and never timing denominators.

Cold records retain worker import/context and input-loading time, first-public-
batch latency, and the first complete qualifying cohort. Recreate one reusable
period grid per shared configuration/worker from sealed `grid_kwargs` and demand
byte equality with the supplied array. Its measured construction time is
included in startup amortization. Historical inputs lacking a recipe are marked
array-only and cannot support an all-preparation-included claim. Cold-amortized
rates conservatively charge the complete validation warmup, including hashes
and each worker's duplicate qualifying inputs; those costs are identified
separately from first-API latency.
Worker processes are fresh, while existing filesystem compiler/kernel caches
remain available. Report these as process-cold latencies, including any actual
first-use compilation or guard canary cost, without claiming an empty disk
cache or a first-ever installation measurement.
The guarded short-row CUB module is a specific exception to filesystem reuse:
direct NVCC compilation explicitly disables flush-to-zero to match the native
CUB wheel; only the resulting process/context module is cached in memory.
Each fresh supported worker/context therefore pays that compilation and startup
canary, even when unrelated CuPy filesystem kernel caches are warm.
Record the guarded short-prefix status after warmup, after the measured queues,
and before worker teardown: actual dispatch/fallback counts, guard or compiler
failure reason, context/device, and compilation/canary time. Reading the status
must not compile or activate the optimization. The baseline's missing helper
is recorded explicitly; native GTLS and BLS mark it not applicable.

Sample device memory, worker RSS and container memory every 0.1 seconds. GPU
and container sampling spans startup, qualification, queues, and teardown;
worker RSS sampling begins when the worker pool reports ready, supplemented
by lifetime RSS high-water marks. Reject observed foreign GPU processes during
that interval, including the explicit checks immediately before/after queues.
Sampled
peaks are lower bounds; also retain worker lifetime RSS high-water marks and
allocator reservation sizes. A container's cumulative memory high-water mark
can include earlier configurations and is labeled accordingly. Record the
bundle's actual hourly price and compute measured-queue and cold-amortized cost
projections. Million-source costs are projections, excluding data acquisition,
survey preprocessing and vetting.

## Reporting

`measurements.csv` contains every attempted configuration, rates, cold timing,
projected costs and memory. The one performance figure uses qualified independent
measurements only: median completed light curves per second with the observed
range across three repetitions. Its four panels show dense TESS, separated TESS,
ZTF and the varied-size queue, with separate y scales explicitly labeled. Display
baseline, optimized TLS, public GTLS, and the science-selected BLS. Use logarithmic y scales to accommodate
different method costs and identify these scales on the figure.

Predeclared failure handling: retain a visibly missing bar labeled **no qualifying
result** and its reason if a competitor fails qualification or has no eligible
development setting. Continue independent remaining backends and panels; do not
abort the entire measurement merely because one competitor fails its fresh
single-worker reference. Every planned panel remains visible, with no failed
result entering a rate or speedup denominator, no relaxed gate, no post-hoc
case subset, and no held-out retuning. A failed baseline/optimized paired gate
suppresses the optimized bar and ratio in that panel. If either result is
unavailable, no baseline/optimized ratio is reported. All qualified displayed
results must have identical GPU/CPU/memory receipts. These finite three-cadence timing workloads do
not establish universal survey throughput or recovery equivalence.
