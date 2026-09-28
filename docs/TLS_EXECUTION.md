# TLS execution modes

The default `method='reference'` uses the full observation-level TLS search.
Its `execution='baseline'` implementation preserves the backend, host math and
search kernel from commit `6ced75d`.

```python
from cuvarbase.tls import tls_search_gpu, tls_search_batch

result = tls_search_gpu(t, flux, error, periods=periods)
experimental = tls_search_gpu(t, flux, error, periods=periods,
                              execution='experimental')
survey = tls_search_batch(lightcurves, periods=periods,
                          execution='experimental', return_arrays=False)
```

The experimental mode opts into the survey optimization bundle: smaller host
allocations for duration groups, vectorized stable candidate ranking, packed
winner transfers, omitted unused refinement preparation, and guarded batched
short-row scans. It retains the observation-level trial policy. Approximate
binned TLS remains a separate explicit `method='binned'` choice.

The precursor optimization bundle failed the frozen zero-mismatch numerical
qualification: 5,111 of 5,120 original held-out comparisons were exact, with
nine chi2/SDE differences. Selected periods, recovery/alias flags and decisions
at both frozen thresholds agreed on those original comparisons. Original
development qualification was 79 of 80. These results do not establish universal
numerical equivalence or sensitivity preservation; the observed differences
have not been isolated to a particular optimization. Disabling only the
short-row kernel does not cover the known long-row differences.

The [release wiring checks](../benchmarks/results/tls_survey_2026-09-10/release-validation/README.md) passed all 24 paired numerical comparisons on eleven fixed development inputs, plus all 86 device tests. Historical survey
receipts describe the preserved precursor sources, not this default-restoring
release. Baseline execution also does not promise bitwise determinism: native
long-row floating-point scans have documented and observed repeatability
limitations. No acceptance tolerance is relaxed by labeling an execution mode.

Every reference result records `search_configuration.execution` and
`search_configuration.experimental_execution`, including null results. Scalar
convenience calls forward the choice, and batches retain it for every observed
curve and FAP permutation. Unknown values and experimental selection with
another method are rejected. The batch permutation FAP remains a white-noise
null; it does not calibrate arbitrary correlated survey noise.

The two backends own separate compiled-module and thread-local prefix caches.
Default execution never dispatches or compiles the experimental short-row
kernel. Shared CUDA allocator/driver history can still affect subsequent work;
switching back to baseline does not reset that history. Use fresh processes and
recorded sources/configurations for numerical comparisons.
