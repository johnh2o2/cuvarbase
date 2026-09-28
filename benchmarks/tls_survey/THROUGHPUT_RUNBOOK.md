# Running the final single-GPU timing campaign

The science controller must finish all GPU work and remote input generation
first. The following paths describe the recorded survey allocation; adjust
paths when reproducing elsewhere while preserving the source and input hashes.
The baseline checkout is revision `6ced75d6d75bfaafa39b78c557fcba86f4651d92`.
The final candidate must match the scientific seal's production sources.
No command here provisions or terminates a resource.

```sh
export PATH="/usr/local/cuda/bin:$PATH"

/workspace/tls-survey/modern/bin/python \
  /workspace/tls-survey/candidate/benchmarks/tls_survey/throughput_campaign.py \
  --stage tune \
  --manifest /workspace/tls-survey/dev-final/manifest.json \
  --output /workspace/tls-survey/evidence/throughput-tune-final \
  --baseline-root /workspace/tls-survey/baseline \
  --candidate-root /workspace/tls-survey/candidate \
  --science-seal /workspace/tls-survey/evidence/seal-final.json \
  --backends baseline candidate gtls bls \
  --hourly-usd 0.49 --max-hours 4
```

This declares the timing plan before the first pilot, runs the separately
tuned five-setting search for each of the four competitors, and freezes one
eligible operating setting per backend. Corrected GTLS is supported for an
explicit separate reproduction, but is not part of this executed campaign.
The CUDA compiler directory must remain on PATH for every competitor and stage:
PyCUDA BLS invokes `nvcc` by name, while CuPy may discover the compiler through
an absolute CUDA toolkit path. Do not set the `NVCC` variable; the guarded
short-prefix path conservatively rejects custom compiler commands.
The first one-worker tuning warmup also serves as the later integration check
of the BLS qualification amendment; no extra held-out smoke population is opened.
Every BLS configuration retains complete pre/post-queue spectra and native
repeat diagnostics, while exact selected endpoints remain required. Preserve
the earlier `smoke-harness-v2`, `smoke-harness-v3` and `bls-native-repeat-v1`
development-only failures/diagnostics alongside the amended timing protocol.

```sh
/workspace/tls-survey/modern/bin/python \
  /workspace/tls-survey/candidate/benchmarks/tls_survey/throughput_campaign.py \
  --stage measure \
  --manifest /workspace/tls-survey/final-campaign/inputs-nulls/manifest.json \
  --output /workspace/tls-survey/evidence/throughput-final \
  --tuning /workspace/tls-survey/evidence/throughput-tune-final/campaign.json \
  --baseline-root /workspace/tls-survey/baseline \
  --candidate-root /workspace/tls-survey/candidate \
  --science-seal /workspace/tls-survey/evidence/seal-final.json \
  --backends baseline candidate gtls bls \
  --hourly-usd 0.49 --max-hours 10

/workspace/tls-survey/modern/bin/python \
  /workspace/tls-survey/candidate/benchmarks/tls_survey/plot_throughput.py \
  /workspace/tls-survey/evidence/throughput-final/campaign.json \
  --exactness /workspace/tls-survey/evidence/exactness-final.json \
  --science-seal /workspace/tls-survey/evidence/seal-final.json \
  --output /workspace/tls-survey/evidence/throughput-final/performance
```

Add `--resume` to an existing tuning or measurement campaign only with unchanged
sources, protocol, seal, and inputs. Completed configurations are reused by
hash; failed configurations remain failed. A failed competitor's fresh
qualification does not prevent the remaining planned panels from running.
The time cap is checked between configurations and does not kill an active GPU
call. It resets per invocation, so the resource owner's outer lifecycle/budget
guard must account for all attempts, preparation, and scientific work.

Before launch, allow roughly **1.5–3 hours for tuning and 4–8 hours for final
measurement/qualification**, approximately **$2.70–5.40 combined at $0.49/hour**.
These are conservative planning estimates, not measured results. In particular,
four-worker strong-BLS development calls for ordinary ZTF took roughly 43 seconds
each under contention; archived public-GTLS single-source ZTF timings were
roughly 15 seconds on another GPU. Every distinct qualifying input runs on every
worker, before and after the sustained queues, so qualification is a material
part of study cost. Actual eligible settings and hardware determine runtime.

Archive both timing directories, their logs, the derived varied-input manifest
and arrays, the original null manifest/arrays, the scientific seal, all source
identities, the figure and its data receipt, and any failure records. The outer
resource owner must archive and terminate the rental on success **or failure**;
these benchmark scripts intentionally do not own resource lifecycle operations.
See [the frozen timing policy](THROUGHPUT_PROTOCOL.md) for numerical gates,
timing boundaries, process-cold accounting, and the missing-competitor policy.
The figure requires the complete held-out exactness receipt and original science
seal, validates all planned regime/split counts, and prominently reports X/N
exact cases. A failed aggregate qualification remains withheld even when a
separate timing cohort supports its own speed ratio. The accompanying figure
CSV and JSON retain that status and both scientific and auxiliary-plan identities.
