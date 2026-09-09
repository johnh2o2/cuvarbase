# Reproducing the September 2026 benchmark audit

The report and archived measurements are in
[`analysis/benchmark-audit-20260906/`](../../analysis/benchmark-audit-20260906/).
The frozen package source is `1032caf029570dc4841db1c594a2cbb1654e8fd8`.
These scripts never substitute the development checkout for that installed
source when running on the GPU host.

## Recompute the audit and plots without a GPU

Use Python 3.11 with NumPy, SciPy, Matplotlib, Astropy 8.0.1, and batman-package.
From the repository root:

```bash
python scripts/benchmark_audit/audit_archives.py
python scripts/benchmark_audit/validate_results.py
python scripts/benchmark_audit/plot_results.py
python scripts/benchmark_audit/summarize_results.py
```

Validation checks identical input hashes, full-output shape/finiteness, and
sampled periodogram values/peak locations against direct float64 GLS fits.
It records failures rather than deleting unfavorable results. The plots require
that validation file and select only eligible, completed measurements. A timeout
or a missing implementation is never represented by a zero or invented timing.

## Rerun on a disposable GPU host

The campaign used one A40 with 48 GB, a Xeon Gold 6342 CPU, and a cgroup CPU quota
of 7.65 cores. GPU runs and CPU competitors execute serially so that they do not
contend with one another. Within a CPU survey job, worker concurrency is allowed
and recorded. Do not run installation, other benchmarks, or CPU validation while
measuring; shared host activity can still affect cloud timings.

Prepare the frozen source archive locally:

```bash
git archive --format=tar -o source-v1.tar 1032caf cuvarbase pyproject.toml README.md LICENSE.txt
```

The archive SHA-256 must be
`19ff05aeb665bf7b4e8159ea7dc9fd4dc358b82fc7c54e5ba1efebd8a6f909ab`.
Copy it and `setup_gpu.sh` to `/tmp/cuvarbase-benchmark-audit/` on the GPU host,
create `source-v1/` and `results/` there, then run `bash setup_gpu.sh`. This builds
separate modern and legacy virtual environments. The legacy package is the
actual PyPI `cuvarbase==0.2.5`; its numerical kernels are unmodified.
The two source tar archives and PyPI comparator wheels are also preserved in
the audit's `sources/` directory.

For the upstream GTLS comparator, archive `src`, `pyproject.toml`, `LICENSE`, and
`README.md` from https://github.com/Farthing-0/GTLS at
`74e449c325792a763dde4fbffab98039c5e8c111`. Extract to `gtls-head/` under the same
remote root, then:

```bash
modern/bin/python -m pip install --no-deps --target gtls-head-install ./gtls-head
```

`run_campaign.py` sets `PYTHONPATH` to this target **only** for the upstream GTLS
process. Other GTLS processes use the installed PyPI 0.4.4. UTF-8 locale is
explicit because GTLS source strings contain non-ASCII comments.

Copy `common.py`, `run_ls.py`, `run_tls.py`, `generate_ls_inputs.py`, and
`run_campaign.py`, plus `collect_evidence.py`, into that remote root, with the archived `inputs/` directory.
The final input files are authoritative: load them unchanged to reproduce the
exact-byte comparisons. To create a new LS dataset rather than reproduce the
existing one, run `generate_ls_inputs.py` once with the modern environment;
never regenerate inputs separately in the two dependency stacks.

TLS inputs were generated locally with:

```bash
python scripts/benchmark_audit/generate_tls_inputs.py --baseline 27 --n-lcs 16 --out analysis/benchmark-audit-20260906/inputs/tls27.npz
python scripts/benchmark_audit/generate_tls_inputs.py --baseline 200 --n-lcs 1 --out analysis/benchmark-audit-20260906/inputs/tls200.npz
python scripts/benchmark_audit/generate_tls_inputs.py --baseline 1500 --n-lcs 1 --out analysis/benchmark-audit-20260906/inputs/tls1500.npz
python scripts/benchmark_audit/generate_tls_inputs.py --baseline 27 --n-lcs 64 --ensemble --out analysis/benchmark-audit-20260906/inputs/tls27_ensemble.npz
```

Run the stages sequentially on the GPU host:

```bash
modern/bin/python run_campaign.py --stage smoke
modern/bin/python run_campaign.py --stage tls
modern/bin/python run_campaign.py --stage ls
modern/bin/python run_campaign.py --stage ensemble
modern/bin/python run_campaign.py --stage tls_followup
modern/bin/python run_campaign.py --stage ls_shared_followup
modern/bin/python collect_evidence.py
```

Inspect smoke JSON statuses as well as return codes. Copy all result JSONs,
NPZs, logs, dependency freezes, GPU/CPU records, controller records and input
files back to the audit directory. Verify transfer checksums before releasing
the host. Run the CPU validation/plotting commands above only after measurements
finish. Raw spectra are retained for TLS; LS retains selected output bins plus
peak results and hashes of full output spectra.

## Timing and comparison boundaries

- First API call is recorded separately; it excludes imports/context startup and
  is not a fresh-container cold-start result. The disk kernel cache is allowed.
- Then one more warmup is discarded. Report median of five LS or three TLS API
  calls, with all samples and ranges. Host preparation, allocation and H2D/D2H
  transfers inside the API call count. Input-file loading and validation do not.
- LS single calls and 32-LC batches use the same first lightcurve. Distinct-time
  and shared-time batches are separate workloads. Every call returns all host
  periodograms. Both cuvarbase precision modes and nifty-ls GPU precisions are
  recorded; the main four-way chart uses float64 throughout.
- TLS uses identical inputs and periods, but GTLS/reference/cuvarbase still
  differ in actual template, duration/epoch discretization, refinement, and
  diagnostic outputs. The figures do not call this equal-sensitivity timing.
- The CPU TLS adapter replaces only the reference library's period-grid factory
  because its public API cannot accept an explicit grid. Returned grids are
  checked. Its search code is unmodified.
- The 64-LC diagnostic distinguishes exact recovery, harmonic recovery, native
  SDE, and a shared current-definition re-score. Sixteen nulls cannot calibrate
  a 1% false-positive rate. No broad completeness claim follows from it.

The initial smoke logs and the discarded LS pilot preserve the adapter/locale
issues and cross-NumPy input-hash mismatch found while preparing this campaign.
They are excluded from the final figures.

The `tls-pypi-initial/` records preserve the first 27-day PyPI GTLS runs: native
non-finite diagnostics caused strict JSON serialization to fail. The follow-up
reruns use the same search and preserve these values explicitly as JSON nulls;
they remain ineligible for the timing comparison. The CPU follow-up covers
additional thread/worker counts, and the shared-time LS follow-up covers both
cuvarbase precisions and additional CPU/GPU configurations.

The source verifier compares the installed v1.0 package against the frozen tar
archive and GTLS's installed Python files against its pinned source archive.
GTLS's packaging omits `GPUFun.cu`, `GPUFun_bak.cu`, and `move.sh`; these omissions
are recorded explicitly. Its runtime CUDA source is the embedded string in
`GPUFun.py`, which is checked byte for byte. Installed PyPI package file hashes
are also saved and checked locally against the archived wheels.

Some initial adapters were corrected during setup: the nifty-ls keyword shape,
GTLS result attribute access, UTF-8 locale, and serialization of non-finite
diagnostics. The immutable-input LS campaign replaced the discarded pilot.
The final harness is archived; logging/timestamp fields were added while the
long TLS run was underway. The numerical libraries were not edited, and each
controller record retains the executed arguments. This is not a claim that
every early harness revision was separately content-addressed before execution.
