# Installed-wheel release gate, September 27

The unchanged release wheel passed all **14 numerical/runtime checks** and all six dependency preflights on an NVIDIA A40. All 86 installed package files matched the previously built wheel byte for byte.

The September 24–25 full GPU suite already passed 2,091 tests, with one expected notebook failure, no unexpected failures, and zero skips. Its separate gate launcher then failed to import `cuvarbase` because the package had not been installed in that environment. This isolated follow-up installed the same wheel and ran the unchanged gate successfully. The original failure receipt remains preserved.

This repairs validation setup; it changes neither numerical sources nor the benchmark qualifications. The GPU was terminated after checksum-verified collection.

[Gate output](release-gate.log) · [Execution receipt](receipt.json) · [Installed-file verification](installation.json) · [Summary and cost](summary.json).

[Full GPU suite output](full-gpu-suite.log) · [JUnit results](full-gpu-suite.xml) · [Tested source inventory](tested-source-inventory.json).

All eight objects in the private R2 prefix `release-validation-20260927/gate-installed-wheel-a3ddc876` passed full SHA256 read-back. [Verification receipt](r2-readback.json).

After review, the 1.0.0 wheel and source distribution were rebuilt to include the updated README. All 86 package files in those artifacts match the GPU-validated wheel exactly; wheel changes are confined to distribution metadata. Those intermediate artifacts passed strict metadata checks, and all 12 focused report/README tests passed. [Intermediate package comparison and artifact hashes](refreshed-package-verification.json).

The final prepared version is **1.0.1**, preserving the earlier June `v1.0.0` tag. Its only package-file change is the version declaration; all numerical sources are unchanged. [Final release preparation](../../../../docs/RELEASE_PREPARATION.md) contains the current artifact verification. Publication remains deferred.
