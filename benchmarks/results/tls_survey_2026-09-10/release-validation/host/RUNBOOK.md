# Local release staging: baseline default, experimental opt-in

The complete detached checkout is `../release-validation-checkout`, based on
`6ced75d6d75bfaafa39b78c557fcba86f4651d92`. It is separate from the active
`v1.0-fixes` working tree. No branch integration, commit, remote operation,
GPU execution, or shared-environment installation has occurred.

`assembly.json` pins all 82 precursor package sources before copying only the
nine differing package/test files. The reviewed eight-file production overlay
was then applied exactly. `validation.json` verifies all 82 original sources
remain unchanged and records all 86 staged package source hashes. The three
default backend/math/kernel files are byte-identical to baseline 6ced75d.
`release-staging.patch` is the complete tracked-plus-new source/doc patch from
that baseline. The original prototype's source snapshots and inventories remain
unchanged under `../default-preserving-release-prototype/`.

The staged implementation uses `execution='baseline'` by default and requires
`execution='experimental'` for the optimization bundle. Invalid selectors and
experimental use with another method are rejected. Result metadata, scalar,
convenience, direct frontend, batch and FAP paths retain that selection. Engine
module caches and thread-local prefix caches are separate. The experimental
math adapter imports only five unchanged baseline helpers and contains the two
optimized functions; no global rebinding is used.

Existing default tests remain. Shared mathematical and GPU prefix contracts
now run for both engines; packing and guarded short-prefix tests explicitly use
the experimental engine. New host tests exercise complete engine-module loading
and cache orchestration with fake CUDA dependencies, plus immutable source pins.
A separate GPU regression exercises real compiled module/graph-buffer isolation
and proves baseline does not dispatch the short-prefix cache. GPU regressions
are authored but not executed here.

Host results: focused 219 passed; full CPU selection 872 passed, 18 skipped,
1117 deselected, one existing xfail. Device modules skip without CuPy. Exact
summaries are in `host-suite.xml`. The first no-isolation wheel attempt in the
old shared host environment emitted invalid UNKNOWN metadata; its artifact is
retained in `wheels/UNKNOWN-0.0.0-py3-none-any.whl` and is not a release wheel.
A separate build-only virtual environment with setuptools77.0.3 and wheel0.45.1
produced `wheels/cuvarbase-1.0.0-py3-none-any.whl`. `verify_wheel.py` verified
its metadata and every staged Python/CUDA/header source byte, then extracted it
to a fresh directory. All 219 focused tests also passed with imports from that
unpacked wheel (`wheel-tests.xml`). No package was installed into the host test
environment. Both environment freezes are retained.

Executed commands (all local):

```sh
cd ../release-validation-checkout
../local-env/bin/python -m pytest cuvarbase/tests/test_tls_reference_frontend.py cuvarbase/tests/test_tls_reference_math.py cuvarbase/tests/test_tls_execution_isolation.py cuvarbase/tests/test_kernel_inventory.py -q
../local-env/bin/python -m pytest -m 'not gpu' -q --junitxml=../release-validation-evidence/host-suite.xml
../release-validation-build-env/bin/python -m pip wheel --no-deps --no-build-isolation --no-cache-dir --disable-pip-version-check --wheel-dir ../release-validation-evidence/wheels .
cd ../release-validation-evidence
PYTHONPATH="$PWD/unpacked-wheel" ../local-env/bin/python -m pytest --pyargs cuvarbase.tests.test_tls_reference_frontend cuvarbase.tests.test_tls_reference_math cuvarbase.tests.test_tls_execution_isolation cuvarbase.tests.test_kernel_inventory -q --junitxml=wheel-tests.xml
```

Do not overwrite the retained XML/archive outputs when reproducing; use fresh
result paths. `verify_wheel.py CHECKOUT WHEEL FRESH_OUTPUT` prints its verification
receipt. `wheel-verification.json` records the exact original invocation paths.

Remaining validation is the already predeclared bounded GPU integration plan at
`../default-preserving-release-prototype/development-validation-plan.json`.
Its eleven fixed development IDs and exact NPZ pins, installation prerequisites,
separate fresh-process controls and 900-second GPU cap remain unchanged. Follow
that prototype README after the original primary → supplement → collection →
provider-termination sequence finishes. Setup/provisioning costs are separate
and remain within the existing authorization; no new allowance or GPU rental
is created here. Existing device prefix/short-prefix tests and selected TLS
contracts must run from this full checkout; record both dispatch branches and
all failures. Root approval is required for the later operational handoff.

This staging work does not requalify the optimized algorithm. Original
qualification remains 79/80 development and 5111/5120 held out, with nine
chi2/SDE differences and no changed selected-period or frozen-threshold decisions
in those original comparisons. The zero-mismatch gate failed. The new routing
has no historical benchmark result of its own. Baseline restoration does not
promise native floating-point determinism or undo allocator/driver history.
