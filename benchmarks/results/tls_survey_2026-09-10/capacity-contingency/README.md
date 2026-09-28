# Unlaunched capacity contingency

Prepared only. No provider API request, mutation, process launch, signal, or
change to the original guard/controller/sidecar was performed in preparation.
Root must review the source, plan and test receipt before using the launch file.

The only authorized pod is `survey01` / `okideq277lpb4a`. This fallback retains
the original study rental cap of **$30**, with its **$29.90 trigger**. It is not
a new compute allowance. The current original node inventory has one pod at
$0.49/hour, created at epoch `1789080637.819206`, and no completed-node entries.
The original `cloud.spend()` evaluated at the plan's snapshot epoch exactly
matches the independently derived $15.336237914995353 rental estimate.

The fixed trigger is **2026-09-13 11:51:51.288594 UTC**, epoch
`1789300311.2885938`. The $30 accounting time is **12:04:05.982471 UTC**,
about 12 minutes 15 seconds later. No retry or restart recalculates a fresh
allowance. The pinned formula includes any completed-node estimates, although
there are none in this inventory. This reproduces the existing study rental
accounting; it is not a new reconciliation of the user's overall budget.

`capacity-plan-v1.json` pins the standalone source, unchanged `ops/cloud.py`,
complete node-record bytes and inventory, active pod ID, rate, accounting
snapshot, and derived cutoff. At startup the code rechecks all of these,
imports the exact verified cloud bytes without a bytecode-cache write, reads
the cloud configuration once into memory, and verifies the provider's current
pod identity/rate. Unrelated provider pods are never mutation targets. An
already absent owned pod produces an `already_absent` startup receipt and
exits without a mutation. A changed local record, source, inventory, rate, or
plan fails startup rather than silently updating the reviewed plan.

After startup validation, the one-time exclusive readiness receipt is flushed
and fsynced at `startup-readiness-v1.json`. It contains the Python process PID,
plan/source/accounting identities, and fixed wall/monotonic deadlines; it never
contains credentials. **That readiness file does not exist yet.** If writing
it fails, startup exits without arming or issuing a mutation. Root should
verify its complete JSON, expected identities and live process after launch;
file existence alone is insufficient. A partial or existing receipt refuses
another launch and must be inspected before any separately reviewed retry.

Once armed, the loop neither reads local accounting/configuration nor writes
state or logs. It checks provider presence at intervals of at most 20 seconds
(apart from bounded request time) so normal collection can end it early. It
attempts termination at the earlier of the fixed wall cutoff and the startup
monotonic deadline, including when the wall clock moves backward. At cutoff a
failed presence query cannot prevent a termination request. Failed mutations
and failed absence verification retry; only verified provider absence permits
successful exit. Even if a previous mutation succeeded but its verification
failed, the next iteration checks again. Removed pods are not replaced with
new IDs. The unchanged cloud API function keeps curl's `--max-time 30`; its
in-memory subprocess wrapper adds a 35-second parent timeout.

`launch-command-v1.sh` contains the concrete, unexecuted command. Python
`subprocess.Popen` passes the exact caffeinate/Python/guard/plan argument list
with `start_new_session=True` and all three child streams set to `DEVNULL`.
The launcher prints only the spawned wrapper PID and creates no PID/log file.
Thus no disk log can fill or kill the guard. The command wraps the fallback
in its own `caffeinate -i -s` process;
the original guard's existing wake lock may release if that guard crashes.
The new wrapper exits when this fallback exits. No original guard PID, wake
lock, source, or collector identity is changed.

The fallback does not claim a provider can be terminated through a sustained
network/provider outage, or while the host is shut down or forcibly asleep.
Its unchanged trigger and roughly 12-minute margin are retained, and request
retries continue without local-write dependencies. The normal collector still
owns the usual evidence verification and accounting receipt updates. This
fallback intentionally performs no such writes after arming, even if it has
to enforce the cap before evidence collection completes.

Offline tests use only fake provider calls and temporary fixtures inside this
directory. `tests-v3.json` and `tests-v3.log` retain the final 27-test
command/results (earlier v1/v2 receipts remain available);
`preparation-receipt-v1.json` records the no-launch preparation and accounting
cross-check. No readiness receipt has been manufactured by the tests.
