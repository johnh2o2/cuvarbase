# Background job monitoring

The September 24 collector verified the evidence and stopped the rental even
though the extra release gate failed. Collection success did not mean release
success. The September 27 monitor checks those outcomes separately and brings
actionable changes back to the same Codex conversation.

`tools/watch_jobs.py` polls every 60 seconds in a background observer started
from Codex's authorized project context. The independent macOS LaunchAgent
`com.cuvarbase.job-monitor` runs a copy of `tools/watch_monitor.py` every 60
seconds to check that observer's heartbeat. Its configuration, durable state,
delivery receipts, checkpoints, and acknowledgements are in:

```
/Users/johnhoffman/Library/Application Support/cuvarbase-job-monitor/
```

The monitor:

- Checks the remote controller, progress files, step exit codes, stage deadlines,
  release validation, qualified panel counts, collection, shutdown, and backup.
- Flags three consecutive connection failures or 15 minutes without log or
  checkpoint activity. Each job can declare a different inactivity threshold.
- Restarts a dead budget guard or collector at most three times per hour,
  preserving the original rental identity, deadline, and budget.
- Keeps failures visible after successful collection. A job needs passing
  required steps, verified local evidence, verified rental shutdown, and verified
  R2 backup before it becomes ready for review.
- Queues one follow-up per distinct incident into the existing conversation
  using the installed `codex queue` command. Multiple findings in one poll are
  combined. A failed delivery remains pending and retries with backoff.
- Requests a desktop notification for new incidents and for a queued follow-up
  still unacknowledged after ten minutes. Notification visibility depends on
  macOS notification settings and Focus mode.

Polling makes no model calls. Follow-up turns use the existing account and
permissions. The monitor creates no rentals, retries no numerical experiments,
changes no qualification thresholds, deletes no evidence, and publishes no
release. A completed failed experiment stays a failed experiment; operational
repairs receive separate receipts.

The LaunchAgent reloads at login. If the observer has no successful poll for
three minutes, the watchdog queues a recovery turn in this conversation. The
observer and watchdog have separate code, logs, locks, and delivery state.
macOS prevents a login service from reading the Documents project directly;
the watchdog reads only its own Application Support state and asks Codex to
restore the project observer using its existing authorized access.

The Mac must be powered on and online. Budget
guards hold an idle-sleep assertion during paid jobs; closing the lid or losing
power can still interrupt local supervision. Remote controllers have their own
bounded execution times, and queued follow-ups wait if Codex is unavailable or
the account is rate-limited. This is local supervision, not an always-on cloud
monitor. [Official scheduled-task documentation](https://learn.chatgpt.com/docs/automations?surface=app)
also describes the host availability requirements for local scheduled work.

## Operations

Read `status.json` for the latest poll, `state.json` for incidents and delivery
receipts, `checkpoints/` for the last remote states, `observer.log` for polling
failures, and `watchdog.json` for independent health checks. The watchdog's
own errors go to `launchd.err.log`. `monitor-error.json` records an attempt to
bring a polling failure back into the conversation.

```sh
python3 tools/watch_jobs.py --config '/Users/johnhoffman/Library/Application Support/cuvarbase-job-monitor/config.json' status
```

Register every new long-running cuvarbase rental in `config.json` before leaving
it unattended. Each job specifies a unique `id`, local evidence `path`,
`remote_root`, exact controller script path, `required_steps`, `step_limits`,
`stall_seconds`, and whether verified cloud backup is required. Its existing
`ops/rental.py` owns budget enforcement and `ops/monitor.py` owns collection;
neither may reset a deadline on restart. Keep configuration free of credentials.

When a monitor follow-up arrives, acknowledge its event with the command in the
message, inspect the evidence, and continue the authorized work. Only close the
job's review after inspecting its actual outcome and preserving its artifacts:

```sh
python3 tools/watch_jobs.py --config '/Users/johnhoffman/Library/Application Support/cuvarbase-job-monitor/config.json' review --job JOB_ID --outcome 'Reviewed result and remaining limitations'
```

To disable the independent heartbeat watchdog without touching a GPU job's
existing guard:

```sh
launchctl bootout gui/$(id -u)/com.cuvarbase.job-monitor
```

The supervisor's tests reproduce the archived-but-failed validation case,
missing evidence or backup, dead/stalled jobs, notification delivery failures,
deduplication across restarts, actual guard/collector process recovery, and an
independent watchdog detecting and recovering from a missing heartbeat.
