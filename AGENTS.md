# Long-running cuvarbase work

The user has requested automatic monitoring so they do not have to return to
ask whether a background job failed. Follow [docs/JOB_MONITORING.md](docs/JOB_MONITORING.md).

- Before leaving a long-running or paid cloud job unattended, register it with
  the persistent job monitor and verify its budget guard and collector are live.
- A process exiting, an archive copying, or a finalizer completing is not enough
  to call the work successful. Inspect required step results and scientific
  qualifications separately from preservation and shutdown receipts.
- Handle queued monitoring events in this conversation: acknowledge receipt,
  inspect the checkpoint, and continue already-authorized work. No new permission
  is required for routine fixes within existing scope and budgets.
- Preserve completed and partial experiments. Never silently replace failed
  numerical results, weaken their gates, or rerun them until they pass. Record
  operational repairs separately and resume from verified checkpoints.
- Verify the actual command's environment and package import before expensive
  stages. Source-tree pytest and a standalone script can have different import
  paths; install the intended wheel or explicitly use the intended source path.
- Keep final release review open until required validation and backup checks
  have been reviewed. Acknowledge failures honestly rather than marking them done.
