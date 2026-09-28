# September 24 follow-up status

The campaign has been collected and the owned GPU rental terminated. 11 of 16 panels have reportable results under their declared contracts.

[Timing report](REPORT.md) · [Completed review and retained failures](REVIEW.md) · [Completion, GPU validation and cost receipt](completion.json).

The original experimental exactness and BLS qualification failures remain unchanged. BLS execution rates do not gain numerical qualification.

R2 bucket `cuvarbase`, prefix `throughput-followup-20260924/resumed-65a59504`: all 7 objects passed full SHA256 read-back. Local archives remain intact. Benchmark review is complete; release publication is pending.

The expanded GPU suite passed 2,091 tests, with one expected failure and zero skips. The additional gate initially failed because the validation launcher had not made the package importable. A [separate September 27 check](../release-gate-20260927/README.md) installed the unchanged wheel and passed all 14 numerical/runtime checks plus six dependency preflights. Its evidence is also verified in R2 and its rental is terminated. The original failed receipt remains intact.

[Persistent monitoring](../../../../docs/JOB_MONITORING.md) now distinguishes validation failures from successful collection and queues follow-up work into the existing conversation.
