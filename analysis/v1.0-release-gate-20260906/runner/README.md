# Phase 5 execution scripts

`gate.sh` ran the initial source gate through packaging and started artifact
setup. The initial `/workspace` artifact setup was deliberately interrupted
for shared-filesystem latency after its wheel environments installed; its
original helper versions are in `initial-workspace-venvs/`.

`gate_resume.sh` resumed at artifact setup using the current
`artifact_setup.sh`, `verify_artifact_paths.py`, and `wheel_run.py`, whose
fresh isolated venvs reside on local `/tmp`. Logs remained under
`/workspace/logs` and were copied back. `check_pkg_info.py` and
`verify_docs.py` were used by the original source/build gate.

These scripts live outside the frozen clone and do not modify its tracked
files. The clone was always checked out at T. No working-tree rsync was used.
