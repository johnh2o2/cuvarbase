#!/usr/bin/env python3
"""Phase 5 merge rehearsal. Prepared separately; run only after T' is pushed.

Invoke with the Phase 5 local venv's Python. An optional positional SHA
must equal the current HEAD. Logs stay under .git/phase5/rehearsal/.
No fetch, push, tag operation, master checkout, or main-checkout edit occurs.
On failure, the temporary checkout and rehearsal branch remain for diagnosis.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import traceback
import zipfile


FROZEN_T = "1032caf029570dc4841db1c594a2cbb1654e8fd8"
FROZEN_TREE = "b023c3e8d163010dbae2fc0b7cd5204ca04384d1"
BRANCH = "rehearsal-1.0.0"
EXPECTED_CONFLICTS = {
    "README.rst",
    "cuvarbase/lombscargle.py",
    "cuvarbase/pdm.py",
    "cuvarbase/utils.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tp", nargs="?", default="HEAD", help="current T' HEAD")
    args = parser.parse_args()
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    logs = repo / ".git" / "phase5" / "rehearsal"
    logs.mkdir(parents=True, exist_ok=True)
    summary_path = logs / "summary.json"
    if summary_path.exists():
        raise SystemExit(
            "Existing rehearsal/summary.json: inspect prior result before rerunning; "
            "this helper will not overwrite its evidence or clean a prior checkout."
        )
    build_python = repo / ".git" / "phase5" / "local-venv" / "bin" / "python"
    state = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_T": FROZEN_T,
        "frozen_tree": FROZEN_TREE,
        "branch": BRANCH,
        "status": "running",
        "outcomes": {},
        "commands": [],
    }
    checkout = None
    temporary_root = None
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_EDITOR"] = "true"
    env["GIT_MERGE_AUTOEDIT"] = "no"
    env.pop("PYTHONPATH", None)

    def save():
        summary_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    def check(name, condition, detail=None):
        state["outcomes"][name] = {"passed": bool(condition), "detail": detail}
        save()
        if not condition:
            raise RuntimeError(name + (": " + str(detail) if detail else ""))

    def run(name, command, cwd=None, allowed=(0,)):
        cwd = Path(cwd) if cwd is not None else repo
        command = [str(item) for item in command]
        log_path = logs / (name + ".log")
        if log_path.exists():
            raise RuntimeError("Refusing to overwrite log " + str(log_path))
        with log_path.open("w") as log:
            log.write("cwd: " + str(cwd) + "\ncommand: " + shlex.join(command) + "\n\n")
            log.flush()
            result = subprocess.run(
                command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True,
            )
            log.write(result.stdout)
            log.write("\nEXIT_CODE=" + str(result.returncode) + "\n")
        state["commands"].append({
            "name": name,
            "command": command,
            "cwd": str(cwd),
            "exit_code": result.returncode,
            "log": log_path.name,
        })
        save()
        if result.returncode not in allowed:
            raise RuntimeError(name + " failed; inspect " + str(log_path))
        return result

    try:
        save()
        check("local_build_python_present", build_python.is_file(), str(build_python))
        tp = run("resolve_tp", ["git", "rev-parse", args.tp + "^{commit}"]).stdout.strip()
        head = run("resolve_head", ["git", "rev-parse", "HEAD"]).stdout.strip()
        state["T_prime"] = tp
        check("requested_tp_is_current_head", tp == head, {"requested": tp, "HEAD": head})
        check("tp_is_later_than_T", tp != FROZEN_T)
        parents = run("tp_parents", ["git", "show", "-s", "--format=%P", tp]).stdout.split()
        check("tp_is_single_child_of_T", parents == [FROZEN_T], parents)
        tree = run("tp_tree", ["git", "rev-parse", tp + "^{tree}"]).stdout.strip()
        state["T_prime_tree"] = tree
        frozen_tree = run("frozen_tree", ["git", "rev-parse", FROZEN_T + "^{tree}"]).stdout.strip()
        check("frozen_tree_matches", frozen_tree == FROZEN_TREE, frozen_tree)
        unchanged = run(
            "analysis_only", ["git", "diff", "--exit-code", FROZEN_T, tp, "--", ".", ":!analysis"],
            allowed=(0, 1),
        )
        check("tp_is_analysis_only", unchanged.returncode == 0)
        remote_tip = run("origin_release_tip", ["git", "rev-parse", "origin/v1.0-fixes"]).stdout.strip()
        check("tp_matches_origin_release_tip", tp == remote_tip, remote_tip)
        origin_master = run("origin_master", ["git", "rev-parse", "origin/master"]).stdout.strip()
        state["origin_master"] = origin_master
        local_master = run("local_master_before", ["git", "rev-parse", "master"]).stdout.strip()
        state["local_master_before"] = local_master
        exists = run(
            "rehearsal_branch_absent", ["git", "show-ref", "--verify", "--quiet", "refs/heads/" + BRANCH],
            allowed=(0, 1),
        )
        check("throwaway_branch_does_not_exist", exists.returncode == 1)
        preview = run(
            "merge_tree", ["git", "merge-tree", "--write-tree", origin_master, tp], allowed=(0, 1),
        )
        lines = [line for line in preview.stdout.splitlines() if line.startswith("CONFLICT")]
        matched = [re.search(r"Merge conflict in (.+)$", line) for line in lines]
        conflict_paths = [match.group(1) for match in matched if match]
        state["merge_tree_conflict_lines"] = lines
        check(
            "merge_tree_has_expected_four_conflicts",
            preview.returncode == 1 and len(lines) == 4
            and len(conflict_paths) == 4 and set(conflict_paths) == EXPECTED_CONFLICTS,
            lines,
        )
        temporary_root = Path(tempfile.mkdtemp(prefix="cuvarbase-phase5-rehearsal-"))
        checkout = temporary_root / "checkout"
        state["temporary_root"] = str(temporary_root)
        state["checkout"] = str(checkout)
        save()
        run("worktree_add", ["git", "worktree", "add", "-b", BRANCH, checkout, origin_master])
        merged = run(
            "merge", ["git", "-c", "rerere.enabled=false", "merge", "--no-ff", "--no-commit", tp],
            cwd=checkout, allowed=(0, 1),
        )
        unmerged = run(
            "unmerged_paths", ["git", "diff", "--name-only", "--diff-filter=U"], cwd=checkout,
        ).stdout.splitlines()
        check(
            "actual_merge_has_expected_four_conflicts",
            merged.returncode == 1 and len(unmerged) == 4 and set(unmerged) == EXPECTED_CONFLICTS,
            unmerged,
        )
        run("take_release_side", ["git", "checkout", "--theirs", "--"] + sorted(EXPECTED_CONFLICTS), cwd=checkout)
        run("stage_resolution", ["git", "add", "--"] + sorted(EXPECTED_CONFLICTS), cwd=checkout)
        remaining = run(
            "unmerged_paths_after_resolution", ["git", "diff", "--name-only", "--diff-filter=U"], cwd=checkout,
        ).stdout.splitlines()
        check("all_conflicts_resolved", not remaining, remaining)
        run("merge_commit", ["git", "commit", "-m", "rehearsal: cuvarbase 1.0.0 gate at " + tp], cwd=checkout)
        merge_commit = run("merge_sha", ["git", "rev-parse", "HEAD"], cwd=checkout).stdout.strip()
        merge_tree = run("merge_tree_sha", ["git", "rev-parse", "HEAD^{tree}"], cwd=checkout).stdout.strip()
        merge_parents = run("merge_parents", ["git", "show", "-s", "--format=%P", "HEAD"], cwd=checkout).stdout.split()
        state["merge_commit"] = merge_commit
        state["merge_tree"] = merge_tree
        check("merge_has_expected_two_parents", merge_parents == [origin_master, tp], merge_parents)
        run("tree_identity_diff", ["git", "diff", "--exit-code", tp, "HEAD"], cwd=checkout)
        check("merge_tree_identical_to_tp", merge_tree == tree, {"merge_tree": merge_tree, "T_prime_tree": tree})
        run("build", [build_python, "-m", "build"], cwd=checkout)
        artifacts = sorted((checkout / "dist").iterdir())
        check(
            "expected_artifacts_built",
            [p.name for p in artifacts] == ["cuvarbase-1.0.0-py3-none-any.whl", "cuvarbase-1.0.0.tar.gz"],
            [p.name for p in artifacts],
        )
        run("twine_check", [build_python, "-m", "twine", "check", "--strict"] + artifacts, cwd=checkout)
        readme = run("frozen_readme", ["git", "show", FROZEN_T + ":README.md"]).stdout
        with tarfile.open(checkout / "dist" / "cuvarbase-1.0.0.tar.gz", "r:gz") as archive:
            pkginfo = archive.extractfile("cuvarbase-1.0.0/PKG-INFO").read().decode()
        with zipfile.ZipFile(checkout / "dist" / "cuvarbase-1.0.0-py3-none-any.whl") as archive:
            metadata = archive.read("cuvarbase-1.0.0.dist-info/METADATA").decode()
        (logs / "PKG-INFO.txt").write_text(pkginfo)
        (logs / "wheel-METADATA.txt").write_text(metadata)
        for label, text in (("sdist", pkginfo), ("wheel", metadata)):
            check(label + "_readme_exactly_matches_T", text.split("\n\n", 1)[1].rstrip("\n") == readme.rstrip("\n"))
            check(label + "_pypi_install_present", "pip install cuvarbase" in text)
            check(label + "_forbidden_banner_absent", "Until v1.0.0" not in text)
            check(label + "_forbidden_git_install_absent", "git+https" not in text)
            check(label + "_no_unpermitted_025_mentions", not any("0.2.5" in line and "since 0.2.5" not in line for line in text.splitlines()))
            check(label + "_measured_counts_present", "1,785 passed + 1 xfailed of 1,786 collected" in text)
            check(label + "_phase2_count_absent", "1,582" not in text)
            check(label + "_xfail_named", "test_notebook_code_cells_compile_without_warnings[Phase Dispersion Minimization.ipynb]" in text)
        state["artifacts"] = [{
            "name": p.name, "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        } for p in artifacts]
        run("tracked_checkout_clean", ["git", "diff", "--exit-code", "HEAD"], cwd=checkout)
        after_head = run("main_head_after", ["git", "rev-parse", "HEAD"]).stdout.strip()
        after_master = run("local_master_after", ["git", "rev-parse", "master"]).stdout.strip()
        check("main_head_unchanged", after_head == tp, after_head)
        check("local_master_unchanged", after_master == local_master, after_master)
        # --force only removes build outputs in this helper's own disposable checkout.
        check("cleanup_path_is_owned", checkout.parent == temporary_root and checkout != repo)
        run("worktree_remove", ["git", "worktree", "remove", "--force", checkout])
        run("branch_delete", ["git", "branch", "-D", BRANCH])
        temporary_root.rmdir()
        state["cleanup"] = "temporary checkout, temporary directory, and rehearsal branch removed"
        run("remaining_worktrees", ["git", "worktree", "list"])
        state["status"] = "passed"
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({key: state[key] for key in (
            "status", "T_prime", "T_prime_tree", "origin_master", "merge_commit", "merge_tree", "cleanup"
        )}, indent=2))
        print("Evidence: " + str(logs))
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        state["cleanup"] = "not attempted after failure; inspect recorded checkout and branch"
        (logs / "failure.txt").write_text(traceback.format_exc())
        save()
        if checkout is not None and checkout.exists():
            result = subprocess.run(
                ["git", "status", "--short"], cwd=checkout, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            (logs / "failure_checkout_status.log").write_text(result.stdout)
        print("Rehearsal failed: " + str(exc), file=sys.stderr)
        print("Evidence and any checkout preserved: " + str(logs), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
