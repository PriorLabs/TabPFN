# ruff: noqa: T201, S603
#  Copyright (c) Prior Labs GmbH 2026.

r"""Run a command under `srun`, re-submitting it when the allocation is lost.

The benchmarks in this directory take hours on tables of tens of GB, and the
partitions that have that much memory going spare are spot ones. A spot node is
reclaimed whenever the cluster wants it back, which kills the step mid-run --
several hours in, with nothing recorded. Slurm's own `--requeue` only applies to
batch jobs, not to the interactive `srun` these are run under.

So: submit, and if the *allocation* died rather than the command, submit again.
The two are told apart by what Slurm says on the way out, not by the exit code --
a preempted step and a command that returned 1 both surface as a non-zero exit,
and re-running a genuine failure would just burn the node again. Anything Slurm
does not label as its own is passed straight back to the caller.

Attempts stop at `--attempts`, and the wait between them grows, so a partition
that has nothing to give is not hammered.

Usage -- everything after `--` is the command, and the srun flags come before it:

    uv run scripts/srun_retry.py -p cpuhighmem16spot --mem=0 --time=05:00:00 -- \
        uv run scripts/bench_clean_data.py --mix all --reference main

A re-submission restarts the command; it does not pick up where the last one got
to. What the lost run recorded is kept, so the next attempt gates against those
baselines rather than overwriting them unseen, and the reference worktree
`--reference` built is reused. Wrap each stage of a long run in its own call if
losing the whole thing to a late preemption is not acceptable.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time

# What Slurm prints when the allocation goes away underneath the step, as opposed
# to the step itself failing. Matched against the run's stderr.
ALLOCATION_LOST = (
    "PREEMPTION",
    "preempted",
    "DUE TO NODE FAILURE",
    "Unable to allocate resources",
    "job allocation",
    "Job allocation",
    "CANCELLED AT",
    "task 0: Killed",
    "error: Node failure",
)

DEFAULT_ATTEMPTS = 6
FIRST_BACKOFF_S = 30.0
BACKOFF_GROWTH = 2.0
MAX_BACKOFF_S = 600.0


def allocation_was_lost(stderr: str) -> str | None:
    """The phrase that says Slurm took the allocation away, if it is there."""
    return next((phrase for phrase in ALLOCATION_LOST if phrase in stderr), None)


def run_once(srun_flags: list[str], command: list[str]) -> tuple[int, str]:
    """One `srun`, with stderr teed so it can be both seen and matched on.

    stdout goes straight through untouched: these runs print tables and progress
    that are worth watching live, and buffering them to inspect afterwards would
    hide a run that is doing nothing.
    """
    full = ["srun", *srun_flags, *command]
    print(f"\n+ {shlex.join(full)}", flush=True)
    with subprocess.Popen(full, stderr=subprocess.PIPE, text=True) as child:
        captured = []
        assert child.stderr is not None
        for line in child.stderr:
            print(line, end="", file=sys.stderr, flush=True)
            captured.append(line)
    return child.returncode, "".join(captured)


def main() -> int:
    """Submit until the command runs to completion or the attempts run out."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help=f"Submissions before giving up. Default {DEFAULT_ATTEMPTS}.",
    )
    known, rest = parser.parse_known_args()

    if "--" not in rest:
        parser.error("separate the srun flags from the command with `--`")
    split = rest.index("--")
    srun_flags, command = rest[:split], rest[split + 1 :]
    if not command:
        parser.error("no command given after `--`")

    backoff = FIRST_BACKOFF_S
    for attempt in range(1, known.attempts + 1):
        exit_code, stderr = run_once(srun_flags, command)
        if exit_code == 0:
            return 0

        lost = allocation_was_lost(stderr)
        if lost is None:
            print(
                f"\nsrun_retry: the command exited {exit_code} and Slurm did not "
                "report losing the allocation, so this is the command's own "
                "failure -- not re-submitting.",
                file=sys.stderr,
            )
            return exit_code

        if attempt == known.attempts:
            print(
                f"\nsrun_retry: allocation lost ({lost!r}) on the last of "
                f"{known.attempts} attempts.",
                file=sys.stderr,
            )
            return exit_code

        print(
            f"\nsrun_retry: allocation lost ({lost!r}) on attempt {attempt} of "
            f"{known.attempts}; re-submitting in {backoff:.0f}s. Whatever the run "
            "recorded before it died is kept, and is gated against rather than "
            "re-measured.",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(backoff)
        backoff = min(backoff * BACKOFF_GROWTH, MAX_BACKOFF_S)

    return 1


if __name__ == "__main__":
    sys.exit(main())
