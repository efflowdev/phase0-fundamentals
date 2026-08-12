"""Where the kernel actually kills the embedding job, as a function of mem_limit.

Run once by hand and the answer looks obvious: set the limit under what the
process needs and it dies. Run it as a sweep and the interesting part shows up —
the job survives at a limit *below* the resident set it reports, because a large
part of that RSS is the memory-mapped ONNX model, and file-backed clean pages get
evicted under pressure rather than counted against you at the moment of the kill.

Which makes "peak RSS" the wrong number to size a container from, in the
direction that hurts: it over-estimates, you set a limit with headroom on top of
an already-inflated figure, and you pay for memory the workload never needed.

Two failure shapes to watch for in the output, because they look different in a
log and both look like a hang from outside:

  * killed during model load  — a step. The last line is the load line.
  * killed mid-batch          — a ramp. The last line is a progress line, and
                                which one it is moves with the limit.

Emits runs/oom_sweep.json. Requires the app image built:
    docker compose -f day6_stack/docker-compose.yml build app
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPOSE = HERE / "docker-compose.yml"
RUNS = HERE / "runs"
CONTAINER = "phase0-embed-1"

DEFAULT_LIMITS = ["240m", "256m", "272m", "288m", "304m", "320m", "2g"]
PROGRESS = re.compile(r"embed-1\s+\|\s+(.*\S)\s*$")


def compose(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    # stderr folded into stdout by the OS, not concatenated afterwards. The
    # first version of this did `proc.stdout + proc.stderr`, which appends every
    # stderr line after every stdout line regardless of when they happened —
    # so the "last line before the kill" was reliably onnxruntime's startup
    # warning, and the load/batch classification below was reliably wrong.
    return subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
    )


def run_at(limit: str) -> dict[str, object]:
    compose("rm", "-fs", "embed")
    proc = compose("up", "embed", env={"EMBED_MEM": limit})
    combined = proc.stdout

    inspect = subprocess.run(
        [
            "docker", "inspect", CONTAINER,
            "--format", "{{.State.ExitCode}} {{.State.OOMKilled}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    exit_code, oom = (inspect.stdout.strip().split() + ["?", "?"])[:2]

    # Only the job's own instrumented lines count as progress. Anything else the
    # container writes — onnxruntime warnings, HF download bars — is noise that
    # would otherwise become "the last line before the kill".
    lines = [
        m.group(1)
        for line in combined.splitlines()
        if (m := PROGRESS.match(line.strip())) and "rss" in m.group(1)
    ]
    compose("rm", "-fs", "embed")

    killed = oom == "true"
    last_line = lines[-1] if lines else ""
    # Where it died, not just that it died. "load" and "batch" are different
    # bugs with different fixes: one is the model, the other is the batch size.
    died_during = None if not killed else ("load" if "model loaded" in last_line else "batch")
    return {
        "limit": limit,
        "oom_killed": killed,
        "exit_code": int(exit_code) if exit_code.isdigit() else None,
        "last_line": last_line,
        "died_during": died_during,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limits", nargs="*", default=DEFAULT_LIMITS)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "runs per limit. The boundary is not a clean line — whether a limit "
            "kills depends on how much of the mmap'd model the kernel can evict, "
            "which depends on what else is resident. One run per limit reports a "
            "threshold that does not reproduce."
        ),
    )
    args = parser.parse_args()

    results = []
    print(f"{'limit':<8} {'outcome':<10} {'exit':<5} {'died during':<12} last line")
    print("-" * 88)
    for limit in args.limits:
        for _ in range(args.repeat):
            row = run_at(limit)
            results.append(row)
            outcome = "OOMKilled" if row["oom_killed"] else "survived"
            exit_code = str(row["exit_code"])
            where = str(row["died_during"] or "-")
            print(
                f"{row['limit']:<8} {outcome:<10} {exit_code:<5} "
                f"{where:<12} {row['last_line']}"
            )

    RUNS.mkdir(exist_ok=True)
    (RUNS / "oom_sweep.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {RUNS / 'oom_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
