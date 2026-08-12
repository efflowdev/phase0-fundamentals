"""Build-time and image-size measurements for the two Dockerfiles.

"Multi-stage builds are better" is not a finding; the finding is *how much*, on
*which* axis, and *which* of the three tricks is doing the work. There are three
and they are usually conflated:

  * layer ordering — dependencies installed before source is copied
  * the multi-stage split — uv and the toolchain excluded from the runtime image
  * the BuildKit cache mount — wheels survive even when the layer above them dies

The scenarios below separate them. The one worth reading twice is
`no_layer_cache`: it is a `--no-cache` build, so every layer re-runs, but the
cache mount survives. Whatever that build saves over `cold` is exactly what the
cache mount is worth, and it is the number the roadmap's `--mount=type=cache`
line is actually buying.

Every timing is one run, not a mean. Build timings are dominated by effects far
larger than their variance, and a script that took ten minutes to answer would
not get run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"

# Three variants, because two would only have shown that the textbook pattern
# beats the careless one. The third exists because the first measurement said
# the textbook pattern LOST on a source change, and the fix had to be measured
# rather than assumed.
VARIANTS = {
    "naive": ("day6_stack/Dockerfile.naive", "phase0-app:naive"),
    "multi": ("day6_stack/Dockerfile", "phase0-app:local"),
    "split": ("day6_stack/Dockerfile.split", "phase0-app:split"),
}

# The file edited to simulate a code change. Any source file does; this one is
# copied by both Dockerfiles.
SOURCE_PROBE = HERE / "healthcheck.py"
# Invalidating the dependency layer means changing what the first COPY sees.
# A comment in pyproject.toml does that without changing a single dependency,
# so the scenario measures cache behaviour and not a different resolution.
DEPS_PROBE = ROOT / "pyproject.toml"


@dataclass
class Result:
    scenario: str
    dockerfile: str
    seconds: float
    note: str = ""


@contextmanager
def temporarily_appended(path: Path, line: str):
    """Edit a file, yield, restore the exact original bytes.

    Byte-for-byte restoration matters more than it looks: leaving the probe
    comment behind would poison the next run's cache and quietly turn every
    later 'warm' measurement into a cold one.
    """
    original = path.read_bytes()
    try:
        path.write_bytes(original + line.encode())
        yield
    finally:
        path.write_bytes(original)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    # check=False: some callers expect a non-zero exit (the cross-arch build is
    # allowed to be refused) and read returncode themselves.
    return subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, check=False, **kwargs
    )


def build(dockerfile: str, tag: str, *, extra: list[str] | None = None) -> float:
    cmd = [
        "docker", "buildx", "build",
        "--file", dockerfile,
        "--tag", tag,
        "--load",
        *(extra or []),
        ".",
    ]
    started = time.perf_counter()
    proc = run(cmd)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(f"build failed ({dockerfile}):\n{proc.stderr[-3000:]}")
    return elapsed


def image_size_mb(tag: str) -> dict[str, float]:
    """Two sizes, because the obvious one is wrong and quietly so.

    `docker image inspect -f {{.Size}}` reports 114 MB for an image that
    `docker images` reports as 530 MB. Neither is lying: colima runs the
    containerd image store, where inspect returns the size of the content in the
    store (compressed layer blobs) and `docker images` returns the unpacked
    size. They differ by ~4.4x here.

    Which one matters depends on the question. Compressed is what crosses the
    network on a pull and what a registry bills for; unpacked is what the disk
    holds and what a cold start has to expand. Reporting one number labelled
    "image size" is how you end up comparing a pull cost against a disk cost.
    """
    inspected = run(["docker", "image", "inspect", "-f", "{{.Size}}", tag])
    inspected.check_returncode()

    listed = run(["docker", "images", tag, "--format", "{{.Size}}"])
    listed.check_returncode()
    return {
        "content_mb": round(int(inspected.stdout.strip()) / (1024 * 1024), 1),
        "unpacked_mb": round(_parse_docker_size(listed.stdout.strip()), 1),
    }


def _parse_docker_size(text: str) -> float:
    """'530MB' -> 530.0. Docker prints decimal units here, not binary ones."""
    match = re.match(r"([\d.]+)\s*([kKMGT]?B)", text)
    if not match:
        return float("nan")
    value, unit = float(match.group(1)), match.group(2).upper()
    factor = {"B": 1 / 1e6, "KB": 1 / 1e3, "MB": 1.0, "GB": 1e3, "TB": 1e6}[unit]
    return value * factor


def context_size_mb() -> tuple[float, float]:
    """How much the .dockerignore has to hold back — NOT bytes transferred.

    The legacy builder tarred the whole context and shipped it before doing
    anything, which is where "keep your context small" comes from. BuildKit does
    not: it resolves each COPY against the (dockerignore-filtered) context and
    transfers only the subset that instruction needs. A plain-progress build here
    reports `transferring context: 6.03kB`, not 260 MB.

    So this is the exposure, not the cost: the repo is 260 MB on disk and 4.2 MB
    tracked, and the gap is .venv, .git and the derived indexes. Without the
    dockerignore a `COPY . .` — which Dockerfile.naive has — pulls the lot.
    """
    total = sum(f.stat().st_size for f in ROOT.rglob("*") if f.is_file())

    proc = run(["git", "ls-files", "-z"])
    tracked = 0.0
    if proc.returncode == 0:
        for name in proc.stdout.split("\0"):
            path = ROOT / name
            if name and path.is_file():
                tracked += path.stat().st_size
    return total / (1024 * 1024), tracked / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-prune",
        action="store_true",
        help="keep existing build cache; the 'cold' number is then meaningless",
    )
    parser.add_argument("--skip-crossarch", action="store_true")
    args = parser.parse_args()

    results: list[Result] = []

    for variant, (dockerfile, tag) in VARIANTS.items():
        # Pruned before EACH variant, not once at the start. multi and split
        # share a uv cache mount id, so measuring split's cold build after
        # multi's would have handed it a warm wheel cache and made the third
        # variant look faster for a reason that has nothing to do with it.
        if not args.skip_prune:
            print(f"\n--- {variant}: pruning build cache ---")
            run(["docker", "buildx", "prune", "-af"])

        print(f"{variant}: cold")
        results.append(Result("cold", variant, build(dockerfile, tag)))

        print(f"{variant}: nothing changed")
        results.append(Result("noop", variant, build(dockerfile, tag)))

        print(f"{variant}: one source line changed")
        with temporarily_appended(SOURCE_PROBE, "\n# build-cache probe\n"):
            results.append(Result("source_change", variant, build(dockerfile, tag)))

        print(f"{variant}: dependency layer invalidated")
        with temporarily_appended(DEPS_PROBE, "\n# build-cache probe\n"):
            results.append(Result("deps_change", variant, build(dockerfile, tag)))

        print(f"{variant}: --no-cache, uv cache mount warm")
        results.append(
            Result(
                "no_layer_cache",
                variant,
                build(dockerfile, tag, extra=["--no-cache"]),
                "vs `cold`: what the cache mount is worth (naive has none)",
            )
        )

    crossarch: dict[str, object] = {}
    if not args.skip_crossarch:
        print("\nsplit: --platform linux/amd64 (emulated through Rosetta)")
        dockerfile, tag = VARIANTS["split"]
        try:
            seconds = build(
                dockerfile, f"{tag}-amd64", extra=["--platform", "linux/amd64"]
            )
            crossarch = {"ok": True, "seconds": round(seconds, 1)}
            results.append(Result("crossarch_amd64", "split", seconds))
        except RuntimeError as exc:
            # The default `docker` driver builds only for the host platform.
            # Recording the refusal is more useful than hiding it.
            crossarch = {"ok": False, "error": str(exc)[-600:]}
            print("     refused — see runs/build_measurements.json")

    sizes = {variant: image_size_mb(tag) for variant, (_, tag) in VARIANTS.items()}
    ctx_all, ctx_tracked = context_size_mb()

    scenarios = ["cold", "noop", "source_change", "deps_change", "no_layer_cache"]
    print(f"\n{'scenario':<16}" + "".join(f"{v:>10}" for v in VARIANTS))
    print("-" * (16 + 10 * len(VARIANTS)))
    for scenario in scenarios:
        row = f"{scenario:<16}"
        for variant in VARIANTS:
            match = next(
                (r for r in results if r.scenario == scenario and r.dockerfile == variant),
                None,
            )
            row += f"{match.seconds:>10.1f}" if match else f"{'-':>10}"
        print(row)

    print(f"\n{'image size':<16}" + "".join(f"{v:>10}" for v in VARIANTS))
    for key in ("unpacked_mb", "content_mb"):
        print(
            f"{key:<16}" + "".join(f"{sizes[v][key]:>10.1f}" for v in VARIANTS)
        )
    # Deliberately not called "build context": BuildKit transfers per-COPY
    # subsets, and labelling this as bytes-on-the-wire would be a lie by caption.
    print(f"\nrepo   on disk   {ctx_all:>8.1f} MB   (what `COPY . .` would expose)")
    print(f"       tracked   {ctx_tracked:>8.1f} MB")

    RUNS.mkdir(exist_ok=True)
    payload = {
        "results": [asdict(r) | {"seconds": round(r.seconds, 1)} for r in results],
        "image_size_mb": sizes,
        "context_mb": {"on_disk": round(ctx_all, 1), "tracked": round(ctx_tracked, 1)},
        "crossarch": crossarch,
    }
    (RUNS / "build_measurements.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {RUNS / 'build_measurements.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
