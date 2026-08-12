"""Day 4's embedding workload, run inside a container with a memory ceiling.

The point is not the embeddings — day 4 already has those. The point is what a
container OOM kill looks like from the inside, which is: nothing. No exception,
no traceback, no log line, no stack. The process is sent SIGKILL by the kernel
and simply stops mid-sentence, and the only evidence is an exit code of 137
(128 + 9) that Docker reports afterwards. An embedding model dying this way at
1am reads as a hang, not as a failure, and that is why it is worth having seen
once on purpose.

Two ceilings matter and they are different numbers: the resident set while the
ONNX model loads, and the resident set while 500 documents are in flight. The
first is a step, the second is a ramp, and only the second scales with the batch.

Reads the committed ESCI cache directly rather than importing day 4's corpus
builder: this job wants texts, not a retrieval corpus, and corpus.build() would
drag in query grouping and relevance judgments that a memory test never touches.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "day4_search" / "cache" / "esci_rows.jsonl"
MODEL = "BAAI/bge-small-en-v1.5"


def rss_mb() -> float:
    """Resident set size in MB, on Linux and macOS.

    ru_maxrss is bytes on Darwin and kilobytes on Linux — the same field, two
    units, and getting it wrong gives you a number 1024x off in either direction.
    Linux also gets the live value from /proc rather than the high-water mark,
    because watching it climb is the whole exercise.
    """
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


CGROUP_FILES = (
    (Path("/sys/fs/cgroup/memory.max"), "max"),  # cgroup v2
    (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), None),  # cgroup v1
)


def cgroup_limit_mb(candidates=CGROUP_FILES) -> float | None:
    """The ceiling the kernel will actually enforce, as the container sees it.

    Worth printing, because `mem_limit` in compose and the number inside the
    container are only equal when nothing is silently reinterpreting the value —
    and a limit below what the runtime rounds to is a common way to set a limit
    that does nothing.

    `candidates` is a parameter so the two encodings can be tested on real bytes
    without a container: v2 writes the literal string "max" when unlimited, v1
    writes a sentinel just under 2^63, and reading either as a number gives you
    a limit of nine million terabytes.
    """
    for path, unlimited in candidates:
        if path.exists():
            raw = path.read_text().strip()
            if raw == unlimited:
                return None
            value = int(raw)
            return None if value > 2**62 else value / (1024 * 1024)
    return None


def load_titles(limit: int, cache: Path = CACHE) -> list[str]:
    """Unique product titles, in file order.

    Keyed by product_id and not by title: ESCI rows are (query, product) pairs,
    so a product judged against four queries is four rows, and embedding it four
    times would inflate the workload without changing what it measures.
    """
    if not cache.exists():
        raise SystemExit(
            f"no ESCI cache at {cache} — run `uv run python day4_search/corpus.py` first"
        )
    seen: dict[str, str] = {}
    with cache.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            title = (row.get("product_title") or "").strip()
            if title and row.get("product_id") not in seen:
                seen[row["product_id"]] = title
            if len(seen) >= limit:
                break
    return list(seen.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    limit = cgroup_limit_mb()
    print(f"cgroup memory limit : {limit:.0f} MB" if limit else "cgroup limit : none")
    print(f"fastembed cache     : {os.environ.get('FASTEMBED_CACHE_PATH', '<default>')}")
    print(f"rss at start        : {rss_mb():.0f} MB", flush=True)

    titles = load_titles(args.limit)
    print(f"loaded {len(titles)} titles, rss {rss_mb():.0f} MB", flush=True)

    started = time.perf_counter()
    from fastembed import TextEmbedding  # imported late so the RSS step is visible

    print(f"imported fastembed, rss {rss_mb():.0f} MB", flush=True)

    embedder = TextEmbedding(model_name=MODEL)
    print(
        f"model loaded in {time.perf_counter() - started:.1f}s, rss {rss_mb():.0f} MB",
        flush=True,
    )

    peak = rss_mb()
    done = 0
    for _vector in embedder.embed(titles, batch_size=args.batch):
        done += 1
        if done % 50 == 0:
            peak = max(peak, rss_mb())
            print(f"  {done:>4}/{len(titles)} embedded, rss {rss_mb():.0f} MB", flush=True)

    peak = max(peak, rss_mb())
    print(
        f"done: {done} vectors in {time.perf_counter() - started:.1f}s, "
        f"peak rss {peak:.0f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
