"""Run N slow async calls under a concurrency cap, and survive the run.

This is the shape every eval, every labelling pass and every batch extraction in
this repo uses. Day 3 wrote it inline for one experiment; it lives here now
because day 5 needed it too, and P1 and P5 will need it again.

Four properties, each of which exists because of a specific way the naive
version loses your work:

* **Bounded concurrency.** `asyncio.gather` over 500 items opens 500 connections
  and gets you rate-limited into oblivion. A `Semaphore` caps what is in flight
  without capping what is queued.
* **Failures are data, not exceptions.** `gather(..., return_exceptions=True)` is
  the well-known half of this. The other half is that a `Result` with `ok=False`
  can be counted, grouped and retried, while an exception object in a list can
  only be printed. One bad item out of 500 should cost you one row.
* **Checkpointing as you go.** Each completed item is appended to a JSONL file
  the moment it lands. A crash at item 480 loses at most one line, not 480 paid
  API calls.
* **Resume skips successes only.** Failures are re-attempted on the next run,
  because the usual reason for a failure is a rate limit or a timeout, and the
  usual fix is running it again.

JSONL rather than SQLite on purpose: the runner should not need to know the
shape of what you are running. Analysis code loads the file into whatever store
it wants afterwards.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONCURRENCY = 8


@dataclass(frozen=True)
class Result:
    id: str
    ok: bool
    value: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0
    resumed: bool = False

    def to_json(self) -> str:
        # default=str so one unserialisable value degrades to a string instead
        # of killing a run that has already paid for 300 calls. Values must be
        # JSON round-trippable for `resume` to hand them back unchanged.
        return json.dumps(
            {
                "id": self.id,
                "ok": self.ok,
                "value": self.value,
                "error": self.error,
                "elapsed_ms": round(self.elapsed_ms, 3),
            },
            ensure_ascii=False,
            default=str,
        )

    @classmethod
    def from_json(cls, line: str) -> Result:
        data = json.loads(line)
        return cls(
            id=data["id"],
            ok=bool(data["ok"]),
            value=data.get("value"),
            error=data.get("error"),
            elapsed_ms=float(data.get("elapsed_ms") or 0.0),
            resumed=True,
        )


def load_checkpoint(path: Path | str | None) -> dict[str, Result]:
    """Last line wins, so a re-attempted failure overwrites its earlier row."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}

    done: dict[str, Result] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            result = Result.from_json(line)
        except (json.JSONDecodeError, KeyError):
            # A torn final line is exactly what a crash mid-append looks like.
            # Skipping it costs one redundant call; refusing to start costs the
            # whole run.
            continue
        done[result.id] = result
    return done


class _Appender:
    """Serialises writes so two coroutines cannot interleave a line.

    A single `write` of a short string would almost certainly be atomic anyway,
    but "almost certainly" is how you get a corrupt checkpoint at 3am, and the
    lock costs nothing next to a network call.
    """

    def __init__(self, path: Path | str | None) -> None:
        self._path = Path(path) if path else None
        self._lock = asyncio.Lock()
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    async def append(self, result: Result) -> None:
        if not self._path:
            return
        async with self._lock:
            with self._path.open("a") as handle:
                handle.write(result.to_json() + "\n")
                handle.flush()


async def run_all[T](
    items: Sequence[T],
    work: Callable[[T], Awaitable[Any]],
    *,
    key: Callable[[T], str],
    concurrency: int = DEFAULT_CONCURRENCY,
    checkpoint: Path | str | None = None,
    resume: bool = True,
    retry_failed: bool = True,
    progress: bool = True,
    desc: str = "",
) -> list[Result]:
    """Run `work` over `items`, at most `concurrency` at a time.

    Returns one `Result` per item, in the order the items were given —
    including items served from the checkpoint, which carry `resumed=True`.
    """
    done = load_checkpoint(checkpoint) if resume else {}
    appender = _Appender(checkpoint)

    def already_done(item: T) -> Result | None:
        previous = done.get(key(item))
        if previous is None:
            return None
        if previous.ok or not retry_failed:
            return previous
        return None

    pending = [item for item in items if already_done(item) is None]
    semaphore = asyncio.Semaphore(max(1, concurrency))
    bar = _make_bar(len(pending), progress, desc)

    async def one(item: T) -> Result:
        item_id = key(item)
        async with semaphore:
            started = time.perf_counter()
            try:
                value = await work(item)
                result = Result(
                    item_id, True, value, None, (time.perf_counter() - started) * 1000
                )
            except Exception as exc:  # noqa: BLE001 - the whole point is to record it
                # CancelledError is a BaseException in 3.8+, so Ctrl-C and
                # task cancellation still propagate rather than being logged as
                # a failed item.
                result = Result(
                    item_id,
                    False,
                    None,
                    f"{type(exc).__name__}: {exc}",
                    (time.perf_counter() - started) * 1000,
                )
        await appender.append(result)
        if bar is not None:
            bar.update(1)
        return result

    try:
        fresh = await asyncio.gather(*(one(item) for item in pending))
    finally:
        if bar is not None:
            bar.close()

    by_id = {result.id: result for result in fresh}
    by_id.update({k: v for k, v in done.items() if k not in by_id})
    return [by_id[key(item)] for item in items if key(item) in by_id]


def _make_bar(total: int, progress: bool, desc: str):
    if not progress or total == 0:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    # disable=None means "off when stderr is not a TTY", so piping a run into a
    # file or a pipeline gets clean output instead of 500 redrawn bar frames.
    return tqdm(
        total=total,
        desc=desc or "running",
        unit="call",
        leave=False,
        disable=None,
    )


def summarize(results: Sequence[Result]) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    fresh = [r for r in results if not r.resumed]
    latencies = sorted(r.elapsed_ms for r in fresh) or [0.0]
    return {
        "total": len(results),
        "ok": len(ok),
        "failed": len(failed),
        "resumed": len(results) - len(fresh),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)],
    }
