"""Where does raising the concurrency stop helping?

    uv run python day5_async/benchmark.py                  # both arms
    uv run python day5_async/benchmark.py --arm fake       # no Ollama needed

Two arms, because "concurrency is faster" is not a finding and the interesting
question is *which side* runs out first.

* **fake** — the work function is `asyncio.sleep`. Nothing leaves the process,
  so this measures the ceiling of the runner itself. Speedup should track the
  concurrency level almost exactly; if it does not, the harness is the problem.
* **ollama** — real calls to a local model. This measures the *server*. Ollama
  serialises requests beyond `OLLAMA_NUM_PARALLEL` (which defaults to a small
  number), so the curve is expected to flatten early no matter what the client
  asks for.

The gap between the two curves is the whole point: it tells you whether a
semaphore of 32 in your eval script is doing anything at all, and it is the
number you use to size that semaphore in every later project.
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx

from phase0.ollama import MODEL, chat
from phase0.runner import run_all

LEVELS = (1, 2, 4, 8, 16, 32)
FAKE_LATENCY_S = 0.2
PROMPT = "Reply with exactly one word: ok"


async def timed(items, work, concurrency: int, desc: str) -> float:
    started = time.perf_counter()
    results = await run_all(
        items,
        work,
        key=str,
        concurrency=concurrency,
        checkpoint=None,  # benchmarking the runner, not its resume path
        progress=False,
        desc=desc,
    )
    elapsed = time.perf_counter() - started
    failed = sum(1 for r in results if not r.ok)
    if failed:
        print(f"    ({failed} of {len(results)} failed at concurrency {concurrency})")
    return elapsed


async def run_arm(name: str, n: int, levels: tuple[int, ...], work_factory) -> None:
    print(f"\n{name} · {n} calls\n")
    header = f"{'concurrency':>11}  {'wall':>8}  {'calls/s':>9}  {'speedup':>8}  {'efficiency':>10}"
    print(header)
    print("-" * len(header))

    baseline = None
    for level in levels:
        async with httpx.AsyncClient() as client:
            elapsed = await timed(range(n), work_factory(client), level, name)
        baseline = baseline if baseline is not None else elapsed
        speedup = baseline / elapsed if elapsed else 0.0
        print(
            f"{level:>11}  {elapsed:>7.2f}s  {n / elapsed:>9.1f}  "
            f"{speedup:>7.2f}x  {100 * speedup / level:>9.0f}%"
        )


def fake_factory(_client):
    async def work(_item) -> str:
        await asyncio.sleep(FAKE_LATENCY_S)
        return "ok"

    return work


def ollama_factory(client):
    async def work(item) -> str:
        # A distinct seed per item so nothing is served from a cache and every
        # call does real work.
        return await chat(client, PROMPT, seed=int(item), num_predict=4)

    return work


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("fake", "ollama", "both"), default="both")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    levels = tuple(level for level in LEVELS if level <= args.n)

    if args.arm in ("fake", "both"):
        ideal = args.n * FAKE_LATENCY_S
        print(f"\nfake arm: {args.n} x {FAKE_LATENCY_S}s = {ideal:.1f}s sequential")
        await run_arm("fake (asyncio.sleep)", args.n, levels, fake_factory)

    if args.arm in ("ollama", "both"):
        await run_arm(f"ollama ({args.model})", args.n, levels, ollama_factory)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
