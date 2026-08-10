"""Where does a matrix and an argsort stop being enough?

    uv run python day4_search/bench.py
    uv run python day4_search/bench.py --max 100000        # skip the 1.5 GB step

Brute-force search is O(n·d) for the scoring matmul plus the ranking cost, and
both terms are measured here at 1k / 10k / 100k / 1M documents. Random vectors
are fine: the arithmetic does not care what the numbers mean, and timing is all
this answers.

Two things to read off the table. **Memory** grows linearly and unforgivingly —
384 float32 per document is 1.5 KB, so a million documents is 1.5 GB of RAM that
has to stay resident. **Ranking overtakes scoring**: `argsort` is O(n log n) over
the whole corpus to return ten rows, while `argpartition` is O(n), and the gap
between those two columns is why the search function uses the second one.
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np

DIM = 384
SIZES = (1_000, 10_000, 100_000, 1_000_000)
REPEATS = 7


def median_ms(fn, repeats: int = REPEATS) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000)
    return float(np.median(timings))


def bench(n: int, k: int = 10) -> dict[str, float]:
    rng = np.random.default_rng(seed=n)
    matrix = rng.standard_normal((n, DIM), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    query = rng.standard_normal(DIM).astype(np.float32)
    query /= np.linalg.norm(query)

    scores = matrix @ query  # warm the result shape for the ranking benchmarks

    result = {
        "n": n,
        "memory_mb": matrix.nbytes / 1e6,
        "matmul_ms": median_ms(lambda: matrix @ query),
        "argsort_ms": median_ms(lambda: np.argsort(-scores)[:k]),
        "argpartition_ms": median_ms(lambda: np.argpartition(-scores, k - 1)[:k]),
    }
    result["total_ms"] = result["matmul_ms"] + result["argpartition_ms"]
    result["qps"] = 1000.0 / max(result["total_ms"], 1e-9)

    # No `del` here: the lambdas above close over `matrix` and `scores`, and
    # deleting the names would leave those closures referencing unbound locals.
    # Returning drops the last reference anyway; the collect is for the 1.5 GB
    # step, so the next size does not allocate on top of it.
    gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=max(SIZES))
    args = parser.parse_args()

    sizes = [n for n in SIZES if n <= args.max]
    print(f"\nbrute-force search, {DIM} dimensions, float32, median of {REPEATS}\n")
    header = (
        f"{'docs':>10}  {'memory':>9}  {'matmul':>9}  {'argsort':>9}  "
        f"{'argpart':>9}  {'total':>9}  {'queries/s':>10}"
    )
    print(header)
    print("-" * len(header))
    for n in sizes:
        r = bench(n)
        print(
            f"{n:>10,}  {r['memory_mb']:>8.1f}M  {r['matmul_ms']:>8.2f}m  "
            f"{r['argsort_ms']:>8.2f}m  {r['argpartition_ms']:>8.2f}m  "
            f"{r['total_ms']:>8.2f}m  {r['qps']:>10,.0f}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
