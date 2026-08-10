"""Look at the results with your own eyes, side by side.

    uv run python day4_search/search.py "running shoes for women"
    uv run python day4_search/search.py --break-it        # the identifier failure
    uv run python day4_search/search.py --break-it -n 5

The `--break-it` mode is the screenshot: pick products at random, query their
exact SKU, and watch dense retrieval return confident, plausible, wrong
neighbours while BM25 returns the row you asked for. This is the motivating
failure for hybrid retrieval in P1, and meeting it here is why that project will
make sense.
"""

from __future__ import annotations

import argparse

import corpus as corpus_mod
from dense import DenseIndex, Hit
from lexical import BM25Index


def show(label: str, hits: list[Hit], target: str | None = None) -> None:
    print(f"\n  {label}")
    if not hits:
        print("    (nothing)")
        return
    for rank, hit in enumerate(hits, start=1):
        marker = " ←" if target and hit.asin == target else ""
        title = hit.title if len(hit.title) <= 62 else hit.title[:61] + "…"
        print(f"    {rank:>2}. {hit.score:6.3f}  {hit.asin}  {title}{marker}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="free-text query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("-n", type=int, default=3, help="products to probe")
    parser.add_argument("--break-it", action="store_true")
    args = parser.parse_args()

    corpus = corpus_mod.build()
    dense = DenseIndex.open_or_build(corpus)
    bm25 = BM25Index.build(corpus)

    if args.break_it:
        # Evenly spaced through the catalogue rather than random, so the demo
        # reproduces and the write-up can quote it.
        step = max(len(corpus.products) // (args.n + 1), 1)
        for i in range(1, args.n + 1):
            product = corpus.products[min(i * step, len(corpus.products) - 1)]
            print(f"\n{'=' * 78}")
            print(f"query: {product.asin}   (wanted: {product.title[:52]})")
            show("dense", dense.search(product.asin, args.k), product.asin)
            show("bm25 ", bm25.search(product.asin, args.k), product.asin)
        print()
        return 0

    if not args.query:
        parser.error("give a query, or pass --break-it")

    query = " ".join(args.query)
    print(f"\nquery: {query!r}")
    show("dense", dense.search(query, args.k))
    show("bm25 ", bm25.search(query, args.k))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
