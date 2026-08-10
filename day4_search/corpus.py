"""500 real products out of Amazon ESCI, without downloading Amazon ESCI.

The full dataset is 1.8 GB of parquet for 2 million judged query-product pairs.
We need 500 products. HuggingFace's datasets-server exposes a paging JSON API
over the same parquet, 100 rows per request, so a few megabytes of HTTP replaces
the download entirely. Rows are cached to JSONL on first run — the corpus has to
be identical between the build and the evaluation or every number below is
meaningless.

Why this dataset and not 500 invented titles: ESCI ships **human relevance
judgments**. Each query is labelled against its candidate products as Exact,
Substitute, Complement or Irrelevant, which is what turns "does this search feel
right" into recall@k. It is also the dataset P1 benchmarks on, so this loader is
P1's loader, written five weeks early.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DATASET = "tasksource/esci"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
CACHE = Path(__file__).resolve().parent / "cache" / "esci_rows.jsonl"

PAGE = 100  # the API's maximum
DATASET_ROWS = 2_027_874
BLOCKS = 16  # sampling points spread evenly across the dataset
PAGES_PER_BLOCK = 4  # 400 contiguous rows at each point
TARGET_PRODUCTS = 500
MIN_JUDGED = 3  # a query judged against 1 product cannot discriminate anything
LOCALE = "us"


@dataclass(frozen=True)
class Product:
    asin: str
    title: str
    brand: str
    color: str

    @property
    def text(self) -> str:
        """What actually gets indexed.

        The SKU is in the indexed text on purpose. Real catalogue search fields
        carry it, and putting it here makes the identifier failure honest: the
        token *is* present and retrievable, so when dense search misses it that
        is a property of the embedding, not of the data.
        """
        parts = [self.title]
        meta = [f"SKU: {self.asin}"]
        if self.brand:
            meta.append(f"Brand: {self.brand}")
        if self.color:
            meta.append(f"Colour: {self.color}")
        parts.append(" | ".join(meta))
        return "\n".join(parts)


@dataclass(frozen=True)
class Query:
    query_id: int
    text: str
    exact: tuple[str, ...]
    substitute: tuple[str, ...]

    @property
    def relevant(self) -> set[str]:
        """Exact only. Substitutes are deliberately not counted as hits.

        ESCI's own guidance treats Substitute as "reasonable but not what was
        asked for", and folding it into the relevant set inflates every score in
        the table by rewarding near-misses.
        """
        return set(self.exact)


@dataclass(frozen=True)
class Corpus:
    products: list[Product]
    queries: list[Query]

    @property
    def asins(self) -> list[str]:
        return [p.asin for p in self.products]

    @property
    def texts(self) -> list[str]:
        return [p.text for p in self.products]

    def index_of(self) -> dict[str, int]:
        return {p.asin: i for i, p in enumerate(self.products)}


# Only what `build` reads. ESCI rows also carry product_description,
# product_bullet_point and product_text, which are long, unused, and take the
# cache from ~1 MB to 14 MB — too big to commit, and the cache has to be
# committed or the corpus is not reproducible when the API moves.
CACHED_FIELDS = (
    "query_id",
    "query",
    "product_id",
    "product_locale",
    "product_title",
    "product_brand",
    "product_color",
    "esci_label",
)


def _project(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in CACHED_FIELDS}


def _get_page(client: httpx.Client, offset: int) -> list[dict[str, Any]]:
    params = {
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": PAGE,
    }
    for attempt in range(4):
        response = client.get(ROWS_URL, params=params)
        if response.status_code < 400:
            return [item["row"] for item in response.json().get("rows", [])]
        # The public endpoint throttles; a flat retry is enough, because this
        # runs once and the cache serves every run after it.
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"datasets-server refused offset {offset}")


def fetch_rows(*, refresh: bool = False) -> list[dict[str, Any]]:
    """Sample contiguous blocks spread across the whole dataset, then cache.

    Not the first N rows. ESCI is ordered by query, so any contiguous prefix is a
    handful of adjacent topics — the first 4000 rows are bathroom fans and
    children's books. A 500-product catalogue of near-duplicates would make every
    retriever look bad for reasons that have nothing to do with retrieval, and it
    fails the "corpus you can judge by eye" test the exercise depends on.

    Blocks rather than single pages because query groups have to survive intact:
    the first and last group in each block is dropped, since a block boundary
    almost certainly cut it in half, and a half-judged query silently caps its
    own recall.
    """
    if CACHE.exists() and not refresh:
        return [json.loads(line) for line in CACHE.read_text().splitlines() if line]

    block_rows = PAGE * PAGES_PER_BLOCK
    stride = DATASET_ROWS // BLOCKS
    rows: list[dict[str, Any]] = []

    with httpx.Client(timeout=60.0) as client:
        for block in range(BLOCKS):
            start = block * stride
            block_buffer: list[dict[str, Any]] = []
            for page_start in range(start, start + block_rows, PAGE):
                page = _get_page(client, page_start)
                if not page:
                    break
                block_buffer.extend(page)

            if not block_buffer:
                continue
            first_qid = block_buffer[0]["query_id"]
            last_qid = block_buffer[-1]["query_id"]
            rows.extend(
                r
                for r in block_buffer
                if r["query_id"] != first_qid and r["query_id"] != last_qid
            )

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_project(row), ensure_ascii=False) + "\n")
    return rows


def build(
    rows: list[dict[str, Any]] | None = None,
    *,
    target_products: int = TARGET_PRODUCTS,
) -> Corpus:
    """Group rows into queries and take whole query groups until the pool is full.

    Whole groups, never partial ones. If a query's judged products were split by
    the 500-product cap, its recall would be capped below 1.0 by the loader
    rather than by the retriever, and the evaluation would be measuring this
    function.
    """
    rows = rows if rows is not None else fetch_rows()

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("product_locale") != LOCALE:
            continue
        if not (row.get("product_title") or "").strip():
            continue
        grouped[row["query_id"]].append(row)

    products: dict[str, Product] = {}
    queries: list[Query] = []

    # Fetch order, not query_id order. ESCI numbers queries alphabetically, so
    # sorting by query_id throws away the spread sampling above and refills the
    # corpus from the alphabetical head — every query starting with a digit.
    for query_id, group in grouped.items():
        if len(group) < MIN_JUDGED:
            continue

        exact = tuple(r["product_id"] for r in group if r["esci_label"] == "Exact")
        if not exact:
            continue

        candidates = {
            r["product_id"]: Product(
                asin=r["product_id"],
                title=(r["product_title"] or "").strip(),
                brand=(r["product_brand"] or "").strip(),
                color=(r["product_color"] or "").strip(),
            )
            for r in group
        }
        new = [a for a in candidates if a not in products]
        if len(products) + len(new) > target_products:
            continue  # skip this group, try the next — do not truncate it

        products.update(candidates)
        queries.append(
            Query(
                query_id=query_id,
                text=group[0]["query"].strip(),
                exact=exact,
                substitute=tuple(
                    r["product_id"] for r in group if r["esci_label"] == "Substitute"
                ),
            )
        )
        if len(products) >= target_products:
            break

    ordered = sorted(products.values(), key=lambda p: p.asin)
    return Corpus(products=ordered, queries=queries)


if __name__ == "__main__":
    corpus = build()
    judged = sum(len(q.exact) for q in corpus.queries)
    print(f"{len(corpus.products)} products, {len(corpus.queries)} queries")
    print(f"{judged} Exact judgments, {judged / len(corpus.queries):.1f} per query")
    print(f"\ncached at {CACHE}")
    print("\nsample document as indexed:\n")
    print(corpus.products[0].text)
