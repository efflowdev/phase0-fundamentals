"""Dense against BM25, on two query types that behave nothing alike.

    uv run python day4_search/evaluate.py
    uv run python day4_search/evaluate.py --markdown

**Natural-language queries** come from ESCI with human judgments; a hit is a
product a human labelled Exact for that query. **Identifier queries** are the
500 ASINs, each looking for the one product it belongs to — a query set with
perfect ground truth, by construction, and no semantics whatsoever.

Three metrics, because one hides too much:

* `recall@k` — what fraction of the relevant products made the top k. Harsh when
  a query has 34 relevant products and k is 10, so read it alongside the others.
* `hit@k` — did *anything* relevant make the top k. This is what a shopper
  actually experiences.
* `MRR@10` — 1/rank of the first relevant result, averaged. Rewards putting the
  right answer first rather than ninth.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import corpus as corpus_mod
from dense import DenseIndex
from lexical import BM25Index


@dataclass(frozen=True)
class Scores:
    n_queries: int
    recall_at_1: float
    recall_at_10: float
    hit_at_1: float
    hit_at_10: float
    mrr_at_10: float
    recall_at_10_ceiling: float

    @property
    def recall_at_10_pct_of_ceiling(self) -> float:
        if self.recall_at_10_ceiling <= 0:
            return 0.0
        return self.recall_at_10 / self.recall_at_10_ceiling


def recall_ceiling(cases: list[tuple[str, set[str]]], k: int) -> float:
    """The best recall@k any retriever could possibly score on this query set.

    A query with 34 relevant products cannot exceed 10/34 at k=10, no matter how
    perfect the ranking. So recall@10 is bounded above by something that has
    nothing to do with the retriever, and quoting it as a percentage of 100% is
    misleading by construction — 0.68 against a ceiling of 0.80 is 85% of
    achievable, not 68% of perfect.

    Reporting the raw number without this column is the most common mistake in
    eval write-ups, and it is always in the flattering-to-nobody direction: it
    makes a good retriever look broken.
    """
    if not cases:
        return 0.0
    return sum(
        min(k, len(relevant)) / len(relevant) for _, relevant in cases if relevant
    ) / len(cases)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    for position, asin in enumerate(ranked[:k], start=1):
        if asin in relevant:
            return 1.0 / position
    return 0.0


def evaluate(search, cases: list[tuple[str, set[str]]], k: int = 10) -> Scores:
    """`search` is any callable (query, k) -> list[Hit]."""
    totals = [0.0] * 5
    for query, relevant in cases:
        ranked = [hit.asin for hit in search(query, k)]
        totals[0] += recall_at_k(ranked, relevant, 1)
        totals[1] += recall_at_k(ranked, relevant, 10)
        totals[2] += hit_at_k(ranked, relevant, 1)
        totals[3] += hit_at_k(ranked, relevant, 10)
        totals[4] += reciprocal_rank(ranked, relevant, 10)

    n = max(len(cases), 1)
    return Scores(len(cases), *[t / n for t in totals], recall_ceiling(cases, 10))


def natural_language_cases(corpus: corpus_mod.Corpus) -> list[tuple[str, set[str]]]:
    in_corpus = set(corpus.asins)
    cases = []
    for query in corpus.queries:
        relevant = query.relevant & in_corpus
        if relevant:
            cases.append((query.text, relevant))
    return cases


def identifier_cases(corpus: corpus_mod.Corpus) -> list[tuple[str, set[str]]]:
    """Every ASIN, looking for itself. Ground truth by construction."""
    return [(p.asin, {p.asin}) for p in corpus.products]


def table(rows: list[list[str]], headers: list[str], markdown: bool) -> str:
    widths = [
        max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))
    ]
    if markdown:
        head = (
            "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
        )
        rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
        body = [
            "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |"
            for r in rows
        ]
        return "\n".join([head, rule, *body])
    head = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * w for w in widths)
    body = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(r)) for r in rows]
    return "\n".join([head, rule, *body])


def row(name: str, query_set: str, s: Scores) -> list[str]:
    return [
        name,
        query_set,
        str(s.n_queries),
        f"{s.recall_at_1:.3f}",
        f"{s.recall_at_10:.3f}",
        f"{s.recall_at_10_ceiling:.3f}",
        f"{s.recall_at_10_pct_of_ceiling * 100:.0f}%",
        f"{s.hit_at_1:.3f}",
        f"{s.hit_at_10:.3f}",
        f"{s.mrr_at_10:.3f}",
    ]


def hub_report(index, cases: list[tuple[str, set[str]]], k: int = 10) -> list[tuple]:
    """How concentrated are the top-k slots across a whole query set?

    A retriever behaving sanely spreads its results: 500 queries x 10 slots over
    500 documents should touch most of the corpus. Hubness is the opposite —
    a handful of vectors sit near the centre of the embedding space and turn up
    as neighbours for queries they have nothing to do with, which is exactly what
    out-of-distribution inputs like a random product code produce.

    Returns (asin, title, appearances, times_at_rank_1), most frequent first.
    """
    from collections import Counter

    appearances: Counter[str] = Counter()
    at_rank_1: Counter[str] = Counter()
    titles: dict[str, str] = {}
    for query, _ in cases:
        for rank, hit in enumerate(index.search(query, k), start=1):
            appearances[hit.asin] += 1
            titles[hit.asin] = hit.title
            if rank == 1:
                at_rank_1[hit.asin] += 1
    return [
        (asin, titles[asin], count, at_rank_1[asin])
        for asin, count in appearances.most_common()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="re-embed the corpus")
    parser.add_argument(
        "--hubs",
        action="store_true",
        help="how concentrated dense results are across the identifier queries",
    )
    args = parser.parse_args()

    corpus = corpus_mod.build()
    dense = (
        DenseIndex.build(corpus) if args.rebuild else DenseIndex.open_or_build(corpus)
    )
    if args.rebuild:
        dense.save()
    bm25 = BM25Index.build(corpus)

    nl = natural_language_cases(corpus)
    ids = identifier_cases(corpus)

    print(f"\n## {len(corpus.products)} products · {len(nl)} judged queries\n")
    rows = [
        row("dense (bge-small)", "natural language", evaluate(dense.search, nl)),
        row("bm25", "natural language", evaluate(bm25.search, nl)),
        row("dense (bge-small)", "identifier (ASIN)", evaluate(dense.search, ids)),
        row("bm25", "identifier (ASIN)", evaluate(bm25.search, ids)),
    ]
    headers = [
        "retriever",
        "query set",
        "n",
        "recall@1",
        "recall@10",
        "r@10 max",
        "% of max",
        "hit@1",
        "hit@10",
        "MRR@10",
    ]
    print(table(rows, headers, args.markdown))
    print()

    if args.hubs:
        report = hub_report(dense, ids)
        slots = len(ids) * 10
        top10 = sum(count for _, _, count, _ in report[:10])
        print(f"### Hubness — dense, {len(ids)} identifier queries x 10 slots\n")
        print(f"documents that ever appear: {len(report)} of {len(corpus.products)}")
        print(
            f"top 10 documents hold {100 * top10 / slots:.1f}% of all slots "
            f"(uniform would be {100 * 10 / len(corpus.products):.1f}%)\n"
        )
        hub_rows = [
            [asin, str(count), str(rank1), title[:48]]
            for asin, title, count, rank1 in report[:6]
        ]
        print(
            table(
                hub_rows,
                ["asin", "in top 10", "at rank 1", "title"],
                args.markdown,
            )
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
