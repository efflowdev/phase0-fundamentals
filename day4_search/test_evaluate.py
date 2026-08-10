from __future__ import annotations

import math

from dense import Hit
from evaluate import (
    evaluate,
    hit_at_k,
    recall_at_k,
    recall_ceiling,
    reciprocal_rank,
)


def test_recall_is_a_fraction_of_the_relevant_set():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"a", "z"}, 10) == 0.5
    assert recall_at_k(ranked, {"a", "b"}, 10) == 1.0
    assert recall_at_k(ranked, {"a", "b"}, 1) == 0.5


def test_recall_at_10_is_capped_by_k_when_relevant_sets_are_large():
    """A query with 34 relevant products cannot exceed 10/34 at k=10. This is
    why recall alone would make the natural-language rows look like failures."""
    ranked = [f"r{i}" for i in range(10)]
    relevant = {f"r{i}" for i in range(34)}
    assert math.isclose(recall_at_k(ranked, relevant, 10), 10 / 34)


def test_hit_asks_a_different_question_than_recall():
    ranked = ["a", "b", "c"]
    relevant = {"c", "x", "y", "z"}
    assert hit_at_k(ranked, relevant, 10) == 1.0  # the shopper found something
    assert recall_at_k(ranked, relevant, 10) == 0.25  # most of it was missed


def test_reciprocal_rank_rewards_position():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_reciprocal_rank_ignores_results_past_k():
    ranked = [f"x{i}" for i in range(12)] + ["target"]
    assert reciprocal_rank(ranked, {"target"}, k=10) == 0.0


def test_empty_relevant_set_scores_zero_rather_than_dividing_by_zero():
    assert recall_at_k(["a"], set(), 10) == 0.0


def test_evaluate_averages_over_queries():
    def fake_search(query: str, k: int) -> list[Hit]:
        table = {
            "first": ["a", "b"],
            "second": ["z", "y"],
        }
        return [Hit(asin, 1.0, asin) for asin in table[query][:k]]

    scores = evaluate(fake_search, [("first", {"a"}), ("second", {"q"})])
    assert scores.n_queries == 2
    assert scores.hit_at_1 == 0.5  # one query found it, one did not
    assert scores.mrr_at_10 == 0.5


def test_recall_ceiling_is_bounded_by_the_size_of_the_relevant_set():
    """A query with 34 relevant products cannot exceed 10/34 at k=10, however
    good the retriever is."""
    cases = [("a", {f"r{i}" for i in range(34)}), ("b", {"x", "y"})]
    ceiling = recall_ceiling(cases, 10)
    assert math.isclose(ceiling, (10 / 34 + 1.0) / 2)


def test_ceiling_is_one_when_every_query_has_few_relevant_docs():
    cases = [("a", {"x"}), ("b", {"y", "z"})]
    assert recall_ceiling(cases, 10) == 1.0


def test_percent_of_ceiling_rescales_a_misleading_raw_number():
    scores = evaluate(
        lambda q, k: [Hit("r0", 1.0, "r0")],
        [("a", {f"r{i}" for i in range(10)})],
    )
    assert math.isclose(scores.recall_at_10, 0.1)
    assert math.isclose(scores.recall_at_10_ceiling, 1.0)
    assert math.isclose(scores.recall_at_10_pct_of_ceiling, 0.1)
