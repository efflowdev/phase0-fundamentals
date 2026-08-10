from __future__ import annotations

import math

from sample import TokenLogprob
from score import (
    EMPTY,
    chosen_margin,
    is_off_top,
    score,
    tokens_from_json,
    tokens_to_json,
)


def tok(prob: float, top: list[tuple[str, float]] | None = None, text: str = "x"):
    """Build a token from probabilities rather than logprobs — easier to read."""
    return TokenLogprob(
        token=text,
        logprob=math.log(prob),
        top=[(t, math.log(p)) for t, p in (top or [])],
    )


def test_empty_input_scores_to_empty():
    assert score(None) == EMPTY
    assert score([]) == EMPTY


def test_mean_probability_is_arithmetic():
    result = score([tok(0.5), tok(0.5)])
    assert result.mean_prob == 0.5
    assert result.n_tokens == 2


def test_perplexity_and_mean_disagree_on_the_same_tokens():
    """Two responses, same perplexity, very different mean probability.

    This is the reason both are reported. Perplexity is driven by the geometric
    mean, so one near-zero token dominates it; the arithmetic mean lets a run of
    confident filler tokens hide that same token.
    """
    flat = score([tok(0.5), tok(0.5)])
    spiky = score([tok(0.25), tok(1.0)])

    assert math.isclose(flat.perplexity, 2.0, rel_tol=1e-9)
    assert math.isclose(spiky.perplexity, 2.0, rel_tol=1e-9)
    assert spiky.mean_prob > flat.mean_prob


def test_top_margin_uses_the_two_best_alternatives():
    result = score([tok(0.6, [("a", 0.6), ("b", 0.3), ("c", 0.1)], text="a")])
    assert math.isclose(result.mean_top_margin, 0.3, abs_tol=1e-9)


def test_tokens_without_a_runner_up_are_excluded_from_the_margin_mean():
    """A missing runner-up is unknown, not zero — averaging it in would report
    maximum certainty exactly where there is least information."""
    tokens = [
        tok(0.6, [("a", 0.6), ("b", 0.3)], text="a"),
        tok(0.9),  # no alternatives at all
    ]
    result = score(tokens)
    assert math.isclose(result.mean_top_margin, 0.3, abs_tol=1e-9)


def test_chosen_margin_is_negative_when_sampling_beats_the_argmax():
    sampled_runner_up = tok(0.3, [("a", 0.6), ("b", 0.3)], text="b")
    assert math.isclose(chosen_margin(sampled_runner_up), -0.3, abs_tol=1e-9)
    assert is_off_top(sampled_runner_up)

    greedy = tok(0.6, [("a", 0.6), ("b", 0.3)], text="a")
    assert math.isclose(chosen_margin(greedy), 0.0, abs_tol=1e-9)
    assert not is_off_top(greedy)


def test_off_top_count_is_the_direct_measure_of_temperature():
    tokens = [
        tok(0.6, [("a", 0.6), ("b", 0.3)], text="a"),
        tok(0.3, [("a", 0.6), ("b", 0.3)], text="b"),
        tok(0.3, [("a", 0.6), ("b", 0.3)], text="b"),
    ]
    result = score(tokens)
    assert result.n_off_top == 2
    assert result.mean_chosen_margin < 0


def test_weakest_token_is_located_not_just_valued():
    tokens = [tok(0.99, text="The"), tok(0.02, text="2019"), tok(0.95, text="/")]
    result = score(tokens)
    assert result.min_prob_token == "2019"
    assert result.min_prob_index == 1
    assert math.isclose(result.min_prob, 0.02, rel_tol=1e-9)


def test_logprobs_survive_a_round_trip_through_sqlite():
    tokens = [tok(0.6, [("a", 0.6), ("b", 0.3)], text="a"), tok(0.9, text="b")]
    restored = tokens_from_json(tokens_to_json(tokens))
    assert restored == tokens
    assert tokens_to_json(None) is None
    assert tokens_from_json(None) is None
