from __future__ import annotations

import math

import numpy as np
from variance import (
    common_prefix_chars,
    mean_pairwise_cosine,
    modal_share,
    normalize,
    summarize,
)


def test_normalize_folds_the_differences_that_are_not_disagreements():
    assert normalize("  GBP.  ") == "gbp"
    assert normalize("a\n\n  b") == "a b"
    assert normalize('"2011/83/EU"') == "2011/83/eu"


def test_normalize_folds_typography_that_nfkc_leaves_alone():
    assert normalize("It’s fine") == normalize("It's fine")
    assert normalize("14–day") == normalize("14-day")
    assert normalize("ﬁnal") == "final"  # this one NFKC does handle


def test_normalize_does_not_reorder_or_drop_words():
    """Conservative on purpose: two strings that normalise equal are the same
    answer, not merely a similar one."""
    assert normalize("red blue") != normalize("blue red")
    assert normalize("not in stock") != normalize("in stock")


def test_modal_share_measures_agreement_with_the_mode_not_with_truth():
    assert modal_share(["a", "a", "a"]) == 1.0
    assert modal_share(["a", "a", "b", "c"]) == 0.5
    assert modal_share([]) == 0.0


def test_common_prefix_stops_at_the_first_divergence():
    assert common_prefix_chars(["Canberra", "Canberra"]) == 8
    assert common_prefix_chars(["Canberra", "Canbera"]) == 6
    assert common_prefix_chars(["abc", "xyz"]) == 0
    assert common_prefix_chars(["abc", "abcdef"]) == 3  # prefix, not equality
    assert common_prefix_chars([]) == 0


def test_cosine_of_orthogonal_and_identical_vectors():
    identical = np.array([[1.0, 0.0], [1.0, 0.0]])
    orthogonal = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert math.isclose(mean_pairwise_cosine(identical), 1.0, abs_tol=1e-6)
    assert math.isclose(mean_pairwise_cosine(orthogonal), 0.0, abs_tol=1e-6)


def test_cosine_ignores_magnitude_because_rows_are_normalised_first():
    scaled = np.array([[3.0, 0.0], [0.5, 0.0]])
    assert math.isclose(mean_pairwise_cosine(scaled), 1.0, abs_tol=1e-6)


def test_cosine_averages_the_upper_triangle_only():
    """Three vectors, two identical: the diagonal would drag the mean to 1."""
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # pairs: (0,1)=1, (0,2)=0, (1,2)=0 → 1/3
    assert math.isclose(mean_pairwise_cosine(matrix), 1 / 3, abs_tol=1e-6)


def test_a_single_sample_has_no_pairs():
    assert mean_pairwise_cosine(np.array([[1.0, 0.0]])) == 1.0


def test_summarize_separates_answer_changes_from_formatting_changes():
    outputs = ["GBP", "GBP.", " gbp ", "USD"]
    result = summarize(outputs, embeddings=np.zeros((1, 384), dtype=np.float32))

    assert result.n == 4
    assert result.distinct == 4  # every string differs
    assert result.distinct_normalized == 2  # but there are only two answers
    assert result.modal_share == 0.25
    assert result.modal_share_normalized == 0.75


def test_summarize_on_an_empty_group():
    result = summarize([])
    assert result.n == 0
    assert result.mean_pairwise_cosine == 1.0
