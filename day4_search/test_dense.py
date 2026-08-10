from __future__ import annotations

import math

import numpy as np
import pytest
from dense import DenseIndex, l2_normalize, top_k


def test_normalisation_turns_cosine_into_a_dot_product():
    """The claim the whole module rests on: once rows are unit length, the
    matrix-vector product *is* the vector of cosine similarities."""
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((6, 8)).astype(np.float32)
    query = rng.standard_normal(8).astype(np.float32)

    dots = l2_normalize(matrix) @ (query / np.linalg.norm(query))
    for i in range(matrix.shape[0]):
        cosine = float(
            matrix[i] @ query / (np.linalg.norm(matrix[i]) * np.linalg.norm(query))
        )
        assert math.isclose(dots[i], cosine, abs_tol=1e-6)


def test_normalisation_survives_a_zero_row():
    """A zero vector has no direction; the clamp keeps it from becoming NaN and
    silently poisoning every score in the corpus."""
    matrix = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    out = l2_normalize(matrix)
    assert not np.isnan(out).any()
    assert math.isclose(float(np.linalg.norm(out[1])), 1.0, abs_tol=1e-6)


def test_top_k_returns_the_best_k_in_order():
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3], dtype=np.float32)
    assert list(top_k(scores, 3)) == [1, 3, 2]


def test_top_k_handles_k_larger_than_the_corpus():
    scores = np.array([0.2, 0.8], dtype=np.float32)
    assert list(top_k(scores, 10)) == [1, 0]


def test_index_round_trips_through_disk(tmp_path):
    index = DenseIndex(
        matrix=l2_normalize(np.eye(3, 4, dtype=np.float32)),
        asins=["B001", "B002", "B003"],
        titles=["one", "two", "three"],
    )
    index.save(tmp_path)
    restored = DenseIndex.load(tmp_path)

    assert restored.asins == index.asins
    assert restored.titles == index.titles
    assert np.allclose(restored.matrix, index.matrix)


def test_a_mismatched_index_raises_instead_of_searching(tmp_path):
    """Row i of the matrix *is* row i of the table and nothing else ties them
    together. A silent mismatch returns confident, wrong products."""
    index = DenseIndex(
        matrix=np.eye(3, 4, dtype=np.float32),
        asins=["B001", "B002", "B003"],
        titles=["one", "two", "three"],
    )
    index.save(tmp_path)
    np.save(tmp_path / "dense.npy", np.eye(2, 4, dtype=np.float32))

    with pytest.raises(ValueError, match="corrupt"):
        DenseIndex.load(tmp_path)
