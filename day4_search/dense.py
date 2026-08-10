"""Semantic search with no vector database: a matrix and an argpartition.

The whole of "vector search" is here, and it is two lines. L2-normalise every
row once at build time, and cosine similarity collapses into a dot product —
so scoring the entire corpus against a query is a single matrix-vector multiply,
and ranking is a partial sort. Everything a vector database sells on top of that
is real (approximate nearest neighbours, metadata filtering, incremental
updates, sharding, replication) but it is *on top* of this, not instead of it.

Why cosine and not Euclidean: the embedding of a long product description has a
larger magnitude than the embedding of a three-word title, and Euclidean distance
would read that difference in length as a difference in meaning. Normalising
throws magnitude away and keeps direction, which is where the semantics live.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from corpus import Corpus

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
DEFAULT_DIR = Path(__file__).resolve().parent / "index"


@dataclass(frozen=True)
class Hit:
    asin: str
    score: float
    title: str


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=MODEL)


def embed_documents(texts: list[str]) -> np.ndarray:
    return np.array(list(_embedder().embed(texts)), dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Queries go through a different code path than documents.

    BGE is trained asymmetrically: passages are embedded bare, queries are
    embedded behind an instruction prefix ("Represent this sentence for
    searching relevant passages:"). `query_embed` applies it. Skipping this
    costs real recall, and it is invisible — the vectors still have the right
    shape and the results still look plausible.
    """
    return np.array(next(iter(_embedder().query_embed(text))), dtype=np.float32)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, ordered.

    `argpartition` is O(n) and only guarantees that the k best land in the first
    k slots, unordered; sorting just those k is O(k log k). A full `argsort` is
    O(n log n) over the whole corpus to get ten rows. At 500 documents this is
    noise — at a million it is the difference measured in bench.py.
    """
    k = min(k, scores.shape[0])
    partitioned = np.argpartition(-scores, k - 1)[:k]
    return partitioned[np.argsort(-scores[partitioned])]


@dataclass
class DenseIndex:
    matrix: np.ndarray  # (n, DIM), rows already L2-normalised
    asins: list[str]
    titles: list[str]

    @classmethod
    def build(cls, corpus: Corpus) -> DenseIndex:
        matrix = l2_normalize(embed_documents(corpus.texts))
        return cls(
            matrix=matrix,
            asins=corpus.asins,
            titles=[p.title for p in corpus.products],
        )

    def search(self, query: str, k: int = 10) -> list[Hit]:
        vector = embed_query(query)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        scores = self.matrix @ vector  # cosine, because both sides are unit norm
        return [
            Hit(self.asins[i], float(scores[i]), self.titles[i])
            for i in top_k(scores, k)
        ]

    # ---- persistence: this is the part a vector database mostly is ----

    def save(self, directory: Path = DEFAULT_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "dense.npy", self.matrix)

        conn = sqlite3.connect(directory / "catalog.db")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                row_idx INTEGER PRIMARY KEY,
                asin    TEXT NOT NULL UNIQUE,
                title   TEXT NOT NULL
            );
            """
        )
        conn.execute("DELETE FROM documents")
        conn.executemany(
            "INSERT INTO documents (row_idx, asin, title) VALUES (?,?,?)",
            list(zip(range(len(self.asins)), self.asins, self.titles, strict=True)),
        )
        conn.commit()
        conn.close()

    @classmethod
    def load(cls, directory: Path = DEFAULT_DIR) -> DenseIndex:
        matrix = np.load(directory / "dense.npy")
        conn = sqlite3.connect(directory / "catalog.db")
        rows = conn.execute(
            "SELECT asin, title FROM documents ORDER BY row_idx"
        ).fetchall()
        conn.close()
        if len(rows) != matrix.shape[0]:
            # Row i of the matrix *is* row i of the table. Nothing else ties
            # them together, so a mismatch means silently wrong search results.
            raise ValueError(
                f"index is corrupt: {matrix.shape[0]} vectors, {len(rows)} documents"
            )
        return cls(
            matrix=matrix,
            asins=[r[0] for r in rows],
            titles=[r[1] for r in rows],
        )

    @classmethod
    def open_or_build(cls, corpus: Corpus, directory: Path = DEFAULT_DIR) -> DenseIndex:
        try:
            return cls.load(directory)
        except (FileNotFoundError, sqlite3.OperationalError, ValueError):
            index = cls.build(corpus)
            index.save(directory)
            return index
