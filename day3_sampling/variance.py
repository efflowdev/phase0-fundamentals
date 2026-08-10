"""How far apart are n samples of the same prompt?

Four measures, each blind to something the next one catches:

* **distinct count** — how many unique strings. Saturates instantly: on an
  open-ended prompt it is 20/20 whether the answers agree in substance or not.
* **modal share** — the fraction of samples equal to the most common one. This is
  the "exact-match rate" the spec asks for, made explicit: matched *against what*
  has to be answered, and the mode is the only choice that needs no ground truth.
  Reported raw and after normalisation, because the gap between the two is the
  difference between "the model changed its answer" and "the model changed its
  punctuation".
* **common prefix** — how many characters all n samples share before the first
  one diverges. Cheap, and it says *when* sampling starts to matter rather than
  whether it did.
* **mean pairwise cosine** — the only one that survives on open-ended text, where
  every sample is unique and none of the string measures can tell agreement from
  paraphrase.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_PUNCT_EDGES = re.compile(r"^[\s\"'`*_(\[]+|[\s\"'`*_)\].,!?:;]+$")
_WHITESPACE = re.compile(r"\s+")

# NFKC does not touch these — it folds ligatures and full-width forms, not
# typography. But a model that emits a curly apostrophe on run 7 and a straight
# one on run 8 has not changed its answer, so they are folded by hand.
_TYPOGRAPHIC = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "–": "-",
        "—": "-",
        "−": "-",
    }
)


@dataclass(frozen=True)
class Variance:
    n: int
    distinct: int
    distinct_normalized: int
    modal_share: float
    modal_share_normalized: float
    common_prefix_chars: int
    mean_pairwise_cosine: float
    mean_chars: float


def normalize(text: str) -> str:
    """Strip the differences that are not disagreements.

    NFKC first (ligatures, full-width forms), then typography, then whitespace,
    edge punctuation and case. Deliberately conservative: nothing here reorders
    or drops words, so two strings that normalise equal really are the same
    answer — which is what lets the gap between `distinct` and
    `distinct_normalized` be read as "same answer, different formatting".
    """
    text = unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHIC)
    text = _WHITESPACE.sub(" ", text).strip()
    text = _PUNCT_EDGES.sub("", text)
    return text.casefold()


def modal_share(outputs: list[str]) -> float:
    if not outputs:
        return 0.0
    return Counter(outputs).most_common(1)[0][1] / len(outputs)


def common_prefix_chars(outputs: list[str]) -> int:
    if not outputs:
        return 0
    shortest = min(outputs, key=len)
    for i, char in enumerate(shortest):
        if any(other[i] != char for other in outputs):
            return i
    return len(shortest)


def mean_pairwise_cosine(matrix: np.ndarray) -> float:
    """Mean cosine over the upper triangle, diagonal excluded.

    Rows are L2-normalised first, which turns cosine similarity into a plain dot
    product — so the whole n×n similarity matrix is one matmul. That collapse is
    also the entire trick behind vector search, met here a day early.
    """
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return 1.0
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.clip(norms, 1e-12, None)
    similarity = normalized @ normalized.T
    upper = np.triu_indices(similarity.shape[0], k=1)
    return float(similarity[upper].mean())


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """(n, 384) float matrix. First call downloads ~130 MB of ONNX weights."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return np.array(list(_embedder().embed(texts)), dtype=np.float32)


def summarize(outputs: list[str], embeddings: np.ndarray | None = None) -> Variance:
    """Embeddings are injected so the string measures stay testable offline."""
    if not outputs:
        return Variance(0, 0, 0, 0.0, 0.0, 0, 1.0, 0.0)

    normalized = [normalize(o) for o in outputs]
    if embeddings is None:
        embeddings = embed(outputs)

    return Variance(
        n=len(outputs),
        distinct=len(set(outputs)),
        distinct_normalized=len(set(normalized)),
        modal_share=modal_share(outputs),
        modal_share_normalized=modal_share(normalized),
        common_prefix_chars=common_prefix_chars(outputs),
        mean_pairwise_cosine=mean_pairwise_cosine(embeddings),
        mean_chars=sum(len(o) for o in outputs) / len(outputs),
    )
