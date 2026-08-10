"""Okapi BM25, hand-rolled, as the control that dense retrieval is measured against.

BM25 is a bag of words with two corrections, and both of them are the reason it
still beats neural retrieval on identifier lookups forty years later:

* **IDF** — a term appearing in one document out of 500 carries far more signal
  than one appearing in 400. An ASIN appears in exactly one, so its IDF is close
  to the maximum the formula can produce.
* **Saturation** — `tf * (k1 + 1) / (tf + k1 * ...)` flattens as term frequency
  grows, so a document that says "adidas" nine times does not outrank one that
  says it twice and is actually about the shoe.

An embedding has neither property. It compresses a document into 384 floats
trained to preserve *meaning*, and a product code has no meaning to preserve —
it is an arbitrary string whose entire value is that it matches exactly.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from corpus import Corpus
from dense import Hit

K1 = 1.5
B = 0.75

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs.

    Deliberately crude — no stemming, no stopword list. `B07XJ8C8F5` survives as
    one token, which is the entire point of the comparison; a stemmer would not
    touch it either, but every extra transform is one more thing to explain when
    the numbers come out.
    """
    return _TOKEN.findall(text.lower())


@dataclass
class BM25Index:
    asins: list[str]
    titles: list[str]
    postings: dict[str, dict[int, int]] = field(default_factory=dict)
    doc_len: list[int] = field(default_factory=list)
    avgdl: float = 0.0

    @classmethod
    def build(cls, corpus: Corpus) -> BM25Index:
        index = cls(
            asins=corpus.asins,
            titles=[p.title for p in corpus.products],
        )
        for doc_idx, text in enumerate(corpus.texts):
            tokens = tokenize(text)
            index.doc_len.append(len(tokens))
            for term, count in Counter(tokens).items():
                index.postings.setdefault(term, {})[doc_idx] = count
        index.avgdl = sum(index.doc_len) / max(len(index.doc_len), 1)
        return index

    @property
    def n_docs(self) -> int:
        return len(self.asins)

    def idf(self, term: str) -> float:
        """Lucene's variant: always positive, unlike the textbook formula.

        The classic `log((N - df + 0.5) / (df + 0.5))` goes negative once a term
        appears in more than half the corpus, which lets a common word actively
        subtract from a document's score. The `1 +` floors it at zero instead.
        """
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[Hit]:
        scores: dict[int, float] = {}
        for term in tokenize(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf(term)
            for doc_idx, tf in posting.items():
                norm = 1 - B + B * self.doc_len[doc_idx] / max(self.avgdl, 1e-9)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * (
                    tf * (K1 + 1) / (tf + K1 * norm)
                )

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            Hit(self.asins[doc_idx], score, self.titles[doc_idx])
            for doc_idx, score in ranked
        ]
