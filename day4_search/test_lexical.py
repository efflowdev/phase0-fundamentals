from __future__ import annotations

from corpus import Corpus, Product
from lexical import BM25Index, tokenize


def tiny() -> Corpus:
    return Corpus(
        products=[
            Product("B00AAA1111", "Adidas Running Shoes for Women", "Adidas", "Black"),
            Product("B00BBB2222", "Adidas Adidas Adidas Slides", "Adidas", "White"),
            Product("B00CCC3333", "Nike Running Shoes for Men", "Nike", "Blue"),
        ],
        queries=[],
    )


def test_tokenizer_keeps_identifiers_whole():
    assert tokenize("SKU: B07XJ8C8F5") == ["sku", "b07xj8c8f5"]
    assert tokenize("2-inch mat!") == ["2", "inch", "mat"]


def test_a_rare_identifier_ranks_its_own_document_first():
    """The whole reason BM25 survives: one document contains the token, so its
    IDF is near maximal and nothing else scores at all."""
    index = BM25Index.build(tiny())
    hits = index.search("B00CCC3333", k=3)
    assert hits[0].asin == "B00CCC3333"
    assert len(hits) == 1  # no other document contains the term


def test_unknown_terms_score_nothing_rather_than_erroring():
    index = BM25Index.build(tiny())
    assert index.search("kombucha", k=3) == []


def test_idf_is_never_negative():
    """Lucene's `1 +` variant. The textbook formula goes negative for terms in
    more than half the corpus, letting a common word subtract from a score."""
    index = BM25Index.build(tiny())
    assert index.idf("adidas") >= 0.0  # in 2 of 3 documents
    assert index.idf("nike") > index.idf("adidas")
    assert index.idf("absent") == 0.0


def test_term_frequency_saturates():
    """Doc B says "adidas" three times, doc A once. B should rank higher, but
    nothing like three times higher — that is what k1 buys."""
    index = BM25Index.build(tiny())
    scores = {h.asin: h.score for h in index.search("adidas", k=3)}
    assert scores["B00BBB2222"] > scores["B00AAA1111"]
    assert scores["B00BBB2222"] < 3 * scores["B00AAA1111"]


def test_longer_documents_are_penalised():
    """Same single match, different document lengths: b=0.75 favours the shorter
    document, because a match in 5 words says more than a match in 50."""
    corpus = Corpus(
        products=[
            Product("B001", "kombucha", "", ""),
            Product("B002", "kombucha " + "filler " * 40, "", ""),
        ],
        queries=[],
    )
    index = BM25Index.build(corpus)
    hits = index.search("kombucha", k=2)
    assert hits[0].asin == "B001"
