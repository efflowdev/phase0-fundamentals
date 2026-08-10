# Day 4 — Embeddings and Semantic Search

Semantic search over 500 real products, with no vector database: embed the catalogue once with
`bge-small-en-v1.5`, L2-normalise the rows so cosine similarity collapses into a dot product, and
rank with one matrix-vector multiply and an `argpartition`. The corpus is a slice of Amazon ESCI —
real titles, real ASINs, and human relevance judgments — pulled through HuggingFace's datasets-server
API rather than by downloading the 1.8 GB of parquet that would otherwise be needed for 500 rows. The
labels are the reason for choosing it: they turn "does this feel right" into recall@k, and BM25,
hand-rolled in sixty lines, is the control that dense retrieval gets measured against.

**Dense wins natural language, and it is not close on the queries that need it.** For *"something to
plug two headphones into one phone"* — a query sharing almost no vocabulary with any product title —
dense returns the Syncwire headphone splitter at 0.689 and a StarTech splitter second. BM25 returns a
rubber grommet and a zinc supplement, matching on "plug" and "into", and only reaches the splitter at
rank 3.

| retriever | query set | n | recall@1 | recall@10 | r@10 max | % of max | hit@1 | hit@10 | MRR@10 |
|---|---|---|---|---|---|---|---|---|---|
| dense (bge-small) | natural language | 66 | 0.203 | 0.682 | 0.930 | 73% | 0.652 | 0.970 | 0.767 |
| bm25 | natural language | 66 | 0.184 | 0.601 | 0.930 | 65% | 0.576 | 0.879 | 0.696 |
| dense (bge-small) | identifier (ASIN) | 500 | 0.006 | 0.096 | 1.000 | 10% | 0.006 | 0.096 | 0.027 |
| bm25 | identifier (ASIN) | 500 | **1.000** | 1.000 | 1.000 | 100% | 1.000 | 1.000 | 1.000 |

**`recall@1` of 0.203 is not a failure, and reporting it without its ceiling would be dishonest.**
Queries here have a median of 4 relevant products and a maximum of 34, and only one document fits in
one slot — so recall@1 is bounded above by a number that has nothing to do with the retriever. That
bound is 0.310 on this query set, which makes dense's 0.203 worth 65% of what is achievable. Note it
is *not* one over the mean relevant-set size: the ceiling is the average of each query's own 1/R, and
averaging reciprocals is not the reciprocal of the average — 1/6.86 would say 0.146 and understate
the bound by half. At k=10 the ceiling is 0.930, so 0.682 is 73% of achievable rather than 68% of
perfect. The `% of max` column exists because the raw number invites exactly the wrong conclusion.

**Dense retrieval cannot find a product code.** Querying each of the 500 ASINs for its own product,
dense returned the right one 3 times out of 500. BM25 returned it 500 times out of 500, always at
rank 1. The SKU is inside the indexed text for both retrievers, so this is not a missing-data
artefact — an embedding is a lossy compression trained to preserve meaning, and `B01N1G8OX8` has no
meaning to preserve, while BM25's IDF term makes a token appearing in exactly one document out of 500
dominate every other signal in the corpus.

**Dense also loses where you would expect it to win.** *"womens white trainers"* returns three pairs
of adidas training **trousers**, because the model reads British "trainers" as a cousin of
"training". BM25 finds actual sneakers at ranks 2 and 3. Neither retriever is strictly better than
the other, which is the argument for running both.

**Brute force is fine much longer than people assume.** Timing the matmul plus ranking over random
vectors:

| docs | memory | matmul | argsort | argpartition | total | queries/s |
|---|---|---|---|---|---|---|
| 1,000 | 1.5 MB | 0.01 ms | 0.01 ms | 0.00 ms | 0.02 ms | 65,219 |
| 10,000 | 15 MB | 0.24 ms | 0.51 ms | 0.06 ms | 0.30 ms | 3,317 |
| 100,000 | 154 MB | 3.81 ms | 7.04 ms | 0.53 ms | 4.34 ms | 231 |
| 1,000,000 | 1,536 MB | 39.95 ms | 86.36 ms | 7.62 ms | 47.57 ms | 21 |

A hundred thousand documents is 4 ms and 154 MB of RAM. The ranking choice matters more than it
looks: at a million documents `argsort` costs 86 ms to return ten rows, against 7.6 ms for
`argpartition` — O(n log n) over the whole corpus versus O(n) quickselect — which would have more
than doubled total latency for one lazy line.

The surprise: **dense retrieval's failure on identifiers is not evenly distributed, it collapses onto
one document.** Across 500 identifier queries and 5,000 top-10 slots, only 243 of the 500 products
ever appear at all, the top ten hold 42.8% of every slot where a uniform spread would be 2.0%, and a
single product — a UGREEN audio cable with a Japanese-language title — took rank 1 for 342 of the 500
queries. The second-place hub is also Japanese. This is hubness: in high-dimensional space a few
vectors sit where out-of-distribution inputs land, and a random alphanumeric string given to an
English-trained model is about as out-of-distribution as an input gets. The practical consequence is
that a dense-only catalogue search does not degrade gracefully on unfamiliar queries — it returns the
same confident wrong answer to all of them.

## Reproducing

```bash
uv run python day4_search/corpus.py                        # build/inspect the 500-product corpus
uv run python day4_search/evaluate.py --markdown --hubs    # the tables above
uv run python day4_search/search.py --break-it             # the identifier failure, by eye
uv run python day4_search/bench.py                         # the scale curve
```
