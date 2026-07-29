# Day 1 — Byte Pair Encoding (BPE)

This day focused on building a small byte-pair encoding pipeline from scratch and comparing it to tiktoken. The implementation trains a set of merge rules from a corpus, encodes text into token IDs, and decodes those IDs back to the original string.

The main measurement was token-count behavior across different corpora. The script compares the number of tokens produced by the custom BPE implementation with those from tiktoken and also checks round-trip decoding for both the training corpus and a few multilingual examples. The results showed that the custom implementation is functional, but its token counts can diverge noticeably depending on the corpus and merge budget.

The biggest surprise was how much the vocabulary learned from the data shapes the final tokenization. Even with the same basic algorithm, different text distributions led to different merge patterns and noticeably different token counts. That made the tradeoff between compression and simplicity feel much more concrete than it does in a purely theoretical explanation.
