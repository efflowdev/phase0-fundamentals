from collections.abc import Sequence

from utils import (
    BASE_TOKEN_ID,
    apply_merge_to_tokens,
    build_merge_map,
    count_pairs,
    flatten_tokens,
    tokenize,
)


def build_decoded_vocab(merges: Sequence[tuple[int, int]]) -> dict[int, bytes]:
    """Build the byte vocabulary for decoding token IDs."""
    merge_map = build_merge_map(merges)
    decoded = {token_id: bytes([token_id]) for token_id in range(BASE_TOKEN_ID)}

    for (left, right), new_id in merge_map.items():
        decoded[new_id] = decoded[left] + decoded[right]

    return decoded


def train(corpus: str, vocab_size: int) -> list[tuple[int, int]]:
    """Learn a sequence of byte-pair merges from a corpus."""
    if vocab_size <= BASE_TOKEN_ID:
        return []

    tokens = tokenize(corpus)
    merges: list[tuple[int, int]] = []
    next_token = BASE_TOKEN_ID
    max_merges = vocab_size - BASE_TOKEN_ID

    while len(merges) < max_merges:
        pair_counts = count_pairs(tokens)
        if not pair_counts:
            break

        best_pair, count = pair_counts.most_common(1)[0]
        if count < 2:
            break

        merges.append(best_pair)
        tokens = apply_merge_to_tokens(tokens, best_pair, next_token)
        next_token += 1

    return merges


def encode(text: str, merges: Sequence[tuple[int, int]]) -> list[int]:
    """Encode text into token IDs using the learned merge rules."""
    tokens = tokenize(text)
    merge_map = build_merge_map(merges)

    while True:
        pair_counts = count_pairs(tokens)
        available_pairs = [pair for pair in pair_counts if pair in merge_map]
        if not available_pairs:
            break

        pair = min(available_pairs, key=lambda current: merge_map[current])
        token_id = merge_map[pair]
        tokens = apply_merge_to_tokens(tokens, pair, token_id)

    return flatten_tokens(tokens)


def decode(ids: Sequence[int], merges: Sequence[tuple[int, int]]) -> str:
    """Decode token IDs back into text using the learned merge rules."""
    decoded = build_decoded_vocab(merges)
    return b"".join(decoded[token_id] for token_id in ids).decode("utf-8")
