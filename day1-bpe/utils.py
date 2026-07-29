import re
from collections import Counter
from collections.abc import Iterator, Sequence
from itertools import pairwise

TOKEN_PATTERN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\w\s]+|\s+")
BASE_TOKEN_ID = 256


def tokenize(text: str) -> list[list[int]]:
    """Split text into UTF-8 byte tokens using the regex-based chunking."""
    words = TOKEN_PATTERN.findall(text)
    return [list(word.encode("utf-8")) for word in words]


def iter_pairs(token_list: Sequence[int]) -> Iterator[tuple[int, int]]:
    """Yield adjacent token pairs from a token list."""
    return pairwise(token_list)


def count_pairs(token_lists: Sequence[Sequence[int]]) -> Counter[tuple[int, int]]:
    """Count adjacent token pairs across a collection of token lists."""
    pair_counts: Counter[tuple[int, int]] = Counter()
    for token_list in token_lists:
        pair_counts.update(iter_pairs(token_list))
    return pair_counts


def build_merge_map(
    merges: Sequence[tuple[int, int]], base_token_id: int = BASE_TOKEN_ID
) -> dict[tuple[int, int], int]:
    """Create a mapping from merge pairs to token IDs."""
    return {pair: base_token_id + index for index, pair in enumerate(merges)}


def apply_merge_to_tokens(
    token_lists: Sequence[Sequence[int]], best_pair: tuple[int, int], new_token: int
) -> list[list[int]]:
    """Apply a single merge rule to every token list."""
    return [
        merge_pair(list(token_list), best_pair, new_token) for token_list in token_lists
    ]


def flatten_tokens(token_lists: Sequence[Sequence[int]]) -> list[int]:
    """Flatten a collection of token lists into a single list of token ids."""
    return [token for token_list in token_lists for token in token_list]


def merge_pair(
    token_list: Sequence[int], best_pair: tuple[int, int], new_token: int
) -> list[int]:
    """Merge a specific pair into a new token across a token list."""
    merged: list[int] = []
    index = 0

    while index < len(token_list):
        if (
            index < len(token_list) - 1
            and token_list[index] == best_pair[0]
            and token_list[index + 1] == best_pair[1]
        ):
            merged.append(new_token)
            index += 2
        else:
            merged.append(token_list[index])
            index += 1

    return merged
