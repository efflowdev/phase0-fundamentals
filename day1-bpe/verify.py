import argparse
from pathlib import Path

import tiktoken
from bpe import build_decoded_vocab, decode, encode, train

enc = tiktoken.get_encoding("cl100k_base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the BPE implementation against tiktoken for multiple corpora."
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=["corpus.txt", "corpus_ru.txt", "corpus_ro.txt"],
        help="Corpus text files to compare",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=1000, help="Number of merges to learn"
    )
    return parser.parse_args()


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def explain_token_fragments(
    token_ids: list[int], merges: list[tuple[int, int]]
) -> list[str]:
    decoded = build_decoded_vocab(merges)
    return [decoded[token_id].decode("utf-8") for token_id in token_ids]


def verify_corpus(path: str, vocab_size: int) -> None:
    content = load_text(path)
    merges = train(content, vocab_size)

    tiktoken_ids = enc.encode(content)
    my_ids = encode(content, merges)

    print(f"\n=== {Path(path).name} ===")
    print("Ticktoken tokens:", len(tiktoken_ids))
    print("My tokens:", len(my_ids))
    print(
        "Tiktoken decode match"
        if enc.decode(tiktoken_ids) == content
        else "Tiktoken decode mismatch"
    )
    print(
        "My decode match" if decode(my_ids, merges) == content else "My decode mismatch"
    )


def run_general_checks(merges: list[tuple[int, int]]) -> None:
    difficult_text = (
        "Héllo wörld! 日本語のテスト émoji: 🚀🔥👍 — mixed ASCII, ünïcödé, "
        "Кириллица, and\ttabs\nand newlines."
    )
    difficult_ids = encode(difficult_text, merges)
    print("\n")
    print(
        "Difficult text roundtrip pass"
        if decode(difficult_ids, merges) == difficult_text
        else "Difficult text roundtrip fails"
    )

    strawberry_ids = encode("strawberry", merges)
    print("Token_ids:", strawberry_ids)
    print("Token count:", len(strawberry_ids))
    print("Decoded token pieces:", explain_token_fragments(strawberry_ids, merges))


def run_sliced_token_list(path: str, vocab_size: int) -> None:
    content = load_text(path)
    merges = train(content, vocab_size)

    tiktoken_ids = enc.encode(content)
    my_ids = encode(content, merges)

    trimmed_tiktoken_ids = tiktoken_ids[:20]
    trimmed_my_ids = my_ids[:20]

    sliced_tiktoken_content = enc.decode(trimmed_tiktoken_ids)
    sliced_my_content = decode(trimmed_my_ids, merges)

    print(f"\n=== {Path(path).name} ===")
    print('Sliced content Tiktoken:\n', sliced_tiktoken_content)
    print('My sliced content: \n', sliced_my_content)


def main() -> None:
    args = parse_args()
    first_content = load_text(args.files[0]) if args.files else ""
    first_merges = train(first_content, args.vocab_size) if first_content else []

    for path in args.files:
        verify_corpus(path, args.vocab_size)
        run_sliced_token_list(path, args.vocab_size)

    run_general_checks(first_merges)


if __name__ == "__main__":
    main()
