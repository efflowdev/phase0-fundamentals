import pytest
from sse import Parser


def test_multiline_data_and_comments_are_combined() -> None:
    raw = b"data: line one\r\ndata: line two\r\n\r\n: ping\r\ndata: final line\r\n\r\n"

    parser = Parser()
    assert parser.feed(raw) == ["line one\nline two", "final line"]


def test_done_payload_is_returned() -> None:
    parser = Parser()
    assert parser.feed(b"data: [DONE]\n\n") == ["[DONE]"]


@pytest.mark.parametrize("separator", [b"\n", b"\r\n"])
@pytest.mark.parametrize("chunk_size", [1, 3, 7, 13, 64])
def test_chunking_is_invariant(separator: bytes, chunk_size: int) -> None:
    raw = b"data: a" + separator + b"data: b" + separator + separator

    parser = Parser()
    events: list[str] = []
    for start in range(0, len(raw), chunk_size):
        events.extend(parser.feed(raw[start : start + chunk_size]))

    assert events == ["a\nb"]


def test_truncated_cr_is_resolved_on_next_chunk() -> None:
    parser = Parser()

    assert parser.feed(b"data: a\r") == []
    assert parser.feed(b"\ndata: b\r\n\r\n") == ["a\nb"]


def test_eof_flushes_pending_cr() -> None:
    parser = Parser()

    assert parser.feed(b"data: a\rdata: b\r") == []
    assert parser.eof() == ["a\nb"]


def test_empty_data_value_is_kept() -> None:
    parser = Parser()

    assert parser.feed(b"data:\n\n") == [""]
