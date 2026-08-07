import json

from accumulate import ChunkAccumulator
from events import Done, MalformedToolCall, TextDelta, ToolCall


def _chunk(delta: dict, finish_reason: str | None = None) -> str:
    return json.dumps(
        {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    )


def _text(content: str) -> str:
    return _chunk({"content": content})


def _open_call(index: int, call_id: str, name: str) -> str:
    return _chunk(
        {
            "tool_calls": [
                {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }
            ]
        }
    )


def _args(index: int, fragment: str) -> str:
    return _chunk(
        {"tool_calls": [{"index": index, "function": {"arguments": fragment}}]}
    )


def _finish(reason: str) -> str:
    return _chunk({}, reason)


def _drain(payloads: list[str]) -> list:
    accumulator = ChunkAccumulator()
    events = []
    for payload in payloads:
        events.extend(accumulator.feed(payload))
    events.extend(accumulator.eof())
    return events


def test_text_passes_straight_through() -> None:
    assert _drain([_text("Hel"), _text("lo"), _finish("stop")]) == [
        TextDelta("Hel"),
        TextDelta("lo"),
        Done("stop"),
    ]


def test_arguments_are_reassembled_from_fragments() -> None:
    events = _drain(
        [
            _open_call(0, "call_a", "get_weather"),
            _args(0, '{"ci'),
            _args(0, 'ty": "Par'),
            _args(0, 'is"}'),
            _finish("tool_calls"),
        ]
    )

    assert events == [
        ToolCall("call_a", "get_weather", {"city": "Paris"}, '{"city": "Paris"}'),
        Done("tool_calls"),
    ]


def test_nothing_is_emitted_before_finish_reason() -> None:
    """The contract: a syntactically complete buffer is still not a finished call.

    Every fragment below leaves a buffer that is valid JSON on its own, and the
    accumulator still emits nothing. Only the provider gets to say the call is
    over.
    """
    accumulator = ChunkAccumulator()

    assert accumulator.feed(_open_call(0, "call_a", "ping")) == []
    assert accumulator.feed(_args(0, "{}")) == []
    assert accumulator.feed(_text("thinking")) == [TextDelta("thinking")]

    assert accumulator.feed(_finish("tool_calls")) == [
        ToolCall("call_a", "ping", {}, "{}"),
        Done("tool_calls"),
    ]


def test_parallel_calls_flush_together_in_index_order() -> None:
    events = _drain(
        [
            _open_call(0, "call_a", "get_weather"),
            _open_call(1, "call_b", "get_time"),
            _args(1, '{"tz": "UTC"}'),
            _args(0, '{"city": "Riga"}'),
            _finish("tool_calls"),
        ]
    )

    assert events == [
        ToolCall("call_a", "get_weather", {"city": "Riga"}, '{"city": "Riga"}'),
        ToolCall("call_b", "get_time", {"tz": "UTC"}, '{"tz": "UTC"}'),
        Done("tool_calls"),
    ]


def test_unparseable_arguments_become_a_malformed_event() -> None:
    events = _drain(
        [
            _open_call(0, "call_a", "get_weather"),
            _args(0, '{"city": "Par'),
            _finish("tool_calls"),
        ]
    )

    assert isinstance(events[0], MalformedToolCall)
    assert events[0].name == "get_weather"
    assert events[0].raw == '{"city": "Par'
    assert events[1] == Done("tool_calls")


def test_non_object_arguments_are_malformed() -> None:
    events = _drain(
        [_open_call(0, "call_a", "ping"), _args(0, "42"), _finish("tool_calls")]
    )

    assert isinstance(events[0], MalformedToolCall)
    assert events[0].error == "arguments were not a JSON object"


def test_truncated_stream_still_surfaces_the_partial_call() -> None:
    """finish_reason "length" means the model was cut off mid-arguments.

    Dropping the buffer would leave the caller with no idea why nothing happened.
    """
    events = _drain([_open_call(0, "call_a", "ping"), _args(0, "{"), _finish("length")])

    assert isinstance(events[0], MalformedToolCall)
    assert events[1] == Done("length")


def test_done_sentinel_flushes_when_no_finish_reason_arrives() -> None:
    events = _drain([_open_call(0, "call_a", "ping"), _args(0, "{}"), "[DONE]"])

    assert events == [ToolCall("call_a", "ping", {}, "{}"), Done("stop")]


def test_eof_flushes_when_neither_arrives() -> None:
    assert _drain([_text("hi")]) == [TextDelta("hi"), Done("stop")]


def test_malformed_chunk_is_skipped_and_neighbours_survive() -> None:
    assert _drain([_text("a"), "{not json", _text("b"), _finish("stop")]) == [
        TextDelta("a"),
        TextDelta("b"),
        Done("stop"),
    ]


def test_usage_only_chunk_is_ignored() -> None:
    usage = json.dumps({"choices": [], "usage": {"total_tokens": 12}})

    assert _drain([_text("a"), usage, _finish("stop")]) == [
        TextDelta("a"),
        Done("stop"),
    ]


def test_events_after_done_are_ignored() -> None:
    accumulator = ChunkAccumulator()

    assert accumulator.feed(_finish("stop")) == [Done("stop")]
    assert accumulator.feed(_text("late")) == []
    assert accumulator.eof() == []


def test_empty_content_deltas_do_not_produce_events() -> None:
    assert _drain([_text(""), _finish("stop")]) == [Done("stop")]
