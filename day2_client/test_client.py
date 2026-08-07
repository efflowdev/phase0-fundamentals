import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing

import client as client_module
import httpx
import pytest
from events import Done, Event, TextDelta, ToolCall


def _sse(*payloads: str) -> list[bytes]:
    return [f"data: {payload}\n\n".encode() for payload in payloads]


def _text_chunk(content: str) -> str:
    return json.dumps(
        {
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": None}
            ]
        }
    )


def _finish_chunk(reason: str = "stop") -> str:
    return json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})


class _Stream(httpx.AsyncByteStream):
    """A response body we can watch. `closed` is the cancellation assertion."""

    def __init__(self, chunks: list[bytes], *, fail_after: int | None = None) -> None:
        self._chunks = chunks
        self._fail_after = fail_after
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for position, chunk in enumerate(self._chunks):
            if self._fail_after is not None and position == self._fail_after:
                raise httpx.ReadError("connection dropped")
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _collect(events: AsyncIterator[Event]) -> list[Event]:
    async def run() -> list[Event]:
        return [event async for event in events]

    return asyncio.run(run())


@pytest.fixture(autouse=True)
def _fast_and_authenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    # _backoff draws from uniform(0, BASE * 2**attempt); a zero base makes every
    # retry instant without touching asyncio.sleep itself.
    monkeypatch.setattr(client_module, "BASE", 0.0)


def _transport(
    responses: list[httpx.Response | BaseException],
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    remaining: Iterator[httpx.Response | BaseException] = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        response = next(remaining)
        if isinstance(response, BaseException):
            raise response
        return response

    return httpx.MockTransport(handler), seen


def _ok(stream: _Stream) -> httpx.Response:
    return httpx.Response(
        200, stream=stream, headers={"content-type": "text/event-stream"}
    )


def test_streams_text_events() -> None:
    stream = _Stream(_sse(_text_chunk("Hel"), _text_chunk("lo"), _finish_chunk()))
    transport, _ = _transport([_ok(stream)])

    assert _collect(client_module.stream_groq([], transport=transport)) == [
        TextDelta("Hel"),
        TextDelta("lo"),
        Done("stop"),
    ]


def test_tool_call_survives_chunks_that_split_an_sse_event() -> None:
    """The two parsers under load at once.

    The wire bytes are re-cut into 5-byte slices, so SSE events, JSON payloads
    and tool-argument fragments all straddle chunk boundaries.
    """
    payloads = [
        json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"city":'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"Riga"}'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _finish_chunk("tool_calls"),
        "[DONE]",
    ]
    raw = b"".join(_sse(*payloads))
    stream = _Stream([raw[i : i + 5] for i in range(0, len(raw), 5)])
    transport, _ = _transport([_ok(stream)])

    assert _collect(client_module.stream_groq([], transport=transport)) == [
        ToolCall("call_a", "get_weather", {"city": "Riga"}, '{"city":"Riga"}'),
        Done("tool_calls"),
    ]


def test_429_is_retried_then_succeeds() -> None:
    ok = _Stream(_sse(_text_chunk("hi"), _finish_chunk()))
    transport, seen = _transport(
        [
            httpx.Response(429, text="slow down", headers={"retry-after": "0"}),
            _ok(ok),
        ]
    )

    assert _collect(client_module.stream_groq([], transport=transport)) == [
        TextDelta("hi"),
        Done("stop"),
    ]
    assert len(seen) == 2


def test_retry_after_beyond_the_cap_fails_fast() -> None:
    transport, seen = _transport(
        [httpx.Response(429, text="nope", headers={"retry-after": "600"})]
    )

    with pytest.raises(httpx.HTTPStatusError):
        _collect(client_module.stream_groq([], transport=transport))

    assert len(seen) == 1, "a 10-minute retry-after should not be slept through"


def test_4xx_that_is_not_429_is_not_retried() -> None:
    transport, seen = _transport([httpx.Response(401, text="bad key")])

    with pytest.raises(httpx.HTTPStatusError):
        _collect(client_module.stream_groq([], transport=transport))

    assert len(seen) == 1


def test_attempts_are_bounded() -> None:
    transport, seen = _transport([httpx.Response(503, text="down")] * 5)

    with pytest.raises(httpx.HTTPStatusError):
        _collect(client_module.stream_groq([], transport=transport))

    assert len(seen) == client_module.MAX_ATTEMPTS


def test_drop_before_the_first_event_is_retried() -> None:
    dropped = _Stream(_sse(_text_chunk("a")), fail_after=0)
    ok = _Stream(_sse(_text_chunk("recovered"), _finish_chunk()))
    transport, seen = _transport([_ok(dropped), _ok(ok)])

    assert _collect(client_module.stream_groq([], transport=transport)) == [
        TextDelta("recovered"),
        Done("stop"),
    ]
    assert len(seen) == 2


def test_drop_after_the_first_event_is_not_retried() -> None:
    """Retrying here would replay the prefix the caller already rendered."""
    dropped = _Stream(_sse(_text_chunk("a"), _text_chunk("b")), fail_after=1)
    transport, seen = _transport([_ok(dropped)])

    with pytest.raises(httpx.ReadError):
        _collect(client_module.stream_groq([], transport=transport))

    assert len(seen) == 1


def _long_stream() -> tuple[_Stream, httpx.MockTransport]:
    stream = _Stream(_sse(*[_text_chunk(str(n)) for n in range(50)], _finish_chunk()))
    transport, _ = _transport([_ok(stream)])
    return stream, transport


def test_aclosing_closes_the_connection_at_the_break() -> None:
    """The documented way to cancel: deterministic, no tick to wait for."""
    stream, transport = _long_stream()
    observed: list[bool] = []

    async def run() -> None:
        seen = 0
        async with aclosing(
            client_module.stream_groq([], transport=transport)
        ) as events:
            async for _ in events:
                seen += 1
                if seen == 3:
                    break
        observed.append(stream.closed)

    asyncio.run(run())
    assert observed == [True]


def test_bare_break_leaks_the_connection() -> None:
    """Pinned deliberately, because it is the trap and it looks like good code.

    `async for` holds its iterator on the enclosing frame's value stack, so the
    generator is still referenced after the break and its finalizer never runs.
    The socket stays open until the loop shuts down.
    """
    stream, transport = _long_stream()
    observed: list[bool] = []

    async def run() -> None:
        seen = 0
        async for _ in client_module.stream_groq([], transport=transport):
            seen += 1
            if seen == 3:
                break
        observed.append(stream.closed)
        await asyncio.sleep(0)
        observed.append(stream.closed)

    asyncio.run(run())
    assert observed == [False, False]
    assert stream.closed, "...and only asyncio.run's shutdown_asyncgens cleans it up"


def test_cancelling_the_task_closes_the_connection_one_tick_later() -> None:
    """The case `break` cannot cover: the aborter is not the consumer.

    Cancelling tears down the consumer's frame, which drops the last reference
    to the generator — so this leaks too, just not for as long.
    """
    stream, transport = _long_stream()
    observed: list[bool] = []

    async def run() -> None:
        started = asyncio.Event()

        async def consume() -> None:
            async for _ in client_module.stream_groq([], transport=transport):
                started.set()
                await asyncio.sleep(1)

        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        observed.append(stream.closed)
        await asyncio.sleep(0)
        observed.append(stream.closed)

    asyncio.run(run())
    assert observed == [False, True]


def test_connect_failure_is_retried() -> None:
    ok = _Stream(_sse(_text_chunk("recovered"), _finish_chunk()))
    transport, seen = _transport([httpx.ConnectError("no route to host"), _ok(ok)])

    assert _collect(client_module.stream_groq([], transport=transport)) == [
        TextDelta("recovered"),
        Done("stop"),
    ]
    assert len(seen) == 2


def test_missing_api_key_fails_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    async def run() -> None:
        agen = client_module.stream_groq([])
        await anext(agen)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        asyncio.run(run())


@pytest.mark.parametrize(
    ("header", "expected"),
    [("0", 0.0), ("2.5", 2.5), ("-5", 0.0)],
)
def test_retry_delay_reads_delta_seconds(header: str, expected: float) -> None:
    assert client_module._retry_delay(header, attempt=0) == expected


def test_retry_delay_reads_an_http_date() -> None:
    delay = client_module._retry_delay("Wed, 21 Oct 2015 07:28:00 GMT", attempt=0)

    assert delay == 0.0, "a date in the past means retry immediately"


def test_retry_delay_falls_back_to_backoff_on_junk() -> None:
    delay = client_module._retry_delay("soon", attempt=3)

    assert delay is not None and 0.0 <= delay <= client_module.CAP
