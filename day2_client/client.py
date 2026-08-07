from __future__ import annotations

import asyncio
import json
import os
import random
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from accumulate import ChunkAccumulator
from events import Done, Event, TextDelta, ToolCall
from sse import Parser

TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
MAX_ATTEMPTS = 5
BASE = 1.0
CAP = 20.0

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"


def _backoff(attempt: int) -> float:
    """Full jitter: uniform over the whole window, not window ± noise.

    Equal-jitter and decorrelated variants exist; full jitter is the one that
    actually de-synchronises a thundering herd, because two clients that failed
    on the same tick pick independent delays over the same range.
    """
    return random.uniform(0, min(CAP, BASE * 2**attempt))


def _retry_delay(retry_after: str | None, attempt: int) -> float | None:
    """Seconds to wait, or None to stop retrying.

    `retry-after` is either delta-seconds or an HTTP-date (RFC 9110), and real
    gateways send both. A value past CAP means the server is telling us it will
    not serve us any time soon — sleeping that long inside a request is worse
    than failing fast, so give up and let the caller decide.
    """
    if retry_after is None:
        return _backoff(attempt)

    try:
        seconds = float(retry_after)
    except ValueError:
        try:
            seconds = (
                parsedate_to_datetime(retry_after) - datetime.now(UTC)
            ).total_seconds()
        except (TypeError, ValueError):
            return _backoff(attempt)

    if seconds > CAP:
        return None
    return max(0.0, seconds)


async def stream_groq(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[Event]:
    """Stream an OpenAI-compatible completion from Groq.

    Cancellation takes no parameter and no token — but it does take `aclosing`::

        async with aclosing(stream_groq(messages)) as events:
            async for event in events:
                ...
                break

    Only `aclose()` unwinds the `async with` blocks below, and only the caller
    can call it. Measured, on this client:

    ==========================  =================  ===============
    how the caller stops        closed at once     after one tick
    ==========================  =================  ===============
    ``break``                   no                 **no**
    ``break`` + ``aclosing``    yes                --
    ``Task.cancel()``           no                 yes
    ``Task.cancel()`` + above   yes                yes
    ==========================  =================  ===============

    Bare `break` holds the socket open until the loop shuts down, because
    `async for` keeps its iterator on the enclosing frame's value stack for as
    long as that frame is alive, so refcounting never drops the generator and
    the asyncgen finalizer never fires. `Task.cancel()` tears the frame down,
    which releases it a tick later.

    So the JS comparison is not the flattering one. Python needs no
    AbortController *object* — a Task is cancellable where a Promise is not —
    but forgetting `aclosing` leaks a connection while looking like textbook
    code, whereas in JS a missing abort is at least visible as a missing
    argument.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    payload: dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_ATTEMPTS):
        final_attempt = attempt == MAX_ATTEMPTS - 1
        parser = Parser()
        accumulator = ChunkAccumulator()
        yielded = False

        try:
            async with (
                httpx.AsyncClient(timeout=TIMEOUT, transport=transport) as client,
                client.stream(
                    "POST",
                    GROQ_URL,
                    headers=headers,
                    json=payload,
                ) as response,
            ):
                if response.status_code == 429 or response.status_code >= 500:
                    # The body has to be read before raise_for_status can quote
                    # it, and a streaming response is not read by default.
                    await response.aread()
                    delay = _retry_delay(response.headers.get("retry-after"), attempt)
                    if delay is None or final_attempt:
                        response.raise_for_status()
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()

                async for chunk in response.aiter_bytes():
                    for item in parser.feed(chunk):
                        for event in accumulator.feed(item):
                            yielded = True
                            yield event

                for item in parser.eof():
                    for event in accumulator.feed(item):
                        yielded = True
                        yield event

                for event in accumulator.eof():
                    yielded = True
                    yield event

            return

        except httpx.TransportError:
            # Connect resets and read timeouts are the whole reason retries exist.
            # But once an event has left this generator the caller has already
            # acted on it, and re-running the request would duplicate the prefix.
            # A stream is not idempotent after its first byte, so we stop here.
            if yielded or final_attempt:
                raise
            await asyncio.sleep(_backoff(attempt))


async def stream_ollama(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> AsyncIterator[Event]:
    """Stream a local Ollama completion — same events, different wire format.

    Two differences worth the trip. Framing is newline-delimited JSON, so there
    is no hand-rolled parser here: `aiter_lines` is enough, because a single
    `\\n` needs none of the multi-line-event machinery SSE does. And tool call
    arguments arrive as a decoded object rather than as fragmented JSON text —
    the partial-arguments problem is a property of the protocol, not of tool
    calling.
    """
    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools

    async with (
        httpx.AsyncClient(timeout=TIMEOUT) as client,
        client.stream("POST", OLLAMA_URL, json=payload) as response,
    ):
        response.raise_for_status()

        async for line in response.aiter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            message = data.get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content:
                yield TextDelta(content)

            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments")
                yield ToolCall(
                    id=call.get("id", ""),
                    name=function.get("name", ""),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    raw=json.dumps(arguments, ensure_ascii=False),
                )

            if data.get("done"):
                yield Done(data.get("done_reason") or "stop")
                return
