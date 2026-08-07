"""Chunk payloads in, events out.

The second of two state machines. `sse.Parser` turns bytes into payload strings
without caring what they mean; this turns payload strings into events without
caring where the bytes came from. Neither touches the network, so both are
testable with no HTTP at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from events import Done, Event, MalformedToolCall, TextDelta, ToolCall


@dataclass
class _PartialToolCall:
    """One tool call under construction.

    `arguments` is a string, not a dict, because that is how it arrives: the
    model's JSON is streamed as text fragments that are individually meaningless.
    """

    id: str = ""
    name: str = ""
    arguments: str = ""


class ChunkAccumulator:
    """Assembles OpenAI-compatible stream chunks into events.

    Text passes straight through — a fragment of text is useful on its own.
    Tool calls do not: they are buffered until the provider reports a
    `finish_reason`, and only then parsed and emitted.
    """

    def __init__(self) -> None:
        self._calls: dict[int, _PartialToolCall] = {}
        self._done = False

    def feed(self, payload: str) -> list[Event]:
        if self._done:
            return []

        if payload == "[DONE]":
            # Some gateways send this sentinel and never a finish_reason.
            return self._flush("stop")

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            # A malformed chunk is not a malformed stream. Skipping one keeps
            # the tokens on either side of it; raising would throw away a
            # response the user already paid for.
            return []

        if not isinstance(chunk, dict):
            return []

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            # Usage-only chunks land here (stream_options.include_usage).
            return []

        choice = choices[0]
        if not isinstance(choice, dict):
            return []

        events: list[Event] = []

        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                events.append(TextDelta(content))
            self._absorb_tool_calls(delta.get("tool_calls"))

        reason = choice.get("finish_reason")
        if isinstance(reason, str) and reason:
            events.extend(self._flush(reason))

        return events

    def eof(self) -> list[Event]:
        """Flush anything still buffered when the byte stream ends.

        Only reachable after the stream ended *without raising*, which means the
        server closed the connection deliberately. A dropped connection surfaces
        as a TransportError further up and never gets here — so whatever is
        buffered at this point is complete, it just came from a provider that
        never sent a finish_reason.
        """
        return self._flush("stop")

    def _absorb_tool_calls(self, fragments: object) -> None:
        if not isinstance(fragments, list):
            return

        for fragment in fragments:
            if not isinstance(fragment, dict):
                continue

            # `index` is the only thing tying fragments together. The id and
            # name arrive once, in the opening fragment; everything after that
            # carries argument text and nothing else.
            index = fragment.get("index", 0)
            if not isinstance(index, int):
                continue

            partial = self._calls.setdefault(index, _PartialToolCall())

            call_id = fragment.get("id")
            if isinstance(call_id, str) and call_id:
                partial.id = call_id

            function = fragment.get("function")
            if not isinstance(function, dict):
                continue

            name = function.get("name")
            if isinstance(name, str) and name:
                partial.name = name

            arguments = function.get("arguments")
            if isinstance(arguments, str):
                partial.arguments += arguments

    def _flush(self, reason: str) -> list[Event]:
        if self._done:
            return []
        self._done = True

        # Flushed even when the reason is "stop" or "length" rather than
        # "tool_calls". A truncated call is worth surfacing; silently dropping
        # buffered work because the reason string was unexpected is not.
        events: list[Event] = [
            self._finish(self._calls[i]) for i in sorted(self._calls)
        ]
        self._calls.clear()
        events.append(Done(reason))
        return events

    @staticmethod
    def _finish(partial: _PartialToolCall) -> Event:
        raw = partial.arguments
        try:
            arguments = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            return MalformedToolCall(partial.id, partial.name, raw, str(exc))

        if not isinstance(arguments, dict):
            return MalformedToolCall(
                partial.id, partial.name, raw, "arguments were not a JSON object"
            )

        return ToolCall(partial.id, partial.name, arguments, raw)
