"""What a stream yields.

The clients used to yield `str`. That works right up until the model calls a
tool, because a tool call is not text and there is nowhere in a string to put
it. So the stream yields tagged events instead and the caller matches on type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A fragment of assistant text, exactly as it came off the wire."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A complete tool call whose arguments parsed cleanly.

    `raw` is kept alongside the parsed dict because when a call misbehaves in
    production, the argument string the model actually produced is the evidence,
    and re-serialising the dict does not reproduce it.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    raw: str


@dataclass(frozen=True, slots=True)
class MalformedToolCall:
    """A tool call whose arguments did not parse.

    Deliberately not an exception. A small model emitting invalid JSON is a
    normal Tuesday, and the correct response is not to tear down the stream —
    it is to hand the error back to the model and let it retry. An exception
    would take that choice away from the agent loop in P2.
    """

    id: str
    name: str
    raw: str
    error: str


@dataclass(frozen=True, slots=True)
class Done:
    """Terminal event. `reason` is the provider's own finish_reason."""

    reason: str


Event = TextDelta | ToolCall | MalformedToolCall | Done
