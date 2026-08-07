"""Runnable proof that the client does what the tests say it does.

    uv run python day2_client/demo.py            # all of it
    uv run python day2_client/demo.py tools      # one section

Sections: text, tools, cancel, ollama.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import aclosing
from pathlib import Path

from client import stream_groq, stream_ollama
from events import Done, Event, MalformedToolCall, TextDelta, ToolCall

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    },
}


def load_env(path: Path) -> None:
    """Six lines instead of python-dotenv, because this week has no frameworks."""
    import os

    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def show(event: Event) -> None:
    match event:
        case TextDelta(text):
            print(text, end="", flush=True)
        case ToolCall(id=call_id, name=name, arguments=arguments):
            print(f"\n  → {name}({arguments})  [{call_id}]")
        case MalformedToolCall(name=name, raw=raw, error=error):
            print(f"\n  ✗ {name} arguments did not parse: {error}\n    raw: {raw!r}")
        case Done(reason):
            print(f"\n  ── finished: {reason}")


async def demo_text() -> None:
    print("\n== text ==")
    messages = [{"role": "user", "content": "Name three Baltic capitals. One line."}]
    async with aclosing(stream_groq(messages)) as events:
        async for event in events:
            show(event)


async def demo_tools() -> None:
    print("\n== tool call ==")
    messages = [{"role": "user", "content": "What is the weather in Riga and in Oslo?"}]
    async with aclosing(stream_groq(messages, tools=[WEATHER_TOOL])) as events:
        async for event in events:
            show(event)


async def demo_cancel() -> None:
    """Stop after 20 text deltas and let `aclosing` shut the socket at the break."""
    print("\n== cancellation ==")
    messages = [{"role": "user", "content": "Count slowly from 1 to 200."}]
    seen = 0
    async with aclosing(stream_groq(messages)) as events:
        async for event in events:
            show(event)
            if isinstance(event, TextDelta):
                seen += 1
                if seen == 20:
                    print(f"\n  ── cancelled after {seen} deltas")
                    break


async def demo_ollama() -> None:
    print("\n== ollama (newline-delimited JSON, not SSE) ==")
    messages = [{"role": "user", "content": "What is the weather in Riga?"}]
    try:
        async with aclosing(stream_ollama(messages, tools=[WEATHER_TOOL])) as events:
            async for event in events:
                show(event)
    except Exception as exc:  # noqa: BLE001 - it is a demo; the reason is the point
        print(f"  skipped: {type(exc).__name__}: {exc}")


SECTIONS = {
    "text": demo_text,
    "tools": demo_tools,
    "cancel": demo_cancel,
    "ollama": demo_ollama,
}


async def main() -> None:
    load_env(Path(__file__).resolve().parents[2] / ".env")
    wanted = sys.argv[1:] or list(SECTIONS)
    for name in wanted:
        if name not in SECTIONS:
            print(f"unknown section {name!r}; pick from {', '.join(SECTIONS)}")
            continue
        await SECTIONS[name]()


if __name__ == "__main__":
    asyncio.run(main())
