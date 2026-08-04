from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx
from sse import Parser

TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)


async def stream_groq(messages: list[dict]) -> AsyncIterator[str]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    parser = Parser()
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def _content_from(payload: str) -> str | None:
        if item == "[DONE]":
            return
        try:
            data = json.loads(item)
        except json.JSONDecodeError:
            return
        delta = data.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if isinstance(content, str) and content:
            yield content

        return content

    async with (
        httpx.AsyncClient(timeout=TIMEOUT) as client,
        client.stream(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as response,
    ):
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            for item in parser.feed(chunk):
                content = _content_from(item)
                if content:
                    yield content

        for item in parser.eof():
            if item == "[DONE]":
                continue
            try:
                data = json.loads(item)
            except json.JSONDecodeError:
                continue
            delta = data.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if isinstance(content, str) and content:
                yield content


async def stream_ollama(messages: list[dict]) -> AsyncIterator[str]:
    payload = {
        "model": "llama3.2",
        "messages": messages,
        "stream": True,
    }

    async with (
        httpx.AsyncClient(timeout=TIMEOUT) as client,
        client.stream(
            "POST",
            "http://localhost:11434/api/chat",
            json=payload,
        ) as response,
    ):
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = data.get("message", {}).get("content")
            if isinstance(content, str) and content:
                yield content
            if data.get("done"):
                break
