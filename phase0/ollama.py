"""Minimal local chat call, for the jobs that only need text back.

Day 3's `sample.complete()` stays where it is: it is a multi-provider sampler
built around logprobs and retry policy, and nothing outside day 3 wants that.
This is the other job — send a prompt to a local model, get a string — which
day 5 and Saturday's structured-output grid both need, so it lives here rather
than being written twice.

`format_schema` is Ollama's constrained-decoding hook: hand it a JSON Schema and
the sampler is restricted to tokens that keep the output parseable against it.
That is one of the three prompt styles Saturday compares, and the interesting
thing about it is that valid JSON is not the same as a valid *document* — a
schema-constrained model still fails cross-field rules like "footwear needs a
size", because those are not expressible as a grammar.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

# Read from the environment because day 6 moves the caller into a container,
# where `localhost` is the container itself and the daemon is a network hop away
# at host.docker.internal. Default unchanged, so nothing outside Docker notices.
BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
URL = f"{BASE_URL}/api/chat"
MODEL = "llama3.2"
TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0)


async def chat(
    client: httpx.AsyncClient,
    prompt: str,
    *,
    model: str = MODEL,
    system: str | None = None,
    format_schema: dict[str, Any] | None = None,
    temperature: float = 0.0,
    seed: int = 42,
    num_predict: int = 512,
) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
        },
    }
    if format_schema is not None:
        payload["format"] = format_schema

    response = await client.post(URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return (response.json().get("message") or {}).get("content") or ""
