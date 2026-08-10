"""Non-streaming completions with logprobs, for both providers.

Day 2's client streams, because streaming is what a product does. This one does
not, because an experiment wants the whole response, the token accounting and the
per-token logprobs as one object — and because a non-streaming request is
*retryable*. Day 2 had to refuse to retry after its first yielded event, since a
half-consumed stream cannot be replayed. Nothing has been handed to a caller here
until the response is complete, so every failure below is safe to retry.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from config import Provider

TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
MAX_ATTEMPTS = 5
BASE = 1.0
CAP = 20.0
TOP_LOGPROBS = 5


@dataclass(frozen=True)
class TokenLogprob:
    """One generated token, with the alternatives the model ranked against it.

    `top` is descending by logprob and may or may not begin with the token that
    was actually chosen — at temperature 0 it always does, above it frequently
    does not, and that gap is the measurement.
    """

    token: str
    logprob: float
    top: list[tuple[str, float]] = field(default_factory=list)

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)


@dataclass(frozen=True)
class Sample:
    output: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    logprobs: list[TokenLogprob] | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _backoff(attempt: int) -> float:
    """Full jitter, same as day 2: uniform over the window, not window ± noise."""
    return random.uniform(0, min(CAP, BASE * 2**attempt))


def _retry_delay(retry_after: str | None, attempt: int) -> float | None:
    """Seconds to wait, or None to give up. Handles both `retry-after` forms."""
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


def _groq_payload(
    provider: Provider,
    prompt_text: str,
    *,
    temperature: float,
    seed: int,
    max_tokens: int,
    want_logprobs: bool,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": provider.model,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    if want_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = TOP_LOGPROBS
    # OpenAI-compatible names live at the top level: top_p, frequency_penalty,
    # presence_penalty. There is no top_k and no repeat_penalty here — the two
    # providers do not expose the same sampler, which is itself the lesson.
    if sampler:
        payload.update(sampler)
    return payload


def _ollama_payload(
    provider: Provider,
    prompt_text: str,
    *,
    temperature: float,
    seed: int,
    max_tokens: int,
    want_logprobs: bool,
    sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": provider.model,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": max_tokens,
        },
    }
    if want_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = TOP_LOGPROBS
    # llama.cpp names, nested under `options`: top_k, top_p, min_p,
    # repeat_penalty, repeat_last_n. Unset ones keep llama.cpp's own defaults,
    # which are not neutral — `repeat_penalty` ships at 1.1, and it is applied
    # to the logits *after* the values reported in `top_logprobs` are read off.
    if sampler:
        payload["options"].update(sampler)
    return payload


def _parse_top(entries: Any) -> list[tuple[str, float]]:
    if not isinstance(entries, list):
        return []
    out: list[tuple[str, float]] = []
    for entry in entries:
        if isinstance(entry, dict) and "token" in entry and "logprob" in entry:
            out.append((entry["token"], float(entry["logprob"])))
    return out


def _parse_groq(
    body: dict[str, Any],
) -> tuple[str, str, int, int, list[TokenLogprob] | None]:
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}

    tokens: list[TokenLogprob] | None = None
    content = (choice.get("logprobs") or {}).get("content")
    if isinstance(content, list):
        tokens = [
            TokenLogprob(
                token=item.get("token", ""),
                logprob=float(item.get("logprob", 0.0)),
                top=_parse_top(item.get("top_logprobs")),
            )
            for item in content
        ]

    return (
        message.get("content") or "",
        choice.get("finish_reason") or "",
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        tokens,
    )


def _parse_ollama(
    body: dict[str, Any],
) -> tuple[str, str, int, int, list[TokenLogprob] | None]:
    message = body.get("message") or {}

    tokens: list[TokenLogprob] | None = None
    raw = body.get("logprobs")
    if isinstance(raw, list):
        tokens = [
            TokenLogprob(
                token=item.get("token", ""),
                logprob=float(item.get("logprob", 0.0)),
                top=_parse_top(item.get("top_logprobs")),
            )
            for item in raw
        ]

    return (
        message.get("content") or "",
        body.get("done_reason") or "",
        int(body.get("prompt_eval_count") or 0),
        int(body.get("eval_count") or 0),
        tokens,
    )


async def complete(
    client: httpx.AsyncClient,
    provider: Provider,
    prompt_text: str,
    *,
    temperature: float,
    seed: int,
    max_tokens: int,
    sampler: dict[str, Any] | None = None,
) -> Sample:
    """One completion. Errors come back as a `Sample`, they do not raise.

    A 480-cell run that dies on cell 300 because one request timed out has
    thrown away 300 paid-for results. Same reasoning as `return_exceptions=True`
    in `asyncio.gather`, applied one level lower so the failure is recorded as a
    row rather than swallowed as `None`.
    """
    headers = {"Content-Type": "application/json"}
    if provider.api_key_env:
        api_key = os.environ.get(provider.api_key_env)
        if not api_key:
            return Sample("", "", 0, 0, 0.0, None, f"{provider.api_key_env} is not set")
        headers["Authorization"] = f"Bearer {api_key}"

    build = _groq_payload if provider.name == "groq" else _ollama_payload
    parse = _parse_groq if provider.name == "groq" else _parse_ollama
    payload = build(
        provider,
        prompt_text,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        want_logprobs=provider.supports_logprobs,
        sampler=sampler,
    )

    last_error = "no attempts made"
    for attempt in range(MAX_ATTEMPTS):
        final_attempt = attempt == MAX_ATTEMPTS - 1
        # Measured per attempt, not across retries: a latency that silently
        # includes a 12-second backoff sleep is not a latency.
        started = time.perf_counter()
        try:
            response = await client.post(
                provider.url, headers=headers, json=payload, timeout=TIMEOUT
            )
            elapsed_ms = (time.perf_counter() - started) * 1000

            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                delay = _retry_delay(response.headers.get("retry-after"), attempt)
                if delay is None or final_attempt:
                    return Sample("", "", 0, 0, elapsed_ms, None, last_error)
                await asyncio.sleep(delay)
                continue

            if response.status_code >= 400:
                # 4xx that is not 429 is a bug in the request, not weather.
                return Sample(
                    "",
                    "",
                    0,
                    0,
                    elapsed_ms,
                    None,
                    f"HTTP {response.status_code}: {response.text[:200]}",
                )

            body = response.json()
            output, finish, p_tokens, c_tokens, logprobs = parse(body)
            return Sample(output, finish, p_tokens, c_tokens, elapsed_ms, logprobs)

        except (httpx.TransportError, json.JSONDecodeError) as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            last_error = f"{type(exc).__name__}: {exc}"
            if final_attempt:
                return Sample("", "", 0, 0, elapsed_ms, None, last_error)
            await asyncio.sleep(_backoff(attempt))

    return Sample("", "", 0, 0, 0.0, None, last_error)


async def probe_logprobs(client: httpx.AsyncClient, provider: Provider) -> str:
    """Ask for logprobs regardless of the config flag and report what came back.

    Worth one request per provider per run: `supports_logprobs` is a claim in a
    config file, and the whole confidence half of this day rests on it.
    """
    headers = {"Content-Type": "application/json"}
    if provider.api_key_env:
        api_key = os.environ.get(provider.api_key_env)
        if not api_key:
            return f"skipped ({provider.api_key_env} is not set)"
        headers["Authorization"] = f"Bearer {api_key}"

    build = _groq_payload if provider.name == "groq" else _ollama_payload
    parse = _parse_groq if provider.name == "groq" else _parse_ollama
    payload = build(
        provider,
        "Reply with exactly: ok",
        temperature=0.0,
        seed=42,
        max_tokens=5,
        want_logprobs=True,
    )

    try:
        response = await client.post(
            provider.url, headers=headers, json=payload, timeout=TIMEOUT
        )
    except httpx.TransportError as exc:
        return f"unreachable ({type(exc).__name__}: {exc})"

    if response.status_code >= 400:
        return f"refused — HTTP {response.status_code}: {response.text.strip()[:180]}"

    _, _, _, _, logprobs = parse(response.json())
    if not logprobs:
        return "accepted the parameter but returned no logprobs"
    return (
        f"supported — {len(logprobs)} tokens, {len(logprobs[0].top)} alternatives each"
    )
