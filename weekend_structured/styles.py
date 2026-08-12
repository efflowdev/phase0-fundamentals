"""One call, for any (target, style) — and the turn that follows a failure.

Two providers with incompatible request shapes and three styles that each mean
something slightly different on each, so the whole point of this module is that
`loop.py` never learns which provider it is talking to.

The part worth reading twice is `repair_turns`. A repair is not "ask again with
an error message appended" — the transcript has to be the one the provider's own
format expects, or the model sees a conversation that never happened:

* prompt / native → `assistant(content)` then `user(errors)`
* tool            → `assistant(tool_calls)` then `tool(errors)`, because a
  function call is answered by a tool result, not by a user complaint. Sending
  a user turn there is valid JSON and a lie about what occurred, and OpenAI's
  API rejects it outright — an assistant message carrying `tool_calls` must be
  followed by a matching `tool` message.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import NATIVE, PROMPT, TOOL, Target

TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
TOOL_NAME = "record_product_attributes"
MAX_HTTP_ATTEMPTS = 5


@dataclass
class Reply:
    """One assistant response, normalised across providers.

    `text` is the candidate JSON — message content for prompt/native, the tool
    call's `arguments` for tool style. `message` is the provider-shaped assistant
    turn, kept verbatim so a repair can replay it.
    """

    text: str
    message: dict[str, Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    http_retries: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class NoToolCall(Exception):
    """Forced tool choice, and the model answered with prose anyway.

    Groq's `gpt-oss-20b` raises this as a 400 server-side; Ollama returns 200
    with an empty `tool_calls`. Same failure, so it is normalised to one
    exception and classified as a violation rather than a transport error.
    """


def user_prompt(product: dict[str, str], schema: dict[str, Any], style: str) -> str:
    """The schema is inlined only for `prompt` style.

    For tool and native styles the schema travels in its own field, and pasting
    it into the text as well would confound the comparison: any difference
    between the columns could then be the enforcement mechanism *or* the extra
    600 tokens of context. Same product text everywhere, schema in exactly one
    place.
    """
    head = f"Product title: {product['title']}\nASIN: {product['asin']}"
    if style != PROMPT:
        return f"{head}\n\nExtract the product attributes."
    body = json.dumps(schema, separators=(",", ":"))
    return f"{head}\n\nJSON Schema:\n{body}\n\nReturn the JSON object."


def build_payload(
    target: Target,
    style: str,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    *,
    seed: int,
    temperature: float,
    num_predict: int,
) -> dict[str, Any]:
    if target.provider == "ollama":
        payload: dict[str, Any] = {
            "model": target.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": num_predict,
            },
        }
        if style == NATIVE:
            payload["format"] = schema
        elif style == TOOL:
            payload["tools"] = [_tool_def(schema)]
        return payload

    payload = {
        "model": target.model,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": num_predict,
    }
    if style == NATIVE:
        # Not `json_schema`: this model answers that with a 400. See config.py.
        payload["response_format"] = {"type": "json_object"}
    elif style == TOOL:
        payload["tools"] = [_tool_def(schema)]
        payload["tool_choice"] = {"type": "function", "function": {"name": TOOL_NAME}}
    return payload


def _tool_def(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Record the attributes extracted from a product title.",
            "parameters": schema,
        },
    }


def parse_reply(target: Target, style: str, body: dict[str, Any]) -> Reply:
    if target.provider == "ollama":
        message = body.get("message") or {}
        usage_in = int(body.get("prompt_eval_count") or 0)
        usage_out = int(body.get("eval_count") or 0)
        finish = str(body.get("done_reason") or "")
    else:
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}
        usage_in = int(usage.get("prompt_tokens") or 0)
        usage_out = int(usage.get("completion_tokens") or 0)
        finish = str(choice.get("finish_reason") or "")

    if style == TOOL:
        calls = message.get("tool_calls") or []
        if not calls:
            raise NoToolCall(f"{target.id} returned no tool call")
        arguments = (calls[0].get("function") or {}).get("arguments")
        # Ollama hands back a decoded dict; OpenAI-compatible APIs hand back a
        # JSON string. Re-serialising the dict keeps `text` one type for the
        # classifier, at the cost of one redundant round trip on Ollama.
        text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    else:
        text = message.get("content") or ""

    return Reply(
        text=text,
        message=message,
        prompt_tokens=usage_in,
        completion_tokens=usage_out,
        finish_reason=finish,
        raw=body,
    )


def repair_turns(
    style: str, text: str, feedback: str, *, provider: str
) -> list[dict[str, Any]]:
    """The two turns that turn a failed attempt into a repair attempt.

    Rebuilt from `text` rather than replayed from the raw response, because the
    checkpoint keeps the candidate JSON and not the provider envelope — a 900-row
    run that stored every response body is a 40 MB file for no gain.

    The asymmetry that cost three episodes on the first smoke run: **Ollama wants
    tool arguments as a decoded object and rejects the string** with
    `Value looks like object, but can't find closing '}' symbol` — a 400 whose
    message describes a parse failure that never happened. OpenAI-compatible APIs
    want exactly the opposite. `parse_reply` normalises the outbound direction to
    a string; this denormalises it on the way back, and the two must stay in step.
    """
    if style != TOOL:
        return [
            {"role": "assistant", "content": text},
            {"role": "user", "content": feedback},
        ]

    if provider == "ollama":
        try:
            arguments: Any = json.loads(text)
        except json.JSONDecodeError:
            # Only reachable if Ollama ever returns arguments we cannot re-encode.
            # Degrading to a plain turn keeps the episode alive; the alternative
            # is a 400 that costs the whole item.
            return [
                {"role": "assistant", "content": text},
                {"role": "user", "content": feedback},
            ]
        tool_message: dict[str, Any] = {
            "role": "tool",
            "content": feedback,
            "tool_name": TOOL_NAME,
        }
        call_id = None
    else:
        arguments = text
        call_id = f"call_{abs(hash(text)) % 10**8:08d}"
        tool_message = {"role": "tool", "content": feedback, "tool_call_id": call_id}

    function: dict[str, Any] = {"name": TOOL_NAME, "arguments": arguments}
    call_envelope: dict[str, Any] = {"type": "function", "function": function}
    if call_id is not None:
        call_envelope["id"] = call_id

    assistant = {"role": "assistant", "content": "", "tool_calls": [call_envelope]}
    return [assistant, tool_message]


async def call(
    client: httpx.AsyncClient,
    target: Target,
    style: str,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    *,
    seed: int,
    temperature: float,
    num_predict: int,
) -> Reply:
    payload = build_payload(
        target,
        style,
        messages,
        schema,
        seed=seed,
        temperature=temperature,
        num_predict=num_predict,
    )
    headers = {"Authorization": f"Bearer {target.key}"} if target.key else {}

    last: Exception | None = None
    for attempt in range(MAX_HTTP_ATTEMPTS):
        try:
            response = await client.post(
                target.url, json=payload, headers=headers, timeout=TIMEOUT
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = exc
            await _backoff(attempt, None)
            continue

        if response.status_code == 200:
            reply = parse_reply(target, style, response.json())
            reply.http_retries = attempt
            return reply

        if response.status_code == 429 or response.status_code >= 500:
            last = httpx.HTTPStatusError(
                f"{response.status_code}: {response.text[:200]}",
                request=response.request,
                response=response,
            )
            await _backoff(attempt, response.headers.get("retry-after"))
            continue

        # A 400 is the request being wrong, and no amount of waiting fixes it.
        response.raise_for_status()

    raise last if last else RuntimeError("unreachable")


async def _backoff(attempt: int, retry_after: str | None) -> None:
    """Full jitter, and honour `retry-after` when the server sends one."""
    if retry_after:
        try:
            await asyncio.sleep(min(float(retry_after), 60.0))
            return
        except ValueError:
            pass
    await asyncio.sleep(random.uniform(0, min(30.0, 0.5 * 2**attempt)))
