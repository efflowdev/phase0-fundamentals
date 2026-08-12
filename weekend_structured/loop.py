"""One product through one (target, style): first try, then two arms.

    attempt 0 ─── valid? ─── yes ──> done, both arms credited
    (temp 0)        │
                    no
                    ├──> repair  arm: transcript + the validation errors
                    └──> control arm: the same prompt again, no errors
                         (both: seeds 43/44, temperature 0.7)

**The control arm is not padding.** "The repair loop recovered 61% of failures"
is only a claim about the feedback if you know what a bare resample recovers.
Without that column, a loop that never actually showed the model its errors would
still report a plausible recovery rate.

**And the control has to be a real resample, which cost a run to learn.** The
first version advanced only the seed — 42 for the first attempt, 43 and 44 for
the retries, identical in both arms — on the theory that the arms would then
differ only in whether the errors were shown. They did. They also produced
identical outputs, because **temperature 0 is greedy decoding and greedy decoding
ignores the seed**. Measured on this machine:

    ollama llama3.2, temp 0,   seeds 42/43/44 -> 1 distinct output of 3
    ollama llama3.2, temp 0.7, seeds 42/43/44 -> 3 distinct outputs of 3
    groq   70b,      temp 0,   seeds 42/43/44 -> 2 distinct outputs of 3

So the local control was 0% by construction — the exact degeneracy the seed
advance was written to avoid — and the Groq control was measuring incidental
non-determinism (day 3's finding, met again) rather than a controlled resample.
It reported 0% recovery in every cell of the smoke run, which reads as "feedback
wins" and is really "nothing was resampled".

Retries therefore run at **temperature 0.7 in both arms**. The first attempt stays
at 0, so tables 1 and 2 are still a deterministic measurement of the mechanisms.
The comparison is fair rather than identical — the repair arm's context is ~200
tokens longer, and length changes sampling on its own — but it is the version
where the feedback has to beat a real second roll instead of a no-op.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import classify, styles
from .classify import Verdict
from .config import (
    MAX_REPAIRS,
    NUM_PREDICT,
    RETRY_TEMPERATURE,
    SYSTEM,
    TEMPERATURE,
    Target,
)

REPAIR = "repair"
CONTROL = "control"
FIRST = "first"


@dataclass
class Attempt:
    seed: int
    text: str
    verdict: Verdict
    temperature: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    http_retries: int = 0
    transport_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.verdict.valid


@dataclass
class Arm:
    name: str
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].ok

    @property
    def total_attempts(self) -> int:
        """Counting the shared first try, because that is what a call costs.

        A cell that succeeds immediately took one attempt; one that needed both
        repairs took three. Reporting the repairs alone would make the expensive
        strategy look free.
        """
        return len(self.attempts)


@dataclass
class Episode:
    target_id: str
    style: str
    product: dict[str, str]
    first: Attempt
    arms: dict[str, Arm]

    @property
    def valid_first_try(self) -> bool:
        return self.first.ok


async def _one_attempt(
    client: httpx.AsyncClient,
    target: Target,
    style: str,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    model: type,
    seed: int,
    temperature: float,
) -> Attempt:
    started = time.perf_counter()
    try:
        reply = await styles.call(
            client,
            target,
            style,
            messages,
            schema,
            seed=seed,
            temperature=temperature,
            num_predict=NUM_PREDICT,
        )
    except styles.NoToolCall as exc:
        # A forced tool call the model declined. That is the enforcement
        # mechanism failing to enforce, which is a finding — so it becomes a
        # violation with its own rule, not a transport error that gets retried.
        return Attempt(
            seed=seed,
            text="",
            verdict=Verdict(valid=False, rules=[classify.NO_TOOL_CALL], detail=[str(exc)]),
            temperature=temperature,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    verdict = classify.classify(reply.text, model)
    return Attempt(
        seed=seed,
        text=reply.text,
        verdict=verdict,
        temperature=temperature,
        latency_ms=(time.perf_counter() - started) * 1000,
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
        finish_reason=reply.finish_reason,
        http_retries=reply.http_retries,
    )


async def episode(
    client: httpx.AsyncClient,
    target: Target,
    style: str,
    product: dict[str, str],
    schema: dict[str, Any],
    model: type,
    *,
    base_seed: int,
    max_repairs: int = MAX_REPAIRS,
) -> Episode:
    base = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": styles.user_prompt(product, schema, style)},
    ]

    first = await _one_attempt(
        client, target, style, base, schema, model, base_seed, TEMPERATURE
    )

    arms = {REPAIR: Arm(REPAIR, [first]), CONTROL: Arm(CONTROL, [first])}
    if first.ok:
        return Episode(target.id, style, product, first, arms)

    # --- repair arm: the transcript grows, each error shown once -------------
    transcript = list(base)
    previous = first
    for step in range(1, max_repairs + 1):
        if previous.transport_error is None:
            transcript = transcript + styles.repair_turns(
                style,
                previous.text,
                classify.feedback(previous.verdict),
                provider=target.provider,
            )
        attempt = await _one_attempt(
            client,
            target,
            style,
            transcript,
            schema,
            model,
            base_seed + step,
            RETRY_TEMPERATURE,
        )
        arms[REPAIR].attempts.append(attempt)
        if attempt.ok:
            break
        previous = attempt

    # --- control arm: same seeds, same temperature, nothing learned ----------
    for step in range(1, max_repairs + 1):
        attempt = await _one_attempt(
            client,
            target,
            style,
            base,
            schema,
            model,
            base_seed + step,
            RETRY_TEMPERATURE,
        )
        arms[CONTROL].attempts.append(attempt)
        if attempt.ok:
            break

    return Episode(target.id, style, product, first, arms)


def to_rows(episode_: Episode) -> list[dict[str, Any]]:
    """One row per dispatch: the shared first try, then each arm's outcome.

    Not one row per attempt — the unique index in `phase0.store` has no `attempt`
    column, so two repair attempts on the same cell would overwrite each other.
    The row is the episode; the per-attempt trail rides in `extra`, which is what
    `store.py`'s docstring reserved `extra_json` for.
    """
    rows = [
        {
            "dispatch": FIRST,
            "output": episode_.first.text,
            "attempt": episode_.first,
            "extra": {
                "valid": episode_.first.ok,
                "valid_first_try": episode_.valid_first_try,
                "rules": episode_.first.verdict.rules,
                "salvaged": episode_.first.verdict.salvaged,
                "detail": episode_.first.verdict.detail,
                "attempts": 1,
            },
        }
    ]
    if episode_.valid_first_try:
        return rows

    for name in (REPAIR, CONTROL):
        arm = episode_.arms[name]
        last = arm.attempts[-1]
        rows.append(
            {
                "dispatch": name,
                "output": last.text,
                "attempt": last,
                "extra": {
                    "valid": arm.resolved,
                    "valid_first_try": False,
                    "rules": last.verdict.rules,
                    "salvaged": last.verdict.salvaged,
                    "detail": last.verdict.detail,
                    "attempts": arm.total_attempts,
                    "trail": [
                        {
                            "seed": a.seed,
                            "temperature": a.temperature,
                            "valid": a.ok,
                            "rules": a.verdict.rules,
                            "latency_ms": round(a.latency_ms, 1),
                        }
                        for a in arm.attempts
                    ],
                },
            }
        )
    return rows
