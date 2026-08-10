"""Providers, prompts and the seed policy — everything the experiment varies.

Kept in one file so the run matrix is readable as data rather than reconstructed
from argument parsing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
DEFAULT_DB = Path(__file__).resolve().parent / "runs.db"
EXPERIMENT = "day3-sampling"


def load_env(path: Path = ENV_PATH) -> None:
    """Six lines instead of python-dotenv, same as day 2. No frameworks week."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    url: str
    supports_logprobs: bool
    api_key_env: str | None
    max_concurrency: int


GROQ = Provider(
    name="groq",
    model="llama-3.3-70b-versatile",
    url="https://api.groq.com/openai/v1/chat/completions",
    # Probed 2026-08-08: every Groq model answers `logprobs` with
    # {"error": {"message": "`logprobs` is not supported with this model"}}.
    # So the confidence half of this day cannot run here at all.
    supports_logprobs=False,
    api_key_env="GROQ_API_KEY",
    # Free tier is ~30 req/min on this model. 8 in flight will still earn 429s
    # on the concurrent arm; that is what day 2's backoff is for, and it is also
    # a caveat for the batching result — see run.py.
    max_concurrency=8,
)

OLLAMA = Provider(
    name="ollama",
    model="llama3.2",
    url="http://localhost:11434/api/chat",
    supports_logprobs=True,
    api_key_env=None,
    # One local model, one GPU. More in flight only lengthens the queue, but
    # queueing is exactly the variable the concurrent arm is probing.
    max_concurrency=4,
)

PROVIDERS = {p.name: p for p in (GROQ, OLLAMA)}


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    max_tokens: int


# Three, not the spec's two. The confidence score needs something to discriminate
# between: a mean probability of 0.94 means nothing until you have seen what the
# same metric reports on an answer the model is inventing.
PROMPTS = [
    Prompt(
        id="easy_factual",
        text=(
            "What is the ISO 4217 currency code for the British pound? "
            "Reply with the code only."
        ),
        max_tokens=16,
    ),
    Prompt(
        id="hard_factual",
        text=(
            "Which EU directive gives online shoppers a 14-day right of "
            "withdrawal? Reply with the directive number only, in the form "
            "YYYY/NN/EU."
        ),
        max_tokens=24,
    ),
    Prompt(
        id="open_ended",
        text=(
            "A customer emails to say the trainers they ordered arrived with a "
            "scuffed sole, 18 days after delivery. Write a two-sentence reply."
        ),
        max_tokens=150,
    ),
]

PROMPTS_BY_ID = {p.id: p for p in PROMPTS}

TEMPERATURES = (0.0, 1.0)
DISPATCHES = ("sequential", "concurrent")


def parse_sampler(spec: str) -> dict[str, object]:
    """`top_k=1,repeat_penalty=1.0` → `{"top_k": 1, "repeat_penalty": 1.0}`.

    Names pass through verbatim rather than being translated, because the two
    providers do not agree on them: Ollama takes llama.cpp's `top_k`, `min_p`
    and `repeat_penalty` nested under `options`, Groq takes OpenAI's `top_p`,
    `frequency_penalty` and `presence_penalty` at the top level, and neither
    exposes the other's. Having to type the provider's own name is the point —
    a wrapper that unified them would hide the difference this day is about.
    """
    sampler: dict[str, object] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if raw.lower() in {"true", "false"}:
            sampler[key] = raw.lower() == "true"
            continue
        try:
            sampler[key] = int(raw)
        except ValueError:
            try:
                sampler[key] = float(raw)
            except ValueError:
                sampler[key] = raw
    return sampler


def seed_for(temperature: float, run_idx: int) -> int:
    """Fixed at greedy, varying above it.

    The policy has to differ by temperature or the experiment measures nothing.
    At temperature 0 a fixed seed is the control: if 20 identical requests with
    one identical seed still disagree, the sampler is not the cause. At
    temperature 1 a fixed seed would make all 20 samples the same output and the
    variance table would read "temperature 1 is deterministic", which is a
    conclusion about the harness, not the model.
    """
    if temperature == 0.0:
        return 42
    return run_idx
