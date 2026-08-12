"""Targets, styles and the constants the grid varies — the matrix as data.

Three models x three prompt styles x 100 products, and the thing that makes the
grid ragged is that **"structured output" is not one mechanism**. Probed
2026-08-12, against this key and these models:

| target        | prompt | tool               | native                      |
|---------------|--------|--------------------|-----------------------------|
| llama3.2:3b   | yes    | yes                | `format=<schema>`, grammar  |
| gemma3:4b     | yes    | probed at startup  | `format=<schema>`, grammar  |
| groq 70b      | yes    | yes, forced        | `response_format=json_object` |

Ollama's `format` takes the whole JSON Schema and constrains the sampler to it.
Groq's `llama-3.3-70b-versatile` answers `json_schema` with a 400 — it supports
only `json_object`, which guarantees the bytes parse and says nothing about the
document. Two cells, same column heading, different guarantees, and reporting
them as one number would be the day-6 image-size mistake again.

(`openai/gpt-oss-20b` on Groq is the inverse: it takes `json_schema` and *fails*
a forced `tool_choice` with "model did not call a tool". No model on this key
supports both. That is why the cloud arm is the 70b and the native column carries
a footnote rather than a third provider.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE.parents[1] / ".env"
DEFAULT_DB = HERE / "runs.db"
CHECKPOINT_DIR = HERE / "runs"
EXPERIMENT = "weekend-structured"

N_PRODUCTS = 100
BASE_SEED = 42
TEMPERATURE = 0.0

# Retries resample; the first attempt does not. Measured 2026-08-12: at
# temperature 0 the seed is inert (greedy decoding returns the same tokens for
# seeds 42/43/44), so a control arm that only advanced the seed recovered 0% by
# construction and made the feedback look effective no matter what it said.
# 0.7 because it is the value that actually produced three distinct outputs from
# three seeds on this hardware, not because it is a round number.
RETRY_TEMPERATURE = 0.7

MAX_REPAIRS = 2
NUM_PREDICT = 512


def load_env(path: Path = ENV_PATH) -> None:
    """Six lines instead of python-dotenv, same as days 2 and 3. No frameworks."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# --- styles -----------------------------------------------------------------
# The schema reaches the model three ways, in ascending order of how much the
# runtime enforces rather than requests.

PROMPT = "prompt"  # schema pasted into the system prompt; nothing enforced
TOOL = "tool"  # schema as a function's `parameters`; the call is forced
NATIVE = "native"  # the provider's own structured-output hook

STYLES = (PROMPT, TOOL, NATIVE)


# --- targets ----------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    id: str
    provider: str
    model: str
    url: str
    concurrency: int
    native: str
    api_key_env: str | None = None
    supports_tools: bool = True

    @property
    def key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    def native_guarantee(self) -> str:
        """What the native column actually promises, for the report footnote."""
        return {
            "schema": "grammar-constrained to the schema",
            "json_object": "valid JSON only; schema not enforced",
        }[self.native]


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")

# One, and the reason is a measurement rather than a preference. Measured
# 2026-08-12 on this machine, same prompt, warm model:
#
#   gemma3:4b   conc=1  28.0 tok/s aggregate, per-call p50 2287ms
#   gemma3:4b   conc=4  32.1 tok/s aggregate, per-call p50 6069ms
#   llama3.2    conc=1  37.2 tok/s aggregate, per-call p50 3517ms
#   llama3.2    conc=4  43.4 tok/s aggregate, per-call p50 9074ms
#
# Four requests do not run four times faster; they run 15% faster in aggregate
# and each one waits ~3x longer. One daemon, one GPU, requests effectively
# serialised — so concurrency here buys queueing, and the queueing lands in
# `latency_ms`. Table 4 would then report how many workers this script used,
# labelled as how long the model takes. 15% of the wall clock is the price of a
# latency column that means what it says. Groq keeps 4: those are independent
# server-side workers and the parallelism is real.
OLLAMA_CONCURRENCY = 1

LLAMA32 = Target(
    id="llama3.2:3b",
    provider="ollama",
    model="llama3.2",
    url=f"{OLLAMA_URL}/api/chat",
    concurrency=OLLAMA_CONCURRENCY,
    native="schema",
)

GEMMA3 = Target(
    id="gemma3:4b",
    provider="ollama",
    model="gemma3:4b",
    url=f"{OLLAMA_URL}/api/chat",
    concurrency=OLLAMA_CONCURRENCY,
    native="schema",
    # Gemma 3 is not tool-trained and Ollama rejects `tools` for it. Left True
    # here so the probe in run.py is what decides, rather than this comment
    # ageing badly the day Ollama adds a template for it.
    supports_tools=True,
)

GROQ70B = Target(
    id="groq/llama-3.3-70b",
    provider="groq",
    model="llama-3.3-70b-versatile",
    url="https://api.groq.com/openai/v1/chat/completions",
    # Free tier is ~30 req/min on this model. Day 2's backoff covers the 429s
    # this still earns; going lower would make the cloud arm the wall clock.
    concurrency=4,
    native="json_object",
    api_key_env="GROQ_API_KEY",
)

TARGETS = (LLAMA32, GEMMA3, GROQ70B)
BY_ID = {target.id: target for target in TARGETS}


SYSTEM = (
    "You extract structured product attributes from catalogue titles. "
    "Reply with a single JSON object and nothing else. "
    "Use null for any attribute the title does not state — never the string "
    "'unknown'."
)
