"""Confidence from logprobs.

Four numbers, because one is not enough to be useful:

* **mean probability** — the arithmetic mean of `exp(logprob)`. Reads high on
  almost everything, because most tokens in an English sentence are structurally
  forced and sit near 1.0.
* **perplexity** — `exp(-mean(logprob))`, the *geometric* mean's reciprocal. One
  catastrophic token drags it up hard, where the arithmetic mean barely notices.
  Reporting both is the point: their disagreement locates the outlier.
* **top margin** — `p(best) - p(second best)`. A token at p=0.40 whose runner-up
  is at 0.39 is a coin flip; at p=0.40 with a runner-up at 0.02 the model is not
  actually torn, it is just spreading mass over a long tail. Absolute probability
  cannot tell those apart.
* **chosen margin** — `p(sampled) - p(best)`. Zero at greedy decoding by
  definition. Negative wherever temperature overrode the model's own preference,
  which makes `n_off_top` a direct count of how often sampling changed the answer
  rather than an inference from output variance.

Scoring is deliberately separate from collection: the runs table stores raw
logprobs, so redefining any of this is a re-read of SQLite, not 480 more calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sample import TokenLogprob


@dataclass(frozen=True)
class Confidence:
    n_tokens: int
    mean_prob: float
    perplexity: float
    mean_top_margin: float
    mean_chosen_margin: float
    n_off_top: int
    min_prob: float
    min_prob_token: str
    min_prob_index: int
    min_top_margin: float
    min_top_margin_token: str


EMPTY = Confidence(0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, "", -1, 0.0, "")


def top_margin(token: TokenLogprob) -> float:
    """p(best) - p(second best), over the alternatives the provider returned.

    With fewer than two alternatives the runner-up is unknown, not zero. Treating
    it as zero would report maximum certainty on exactly the tokens where we have
    the least information, so those tokens are excluded from the mean instead
    (see `score`) and this returns the bare top probability for inspection.
    """
    if not token.top:
        return token.prob
    best = math.exp(token.top[0][1])
    if len(token.top) < 2:
        return best
    return best - math.exp(token.top[1][1])


def chosen_margin(token: TokenLogprob) -> float:
    """p(sampled) - p(best). Negative means sampling beat the argmax."""
    if not token.top:
        return 0.0
    return token.prob - math.exp(token.top[0][1])


def is_off_top(token: TokenLogprob) -> bool:
    """Was the sampled token something other than the model's first choice?

    Compared on the token string rather than on probability, because ties and
    float rounding make `p(chosen) == p(top)` an unreliable test.
    """
    return bool(token.top) and token.token != token.top[0][0]


def score(tokens: list[TokenLogprob] | None) -> Confidence:
    if not tokens:
        return EMPTY

    probs = [t.prob for t in tokens]
    logprobs = [t.logprob for t in tokens]
    n = len(tokens)

    mean_prob = sum(probs) / n
    perplexity = math.exp(-sum(logprobs) / n)

    # Only tokens with a real runner-up contribute to the margin means.
    with_alternatives = [t for t in tokens if len(t.top) >= 2]
    if with_alternatives:
        top_margins = [top_margin(t) for t in with_alternatives]
        mean_top = sum(top_margins) / len(with_alternatives)
        worst_idx = min(range(len(top_margins)), key=top_margins.__getitem__)
        min_top = top_margins[worst_idx]
        min_top_token = with_alternatives[worst_idx].token
    else:
        mean_top, min_top, min_top_token = 0.0, 0.0, ""

    with_top = [t for t in tokens if t.top]
    mean_chosen = (
        sum(chosen_margin(t) for t in with_top) / len(with_top) if with_top else 0.0
    )

    worst_prob_idx = min(range(n), key=probs.__getitem__)

    return Confidence(
        n_tokens=n,
        mean_prob=mean_prob,
        perplexity=perplexity,
        mean_top_margin=mean_top,
        mean_chosen_margin=mean_chosen,
        n_off_top=sum(1 for t in tokens if is_off_top(t)),
        min_prob=probs[worst_prob_idx],
        min_prob_token=tokens[worst_prob_idx].token,
        min_prob_index=worst_prob_idx,
        min_top_margin=min_top,
        min_top_margin_token=min_top_token,
    )


def tokens_from_json(raw: str | None) -> list[TokenLogprob] | None:
    """Rebuild logprobs from the JSON blob the runs table stores."""
    import json

    if not raw:
        return None
    data = json.loads(raw)
    return [
        TokenLogprob(
            token=item["token"],
            logprob=float(item["logprob"]),
            top=[(alt[0], float(alt[1])) for alt in item.get("top", [])],
        )
        for item in data
    ]


def tokens_to_json(tokens: list[TokenLogprob] | None) -> str | None:
    import json

    if tokens is None:
        return None
    return json.dumps(
        [
            {"token": t.token, "logprob": t.logprob, "top": [list(a) for a in t.top]}
            for t in tokens
        ],
        ensure_ascii=False,
    )
