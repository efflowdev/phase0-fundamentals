# Day 3 — Sampling and Determinism

This day measured two things: how much the same prompt varies across repeated calls, and whether a
model's own token probabilities say anything useful about when it is inventing an answer. The harness
runs a matrix — two providers (Groq `llama-3.3-70b-versatile`, local Ollama `llama3.2`), three prompts
(a one-token factual, a factual the small model is likely to get wrong, and an open-ended support
reply), temperatures 0 and 1, and at temperature 0 both sequential and concurrent dispatch — with 20
samples per cell, 360 calls in total. Every call is stored raw in SQLite and the metrics are computed
afterwards from the stored logprobs, so changing the definition of a metric costs a query rather than
another 360 calls.

**Logprobs are not available on Groq.** Every model there refuses the request:
`{"error":{"message":"`logprobs` is not supported with this model","type":"invalid_request_error","param":"logprobs"}}`.
The confidence half of the day therefore runs only against Ollama, which returns both `logprobs` and
`top_logprobs` from `/api/chat`.

**Temperature 0 is deterministic locally and not in the cloud.** All nine Ollama cells at temperature
0 returned a single distinct output across 20 samples. Groq returned one distinct output for both
short prompts, but three distinct outputs for the 350-character support reply — 11, 7 and 2 samples
respectively, from identical payloads carrying an identical seed. Two of those three agree for 183
characters before splitting on a word-order flip: "Please could you reply" against "Could you please
reply". Firing the 20 requests concurrently rather than one at a time produced the same three variants
and no new ones, so this data does not support batching as the mechanism — with the caveat that the
concurrent arm was rate-limited and retried, which scattered the burst and would hide exactly that
effect.

**Temperature only matters where the model is already uncertain.** The one-token answer (`GBP`, mean
token probability 0.997) is identical across all 20 samples at temperature 1 on both providers. The
70B model returned the same directive number 20 times out of 20 at temperature 1. The 3B model
returned 16 distinct answers to the same prompt at the same temperature. Sharpness of the underlying
distribution, not the temperature setting, decides whether sampling changes anything.

**Self-consistency is not correctness, and greedy decoding can be a trap.** `llama3.2` answered
`2008/122/EC` 40 times out of 40 at temperature 0 — stable, confident, and wrong (2008/122/EC is the
Timeshare Directive; the answer is 2011/83/EU, which the 70B gave 60 times out of 60). At temperature
1 the same small model produced `2011/83/EU` exactly once in 20 samples. Every consistency metric on
this page rates the temperature-0 run perfect and the temperature-1 run poor, and only the
temperature-1 run ever found the right answer.

**Confidence separates the prompts and locates the fabrication.** Mean token probability is 0.997 on
the easy factual, 0.786 on the open-ended reply and 0.714 on the hard factual — the model is
measurably less sure of the fact than of the boilerplate. The lowest-probability token in the hard
factual is `'200'` at p=0.415, the first token of the fabricated directive number.

**The surprise: at temperature 0, 6.5% of tokens were not the model's own top choice.** Greedy
decoding should make that impossible. Every swap replaced a word that had already appeared earlier in
the response, which pointed at llama.cpp's default `repeat_penalty=1.1` editing the logits *after* the
values reported in `top_logprobs` are read off. Re-running with `repeat_penalty=1.0` drove the off-top
rate to exactly 0.0% and made the responses both longer (62 → 77 tokens) and more confident (mean
probability 0.786 → 0.834, perplexity 1.39 → 1.25). The number an API hands back as the model's
confidence is a snapshot taken partway down the sampler chain, not a description of what the sampler
actually did.

Two of the 180 Groq calls failed with HTTP 429 after five retries, both in the concurrent arm, so that
cell reports 18 samples rather than 20.

## Reproducing

```bash
uv run python day3_sampling/run.py                    # 360 calls, ~20 min
uv run python day3_sampling/report.py --markdown      # the tables above
uv run python day3_sampling/run.py --providers ollama --temps 0 \
    --sampler repeat_penalty=1.0 --experiment day3-no-repeat-penalty
```
