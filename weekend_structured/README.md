# Weekend — Structured output, and the ceiling on enforcing it

Eight hundred episodes: three models, three ways of demanding a schema, one hundred product titles
each, and a validate-repair-retry loop measured against a control that was told nothing. The
question is not whether a model can return JSON — all of them can — but which failures survive each
enforcement mechanism, and whether showing a model its own validation errors beats simply asking it
again.

The short version of the answer is structural rather than empirical, and it holds before a single
call is made: **a grammar can only enforce what a JSON Schema can express**, and the contract this
day validates against is mostly not expressible. Everything measured below is a consequence of that.

## "Structured output" is three different guarantees wearing one name

Probed 2026-08-12, one throwaway call per cell, against this API key and these daemons:

| target | prompt | tool | native |
|---|---|---|---|
| `llama3.2:3b` (Ollama) | yes | yes, offered | `format=<schema>` — sampler constrained to the grammar |
| `gemma3:4b` (Ollama) | yes | **HTTP 400, "does not support tools"** | `format=<schema>` — same |
| `llama-3.3-70b` (Groq) | yes | yes, forced | `response_format: json_object` — bytes parse, schema not enforced |

Two things fall out of that table and neither was in the plan.

The 3×3 is **eight cells**. Gemma 3 is not tool-trained and Ollama refuses the request outright, so
which enforcement mechanisms exist is a property of the model and belongs in the results rather than
in configuration. `run.py` probes every cell before the run and writes `capabilities.json` with the
provider's own error text, so an empty cell in the report carries its reason.

And the native column is **not one mechanism**. Ollama takes the entire JSON Schema and restricts the
sampler to tokens that keep the output valid against it. Groq's 70b answers `json_schema` with a 400
— it offers only `json_object`, which promises the response parses as JSON and promises nothing
whatsoever about the document inside. Reporting those two as one number would be day 6's image-size
mistake in a new place: two honest measurements of different things, averaged into a measurement of
something that does not exist.

`openai/gpt-oss-20b` on the same key is the exact inverse — it accepts `json_schema` and *fails* a
forced `tool_choice` with `"Tool choice is required, but model did not call a tool"`. No model
available here supports both, so the cloud arm cannot be chosen to make the columns comparable.

## The schema you send is not the schema you validate against

`ProductAttributes` has seven fields and four hand-written rules. `model_json_schema()` — the object
that becomes Ollama's grammar and the tool's `parameters` — carries some of them and silently drops
the rest:

| rule | reaches the model? |
|---|---|
| `pack_quantity` between 1 and 1000 | yes — `minimum`, `maximum` |
| `category` is one of eight values | yes — the enum, via `$ref` |
| `dimensions` is an object or null | yes — the `anyOf` |
| ASIN matches `^[A-Z0-9]{10}$` | **no** — emitted as bare `{"type": "string"}` |
| `"unknown"` is forbidden; absent means `null` | **no** |
| a `dimensions` object cannot be entirely null | **no** |
| footwear and apparel require a size | **no** |

The line is not difficulty. `pack_quantity <= 1000` is expressible and "footwear needs a size" is
not, and the second is barely harder to state in English. The line is whether the constraint is a
*field constraint* or a *validator*: `Field(le=1000)` becomes `maximum`, while `@field_validator` and
`@model_validator` are Python callables with no JSON Schema representation at all. The ASIN case is
the sharpest, because that rule *is* expressible — `Field(pattern=...)` would emit it — and it is
declared in a way that throws the expressibility away.

So constrained decoding has a hard ceiling that the mechanism cannot be blamed for and cannot cross.
`classify.py` carries thirteen rules with an `expressible` flag on each, eight of them true, and
table 2 below reports violations against that split. That column is the day's actual result.

## Results

> **PENDING — do not publish this section as-is.** The 800-episode run was still in flight when this
> file was written. Generate the tables with:
>
> ```
> uv run python -m weekend_structured.report
> ```
>
> and paste each one under its heading below, then replace each italic prompt with what the numbers
> actually show. The prompts are questions to answer, not conclusions to confirm — if the data says
> something else, the data is the finding.

### 1. Valid document on the first try

<!-- paste table 1 -->

*Which mechanism wins, and by how much? Does the ordering hold across all three models, or does the
70b close the gap that enforcement opens for the small ones?*

### 2. Which violations, and could a grammar have caught them

<!-- paste table 2 -->

*The two bottom rows are the point: total violations, and how many of them no schema could express.
Read the ratio down the native column. If native's surviving failures are overwhelmingly
inexpressible, the mechanism has hit its ceiling and further enforcement cannot move it — the
remaining fix is prompting, or a second validation pass, or changing the contract.*

### 3. Repair against a no-feedback control

<!-- paste table 3 -->

*`repair fixed` against `control fixed` is the only comparison that attributes anything. If they are
close, the error text is doing nothing and the loop is paying triple the calls for a second roll of
the dice. `mean attempts` is the price of the whole strategy and the number people leave out.*

### 4. What it cost

<!-- paste table 4 -->

*Local against cloud is a 3B on a laptop against a 70B on someone else's accelerator — read this as
what structure costs, not as which model is faster. The comparison that is fair is between styles
within one row: the tool style ships the schema on every call, and it shows up in `mean in-tok`.*

## What was wrong, and it was the instruments again

Three times. Day 5 ended on this note and day 6 ended on it twice; this is the third day running
where every bug lived in the code that measures rather than the code that runs, and none of them
failed loudly.

**The control arm was zero by construction.** Both arms advanced the seed — 42 for the first
attempt, then 43 and 44 — so that the only difference between them would be whether the validation
errors were shown. That reasoning is sound and the implementation was correct. It was also useless,
because **temperature 0 is greedy decoding and greedy decoding ignores the seed**:

| | distinct outputs from seeds 42/43/44 |
|---|---|
| llama3.2, temperature 0 | **1 of 3** |
| llama3.2, temperature 0.7 | 3 of 3 |
| Groq 70b, temperature 0 | 2 of 3 |

The local control resampled nothing and recovered 0% of failures in every cell of the smoke run,
which reads as a clean win for the feedback and is really a report that nothing was tried. The Groq
control was measuring incidental non-determinism — day 3's finding, met one layer down — rather than
a controlled resample. Retries now run at temperature 0.7 in both arms; the first attempt stays
greedy so tables 1 and 2 remain deterministic. A control that cannot vary is not a control, and it
fails in the direction that flatters the thing being tested.

**Concurrency was being measured as latency.** The runner was set to four in flight against Ollama.
Measured on this machine, warm model, same prompt:

| | aggregate tok/s | per-call p50 |
|---|---|---|
| gemma3:4b, concurrency 1 | 28.0 | 2287 ms |
| gemma3:4b, concurrency 4 | 32.1 | 6069 ms |
| llama3.2, concurrency 1 | 37.2 | 3517 ms |
| llama3.2, concurrency 4 | 43.4 | 9074 ms |

One daemon and one GPU: the requests are effectively serialised, four in flight buy 15% aggregate
throughput, and each call waits about three times longer. That wait is indistinguishable from model
latency once it is written to `latency_ms`, so table 4 would have reported the harness's worker count
under the heading "p50 ms". Ollama now runs at concurrency 1 and Groq stays at 4, where the workers
are independent and the parallelism is real.

**The tool-style repair 400'd on a string that parses.** Ollama returns tool-call arguments already
decoded and requires them decoded on the way back; OpenAI-compatible APIs use a JSON string in both
directions. `parse_reply` normalised the inbound direction so the classifier only ever sees text,
and the repair turn sent that text straight back — earning
`{"error":"Value looks like object, but can't find closing '}' symbol"}`, a parse failure describing
a string that parses perfectly. It cost three episodes and it is the kind of error message that sends
you looking at your JSON rather than at the type you put it in.

Two of these three would have produced a publishable table. That is the whole hazard: a broken
measurement does not crash, it returns a plausible number, and the more sensible the number looks the
longer it survives.

## A note on the directory being a package

Days 1 through 6 are script directories. This one is a package, invoked with `-m`, because
`day3_sampling/config.py` and `weekend_structured/config.py` are both placed on `sys.path` by pytest
and `import config` resolved to whichever was collected first. The weekend imported cleanly when run
alone and failed to import under `uv run pytest` — a bug that depends on collection order.

`pyproject.toml` predicted the general case on day 5: scripts importing each other by accident of
`sys.path[0]` stop working the moment two days share code. This is the other half of it. Two days
that share nothing at all except a *filename* collide exactly as hard, and the flat-script convention
ran out here rather than being outgrown.

`store.py` and `schema.py` moved into `phase0` the same weekend, for day 5's stated reason: the
second caller is what turns a script into a module.

## Reproducing

Needs Ollama running on the host with `llama3.2` and `gemma3:4b` pulled, and `GROQ_API_KEY` in the
repo-root `.env`.

```bash
uv run python -m weekend_structured.run --probe     # capability matrix only, ~10 calls
uv run python -m weekend_structured.run --n 5       # smoke test, a few minutes
uv run python -m weekend_structured.run             # the full grid, ~90 min, resumable
uv run python -m weekend_structured.report          # the four tables
uv run pytest                                       # 186 tests
```

Three things that will make the run look broken and are not:

- `gemma3:4b: skipping tool` is the capability probe working. The reason is in `capabilities.json`.
- Progress appears to stall on the Ollama targets. Concurrency is 1 there on purpose — see above —
  so the local arms are genuinely serial.
- Re-running costs nothing for work already done. The JSONL checkpoint in `runs/` is keyed per
  episode and only failures are retried; `Ctrl-C` loses at most one call.
