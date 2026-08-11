# Day 5 — Python at eval scale

Most of this day's checklist was already standing: `uv`, `pyproject.toml`, `ruff` and `pytest` since
day 1, and `asyncio.gather` behind a `Semaphore` with `return_exceptions=True` since day 3's
concurrent dispatch arm. So rather than re-typing the pattern in a fresh folder, day 5 extracts it —
`phase0/runner.py` is now the harness every later project imports, day 3 is refactored onto it, and
the remaining new material is Pydantic v2 and an actual measurement of where concurrency stops
paying.

**The repo is an installable package now, and it had to become one.** For four days every script
worked by accident of `sys.path[0]` containing its own directory, which silently caps a repo at "no
file may ever be shared". Day 3 and day 5 both need the runner. `phase0/` is the one real package,
declared through hatchling and installed editable by `uv sync`; each `dayN/` directory stays a set of
scripts, because they are exercises, not libraries.

**The refactor is byte-for-byte behaviour-preserving.** `dispatch` in day 3 is now just a concurrency
number — sequential is a semaphore of one — and re-running all 120 Ollama temperature-0 cells against
the committed database gives 120/120 identical outputs, completion counts and logprobs. Those cells
are deterministic, which is what makes them usable as a regression test at all.

**Concurrency stops paying at 4, and past 16 it costs.** Two arms, because "concurrency is faster" is
not a finding — the question is which side saturates first. The fake arm replaces the work with
`asyncio.sleep`, so nothing leaves the process and it measures the runner's own ceiling:

| concurrency | fake speedup | fake efficiency | ollama speedup | ollama efficiency |
|---|---|---|---|---|
| 1 | 1.00× | 100% | 1.00× | 100% |
| 2 | 2.00× | 100% | 1.96× | 98% |
| 4 | 3.99× | 100% | 2.46× | 61% |
| 8 | 7.99× | 100% | 2.63× | 33% |
| 16 | 16.05× | 100% | 2.62× | 16% |
| 32 | 31.96× | 100% | **2.27×** | 7% |

The runner holds 100% efficiency all the way to 32, so everything in the right-hand columns is the
server. Ollama's knee is at 4, it plateaus around 2.6× from 8 onward, and at 32 throughput actually
*falls* — coordination cost for parallelism the server was never going to provide. Nothing errors and
nothing warns; asking for 32 is simply slower than asking for 8.

**Resume turns a three-minute run into 0.38 seconds.** Every completed item is appended to JSONL the
moment it lands, so a crash at item 480 loses one line rather than 480 paid calls. Re-running the
50-product extraction served all 50 from the checkpoint and produced identical numbers. JSONL rather
than SQLite because the runner should not know the shape of what it is running — day 3 keeps its own
SQLite persistence and takes only the concurrency half, which is the test of whether the abstraction
composes.

**Untreated schema violation rate: 70%.** Fifty product titles through `llama3.2` against a
deliberately awkward Pydantic model — nested object, enum, optionals that mean *absent*, a ranged
integer — with no repair loop, because Saturday's job is to measure what a repair loop recovers and
that needs an untreated baseline. Fifteen of fifty validated. The 43 distinct violations across the
35 bad responses:

| violation | count | what it is |
|---|---|---|
| `field:missing` | 22 | a required field absent — mostly the model emitting `asins` for `asin`, plus capitalised `Category` and `Dimensions` |
| `empty_nested_object` | 18 | `{"length_cm":null,"width_cm":null,"height_cm":null}` — structure with no content, where the contract says `null` |
| `field:json_invalid` | 3 | not parseable at all |
| `placeholder_instead_of_null` | **0** | the model never wrote "unknown"; the system prompt's instruction holds |

Two thirds of the failures are key naming, not comprehension. That is a prompt problem, and it is one
line to test.

The surprise: **both of this day's wrong answers came from the measuring code, not the measured
code.** The violation breakdown first reported 10 placeholder errors, then 18, then 0 — because both
of the schema's validators end their message with "use null", and the classifier tested that shared
phrase before the more specific one, filing every empty-`dimensions` error as a placeholder error. A
whole failure mode reported as zero, three runs in a row. Separately, the regression check comparing
the refactor against the committed database reported 80/120 — the query filtered on provider and
temperature but not on `experiment`, so rows from the `repeat_penalty` ablation overwrote the
baseline rows they shared keys with. The 40 "failures" were exactly the `open_ended` cells, which is
the only prompt long enough for a repetition penalty to change, so the bug wore the fingerprint of a
real finding.

Neither crashed. Neither failed a test. Both produced numbers plausible enough to act on, and acting
on the second one would have meant reverting a correct refactor. Keeping raw data and distrusting any
category that reports zero is the only defence, and it is the reason the runner stores full response
text rather than a truncated slice.

## Reproducing

```bash
uv sync                                                   # installs phase0/ editable
uv run python day5_async/schema.py                        # the JSON Schema, i.e. P2's tool definition
uv run python day5_async/benchmark.py --n 32              # the two-arm curve
uv run python day5_async/extract.py --n 50 --concurrency 4
uv run pytest                                             # 117 tests
```
