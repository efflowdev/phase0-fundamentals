# Decision log — phase 0

Five lines per decision: context, options considered, choice, why, what would change my mind.

---

## Seed policy varies by temperature

- **Context.** Every cell runs the same prompt 20 times; the seed has to be set to something.
- **Options.** One fixed seed everywhere; no seed at all; a fixed seed at temperature 0 and a
  per-sample seed above it.
- **Choice.** Fixed seed 42 at temperature 0, `seed = run_idx` at temperature 1.
- **Why.** A fixed seed at temperature 1 would return 20 identical outputs and the variance table
  would then be describing the harness rather than the model. A fixed seed at temperature 0 is the
  opposite case — it is the control, because it removes the sampler as a possible explanation for any
  disagreement that shows up.
- **What would change my mind.** A provider that ignores the seed entirely, in which case the
  temperature-0 arm proves less than it appears to and the control has to come from somewhere else.

---

## Collection is separated from scoring

- **Context.** The run produces per-token logprobs; the report needs confidence metrics derived from
  them.
- **Options.** Compute metrics during the run and store the numbers; store raw logprobs and compute
  metrics at report time.
- **Choice.** Store raw, score later. `run.py` computes no metric and `report.py` makes no network
  call.
- **Why.** The first scorer definition was wrong twice on day 3 alone — margin needed a rule for
  tokens with no runner-up, and the off-top count did not exist at all in the first version. With
  metrics baked in at collection time each of those changes costs 360 API calls; with raw storage it
  costs a re-read.
- **What would change my mind.** Storage volume. Raw logprobs with five alternatives per token are
  roughly 1.9 MB for 360 short calls; at eval scale the same choice would need compression or a
  separate blob store.

---

## SQLite from day 3, on a schema shared with the weekend

- **Context.** Day 3 and the weekend's structured-output grid both produce rows of model calls, and
  `lens` reads them next week.
- **Options.** JSONL now and SQLite later; SQLite from the start with one generic `runs` table.
- **Choice.** One `runs` table now, with day-specific fields in an `extra_json` column.
- **Why.** Every column in the table is a property of *a model call*, not of this experiment, so the
  weekend appends rather than migrates. The alternative means writing the schema twice and giving
  `lens` an importer.
- **What would change my mind.** If the weekend's rows turn out to need real columns rather than
  `extra_json` — querying inside a JSON blob is fine for a handful of fields and unpleasant beyond
  that.

---

## Dispatch is an experimental variable, but only at temperature 0

- **Context.** Temperature-0 nondeterminism is usually attributed to server-side batching, which is a
  claim I wanted to test rather than repeat.
- **Options.** Report the divergence and cite the known causes; vary the seed; vary sequential against
  concurrent dispatch.
- **Choice.** Sequential and concurrent arms at temperature 0, sequential only at temperature 1.
- **Why.** Comparing dispatch modes at temperature 1 measures nothing, because both arms vary by
  design. At temperature 0 with a fixed seed, firing 20 requests together is the cheapest way to put
  the server's batching under load and see whether the disagreement rate moves with it.
- **What would change my mind.** It did not move — but the concurrent arm was rate-limited and
  retried, so the requests were never actually co-batched. A paid tier with headroom would make the
  test conclusive; as it stands the result is inconclusive rather than negative.

---

## "Exact-match rate" is defined as modal share

- **Context.** The spec asks for an exact-match rate across 20 samples, which does not say what the
  match is against.
- **Options.** Match against a ground-truth answer; against the first sample; against the most common
  sample.
- **Choice.** The fraction of samples equal to the most common one, reported both raw and after
  normalisation.
- **Why.** Ground truth does not exist for the open-ended prompt, and matching against the first
  sample makes the metric depend on run order. The mode needs neither. Reporting it twice separates
  "the model changed its answer" from "the model changed its punctuation".
- **What would change my mind.** Nothing for a variance metric — but this number says nothing about
  correctness, which the hard-factual prompt demonstrated by scoring a perfect 100% on an answer that
  was wrong 40 times out of 40.

---

## Three prompts instead of the spec's two

- **Context.** The spec asks for one short factual prompt and one open-ended one.
- **Options.** Two prompts as specified; add a third factual prompt the small model is likely to get
  wrong.
- **Choice.** Three — easy factual, hard factual, open-ended.
- **Why.** A mean token probability of 0.94 means nothing on its own. The confidence metric needs a
  case where the model is fabricating, or there is no way to tell whether the number is measuring
  anything.
- **What would change my mind.** Nothing; it was the right call. The hard factual produced the two
  most interesting results of the day and cost 120 extra calls on free tiers.

---

## A separate non-streaming client, despite day 2's streaming one

- **Context.** Day 2 already has a working hand-rolled streaming client for both providers.
- **Options.** Reuse the streaming client and accumulate; write a non-streaming path for day 3.
- **Choice.** A separate non-streaming `complete()`.
- **Why.** An experiment wants the whole response, the token accounting and the logprobs as one
  object. More importantly, a non-streaming request is retryable: day 2 has to refuse to retry after
  its first yielded event, because a half-consumed stream cannot be replayed, while nothing here is
  handed to a caller until the response is complete.
- **What would change my mind.** Needing time-to-first-token as a metric, which only the streaming
  path can measure.
