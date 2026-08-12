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

---

## Corpus is sampled in blocks spread across ESCI, not the first N rows

- **Context.** 500 products are needed out of a 2-million-row dataset, pulled through a paging API.
- **Options.** Take the first 4000 rows; sample blocks from evenly spaced offsets; sample rows at
  random.
- **Choice.** Sixteen contiguous blocks of 400 rows, spread evenly, with the first and last query
  group in each block discarded.
- **Why.** ESCI is ordered by query, so any contiguous prefix is a handful of adjacent topics — the
  first attempt produced a catalogue of bathroom fans and children's books. A corpus of near
  duplicates makes every retriever look bad for reasons unrelated to retrieval, and it fails the
  "corpus you can judge by eye" test the whole exercise depends on. Blocks rather than single rows
  because query groups must stay intact; the edge groups are dropped because a block boundary almost
  certainly cut them in half, and a half-judged query silently caps its own recall.
- **What would change my mind.** Needing a specific product category, at which point filtering beats
  sampling — but then the corpus stops being representative and the numbers stop transferring.

---

## Query groups are skipped whole, never truncated

- **Context.** The 500-product cap will land mid-group for whichever query crosses it.
- **Options.** Truncate the group and keep the query; drop the query; raise the cap to fit.
- **Choice.** Skip the whole group and continue to the next one.
- **Why.** A truncated group means some of a query's judged products are absent from the index, so
  its recall is capped by the loader rather than by the retriever. Every recall number in the table
  would then be partly a measurement of `corpus.build`, and biased downward by an amount that varies
  per query.
- **What would change my mind.** Nothing at this size. At a corpus where whole-group packing wastes
  significant space, the fix is to record each query's achievable ceiling rather than to truncate.

---

## The SKU is inside the indexed document text

- **Context.** The identifier experiment asks whether dense retrieval can find a product by its code.
- **Options.** Index title only, and query codes that appear nowhere; index title plus a SKU line.
- **Choice.** Every document carries `SKU: <asin>` in its indexed text.
- **Why.** If the code were absent, both retrievers would fail and the comparison would prove
  nothing. With it present, the information is genuinely retrievable and BM25 finds it 500 times out
  of 500 — so dense missing it 497 times is a property of the embedding, not of the data. Real
  catalogue search fields carry the SKU anyway.
- **What would change my mind.** Nothing for this experiment. Worth noting that adding a rare token
  to all 500 documents perturbs their embeddings slightly, which is a small cost paid for a clean
  comparison.

---

## Ranking uses argpartition, not argsort

- **Context.** Scoring produces one float per document; the caller wants the best ten.
- **Options.** `argsort` the whole score vector; `argpartition` for the top k, then sort those k.
- **Choice.** `argpartition`, then sort the k.
- **Why.** `argsort` is O(n log n) over the entire corpus to return ten rows, `argpartition` is O(n)
  quickselect. Invisible at 500 documents; at a million it measured 86 ms against 7.6 ms, which would
  have more than doubled query latency.
- **What would change my mind.** Needing the full ranking rather than a top-k — pagination deep into
  results, or a reranker that consumes several hundred candidates.

---

## Metrics are reported against their achievable ceiling

- **Context.** Queries have 8.7 relevant products on average, and k is 1 or 10.
- **Options.** Report recall@k raw; report only hit@k and MRR; report recall@k alongside its ceiling.
- **Choice.** Add `r@10 max` and `% of max` columns computed from the relevant-set sizes.
- **Why.** Recall@1 cannot exceed 1/8.7 here, and recall@10 cannot exceed 0.930 on this query set, so
  quoting either as a fraction of 100% understates the retriever by an amount that has nothing to do
  with the retriever. This is the most common error in eval write-ups and it always points the same
  way — it makes working systems look broken.
- **What would change my mind.** A query set with exactly one relevant document per query, where the
  ceiling is 1.0 and the column is noise. The identifier query set is exactly that case.

---

## The catalogue lives in its own SQLite file, not day 3's

- **Context.** Day 3 established `runs.db` as the phase-0 database, and `lens` will read it.
- **Options.** Add a `documents` table to `runs.db`; keep a separate `day4_search/index/catalog.db`.
- **Choice.** Separate file.
- **Why.** `runs.db` holds model calls — one row per request, with logprobs and latency. A product
  catalogue is a different entity with a different lifecycle, and mixing them means `lens` has to
  know which tables to ignore.
- **What would change my mind.** P1, where traces and the documents they retrieved genuinely need
  joining. At that point one database with a foreign key beats two files.

---

## The repo became an installable package

- **Context.** Day 3 and day 5 both need the concurrency runner, and until now every script imported
  its neighbours by accident of `sys.path[0]` holding its own directory.
- **Options.** Copy the runner into both days; manipulate `sys.path`; make the project installable
  and import it properly.
- **Choice.** A hatchling build target over a single `phase0/` package, installed editable by
  `uv sync`. Each `dayN/` directory stays a set of scripts.
- **Why.** Two copies drift, and the one that drifts is always the one you are not looking at. The
  `sys.path` version works and is embarrassing to explain. Editable install is what the tooling is
  for, and the split it forces — `phase0/` is a library, `dayN/` are exercises — is the useful part.
- **What would change my mind.** Nothing here. The mistake was leaving it until day 5; it should
  have happened the first time two days shared anything.

---

## The runner checkpoints to JSONL, while day 3 keeps SQLite

- **Context.** Both need resumability, but the runner is generic and day 3's store is not.
- **Options.** One store for both; JSONL in the runner and SQLite in the days that want queries.
- **Choice.** Append-only JSONL in the runner, `checkpoint=None` when day 3 calls it.
- **Why.** A generic runner must not know the shape of what it is running — day 3's SQLite resume key
  is an eight-column tuple, which is right for day 3 and useless to anything else. Append-only also
  means a crash costs one line, and a torn final line is skipped rather than fatal. Passing
  `checkpoint=None` is what proves the abstraction composes instead of being all-or-nothing.
- **What would change my mind.** Needing cross-run queries during a run rather than after it, which
  would justify the write-ahead-log-plus-database shape real labelling pipelines use.

---

## Resume retries failures but never re-runs successes

- **Context.** A resumed run has to decide what a previously recorded failure means.
- **Options.** Skip everything already recorded; retry everything; retry only failures.
- **Choice.** Retry failures, skip successes, with `retry_failed=False` available for the other case.
- **Why.** The usual cause of a failure is a rate limit or a timeout and the usual fix is running it
  again, so treating failures as final means manually re-driving them. Successes are never worth
  re-paying for.
- **What would change my mind.** A deterministic failure — a malformed input that will fail every
  time — where retrying is pure cost. That is what the flag is for.

---

## The schema raises on bad input instead of repairing it

- **Context.** Models answer optional fields with `"unknown"`. The validator could coerce that to
  `None`.
- **Options.** Coerce and move on; raise and record.
- **Choice.** Raise. `mode="before"` turns `""` into `None`, but a placeholder word is a violation.
- **Why.** Coercion would make Saturday's violation-rate table measure the validator rather than the
  model, and it would report a beautiful 0% while the model was ignoring the contract. A repair loop
  is allowed to fix things; a schema is not. The two roles have to stay separate or the measurement
  is meaningless.
- **What would change my mind.** Production, where the goal is a filled database rather than a
  measurement — there, coerce, but count what you coerced.

---

## The benchmark has a fake arm

- **Context.** Measuring whether raising concurrency helps.
- **Options.** Time the real calls at each level; add a second arm with no external dependency.
- **Choice.** Both — `asyncio.sleep` alongside real Ollama calls.
- **Why.** "2.6× at concurrency 8" is ambiguous on its own: a saturated server and a badly written
  harness look identical. The fake arm holding 100% efficiency to 32 is what licenses blaming the
  server. Reach for this shape whenever benchmarking anything with a remote dependency.
- **What would change my mind.** Nothing. It cost twenty lines and it is the difference between a
  number and a conclusion.

---

## Errors are classified by every violation, not the first

- **Context.** A response can break the schema in several ways at once, and Pydantic reports them all.
- **Options.** Take `errors()[0]`; record the whole set.
- **Choice.** The whole set, matched on distinctive substrings, most specific first.
- **Why.** Taking the first error under-counted, and worse, the two validators that both say "use
  null" collided: matching the shared phrase first filed all 22 empty-`dimensions` errors as
  placeholder errors, reporting a whole failure mode as zero across three runs. There is a regression
  test pinning the two messages apart now.
- **What would change my mind.** Matching on message substrings is fragile regardless. Pydantic's
  `ctx` and a custom error code per validator would be sturdier, and is the right fix if this
  classifier grows past a handful of rules.

---

## Ollama runs on the host; the container is opt-in behind a profile

- **Context.** Day 6 stands up a compose stack, and the phase-0 done bar asks for Postgres, Qdrant
  and Ollama from one command. Docker on macOS runs a Linux VM with no path to the Metal GPU.
- **Options.** Ollama as a default compose service; on the host only; on the host by default with a
  containerised service behind a compose profile.
- **Choice.** Host by default, `--profile full` for the containerised one, both measured.
- **Why.** A containerised Ollama on this machine is CPU-only, and the gap is 4.02x — 54.0 tok/s
  against 13.4 on identical weights. Every later day's model calls would pay that. The profile keeps
  the stack complete for anyone cloning it onto Linux, and running both at once is what let the
  number be measured rather than assumed. The literal done bar therefore needs the flag, which is
  stated in the README rather than quietly ignored.
- **What would change my mind.** A Linux host with an NVIDIA GPU, where `nvidia-container-toolkit`
  and `--gpus all` put the accelerator inside the container and the reason for the split disappears.

---

## Postgres+pgvector and Qdrant both run, from the start

- **Context.** P1 is hybrid retrieval, and these two answer it differently. Day 4 already showed
  dense losing to BM25 by 1.000 to 0.006 on identifier queries, so P1 needs both halves.
- **Options.** pgvector only; Qdrant only; both.
- **Why.** Postgres does lexical and vector in one engine with no cross-store join; Qdrant has the
  better ANN implementation and filters during search rather than after it. Which of those matters
  more at 500k products is the question P1 answers, and it cannot be answered by whichever one got
  installed. Idle cost is roughly 30 MB and 150 MB, which is not a reason to prejudge it.
- **Choice.** Both, as default services.
- **What would change my mind.** P1 finding pgvector's HNSW sufficient at the real corpus size — at
  which point the second store is complexity kept for its own sake, and the honest move is to drop
  it and say why.

---

## The virtualenv is copied as its own layer, and the package is not installed

- **Context.** The textbook multi-stage build ends `COPY --from=builder /app /app`. `docker history`
  shows that is a single 244 MB layer holding the virtualenv and the source together.
- **Options.** Keep the textbook build; copy the venv and the source as separate layers, with the
  package on PYTHONPATH rather than installed.
- **Choice.** Separate layers. `docker-compose.yml` builds `Dockerfile.split`; `Dockerfile` stays in
  the repo as the measured comparison.
- **Why.** A one-line source edit rebuilds in 1.0s instead of 9.3s, because the venv layer is
  byte-identical across source edits and stays cached. Layer ordering alone did not deliver this —
  it saved the dependency resolution and then handed the saving back at the stage boundary, which is
  the part the usual advice leaves out. The images are identical in size, 530 MB unpacked.
- **What would change my mind.** `phase0` gaining a console entry point or calling
  `importlib.metadata` — it is on PYTHONPATH here, so it has no distribution metadata in the image.
  There is a live seam in that the package is installed locally and not in the image, and a test
  that passes under `uv run` could still fail in the container.

---

## Qdrant is spoken to over REST, not through qdrant-client

- **Context.** The healthcheck creates a collection, upserts points and runs a filtered search.
- **Options.** `qdrant-client`; raw REST through the httpx client already in the tree.
- **Choice.** REST.
- **Why.** Phase 0's rule is to meet the protocol before the SDK, and the surface used here is three
  endpoints. The client also pulls in grpcio, which is one of the heavier wheels to resolve for a
  second architecture — a cost paid on every cross-arch build for convenience not yet needed.
- **What would change my mind.** P1, where batch upserts of 500k vectors want the client's batching
  and retry behaviour rather than a hand-rolled version of it.

---

## The grid varies products, not repetitions

- **Context.** The spec says "3 models x 3 prompt styles x 100 runs". "100 runs" is ambiguous
  between 100 samples of one product and one sample each of 100 products.
- **Options.** One product sampled 100 times; 100 distinct products once each; 20 products x 5.
- **Choice.** 100 distinct products, one call each, and the *same* 100 products in every cell.
- **Why.** At temperature 0 the repeated-sampling version collapses — most cells return the same
  answer 100 times, so every cell reads 0% or 100% and the table has no dynamic range. A rate over
  distinct inputs is a number that generalises and carries a binomial confidence interval. Holding
  the product set fixed across cells makes the model and style comparisons paired rather than
  independent, which is free precision: any difference between two cells cannot be the products.
- **What would change my mind.** A question about output stability rather than schema compliance —
  "does this model answer the same way twice" is day 3's question, and it needs the other design.

---

## A violation is a violation whether or not a schema could express it

- **Context.** A response can be perfect JSON of exactly the right shape and still say footwear with
  no size, `"colour": "unknown"`, or a dimensions object whose every measurement is null.
- **Options.** Headline on "matches the schema shape", with semantics in a separate column; headline
  on "valid document", meaning parses *and* matches *and* satisfies the cross-field rules.
- **Choice.** Valid document. All three layers must pass to count.
- **Why.** The shape-only definition is the one vendors report, and it is the definition under which
  constrained decoding looks perfect. Counting the semantic failures is what makes the ceiling
  visible: a grammar enforces exactly the JSON-Schema-expressible subset of the contract, so under
  the stricter definition the native column stops improving and the remaining failures concentrate
  in rules no grammar can state. That is the finding; the looser definition hides it.
- **What would change my mind.** A consumer that only needs the shape — if downstream code treats
  `"unknown"` and `null` identically, the semantic rules are not part of the contract and should not
  be in the model.

---

## The repair loop is measured against a no-feedback control

- **Context.** The loop is validate, feed the errors back, retry twice. The natural metric is "the
  repair loop recovered X% of failures".
- **Options.** Report the recovery rate alone; run a second arm that retries without showing the
  errors and report both.
- **Choice.** Two arms. Both start from the same failed first attempt; one sees the validation
  errors, the other only sees the original prompt again.
- **Why.** A second attempt recovers some failures on its own, so the recovery rate alone cannot
  distinguish "the error text worked" from "another sample worked". Without the control, a repair
  loop with the feedback silently disconnected would still report a plausible number — and it would
  be the same number. The control costs roughly a third more calls and is the only thing that makes
  the headline attributable.
- **What would change my mind.** Nothing about this experiment. In production the control is waste;
  it exists to justify the strategy once, not to run alongside it forever.

---

## Retries resample, and the first attempt does not

- **Context.** Both arms retry at seeds 43 and 44 while the first attempt uses seed 42.
- **Options.** Everything at temperature 0, varying only the seed; retries at a temperature that
  actually samples; the control at a different temperature from the repair arm.
- **Choice.** First attempt at temperature 0, every retry in *both* arms at 0.7.
- **Why.** Temperature 0 is greedy decoding and greedy decoding ignores the seed. Measured on this
  machine: llama3.2 at temperature 0 returns 1 distinct output across seeds 42/43/44, and 3 distinct
  outputs at 0.7. So the seed-only control was 0% by construction — it proved the feedback effective
  no matter what the feedback contained, which is the exact failure the control was added to
  prevent. Groq returned 2 of 3 distinct at temperature 0, so the cloud control was measuring
  incidental non-determinism rather than a controlled resample. The first attempt stays greedy
  because tables 1 and 2 are a deterministic measurement of the mechanisms; only the arms need heat,
  and they need the same heat or the comparison becomes feedback-and-sampling against sampling.
- **What would change my mind.** A provider whose temperature 0 is genuinely stochastic across
  seeds, where the resample is real without raising the temperature.

---

## Violations are classified by error location, not by message text

- **Context.** Every custom validator raises `ValueError`, so Pydantic reports them all as
  `value_error` and something has to tell them apart.
- **Options.** Match substrings of the validator messages, as day 5 does; key on the `(loc, type)`
  pair Pydantic attaches to each error.
- **Choice.** `(loc, type)`. No message is ever read.
- **Why.** Day 5's own docstring records what substring matching cost: the placeholder rule and the
  empty-dimensions rule both say "use null", the shorter phrase was tested first, and 22
  nested-object failures were filed as placeholder failures — a whole failure mode reported as zero.
  Probing the real model shows `(loc, type)` is fully determining: `('asin',)`, `('colour',)`,
  `('dimensions',)` and `()` each identify exactly one rule. Rewording a validator can no longer
  re-bucket anything.
- **What would change my mind.** Two custom validators on the same field, which would put two rules
  at one `loc` and force a tiebreak. The honest fix then is `PydanticCustomError` with an explicit
  code, not a return to reading messages.

---

## Prose-wrapped JSON is salvaged for analysis and counted as a violation

- **Context.** Small models answer "Here you go:" and then a fenced code block. Strictly that does
  not parse.
- **Options.** Strict `json.loads` only; strip fences silently and forgive it; strip fences and count
  it.
- **Choice.** Salvage from the first `{` to the last `}`, record `json_in_prose` as a violation, and
  give it its own row in the report.
- **Why.** Strictness would measure the harness — every prose-wrapped response becomes one opaque
  failure and nothing inside it is ever examined, so the prompt column turns into a single number
  with no breakdown. Forgiveness would measure a different contract than the one the system prompt
  states. Doing both separates "cannot follow the schema" from "cannot stop saying hello", which
  need different fixes. Truncated JSON is deliberately *not* repaired: a response cut off by
  `num_predict` is a length failure, and inventing a closing brace would relabel it as a violation
  of whichever field happened to be missing.
- **What would change my mind.** A production consumer that already strips fences, in which case
  `json_in_prose` is a cosmetic property of the transport and does not belong in the headline.

---

## Capabilities are probed at startup, not declared in config

- **Context.** The grid assumes nine cells. `gemma3:4b` answers `tools` with HTTP 400, "does not
  support tools".
- **Options.** Hard-code the supported combinations; drop the tool style entirely so the grid stays
  square; probe each cell with one throwaway call before the run.
- **Choice.** Probe, and write the result to `capabilities.json` with the provider's own error text.
- **Why.** Which enforcement mechanisms a model offers is a result, not configuration — it belongs
  in the output next to the numbers rather than in a comment that ages badly the day Ollama ships a
  tool template for Gemma. It also lets the report print *why* a cell is empty instead of a blank.
  Only transport-level refusals count as unsupported: a model that accepts `tools` and then answers
  in prose is supported and bad at it, which is data.
- **What would change my mind.** A provider charging for the probe calls, at which point the
  capability map wants caching with an explicit refresh rather than a probe per run.

---

## The native column is three different guarantees under one heading

- **Context.** "Native structured output" is the third style. Each provider means something
  different by it.
- **Options.** Report one native column; drop the providers that differ; report the column with the
  guarantee attached per target.
- **Choice.** Keep the column, attach the guarantee. Ollama sends `format=<the whole schema>`, which
  constrains the sampler to the grammar. Groq's `llama-3.3-70b-versatile` answers `json_schema` with
  a 400 and supports only `response_format: json_object`, which guarantees the bytes parse and says
  nothing about the document.
- **Why.** Averaging a grammar-constrained cell with a JSON-mode cell produces a number for a
  mechanism that does not exist, which is day 6's image-size mistake in a new place — two honest
  measurements of different things, reported as one. `openai/gpt-oss-20b` is the inverse case and
  makes the point sharper: it accepts `json_schema` and *fails* a forced `tool_choice` with "model
  did not call a tool". No model on this key supports both, so the cloud arm cannot be chosen to
  make the columns comparable.
- **What would change my mind.** Groq enabling `json_schema` on the 70b, which would make the two
  native cells the same mechanism and the footnote unnecessary.

---

## Ollama runs at concurrency 1, and the reason is a measurement

- **Context.** The runner takes a concurrency cap per target. Day 5's shape suggests 8.
- **Options.** Match the provider's documented parallelism; tune for throughput; set it to 1.
- **Choice.** 1 for both Ollama targets, 4 for Groq.
- **Why.** Measured here, warm model, same prompt: gemma3 does 28.0 tok/s aggregate at concurrency 1
  and 32.1 at 4, while per-call p50 goes from 2287 ms to 6069 ms; llama3.2 does 37.2 against 43.4,
  with p50 from 3517 ms to 9074 ms. One daemon and one GPU means the requests are effectively
  serialised — four in flight buy 15% aggregate throughput and each waits three times longer. That
  wait lands in `latency_ms`, so table 4 would have reported the number of workers this script
  chose, labelled as how long the model takes. 15% of the wall clock is a fair price for a latency
  column that means what it says. Groq keeps 4 because those are independent server-side workers.
- **What would change my mind.** `OLLAMA_NUM_PARALLEL` set high on a machine with the memory to
  honour it, where the parallelism would be real and the measurement would have to be redone.

---

## One database row per dispatch, not per attempt

- **Context.** A failed episode can make five calls: one first try, up to two repairs, up to two
  controls. The unique index in `phase0.store` has no `attempt` column.
- **Options.** Add `attempt` to the index and migrate day 3; write one row per attempt into a new
  table; write one row per dispatch and put the trail in `extra_json`.
- **Choice.** Three rows at most — `first`, `repair`, `control` — with the per-attempt trail as JSON
  in `extra`.
- **Why.** This is what `store.py` reserved `extra_json` for, so `valid_first_try`, `attempts` and
  `error_type` land without a migration and day 3's rows keep working. Two repair attempts on one
  cell would otherwise collide under `INSERT OR REPLACE` and the second would silently overwrite the
  first. The row is the episode; the attempts are its detail.
- **What would change my mind.** Wanting to query across attempts — "how often does attempt 2 fix
  what attempt 1 did not" is answerable from the trail only by unpacking JSON in Python, and if that
  question becomes routine the attempts want their own table.

---

## The tool-call repair turn is provider-shaped in both directions

- **Context.** Repairing a tool-style failure means replaying the model's own call and answering it.
- **Options.** One transcript shape for both providers; separate shapes per provider.
- **Choice.** Separate. Ollama takes `arguments` as a decoded object and a `tool` message carrying
  `tool_name`; OpenAI-compatible APIs take `arguments` as a string and a `tool` message carrying
  `tool_call_id`.
- **Why.** Sending Ollama the string produces HTTP 400, "Value looks like object, but can't find
  closing '}' symbol" — a parse error describing a string that parses perfectly well, and it killed
  three episodes on the first smoke run. `parse_reply` already normalises the inbound direction to a
  string so the classifier sees one type; `repair_turns` has to denormalise on the way back, and the
  two must stay in step. A repair is also not "the errors appended as a user turn": a function call
  is answered by a tool result, and OpenAI rejects an assistant message with `tool_calls` that is
  not followed by a matching `tool` message.
- **What would change my mind.** Nothing on these two providers. A third provider means a third
  branch, and at that point the shapes want a small adapter per provider rather than two
  `if provider ==` tests.

---

## `store.py` and `schema.py` moved into the `phase0` package

- **Context.** The weekend needs day 3's SQLite schema and day 5's Pydantic model. Both live in day
  directories, which are scripts on `sys.path[0]` rather than importable modules.
- **Options.** `sys.path` manipulation from the weekend; copy the files; move them into `phase0`.
- **Choice.** Move. Day 3 and day 5 import them from `phase0` now, and neither file changed.
- **Why.** The same move `runner.py` made on day 5, for the same reason: the second caller is what
  turns a script into a module. Copying would fork the schema the first time a validator changes,
  and the whole point of the weekend's table is that it validates against the model day 5 built.
- **What would change my mind.** `phase0` accumulating domain objects that are not infrastructure —
  `ProductAttributes` is already a borderline case, and P1 wanting a different product schema would
  argue for a `phase0/domain/` split rather than a flat package.

---

## `weekend_structured` is a package, and days 1-6 are not

- **Context.** `day3_sampling/config.py` and `weekend_structured/config.py` are both placed on
  `sys.path` by pytest. `import config` resolved to whichever directory was collected first.
- **Options.** Rename this directory's `config.py`; make the directory a package; convert every day
  to a package.
- **Choice.** Package, with `-m` invocation: `uv run python -m weekend_structured.run`.
- **Why.** The whole weekend failed to import under `uv run pytest` while passing when run alone,
  which is the worst version of this bug — it depends on collection order. pyproject.toml predicted
  the general case on day 5, that scripts importing each other by accident of `sys.path[0]` stop
  working once two days share code; this is the other half of it, where two days that share nothing
  but a *filename* collide just as hard. Renaming would have worked today and left the same trap for
  the next `report.py`.
- **What would change my mind.** Nothing here, but the earlier days are now inconsistent with this
  one. Converting them is a day of churn for no new capability, so the inconsistency stands and this
  entry is the explanation.
