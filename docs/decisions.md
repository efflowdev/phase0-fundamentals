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
