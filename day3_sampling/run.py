"""Run the sampling matrix and write every call to SQLite.

    uv run python day3_sampling/run.py --n 3 --providers ollama   # smoke test
    uv run python day3_sampling/run.py                            # the real one
    uv run python day3_sampling/run.py --resume                   # after a 429 wall

    # sampler ablation — always into its own experiment, never the baseline
    uv run python day3_sampling/run.py --providers ollama --temps 0 \
        --sampler repeat_penalty=1.0 --experiment day3-no-repeat-penalty

The matrix is providers × prompts × {0.0, 1.0} × dispatch × n, where dispatch is
{sequential, concurrent} at temperature 0 and sequential only above it. Comparing
dispatch modes at temperature 1 would measure nothing — both vary by design. At
temperature 0, with a fixed seed and identical payloads, *any* disagreement has
to come from the serving stack, and firing the 20 requests together rather than
one at a time is the cheapest way to put server-side batching under load and see
whether the disagreement rate moves with it.

Caveat to carry into the write-up: Groq's free tier rate-limits the concurrent
arm, so some of those 20 requests are retried seconds later and were never
co-batched with the rest. That biases the concurrent arm *towards* looking like
the sequential one, so a difference that survives it is real; an absence of
difference is not conclusive.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import httpx
from config import (
    DEFAULT_DB,
    DISPATCHES,
    EXPERIMENT,
    PROMPTS,
    PROMPTS_BY_ID,
    PROVIDERS,
    TEMPERATURES,
    Prompt,
    Provider,
    load_env,
    parse_sampler,
    seed_for,
)
from sample import Sample, complete, probe_logprobs
from score import tokens_to_json

from phase0 import store
from phase0.runner import run_all


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def to_run(
    experiment: str,
    provider: Provider,
    prompt: Prompt,
    temperature: float,
    dispatch: str,
    run_idx: int,
    result: Sample,
    sampler: dict[str, object] | None,
) -> store.Run:
    return store.Run(
        experiment=experiment,
        provider=provider.name,
        model=provider.model,
        prompt_id=prompt.id,
        temperature=temperature,
        dispatch=dispatch,
        run_idx=run_idx,
        output=result.output,
        seed=seed_for(temperature, run_idx),
        finish_reason=result.finish_reason,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=result.latency_ms,
        error=result.error,
        logprobs_json=tokens_to_json(result.logprobs),
        # Whatever sampler overrides produced this row travel with it. Without
        # that, two ablation runs in the same table are indistinguishable.
        extra={"sampler": sampler} if sampler else None,
    )


async def run_group(
    client: httpx.AsyncClient,
    conn,
    experiment: str,
    provider: Provider,
    prompt: Prompt,
    temperature: float,
    dispatch: str,
    indices: list[int],
    sampler: dict[str, object] | None,
) -> tuple[int, int]:
    """One cell of the matrix. Returns (completed, failed).

    Dispatch is now just a concurrency number: sequential is a semaphore of one.
    A `gather` behind `Semaphore(1)` creates every task up front but keeps a
    single request in flight, which is the property the determinism experiment
    actually cares about — so the two arms still differ in exactly the way the
    day 3 write-up claims.

    The runner supplies concurrency and failure capture only. Persistence and
    resume stay on SQLite here, because `runs.db` is the committed evidence and
    what `lens` reads; passing `checkpoint=None` is what makes the runner
    composable rather than all-or-nothing.
    """
    concurrency = 1 if dispatch == "sequential" else provider.max_concurrency

    async def one(run_idx: int) -> Sample:
        return await complete(
            client,
            provider,
            prompt.text,
            temperature=temperature,
            seed=seed_for(temperature, run_idx),
            max_tokens=prompt.max_tokens,
            sampler=sampler,
        )

    outcomes = await run_all(
        indices,
        one,
        key=str,
        concurrency=concurrency,
        checkpoint=None,
        progress=False,
    )

    results: list[tuple[int, Sample]] = []
    for run_idx, outcome in zip(indices, outcomes, strict=True):
        if outcome.ok and isinstance(outcome.value, Sample):
            results.append((run_idx, outcome.value))
        else:
            # `complete` returns errors rather than raising, so an exception
            # reaching here is a bug in the harness, not a bad response.
            results.append(
                (
                    run_idx,
                    Sample(
                        "",
                        "",
                        0,
                        0,
                        outcome.elapsed_ms,
                        None,
                        f"harness: {outcome.error}",
                    ),
                )
            )

    failed = 0
    for run_idx, result in results:
        if not result.ok:
            failed += 1
        store.insert(
            conn,
            to_run(
                experiment,
                provider,
                prompt,
                temperature,
                dispatch,
                run_idx,
                result,
                sampler,
            ),
        )
    return len(results), failed


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="samples per cell")
    parser.add_argument("--providers", default="groq,ollama")
    parser.add_argument("--prompts", default=",".join(p.id for p in PROMPTS))
    parser.add_argument(
        "--temps",
        default=",".join(f"{t:g}" for t in TEMPERATURES),
        help="temperatures to sweep",
    )
    parser.add_argument(
        "--sampler",
        default="",
        help=(
            "provider-native sampler overrides, e.g. repeat_penalty=1.0,top_k=1 "
            "for ollama or top_p=0.5 for groq"
        ),
    )
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip cells already stored without an error",
    )
    parser.add_argument("--no-probe", action="store_true")
    args = parser.parse_args()

    load_env()

    try:
        providers = [PROVIDERS[name.strip()] for name in args.providers.split(",")]
        prompts = [PROMPTS_BY_ID[pid.strip()] for pid in args.prompts.split(",")]
        temperatures = [float(t) for t in args.temps.split(",") if t.strip()]
    except (KeyError, ValueError) as exc:
        log(f"unknown provider, prompt or temperature: {exc}")
        return 2

    sampler = parse_sampler(args.sampler)
    if sampler:
        log(f"sampler overrides: {sampler}")
        if args.experiment == EXPERIMENT:
            # Otherwise the ablation lands in the same cells as the baseline and
            # INSERT OR REPLACE quietly overwrites it.
            log("refusing to write an ablation into the baseline experiment;")
            log("pass --experiment day3-<something> to keep them apart")
            return 2

    conn = store.connect(args.db)
    done = store.existing_cells(conn, args.experiment) if args.resume else set()
    if args.resume:
        log(f"resume: {len(done)} cells already stored")

    started = time.perf_counter()
    total = failed_total = skipped = 0

    async with httpx.AsyncClient() as client:
        if not args.no_probe:
            for provider in providers:
                verdict = await probe_logprobs(client, provider)
                log(f"logprobs · {provider.name:7} {provider.model:24} {verdict}")
            log("")

        for provider in providers:
            for prompt in prompts:
                for temperature in temperatures:
                    dispatches = DISPATCHES if temperature == 0.0 else ("sequential",)
                    for dispatch in dispatches:
                        indices = [
                            i
                            for i in range(args.n)
                            if (
                                args.experiment,
                                provider.name,
                                provider.model,
                                prompt.id,
                                "",
                                temperature,
                                dispatch,
                                i,
                            )
                            not in done
                        ]
                        label = (
                            f"{provider.name:7} {prompt.id:13} "
                            f"t={temperature:<4} {dispatch:10}"
                        )
                        if not indices:
                            skipped += args.n
                            log(f"{label} skipped")
                            continue

                        cell_started = time.perf_counter()
                        completed, failed = await run_group(
                            client,
                            conn,
                            args.experiment,
                            provider,
                            prompt,
                            temperature,
                            dispatch,
                            indices,
                            sampler,
                        )
                        elapsed = time.perf_counter() - cell_started
                        total += completed
                        failed_total += failed
                        note = f" · {failed} failed" if failed else ""
                        log(f"{label} {completed:3} calls  {elapsed:6.1f}s{note}")

    stored, stored_failed = store.counts(conn, args.experiment)
    conn.close()

    log("")
    log(
        f"{total} calls in {time.perf_counter() - started:.1f}s "
        f"({failed_total} failed, {skipped} skipped) → {args.db}"
    )
    log(f"table now holds {stored} rows for {args.experiment}, {stored_failed} failed")
    log("next: uv run python day3_sampling/report.py")
    return 1 if failed_total and failed_total == total else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
