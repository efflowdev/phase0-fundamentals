"""The grid: 3 targets x 3 styles x 100 products, resumable.

    uv run python -m weekend_structured.run --probe        # capabilities only
    uv run python -m weekend_structured.run --n 5          # smoke test
    uv run python -m weekend_structured.run                # the real run
    uv run python -m weekend_structured.run                # again: pays for nothing

Three things this file decides, none of which is the loop itself.

**Capabilities are probed, not declared.** `gemma3:4b` answers `tools` with a 400
— "does not support tools" — so the 3x3 is really eight cells, and which cell is
missing is a fact about the model that belongs in the results rather than in a
comment that rots. One cheap call per (target, style) before the run, written to
`capabilities.json` so `report.py` can print *why* a cell is empty instead of a
blank.

**One `run_all` per target, not one for all of them.** Concurrency is a property
of the provider — 4 for Ollama because that is `OLLAMA_NUM_PARALLEL`, 4 for Groq
because the free tier is ~30 req/min — and a single pool would have to take the
minimum for everyone.

**SQLite is written per target, from the checkpoint.** The runner's JSONL is the
crash-safe record; the database is the queryable one. Loading after each target
means a Ctrl-C costs at most the target in flight, and re-running loads the
resumed rows again into an `INSERT OR REPLACE` on the same unique cell, which is
a no-op rather than a duplicate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from phase0 import store
from phase0.runner import run_all, summarize
from phase0.schema import ProductAttributes

from .classify import RULES
from .config import (
    BASE_SEED,
    CHECKPOINT_DIR,
    DEFAULT_DB,
    EXPERIMENT,
    N_PRODUCTS,
    STYLES,
    TARGETS,
    Target,
    load_env,
)
from .loop import episode, to_rows

CACHE = Path(__file__).resolve().parents[1] / "day4_search" / "cache" / "esci_rows.jsonl"
CAPABILITIES = Path(__file__).resolve().parent / "capabilities.json"


def load_products(limit: int) -> list[dict[str, str]]:
    """The same first-N-distinct selection day 5 used, deliberately.

    Day 5's `extract.py` ran this corpus with no repair loop at all, so its
    `runs/extract.jsonl` is an untreated baseline for the same products — but
    only if the selection is identical. Changing the sampling here would silently
    break a comparison that costs nothing to keep.
    """
    seen: dict[str, dict[str, str]] = {}
    for line in CACHE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        asin = row.get("product_id") or ""
        title = (row.get("product_title") or "").strip()
        if not asin or not title or asin in seen:
            continue
        seen[asin] = {"asin": asin, "title": title}
        if len(seen) >= limit:
            break
    return list(seen.values())


async def probe(
    client: httpx.AsyncClient, targets: tuple[Target, ...], styles: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """One throwaway call per cell, to find out what the provider will accept.

    Only transport-level refusals count as unsupported. A model that accepts
    `tools` and then answers in prose is *supported and bad at it*, which is a
    result, not a missing cell.
    """
    schema = ProductAttributes.model_json_schema()
    product = {"asin": "B01N5IB20Q", "title": "Nike Air Zoom Pegasus 38 running shoe"}
    found: dict[str, dict[str, Any]] = {}

    for target in targets:
        found[target.id] = {}
        for style in styles:
            try:
                await episode(
                    client,
                    target,
                    style,
                    product,
                    schema,
                    ProductAttributes,
                    base_seed=BASE_SEED,
                    max_repairs=0,
                )
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text[:200].strip()
                found[target.id][style] = {"supported": False, "reason": detail}
                print(f"  {target.id:22} {style:7} unsupported — {detail}")
                continue
            except Exception as exc:  # noqa: BLE001 — a probe reports, never raises
                found[target.id][style] = {"supported": False, "reason": str(exc)[:200]}
                print(f"  {target.id:22} {style:7} unsupported — {exc}")
                continue
            found[target.id][style] = {"supported": True, "reason": ""}
            print(f"  {target.id:22} {style:7} ok")
    return found


def to_run(
    target: Target, style: str, product: dict[str, str], run_idx: int, row: dict[str, Any]
) -> store.Run:
    attempt = row["attempt"]
    return store.Run(
        experiment=EXPERIMENT,
        provider=target.provider,
        model=target.id,
        prompt_id=product["asin"],
        prompt_style=style,
        # The attempt's own temperature, not the experiment constant: the first
        # try is greedy at 0 and both arms retry at 0.7, and a row that claimed 0
        # for all three would misreport what produced the output it stores.
        temperature=attempt["temperature"],
        seed=attempt["seed"],
        dispatch=row["dispatch"],
        run_idx=run_idx,
        output=row["output"],
        finish_reason=attempt["finish_reason"],
        prompt_tokens=attempt["prompt_tokens"],
        completion_tokens=attempt["completion_tokens"],
        latency_ms=attempt["latency_ms"],
        error=None,
        extra=row["extra"] | {"title": product["title"]},
    )


async def run_target(
    client: httpx.AsyncClient,
    target: Target,
    styles: tuple[str, ...],
    products: list[dict[str, str]],
    conn,
    args: argparse.Namespace,
) -> None:
    schema = ProductAttributes.model_json_schema()
    index = {product["asin"]: i for i, product in enumerate(products)}
    items = [(style, product) for style in styles for product in products]
    if not items:
        return

    async def work(item: tuple[str, dict[str, str]]) -> dict[str, Any]:
        style, product = item
        result = await episode(
            client,
            target,
            style,
            product,
            schema,
            ProductAttributes,
            base_seed=BASE_SEED,
            max_repairs=args.max_repairs,
        )
        # Serialised here rather than in `loop.py` because the runner
        # checkpoints whatever this returns as JSON, and it has to survive a
        # round trip through the file to be resumable.
        return {
            "style": style,
            "asin": product["asin"],
            "rows": [
                {
                    "dispatch": row["dispatch"],
                    "output": row["output"],
                    "extra": row["extra"],
                    "attempt": {
                        "seed": row["attempt"].seed,
                        "temperature": row["attempt"].temperature,
                        "latency_ms": row["attempt"].latency_ms,
                        "prompt_tokens": row["attempt"].prompt_tokens,
                        "completion_tokens": row["attempt"].completion_tokens,
                        "finish_reason": row["attempt"].finish_reason,
                    },
                }
                for row in to_rows(result)
            ],
        }

    checkpoint = CHECKPOINT_DIR / f"grid-{target.id.replace('/', '-')}.jsonl"
    results = await run_all(
        items,
        work,
        key=lambda item: f"{target.id}|{item[0]}|{item[1]['asin']}",
        concurrency=target.concurrency,
        checkpoint=checkpoint,
        resume=not args.no_resume,
        desc=target.id,
    )

    written = 0
    for result in results:
        if not result.ok or not result.value:
            continue
        payload = result.value
        product = products[index[payload["asin"]]]
        for row in payload["rows"]:
            store.insert(
                conn, to_run(target, payload["style"], product, index[payload["asin"]], row)
            )
            written += 1

    stats = summarize(results)
    print(
        f"  {target.id:22} {stats['ok']}/{stats['total']} episodes "
        f"({stats['resumed']} resumed, {stats['failed']} failed), "
        f"{written} rows, p50 {stats['p50_ms']:.0f}ms"
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=N_PRODUCTS)
    parser.add_argument("--targets", nargs="*", default=[t.id for t in TARGETS])
    parser.add_argument("--styles", nargs="*", default=list(STYLES))
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--probe", action="store_true", help="capabilities, then exit")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    load_env()
    targets = tuple(t for t in TARGETS if t.id in args.targets)
    styles = tuple(s for s in STYLES if s in args.styles)
    products = load_products(args.n)

    missing = [t.id for t in targets if t.api_key_env and not t.key]
    if missing:
        print(f"no API key for {', '.join(missing)} — skipping")
        targets = tuple(t for t in targets if t.id not in missing)

    print(f"{len(products)} products, {len(targets)} targets, {len(styles)} styles")
    print(f"rules: {len(RULES)} ({sum(1 for e, _ in RULES.values() if e)} schema-expressible)")

    async with httpx.AsyncClient() as client:
        print("probing capabilities")
        capabilities = await probe(client, targets, styles)
        CAPABILITIES.write_text(json.dumps(capabilities, indent=2) + "\n")
        if args.probe:
            return 0

        conn = store.connect(args.db)
        try:
            for target in targets:
                supported = tuple(
                    style
                    for style in styles
                    if capabilities[target.id][style]["supported"]
                )
                skipped = set(styles) - set(supported)
                if skipped:
                    print(f"  {target.id}: skipping {', '.join(sorted(skipped))}")
                await run_target(client, target, supported, products, conn, args)
        finally:
            total, failed = store.counts(conn, EXPERIMENT)
            conn.close()

    print(f"{total} rows in {args.db} ({failed} carrying an error)")
    print(f"report: uv run python -m weekend_structured.report --db {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
