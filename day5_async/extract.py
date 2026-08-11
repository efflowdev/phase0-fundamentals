"""Extract structured attributes from product titles, at eval scale.

    uv run python day5_async/extract.py --n 50
    uv run python day5_async/extract.py --n 200 --concurrency 8
    uv run python day5_async/extract.py --n 200            # resumes automatically

Every call goes through `phase0.runner`, so this is also the first real exercise
of it: bounded concurrency, a JSONL checkpoint written as results land, failures
recorded as rows, and a rerun that only pays for what is missing.

No repair loop here — a `ValidationError` is recorded and left alone. Saturday's
job is to feed the error text back to the model and measure how much of the
violation rate that recovers, and it can only measure that against the untreated
rate this file produces.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import httpx
from pydantic import ValidationError
from schema import ProductAttributes

from phase0.ollama import MODEL, chat
from phase0.runner import run_all, summarize

CACHE = (
    Path(__file__).resolve().parents[1] / "day4_search" / "cache" / "esci_rows.jsonl"
)
CHECKPOINT = Path(__file__).resolve().parent / "runs" / "extract.jsonl"

SYSTEM = (
    "You extract structured product attributes from catalogue titles. "
    "Reply with a single JSON object and nothing else. "
    "Use null for any attribute the title does not state — never the string "
    "'unknown'."
)


def load_products(limit: int) -> list[dict[str, str]]:
    """Distinct products out of day 4's committed cache.

    Reads the cache file rather than importing day 4's `corpus.build`, because
    the extractor wants raw catalogue rows and none of the query-grouping or
    500-product capping that the retrieval evaluation needs.
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


def build_prompt(product: dict[str, str]) -> str:
    schema = json.dumps(ProductAttributes.model_json_schema(), separators=(",", ":"))
    return (
        f"Product title: {product['title']}\n"
        f"ASIN: {product['asin']}\n\n"
        f"JSON Schema:\n{schema}\n\n"
        "Return the JSON object."
    )


def _name_one(error: dict) -> str:
    """Substring matching on validator messages, ordered most-specific-first.

    Both the placeholder rule and the empty-dimensions rule tell the model to
    "use null", and matching the shorter phrase first swallowed every
    empty-nested-object into the placeholder bucket — 22 of them, a whole
    failure mode reported as zero. The placeholder test is now anchored on its
    own distinctive phrase rather than on the two words they share.
    """
    message = str(error.get("msg", ""))
    if "at least one measurement" in message:
        return "empty_nested_object"
    if "is optional: use null" in message:
        return "placeholder_instead_of_null"
    if "requires a size" in message:
        return "cross_field_rule"
    if "not an ASIN" in message:
        return "bad_identifier"
    return f"field:{error.get('type', 'unknown')}"


def classify_error(exc: Exception) -> list[str]:
    """Every distinct violation in one response, not just the first.

    Pydantic reports all of them and the first version of this took
    `errors()[0]`, which quietly under-counted: a response that both wrote
    "unknown" for a colour *and* emitted an all-null dimensions object was
    filed as one placeholder problem. Saturday's job is to measure which
    violations a repair loop actually recovers, and it cannot do that against a
    breakdown that only ever saw the alphabetically-first cause.

    'Invalid JSON' and 'said unknown instead of null' need completely different
    fixes, and only one of them is a prompting problem.
    """
    if isinstance(exc, json.JSONDecodeError):
        return ["unparseable_json"]
    if isinstance(exc, ValidationError):
        return sorted({_name_one(error) for error in exc.errors()})
    return [type(exc).__name__]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--constrained", action="store_true", help="Ollama `format`")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    products = load_products(args.n)
    format_schema = ProductAttributes.model_json_schema() if args.constrained else None

    async with httpx.AsyncClient() as client:

        async def work(product: dict[str, str]) -> dict[str, object]:
            raw = await chat(
                client,
                build_prompt(product),
                model=args.model,
                system=SYSTEM,
                format_schema=format_schema,
            )
            try:
                parsed = ProductAttributes.model_validate_json(raw)
            except (ValidationError, json.JSONDecodeError) as exc:
                return {
                    "valid": False,
                    # Full text, not a 400-char slice: truncating mid-JSON makes
                    # the checkpoint useless for re-analysis, and re-analysis
                    # without re-paying for the calls is the point of storing it.
                    "error_types": classify_error(exc),
                    "raw": raw,
                }
            return {"valid": True, "attributes": parsed.model_dump(mode="json")}

        results = await run_all(
            products,
            work,
            key=lambda p: p["asin"],
            concurrency=args.concurrency,
            checkpoint=args.checkpoint,
            resume=not args.no_resume,
            desc=f"extract x{args.concurrency}",
        )

    summary = summarize(results)
    payloads = [r.value for r in results if r.ok and isinstance(r.value, dict)]
    valid = [p for p in payloads if p.get("valid")]
    invalid = [p for p in payloads if not p.get("valid")]
    # Counted per response, so the percentages below can exceed the violation
    # rate — one response is allowed to break the schema in three ways.
    errors = Counter(
        error_type for p in invalid for error_type in p.get("error_types", ["?"])
    )

    print(f"\n{summary['total']} products · {summary['resumed']} resumed")
    print(f"transport failures: {summary['failed']}")
    print(f"p50 {summary['p50_ms']:.0f} ms · p95 {summary['p95_ms']:.0f} ms")
    if payloads:
        rate = 100 * len(invalid) / len(payloads)
        print(
            f"\nschema violation rate: {rate:.1f}%  ({len(valid)}/{len(payloads)} ok)"
        )
        print(f"distinct violations across {len(invalid)} bad responses:")
    for error_type, count in errors.most_common():
        print(f"  {count:4}  {error_type}")
    print(f"\ncheckpoint: {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
