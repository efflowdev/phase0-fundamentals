"""The tables, as markdown, from whatever is in the database.

    uv run python -m weekend_structured.report
    uv run python -m weekend_structured.report --db weekend_structured/runs.db

Separated from collection for day 3's reason, which held up twice since: a
scorer definition that turns out to be wrong should cost a query, not 900 calls.
Everything here is derived from `runs` rows and nothing here can be recomputed
only by re-running the models.

Four tables, and the second is the one the day exists for. Table 1 says which
mechanism wins; table 2 says the winner cannot go further, because the violations
it has left are the ones no grammar can express.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from typing import Any

from .classify import EXPRESSIBLE, RULES
from .config import DEFAULT_DB, EXPERIMENT, STYLES

FIRST, REPAIR, CONTROL = "first", "repair", "control"


def load(conn: sqlite3.Connection, experiment: str) -> list[dict[str, Any]]:
    rows = []
    for row in conn.execute(
        "SELECT * FROM runs WHERE experiment = ? ORDER BY model, prompt_style, run_idx",
        (experiment,),
    ):
        record = dict(row)
        record["extra"] = json.loads(record["extra_json"] or "{}")
        rows.append(record)
    return rows


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "—"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    line = "|" + "|".join(headers) + "|"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = ["|" + "|".join(cells) + "|" for cells in rows]
    return "\n".join([line, rule, *body])


def headline(records: list[dict[str, Any]], models: list[str], styles: list[str]) -> str:
    """Valid document on the first try — all three layers, per the decision.

    Parses, matches the schema, and satisfies the cross-field rules. A response
    that is perfect JSON of the right shape and says footwear with no size is
    counted here as a failure, which is what makes the native column's ceiling
    visible in table 2 rather than hidden in a footnote.
    """
    firsts = [r for r in records if r["dispatch"] == FIRST]
    grid: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for record in firsts:
        grid[(record["model"], record["prompt_style"])].append(
            bool(record["extra"].get("valid"))
        )

    body = []
    for model in models:
        cells = [model]
        for style in styles:
            outcomes = grid.get((model, style), [])
            cells.append(
                _pct(sum(outcomes), len(outcomes)) if outcomes else "n/a"
            )
        overall = [ok for style in styles for ok in grid.get((model, style), [])]
        cells.append(f"**{_pct(sum(overall), len(overall))}**")
        body.append(cells)

    footer = ["**all models**"]
    for style in styles:
        outcomes = [ok for model in models for ok in grid.get((model, style), [])]
        footer.append(f"**{_pct(sum(outcomes), len(outcomes))}**")
    every = [ok for outcomes in grid.values() for ok in outcomes]
    footer.append(f"**{_pct(sum(every), len(every))}**")
    body.append(footer)

    return _table(["model", *styles, "all styles"], body)


def violations(records: list[dict[str, Any]], styles: list[str]) -> str:
    """Rule x style, and whether a grammar could ever have caught it.

    Counts violations, not responses — one response that both said "unknown" and
    emitted an all-null nested object appears in two rows. The column totals are
    therefore larger than the failure count, and that is the honest denominator
    for "which violations does repair recover".
    """
    firsts = [r for r in records if r["dispatch"] == FIRST]
    counts: dict[str, Counter] = {style: Counter() for style in styles}
    for record in firsts:
        style = record["prompt_style"]
        if style not in counts:
            continue
        for rule in record["extra"].get("rules", []):
            counts[style][rule] += 1

    body = []
    for rule, (expressible, gloss) in RULES.items():
        cells = [f"`{rule}`", "yes" if expressible else "**no**"]
        total = 0
        for style in styles:
            value = counts[style][rule]
            total += value
            cells.append(str(value) if value else "·")
        if total == 0:
            continue
        cells.append(gloss)
        body.append(cells)

    totals = ["**total**", ""]
    for style in styles:
        totals.append(f"**{sum(counts[style].values())}**")
    totals.append("")
    body.append(totals)

    inexpressible = ["*of which inexpressible*", ""]
    for style in styles:
        inexpressible.append(
            f"**{sum(v for k, v in counts[style].items() if k not in EXPRESSIBLE)}**"
        )
    inexpressible.append("")
    body.append(inexpressible)

    return _table(["rule", "in schema", *styles, ""], body)


def repair(records: list[dict[str, Any]], models: list[str], styles: list[str]) -> str:
    """First try, then the two arms — and the gap between them is the finding.

    `control` is the same prompt resampled at the same seeds with no error text.
    Without that column "repair recovers X%" cannot be distinguished from
    "a second sample recovers X%", and the difference is whether the repair loop
    is worth its latency.
    """
    by_cell: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_cell[(record["model"], record["prompt_style"])][record["dispatch"]].append(
            record
        )

    body = []
    for model in models:
        for style in styles:
            cell = by_cell.get((model, style))
            if not cell:
                continue
            firsts = cell[FIRST]
            failed = [r for r in firsts if not r["extra"].get("valid")]
            if not firsts:
                continue

            fixed = {
                dispatch: sum(1 for r in cell[dispatch] if r["extra"].get("valid"))
                for dispatch in (REPAIR, CONTROL)
            }
            # Mean attempts over every episode: one for each that passed
            # immediately, plus whatever the repair arm actually spent. Averaging
            # the repair arm alone would report the cost of the failures only.
            attempts = [
                r["extra"].get("attempts", 1) for r in firsts if r["extra"].get("valid")
            ] + [r["extra"].get("attempts", 1) for r in cell[REPAIR]]

            body.append(
                [
                    model,
                    style,
                    _pct(len(firsts) - len(failed), len(firsts)),
                    str(len(failed)),
                    f"{fixed[REPAIR]} ({_pct(fixed[REPAIR], len(failed))})",
                    f"{fixed[CONTROL]} ({_pct(fixed[CONTROL], len(failed))})",
                    _pct(len(firsts) - len(failed) + fixed[REPAIR], len(firsts)),
                    f"{sum(attempts) / len(attempts):.2f}" if attempts else "—",
                ]
            )

    return _table(
        [
            "model",
            "style",
            "valid 1st",
            "failures",
            "repair fixed",
            "control fixed",
            "after repair",
            "mean attempts",
        ],
        body,
    )


def cost(records: list[dict[str, Any]], models: list[str], styles: list[str]) -> str:
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        by_cell[(record["model"], record["prompt_style"])].append(record)

    body = []
    for model in models:
        for style in styles:
            cell = by_cell.get((model, style), [])
            if not cell:
                continue
            latencies = sorted(r["latency_ms"] for r in cell)
            out = [r["completion_tokens"] for r in cell]
            inp = [r["prompt_tokens"] for r in cell]
            body.append(
                [
                    model,
                    style,
                    f"{latencies[len(latencies) // 2]:.0f}",
                    f"{latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]:.0f}",
                    f"{sum(inp) / len(inp):.0f}",
                    f"{sum(out) / len(out):.0f}",
                ]
            )
    return _table(
        ["model", "style", "p50 ms", "p95 ms", "mean in-tok", "mean out-tok"], body
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--experiment", default=EXPERIMENT)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    records = load(conn, args.experiment)
    conn.close()

    if not records:
        print(f"no rows for experiment {args.experiment!r} in {args.db}")
        return 1

    models = sorted({r["model"] for r in records})
    styles = [s for s in STYLES if any(r["prompt_style"] == s for r in records)]
    episodes = len([r for r in records if r["dispatch"] == FIRST])

    print(f"# Structured output — {episodes} episodes, {len(records)} rows\n")
    print("## 1. Valid document on the first try\n")
    print(headline(records, models, styles))
    print("\n## 2. Which violations, and could a grammar have caught them\n")
    print(violations(records, styles))
    print("\n## 3. Repair against a no-feedback control\n")
    print(repair(records, models, styles))
    print("\n## 4. What it cost\n")
    print(cost(records, models, styles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
