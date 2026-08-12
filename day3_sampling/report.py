"""Read the runs table and print the measurements. No interpretation.

    uv run python day3_sampling/report.py
    uv run python day3_sampling/report.py --markdown       # paste into the README
    uv run python day3_sampling/report.py --no-embed       # skip fastembed

Scoring happens here rather than in `run.py` on purpose: the database holds raw
logprobs and raw outputs, so changing the definition of a metric is a re-read,
not a re-run.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict

import numpy as np
import variance as var
from config import DEFAULT_DB, EXPERIMENT
from score import Confidence, score, tokens_from_json

from phase0 import store


def table(headers: list[str], rows: list[list[str]], markdown: bool) -> str:
    if not rows:
        return "  (no rows)"
    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    if markdown:
        head = (
            "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
        )
        rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
        body = [
            "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |"
            for row in rows
        ]
        return "\n".join([head, rule, *body])

    head = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * w for w in widths)
    body = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(row)) for row in rows]
    return "\n".join([head, rule, *body])


def group_rows(rows: list[sqlite3.Row]) -> dict[tuple, list[sqlite3.Row]]:
    groups: dict[tuple, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        key = (row["provider"], row["prompt_id"], row["temperature"], row["dispatch"])
        groups[key].append(row)
    return dict(sorted(groups.items()))


def variance_section(groups: dict[tuple, list[sqlite3.Row]], embed: bool) -> str:
    headers = [
        "provider",
        "prompt",
        "temp",
        "dispatch",
        "ok",
        "err",
        "distinct",
        "dist_norm",
        "modal%",
        "modal_norm%",
        "prefix",
        "cosine",
        "chars",
    ]
    rows: list[list[str]] = []
    for (provider, prompt_id, temperature, dispatch), items in groups.items():
        good = [r for r in items if r["error"] is None]
        errors = len(items) - len(good)
        outputs = [r["output"] for r in good]
        if not outputs:
            rows.append(
                [provider, prompt_id, f"{temperature:g}", dispatch, "0", str(errors)]
                + ["-"] * 7
            )
            continue

        embeddings = None
        if not embed or len(set(outputs)) == 1:
            embeddings = np.zeros((1, 384), dtype=np.float32)
        summary = var.summarize(outputs, embeddings)

        rows.append(
            [
                provider,
                prompt_id,
                f"{temperature:g}",
                dispatch,
                str(len(outputs)),
                str(errors),
                str(summary.distinct),
                str(summary.distinct_normalized),
                f"{summary.modal_share * 100:.0f}",
                f"{summary.modal_share_normalized * 100:.0f}",
                str(summary.common_prefix_chars),
                "-"
                if embeddings is not None and not embed
                else f"{summary.mean_pairwise_cosine:.3f}",
                f"{summary.mean_chars:.0f}",
            ]
        )
    return table(headers, rows, MARKDOWN)


def confidence_section(groups: dict[tuple, list[sqlite3.Row]]) -> str:
    headers = [
        "provider",
        "prompt",
        "temp",
        "dispatch",
        "n",
        "tokens",
        "mean_p",
        "perplexity",
        "top_margin",
        "chosen_margin",
        "off_top%",
    ]
    rows: list[list[str]] = []
    for (provider, prompt_id, temperature, dispatch), items in groups.items():
        scores: list[Confidence] = []
        for row in items:
            if row["error"] is not None:
                continue
            tokens = tokens_from_json(row["logprobs_json"])
            if tokens:
                scores.append(score(tokens))
        if not scores:
            continue

        n = len(scores)
        total_tokens = sum(s.n_tokens for s in scores)
        rows.append(
            [
                provider,
                prompt_id,
                f"{temperature:g}",
                dispatch,
                str(n),
                f"{total_tokens / n:.1f}",
                f"{sum(s.mean_prob for s in scores) / n:.3f}",
                f"{sum(s.perplexity for s in scores) / n:.2f}",
                f"{sum(s.mean_top_margin for s in scores) / n:.3f}",
                f"{sum(s.mean_chosen_margin for s in scores) / n:+.3f}",
                f"{100 * sum(s.n_off_top for s in scores) / max(total_tokens, 1):.1f}",
            ]
        )
    if not rows:
        return "  (no logprobs stored — only providers that support them appear here)"
    return table(headers, rows, MARKDOWN)


def weakest_tokens_section(groups: dict[tuple, list[sqlite3.Row]], limit: int) -> str:
    headers = ["provider", "prompt", "temp", "min_p", "token", "at", "output"]
    rows: list[list[str]] = []
    for (provider, prompt_id, temperature, dispatch), items in groups.items():
        if dispatch != "sequential":
            continue
        worst: tuple[float, str, int, str] | None = None
        for row in items:
            if row["error"] is not None:
                continue
            tokens = tokens_from_json(row["logprobs_json"])
            if not tokens:
                continue
            result = score(tokens)
            if worst is None or result.min_prob < worst[0]:
                worst = (
                    result.min_prob,
                    result.min_prob_token,
                    result.min_prob_index,
                    row["output"],
                )
        if worst is None:
            continue
        min_prob, token, index, output = worst
        flat = " ".join(output.split())
        rows.append(
            [
                provider,
                prompt_id,
                f"{temperature:g}",
                f"{min_prob:.3f}",
                repr(token),
                str(index),
                flat[:limit] + ("…" if len(flat) > limit else ""),
            ]
        )
    if not rows:
        return "  (no logprobs stored)"
    return table(headers, rows, MARKDOWN)


def determinism_section(groups: dict[tuple, list[sqlite3.Row]]) -> str:
    """Temperature 0 only, sequential against concurrent, side by side."""
    headers = [
        "provider",
        "prompt",
        "seq distinct",
        "conc distinct",
        "seq modal%",
        "conc modal%",
        "pooled distinct",
    ]
    by_pair: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(dict)
    for (provider, prompt_id, temperature, dispatch), items in groups.items():
        if temperature != 0.0:
            continue
        by_pair[(provider, prompt_id)][dispatch] = [
            r["output"] for r in items if r["error"] is None
        ]

    rows: list[list[str]] = []
    for (provider, prompt_id), arms in sorted(by_pair.items()):
        seq = arms.get("sequential", [])
        conc = arms.get("concurrent", [])
        pooled = seq + conc
        rows.append(
            [
                provider,
                prompt_id,
                str(len(set(seq))) if seq else "-",
                str(len(set(conc))) if conc else "-",
                f"{var.modal_share(seq) * 100:.0f}" if seq else "-",
                f"{var.modal_share(conc) * 100:.0f}" if conc else "-",
                str(len(set(pooled))) if pooled else "-",
            ]
        )
    return table(headers, rows, MARKDOWN)


MARKDOWN = False


def main() -> int:
    global MARKDOWN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--width", type=int, default=70, help="output preview width")
    args = parser.parse_args()
    MARKDOWN = args.markdown

    conn = store.connect(args.db)
    rows = store.fetch(conn, args.experiment)
    conn.close()

    if not rows:
        print(f"no rows for experiment {args.experiment!r} in {args.db}")
        return 1

    groups = group_rows(rows)
    models = sorted({f"{r['provider']}:{r['model']}" for r in rows})

    print(f"\n## {args.experiment} — {len(rows)} calls across {len(models)} models")
    print("   " + ", ".join(models))

    print("\n### Output variance\n")
    print(variance_section(groups, embed=not args.no_embed))

    print("\n### Determinism at temperature 0 — dispatch as the variable\n")
    print(determinism_section(groups))

    print("\n### Confidence from logprobs\n")
    print(confidence_section(groups))

    print("\n### Lowest-confidence token per cell\n")
    print(weakest_tokens_section(groups, args.width))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
