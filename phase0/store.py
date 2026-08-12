"""One `runs` table, shared by day 3 and the weekend's structured-output grid.

The alternative was JSONL now and SQLite on Saturday, which means writing the
schema twice and giving lens an importer on Monday. One table costs ~20 lines
today instead.

It lived in `day3_sampling/` until the weekend actually needed it, which is the
same move `runner.py` made on day 5 and for the same reason: the second caller
is what turns a script into a module. Day 3 imports it from here now, and the
schema did not change.

Generic enough for both days by construction: every column here is a property of
*a model call*, not of this experiment. Anything day-specific goes in
`extra_json` — the weekend's `valid_first_try`, `attempts` and `error_type` land
there without a migration.

`dispatch` is the one column whose *meaning* is per-experiment. Day 3 uses it for
sequential-vs-concurrent; the weekend uses it for which attempt produced the row
(`first`, `repair`, `control`). Both are properties of how the call was
dispatched, and the `experiment` column keeps them from ever being compared.

Raw logprobs are stored rather than computed metrics. Re-scoring is then a query,
not a re-run, which matters the first time a scorer definition turns out to be
wrong after 480 paid calls.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY,
    experiment        TEXT    NOT NULL,
    created_at        TEXT    NOT NULL,
    provider          TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    prompt_id         TEXT    NOT NULL,
    prompt_style      TEXT    NOT NULL DEFAULT '',
    temperature       REAL    NOT NULL,
    seed              INTEGER,
    dispatch          TEXT    NOT NULL,
    run_idx           INTEGER NOT NULL,
    output            TEXT    NOT NULL,
    finish_reason     TEXT    NOT NULL DEFAULT '',
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms        REAL    NOT NULL DEFAULT 0,
    error             TEXT,
    logprobs_json     TEXT,
    extra_json        TEXT
);

-- The cell coordinates of the run matrix. Doubles as the resume key: a rerun
-- skips whatever is already here rather than re-paying for it.
-- `prompt_style` defaults to '' rather than NULL on purpose — SQLite treats
-- NULLs as distinct inside a UNIQUE index, so a nullable column here would
-- silently stop deduplicating for day 3.
CREATE UNIQUE INDEX IF NOT EXISTS runs_cell ON runs (
    experiment, provider, model, prompt_id, prompt_style,
    temperature, dispatch, run_idx
);

CREATE INDEX IF NOT EXISTS runs_experiment ON runs (experiment);
"""


@dataclass
class Run:
    experiment: str
    provider: str
    model: str
    prompt_id: str
    temperature: float
    dispatch: str
    run_idx: int
    output: str
    prompt_style: str = ""
    seed: int | None = None
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    logprobs_json: str | None = None
    extra: dict[str, Any] | None = None

    def cell(self) -> tuple:
        return (
            self.experiment,
            self.provider,
            self.model,
            self.prompt_id,
            self.prompt_style,
            self.temperature,
            self.dispatch,
            self.run_idx,
        )


def connect(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # WAL because report.py will be reading while a long run is still writing.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def insert(conn: sqlite3.Connection, run: Run) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO runs (
            experiment, created_at, provider, model, prompt_id, prompt_style,
            temperature, seed, dispatch, run_idx, output, finish_reason,
            prompt_tokens, completion_tokens, latency_ms, error,
            logprobs_json, extra_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run.experiment,
            datetime.now(UTC).isoformat(timespec="seconds"),
            run.provider,
            run.model,
            run.prompt_id,
            run.prompt_style,
            run.temperature,
            run.seed,
            run.dispatch,
            run.run_idx,
            run.output,
            run.finish_reason,
            run.prompt_tokens,
            run.completion_tokens,
            run.latency_ms,
            run.error,
            run.logprobs_json,
            json.dumps(run.extra, ensure_ascii=False) if run.extra else None,
        ),
    )
    conn.commit()


def existing_cells(conn: sqlite3.Connection, experiment: str) -> set[tuple]:
    """Cells already recorded *without* an error — failures are worth retrying."""
    rows = conn.execute(
        """
        SELECT experiment, provider, model, prompt_id, prompt_style,
               temperature, dispatch, run_idx
        FROM runs WHERE experiment = ? AND error IS NULL
        """,
        (experiment,),
    ).fetchall()
    return {tuple(row) for row in rows}


def fetch(conn: sqlite3.Connection, experiment: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM runs WHERE experiment = ? ORDER BY provider, prompt_id, "
        "temperature, dispatch, run_idx",
        (experiment,),
    ).fetchall()


def counts(conn: sqlite3.Connection, experiment: str) -> tuple[int, int]:
    row = conn.execute(
        "SELECT COUNT(*) AS total, SUM(error IS NOT NULL) AS failed "
        "FROM runs WHERE experiment = ?",
        (experiment,),
    ).fetchone()
    return int(row["total"] or 0), int(row["failed"] or 0)


def insert_many(conn: sqlite3.Connection, runs: Iterable[Run]) -> None:
    for run in runs:
        insert(conn, run)
