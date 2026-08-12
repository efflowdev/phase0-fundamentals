"""Proof the stack is wired, by doing each store's actual job rather than
knocking on its port.

"Is the container up" and "can I store and retrieve a vector" are different
questions, and only the second one is the done bar. So Postgres gets a real
pgvector distance query and Qdrant gets a real collection, upsert and search.

The same script runs from the host and from inside the app container — the only
difference is three environment variables, which is the point of putting the
addresses in the environment in the first place.

Qdrant is spoken to over raw REST rather than through qdrant-client. Phase 0's
rule is that you meet the protocol before the SDK, and Qdrant's REST surface is
three endpoints; the client also drags in grpcio, which is one of the heavier
wheels to build for a second architecture.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from typing import Any

import httpx

PG_DSN = os.environ.get("PG_DSN", "postgresql://phase0:phase0@localhost:5433/phase0")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")

COLLECTION = "phase0_healthcheck"
DIM = 3

def retrying[T](what: str, fn: Callable[[], T], *, attempts: int = 30) -> T:
    """Linear retry, because container start order is not container readiness.

    Postgres has a healthcheck and compose waits on it; Qdrant does not, and the
    app container regularly wins the race against it by a second or two. Failing
    on the first connection refused would make the stack look broken when it is
    merely young.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — any failure here is "not ready yet"
            last = exc
            time.sleep(0.5)
    raise RuntimeError(f"{what} never became ready after {attempts} attempts") from last


# --------------------------------------------------------------------------- pg


def check_postgres() -> dict[str, Any]:
    import psycopg

    def connect() -> psycopg.Connection:
        return psycopg.connect(PG_DSN, connect_timeout=3)

    conn = retrying("postgres", connect)
    with conn, conn.cursor() as cur:
        cur.execute("SELECT version()")
        server = cur.fetchone()[0].split(",")[0]

        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                "pgvector extension is absent. init/01-extensions.sql only runs on a "
                "fresh data directory — `docker compose down -v` and start again."
            )
        vector_version = row[0]

        # A real index-shaped round trip: store three vectors, ask for the
        # nearest to a fourth. `<=>` is cosine distance, the operator P1 uses.
        cur.execute("DROP TABLE IF EXISTS healthcheck_vectors")
        cur.execute(
            "CREATE TABLE healthcheck_vectors (id int PRIMARY KEY, embedding vector(3))"
        )
        cur.executemany(
            "INSERT INTO healthcheck_vectors (id, embedding) VALUES (%s, %s)",
            [(1, "[1,0,0]"), (2, "[0,1,0]"), (3, "[0.9,0.1,0]")],
        )
        cur.execute(
            "SELECT id, embedding <=> %s AS distance "
            "FROM healthcheck_vectors ORDER BY distance LIMIT 1",
            ("[1,0,0]",),
        )
        nearest_id, distance = cur.fetchone()
        cur.execute("DROP TABLE healthcheck_vectors")
    # `with conn` commits and closes in psycopg 3 — there is no close() below.

    if nearest_id != 1:
        raise RuntimeError(f"pgvector returned {nearest_id} as nearest to [1,0,0]")

    return {
        "server": server,
        "pgvector": vector_version,
        "nearest_id": nearest_id,
        "distance": round(float(distance), 6),
    }


# ----------------------------------------------------------------------- qdrant


def check_qdrant() -> dict[str, Any]:
    with httpx.Client(base_url=QDRANT_URL, timeout=10.0) as client:
        version = retrying(
            "qdrant", lambda: client.get("/").raise_for_status().json()
        )["version"]

        client.delete(f"/collections/{COLLECTION}")
        client.put(
            f"/collections/{COLLECTION}",
            json={"vectors": {"size": DIM, "distance": "Cosine"}},
        ).raise_for_status()

        client.put(
            f"/collections/{COLLECTION}/points",
            params={"wait": "true"},
            json={
                "points": [
                    {"id": 1, "vector": [1.0, 0.0, 0.0], "payload": {"tag": "x"}},
                    {"id": 2, "vector": [0.0, 1.0, 0.0], "payload": {"tag": "y"}},
                    {"id": 3, "vector": [0.9, 0.1, 0.0], "payload": {"tag": "x"}},
                ]
            },
        ).raise_for_status()

        # With the payload filter, because filtering during search rather than
        # after it is the actual argument for a dedicated vector database.
        hits = (
            client.post(
                f"/collections/{COLLECTION}/points/search",
                json={
                    "vector": [1.0, 0.0, 0.0],
                    "limit": 1,
                    "with_payload": True,
                    "filter": {"must": [{"key": "tag", "match": {"value": "x"}}]},
                },
            )
            .raise_for_status()
            .json()["result"]
        )
        client.delete(f"/collections/{COLLECTION}")

    if not hits or hits[0]["id"] != 1:
        raise RuntimeError(f"qdrant returned {hits} as nearest to [1,0,0]")

    return {
        "version": version,
        "nearest_id": hits[0]["id"],
        "score": round(float(hits[0]["score"]), 6),
    }


# ----------------------------------------------------------------------- ollama


def check_ollama() -> dict[str, Any]:
    with httpx.Client(base_url=OLLAMA_URL, timeout=15.0) as client:
        try:
            tags = client.get("/api/tags").raise_for_status().json()
        except httpx.ConnectError as exc:
            # Measured, not assumed: under colima's vz networking a container
            # reaches a host daemon bound to 127.0.0.1 anyway, because the
            # gateway behind host.docker.internal is a process on macOS and it
            # originates the connection from the host's own loopback. So a
            # loopback bind is NOT the likely cause here — the daemon is
            # probably just not running. On a Linux host, where host-gateway is
            # a real bridge address and nothing proxies, the bind does matter.
            raise RuntimeError(
                f"no Ollama at {OLLAMA_URL}. Start the host daemon, or the "
                "containerised one with `docker compose --profile full up -d ollama` "
                "and point OLLAMA_URL at http://ollama:11434."
            ) from exc

    return {
        "url": OLLAMA_URL,
        "models": [m["name"] for m in tags.get("models", [])],
    }


def main() -> int:
    checks = {
        "postgres": check_postgres,
        "qdrant": check_qdrant,
        "ollama": check_ollama,
    }

    results: dict[str, Any] = {}
    failed = False

    for name, check in checks.items():
        started = time.perf_counter()
        try:
            payload = check()
            elapsed_ms = (time.perf_counter() - started) * 1000
            results[name] = {"ok": True, "ms": round(elapsed_ms, 1), **payload}
            print(f"  ok    {name:<9} {elapsed_ms:7.1f} ms  {payload}")
        except Exception as exc:  # noqa: BLE001 — report every service, not just the first
            failed = True
            results[name] = {"ok": False, "error": str(exc)}
            print(f"  FAIL  {name:<9} {exc}", file=sys.stderr)

    print(json.dumps(results, indent=2, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
