-- Runs once, on an empty data directory. If the pgdata volume already exists,
-- docker-entrypoint-initdb.d is skipped entirely — which is the usual reason a
-- change here "does nothing" and needs `docker compose down -v`.

CREATE EXTENSION IF NOT EXISTS vector;

-- pg_trgm is the lexical half of P1's hybrid search. Postgres also has
-- full-text search built in (tsvector/ts_rank) with no extension at all; the
-- comparison between the two is P1's problem, not today's.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
