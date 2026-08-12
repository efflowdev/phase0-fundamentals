"""Tests for the two things in this day that can silently regress.

Not the containers — those either come up or they do not, and a test needing a
running Postgres is an integration test wearing a unit test's clothes. What
breaks quietly is the *Dockerfile's layer ordering* (someone moves a COPY up to
fix an unrelated problem, every rebuild goes from seconds to minutes, and
nothing fails) and the small parsers in embed_job.py, whose units are easy to
get wrong by a factor of 1024.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from embed_job import cgroup_limit_mb, load_titles

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((HERE / "docker-compose.yml").read_text())


@pytest.fixture(scope="module")
def dockerfile(compose) -> list[str]:
    """Whichever Dockerfile compose actually builds — not a hardcoded name.

    There are three in this directory and only one is live. Pinning the tests to
    a filename meant that switching compose to Dockerfile.split silently moved
    the real build out from under its own guarantees, while the suite stayed
    green testing a file nothing uses.
    """
    relative = compose["services"]["app"]["build"]["dockerfile"]
    path = HERE.parent / relative
    assert path.exists(), f"compose builds {relative}, which does not exist"
    return path.read_text().splitlines()


def _first_line_matching(lines: list[str], pattern: str) -> int:
    for i, line in enumerate(lines):
        if re.match(pattern, line.strip()):
            return i
    raise AssertionError(f"no line matching {pattern!r}")


def test_dependencies_are_installed_before_source_is_copied(dockerfile):
    """The single property that makes the rebuild fast.

    If a source COPY moves above the dependency sync, every code edit
    re-resolves the whole tree. Nothing fails — it just gets slow, which is
    exactly the kind of regression that survives review.
    """
    deps_sync = _first_line_matching(dockerfile, r"uv sync --frozen --no-install-project")
    # `COPY --chown=... phase0/` in the split build, bare `COPY phase0/` in the
    # other. Matching the literal prefix passed silently on one of them.
    source_copy = _first_line_matching(dockerfile, r"COPY\b.*\bphase0/")
    assert deps_sync < source_copy


def test_the_virtualenv_is_copied_before_the_source(dockerfile):
    """What the measurement actually found, pinned so it cannot regress.

    Layer ordering alone is not enough. `COPY --from=builder /app /app` puts the
    virtualenv and the source in ONE 244 MB layer, so a one-line edit re-copies
    all of it — 9.3s against 1.0s. The venv has to land in its own layer, before
    anything volatile.
    """
    venv_copy = _first_line_matching(dockerfile, r"COPY --from=\w+.*\.venv")
    source_copy = _first_line_matching(dockerfile, r"COPY\b.*\bphase0/")
    assert venv_copy < source_copy


def test_runtime_stage_does_not_carry_uv(dockerfile):
    """The multi-stage split, asserted rather than assumed."""
    runtime_start = _first_line_matching(dockerfile, r"FROM python:.* AS runtime")
    runtime = "\n".join(dockerfile[runtime_start:])
    assert "uv sync" not in runtime
    assert "astral-sh/uv" not in runtime


def test_every_stateful_service_uses_a_named_volume(compose):
    """Weights and data are mounted, never baked in or left in the container.

    The failure this guards against is a service that looks fine until the first
    `docker compose down`, at which point a 2 GB model pull or a whole database
    quietly evaporates.
    """
    named = set(compose["volumes"])
    for service in ("postgres", "qdrant", "ollama", "app"):
        mounts = compose["services"][service].get("volumes", [])
        assert any(m.split(":")[0] in named for m in mounts), service


def test_ollama_is_not_a_default_service(compose):
    """Containerised Ollama is opt-in because it cannot reach Metal.

    Deleting the profile would silently move every local model call onto CPU —
    correct results, several times slower, no error anywhere.
    """
    assert compose["services"]["ollama"]["profiles"] == ["full"]


def test_memory_limit_and_swap_limit_move_together(compose):
    """A memory limit without a swap limit does not kill, it swaps.

    The container gets slow instead of dying, so the OOM demonstration silently
    stops demonstrating anything.
    """
    embed = compose["services"]["embed"]
    assert embed["mem_limit"] == embed["memswap_limit"]


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("268435456\n", 256.0),  # cgroup v2, a real limit
        ("max\n", None),  # cgroup v2, unlimited
        ("9223372036854771712\n", None),  # cgroup v1 sentinel, not 8.8 million TB
    ],
)
def test_cgroup_limit_reads_both_encodings(tmp_path, contents, expected):
    path = tmp_path / "memory.max"
    path.write_text(contents)
    assert cgroup_limit_mb(((path, "max"),)) == expected


def test_load_titles_deduplicates_by_product_id(tmp_path):
    """One product judged against three queries is three rows and one document."""
    cache = tmp_path / "rows.jsonl"
    cache.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"product_id": "A1", "product_title": "Red running shoe"},
                {"product_id": "A1", "product_title": "Red running shoe"},
                {"product_id": "A2", "product_title": "Blue running shoe"},
                {"product_id": "A3", "product_title": "   "},  # dropped: no title
            ]
        )
    )
    assert load_titles(10, cache) == ["Red running shoe", "Blue running shoe"]


def test_load_titles_stops_at_the_limit(tmp_path):
    cache = tmp_path / "rows.jsonl"
    cache.write_text(
        "\n".join(
            json.dumps({"product_id": f"A{i}", "product_title": f"Product {i}"})
            for i in range(100)
        )
    )
    assert len(load_titles(7, cache)) == 7
