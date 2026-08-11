from __future__ import annotations

import asyncio

import pytest

from phase0.runner import Result, load_checkpoint, run_all, summarize


def ident(x):
    return str(x)


@pytest.mark.asyncio
async def test_results_come_back_in_input_order_not_completion_order():
    async def work(n: int) -> int:
        await asyncio.sleep(0.02 if n == 0 else 0.001)  # item 0 finishes last
        return n * 10

    results = await run_all([0, 1, 2], work, key=ident, progress=False)
    assert [r.id for r in results] == ["0", "1", "2"]
    assert [r.value for r in results] == [0, 10, 20]


@pytest.mark.asyncio
async def test_one_failure_does_not_discard_the_successes():
    async def work(n: int) -> int:
        if n == 2:
            raise ValueError("boom")
        return n

    results = await run_all([1, 2, 3], work, key=ident, progress=False)
    assert [r.ok for r in results] == [True, False, True]
    assert results[1].error == "ValueError: boom"
    assert results[1].value is None


@pytest.mark.asyncio
async def test_concurrency_is_actually_bounded():
    in_flight = 0
    peak = 0

    async def work(n: int) -> int:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return n

    await run_all(range(20), work, key=ident, concurrency=3, progress=False)
    assert peak <= 3
    assert peak > 1  # and it is genuinely running things in parallel


@pytest.mark.asyncio
async def test_concurrency_of_one_serialises():
    peak = 0
    in_flight = 0

    async def work(n: int) -> int:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.001)
        in_flight -= 1
        return n

    await run_all(range(5), work, key=ident, concurrency=1, progress=False)
    assert peak == 1


@pytest.mark.asyncio
async def test_cancellation_propagates_instead_of_being_logged_as_a_failure():
    """CancelledError is a BaseException, so `except Exception` must miss it —
    otherwise Ctrl-C during a long run turns into 500 'failed' rows."""

    async def work(n: int) -> int:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_all([1], work, key=ident, progress=False)


@pytest.mark.asyncio
async def test_checkpoint_is_written_as_work_completes(tmp_path):
    path = tmp_path / "ckpt.jsonl"

    async def work(n: int) -> int:
        return n

    await run_all([1, 2, 3], work, key=ident, checkpoint=path, progress=False)
    assert len(path.read_text().splitlines()) == 3
    assert set(load_checkpoint(path)) == {"1", "2", "3"}


@pytest.mark.asyncio
async def test_resume_skips_successes_and_does_not_call_work_again(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    calls: list[int] = []

    async def work(n: int) -> int:
        calls.append(n)
        return n

    await run_all([1, 2], work, key=ident, checkpoint=path, progress=False)
    assert calls == [1, 2]

    calls.clear()
    results = await run_all([1, 2, 3], work, key=ident, checkpoint=path, progress=False)
    assert calls == [3]  # only the new item
    assert [r.resumed for r in results] == [True, True, False]
    assert [r.value for r in results] == [1, 2, 3]


@pytest.mark.asyncio
async def test_resume_retries_failures_by_default(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    attempts: list[int] = []

    async def flaky(n: int) -> int:
        attempts.append(n)
        if len(attempts) == 1:
            raise TimeoutError("rate limited")
        return n

    first = await run_all([1], flaky, key=ident, checkpoint=path, progress=False)
    assert first[0].ok is False

    second = await run_all([1], flaky, key=ident, checkpoint=path, progress=False)
    assert second[0].ok is True
    assert second[0].value == 1


@pytest.mark.asyncio
async def test_retry_failed_false_treats_a_failure_as_final(tmp_path):
    path = tmp_path / "ckpt.jsonl"

    async def always_fails(n: int) -> int:
        raise ValueError("no")

    await run_all([1], always_fails, key=ident, checkpoint=path, progress=False)
    results = await run_all(
        [1],
        always_fails,
        key=ident,
        checkpoint=path,
        retry_failed=False,
        progress=False,
    )
    assert results[0].resumed is True
    assert results[0].ok is False


def test_a_torn_final_line_is_skipped_rather_than_fatal(tmp_path):
    """What a crash mid-append actually looks like on disk."""
    path = tmp_path / "ckpt.jsonl"
    path.write_text(Result("a", True, 1).to_json() + "\n" + '{"id": "b", "ok": tr')
    assert set(load_checkpoint(path)) == {"a"}


def test_last_line_wins_so_a_retried_failure_is_superseded(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    path.write_text(
        Result("a", False, None, "boom").to_json()
        + "\n"
        + Result("a", True, 42).to_json()
        + "\n"
    )
    assert load_checkpoint(path)["a"].value == 42


def test_summarize_separates_fresh_work_from_resumed_rows():
    results = [
        Result("a", True, 1, elapsed_ms=10.0),
        Result("b", True, 2, elapsed_ms=30.0, resumed=True),
        Result("c", False, None, "boom", elapsed_ms=20.0),
    ]
    summary = summarize(results)
    assert summary == {
        "total": 3,
        "ok": 2,
        "failed": 1,
        "resumed": 1,
        "p50_ms": 20.0,
        "p95_ms": 20.0,
    }
