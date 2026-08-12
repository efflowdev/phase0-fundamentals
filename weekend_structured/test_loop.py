"""The arms, and the transcripts they build.

`styles.call` is replaced by a recorder, so these assert on the *conversation*
rather than on any model's behaviour. That is the part that can be wrong without
failing: a repair loop that never actually shows the model the errors still
produces a plausible recovery rate, because a second sample recovers some
failures on its own. That is exactly what the control arm exists to catch, and
these tests check the control arm is really a control.
"""

from __future__ import annotations

import json

import pytest

from phase0.schema import ProductAttributes
from weekend_structured import classify, config, loop, styles

VALID = json.dumps(
    {"asin": "B01N5IB20Q", "brand": "Nike", "category": "footwear", "size": "10"}
)
BAD_ENUM = json.dumps({"asin": "B01N5IB20Q", "category": "sneakers"})
PLACEHOLDER = json.dumps(
    {"asin": "B01N5IB20Q", "category": "other", "colour": "unknown"}
)

PRODUCT = {"asin": "B01N5IB20Q", "title": "Nike Air Zoom Pegasus 38"}
SCHEMA = ProductAttributes.model_json_schema()


class Recorder:
    """Replays a scripted list of responses and keeps every request it saw."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def __call__(
        self, client, target, style, messages, schema, *, seed, temperature, num_predict
    ):
        self.calls.append(
            {
                "style": style,
                "messages": [dict(m) for m in messages],
                "seed": seed,
                "temperature": temperature,
            }
        )
        text = self.script.pop(0) if self.script else BAD_ENUM
        return styles.Reply(text=text, message={"role": "assistant", "content": text})


@pytest.fixture
def patched(monkeypatch):
    def install(script):
        recorder = Recorder(script)
        monkeypatch.setattr(styles, "call", recorder)
        return recorder

    return install


async def run(target=config.LLAMA32, style=config.PROMPT, max_repairs=2):
    return await loop.episode(
        None,
        target,
        style,
        PRODUCT,
        SCHEMA,
        ProductAttributes,
        base_seed=42,
        max_repairs=max_repairs,
    )


# --- the happy path costs one call ------------------------------------------


@pytest.mark.asyncio
async def test_valid_first_try_makes_exactly_one_call(patched):
    recorder = patched([VALID])
    result = await run()
    assert result.valid_first_try
    assert len(recorder.calls) == 1, "a passing first try must not run either arm"
    assert result.arms[loop.REPAIR].resolved
    assert result.arms[loop.CONTROL].resolved


@pytest.mark.asyncio
async def test_first_attempt_is_shared_between_arms(patched):
    """Both arms are credited with one attempt, not two.

    The first call is the same request in both worlds, so paying for it twice
    would inflate the cost of the strategy by a third and understate its
    per-attempt success rate.
    """
    patched([VALID])
    result = await run()
    assert result.arms[loop.REPAIR].total_attempts == 1
    assert result.arms[loop.CONTROL].total_attempts == 1
    assert result.arms[loop.REPAIR].attempts[0] is result.first


# --- the two arms are genuinely different ------------------------------------


@pytest.mark.asyncio
async def test_repair_arm_shows_the_errors_and_control_does_not(patched):
    recorder = patched([BAD_ENUM, VALID, VALID])
    await run()

    first, repair_call, control_call = recorder.calls
    assert len(first["messages"]) == 2

    # The repair arm grew the transcript by the failed answer and the errors.
    assert len(repair_call["messages"]) == 4
    assert repair_call["messages"][2]["content"] == BAD_ENUM
    assert "category" in repair_call["messages"][3]["content"]

    # The control arm sent the original two messages, unchanged.
    assert control_call["messages"] == first["messages"]


@pytest.mark.asyncio
async def test_both_arms_advance_the_seed_together(patched):
    """Attempt 1 must be seed 43 in both arms, attempt 2 seed 44."""
    recorder = patched([BAD_ENUM] * 5)
    await run()

    seeds = [call["seed"] for call in recorder.calls]
    assert seeds[0] == 42
    assert seeds[1:3] == [43, 44], "repair arm"
    assert seeds[3:5] == [43, 44], "control arm"


@pytest.mark.asyncio
async def test_first_attempt_is_greedy_and_every_retry_resamples(patched):
    """The bug a whole smoke run was spent finding.

    Advancing the seed at temperature 0 changes nothing — greedy decoding ignores
    it — so the control arm recovered 0% by construction and the feedback looked
    effective no matter what it contained. Retries must carry a temperature that
    actually resamples, and both arms must carry the *same* one or the comparison
    is between feedback-and-heat versus heat.
    """
    recorder = patched([BAD_ENUM] * 5)
    await run()

    temperatures = [call["temperature"] for call in recorder.calls]
    assert temperatures[0] == 0.0, "the headline table must stay deterministic"
    assert all(t == config.RETRY_TEMPERATURE for t in temperatures[1:])
    assert config.RETRY_TEMPERATURE > 0, "a control at temperature 0 is not a control"
    assert temperatures[1:3] == temperatures[3:5], "arms must differ only in feedback"


@pytest.mark.asyncio
async def test_recorded_temperature_matches_the_call_that_produced_it(patched):
    """`to_rows` writes a `temperature` column, and it has to be the real one."""
    patched([BAD_ENUM, VALID, BAD_ENUM, BAD_ENUM])
    result = await run()
    assert result.first.temperature == 0.0
    assert result.arms[loop.REPAIR].attempts[1].temperature == config.RETRY_TEMPERATURE

    rows = loop.to_rows(result)
    trail = rows[1]["extra"]["trail"]
    assert [step["temperature"] for step in trail] == [0.0, config.RETRY_TEMPERATURE]


@pytest.mark.asyncio
async def test_arms_stop_at_the_first_success(patched):
    recorder = patched([BAD_ENUM, VALID, VALID])
    result = await run()
    assert len(recorder.calls) == 3, "one failure + one repair + one control"
    assert result.arms[loop.REPAIR].total_attempts == 2
    assert result.arms[loop.CONTROL].total_attempts == 2


@pytest.mark.asyncio
async def test_exhausted_arm_is_unresolved_and_costs_three(patched):
    patched([BAD_ENUM] * 5)
    result = await run()
    assert not result.arms[loop.REPAIR].resolved
    assert result.arms[loop.REPAIR].total_attempts == 3
    assert not result.arms[loop.CONTROL].resolved


@pytest.mark.asyncio
async def test_repair_transcript_accumulates_across_two_failures(patched):
    """Attempt 2 sees attempt 1's answer and errors, not only attempt 0's.

    A loop that rebuilds the transcript from the original failure each time
    re-shows a mistake the model has already moved past, and hides the one it
    just made.
    """
    recorder = patched([BAD_ENUM, PLACEHOLDER, VALID])
    await run()
    second_repair = recorder.calls[2]["messages"]
    assert len(second_repair) == 6
    assert second_repair[2]["content"] == BAD_ENUM
    assert second_repair[4]["content"] == PLACEHOLDER
    assert "colour" in second_repair[5]["content"]


# --- style-specific transcript shapes ----------------------------------------


@pytest.mark.asyncio
async def test_tool_style_answers_a_tool_call_with_a_tool_message(patched):
    recorder = patched([BAD_ENUM, VALID, VALID])
    await run(target=config.GROQ70B, style=config.TOOL)

    repair_call = recorder.calls[1]["messages"]
    assistant, tool_result = repair_call[2], repair_call[3]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["function"]["arguments"] == BAD_ENUM
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert "category" in tool_result["content"]


@pytest.mark.asyncio
async def test_ollama_tool_repair_uses_tool_name_not_call_id(patched):
    recorder = patched([BAD_ENUM, VALID, VALID])
    await run(target=config.LLAMA32, style=config.TOOL)
    tool_result = recorder.calls[1]["messages"][3]
    assert tool_result["role"] == "tool"
    assert tool_result["tool_name"] == styles.TOOL_NAME
    assert "tool_call_id" not in tool_result


@pytest.mark.asyncio
async def test_a_declined_forced_tool_call_is_a_violation_not_an_error(monkeypatch):
    """`NoToolCall` has to land in the table, not in the failure column.

    The mechanism failing to enforce is the most interesting thing the tool
    column can report; recording it as a transport error would retry it and then
    drop it out of the violation counts entirely.
    """

    async def refuse(*args, **kwargs):
        raise styles.NoToolCall("no tool call")

    monkeypatch.setattr(styles, "call", refuse)
    result = await run(target=config.GROQ70B, style=config.TOOL, max_repairs=0)
    assert not result.valid_first_try
    assert result.first.verdict.rules == ["no_tool_call"]


# --- what reaches the database -----------------------------------------------


def test_rows_for_a_clean_episode_are_one():
    """A cell that passed has no arms to report."""
    first = loop.Attempt(
        seed=42, text=VALID, verdict=classify.classify(VALID, ProductAttributes)
    )
    episode = loop.Episode(
        "m",
        config.PROMPT,
        PRODUCT,
        first,
        {
            loop.REPAIR: loop.Arm(loop.REPAIR, [first]),
            loop.CONTROL: loop.Arm(loop.CONTROL, [first]),
        },
    )
    rows = loop.to_rows(episode)
    assert [row["dispatch"] for row in rows] == [loop.FIRST]
    assert rows[0]["extra"]["valid_first_try"] is True


@pytest.mark.asyncio
async def test_rows_for_a_failed_episode_carry_both_arms_and_a_trail(patched):
    patched([BAD_ENUM, VALID, BAD_ENUM, BAD_ENUM])
    rows = loop.to_rows(await run())
    assert [row["dispatch"] for row in rows] == [loop.FIRST, loop.REPAIR, loop.CONTROL]

    repair_row = rows[1]
    assert repair_row["extra"]["valid"] is True
    assert repair_row["extra"]["attempts"] == 2
    assert [step["seed"] for step in repair_row["extra"]["trail"]] == [42, 43]

    control_row = rows[2]
    assert control_row["extra"]["valid"] is False
    assert control_row["extra"]["attempts"] == 3
