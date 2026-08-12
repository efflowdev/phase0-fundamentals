"""Request and response shapes, per provider and per style.

These are the assertions that would have caught the two things this layer got
wrong in probing: Groq's 70b answers `json_schema` with a 400 so the native style
must send `json_object`, and Ollama returns tool arguments already decoded while
OpenAI-compatible APIs return them as a string.
"""

from __future__ import annotations

import json

import pytest

from phase0.schema import ProductAttributes
from weekend_structured import config, styles

SCHEMA = ProductAttributes.model_json_schema()
MESSAGES = [{"role": "user", "content": "hi"}]
PRODUCT = {"asin": "B01N5IB20Q", "title": "Nike Air Zoom Pegasus 38"}


def payload(target, style):
    return styles.build_payload(
        target, style, MESSAGES, SCHEMA, seed=42, temperature=0.0, num_predict=512
    )


# --- the schema reaches the model exactly once -------------------------------


def test_schema_is_inlined_only_for_prompt_style():
    """Otherwise the columns differ by ~600 tokens of context as well as by
    mechanism, and any gap between them is unattributable."""
    assert "JSON Schema" in styles.user_prompt(PRODUCT, SCHEMA, config.PROMPT)
    for style in (config.TOOL, config.NATIVE):
        assert "JSON Schema" not in styles.user_prompt(PRODUCT, SCHEMA, style)


def test_prompt_style_sends_no_enforcement_field():
    body = payload(config.LLAMA32, config.PROMPT)
    assert "format" not in body
    assert "tools" not in body


# --- ollama ------------------------------------------------------------------


def test_ollama_native_sends_the_whole_schema_as_a_grammar():
    body = payload(config.LLAMA32, config.NATIVE)
    assert body["format"] == SCHEMA
    assert body["options"]["seed"] == 42


def test_ollama_tool_style_does_not_force_the_call():
    """Ollama has no `tool_choice`, which is itself a difference worth recording:
    the tool column is 'offered a tool' locally and 'forced to call it' on Groq."""
    body = payload(config.LLAMA32, config.TOOL)
    assert body["tools"][0]["function"]["parameters"] == SCHEMA
    assert "tool_choice" not in body


# --- groq --------------------------------------------------------------------


def test_groq_native_is_json_object_not_json_schema():
    """Probed 2026-08-12: `json_schema` returns 400 on llama-3.3-70b.

    This is the single most load-bearing line in `styles.py`, because getting it
    wrong turns the whole cloud native column into transport failures.
    """
    body = payload(config.GROQ70B, config.NATIVE)
    assert body["response_format"] == {"type": "json_object"}


def test_groq_tool_style_forces_the_call():
    body = payload(config.GROQ70B, config.TOOL)
    assert body["tool_choice"]["function"]["name"] == styles.TOOL_NAME
    assert body["tools"][0]["function"]["parameters"] == SCHEMA


def test_groq_uses_max_tokens_and_ollama_uses_num_predict():
    assert "max_tokens" in payload(config.GROQ70B, config.PROMPT)
    assert payload(config.LLAMA32, config.PROMPT)["options"]["num_predict"] == 512


# --- responses ---------------------------------------------------------------


def test_ollama_reply_reads_its_own_token_counts():
    reply = styles.parse_reply(
        config.LLAMA32,
        config.PROMPT,
        {
            "message": {"role": "assistant", "content": "{}"},
            "prompt_eval_count": 120,
            "eval_count": 30,
            "done_reason": "stop",
        },
    )
    assert (reply.text, reply.prompt_tokens, reply.completion_tokens) == ("{}", 120, 30)
    assert reply.finish_reason == "stop"


def test_groq_reply_reads_its_own_token_counts():
    reply = styles.parse_reply(
        config.GROQ70B,
        config.PROMPT,
        {
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        },
    )
    assert (reply.text, reply.prompt_tokens, reply.completion_tokens) == ("{}", 120, 30)


def test_tool_arguments_normalise_to_a_string_from_either_provider():
    """Ollama decodes them, OpenAI-compatible APIs do not. The classifier takes
    text, so one of the two has to be re-serialised and it must not matter which
    provider produced it."""
    decoded = styles.parse_reply(
        config.LLAMA32,
        config.TOOL,
        {"message": {"tool_calls": [{"function": {"name": "f", "arguments": {"asin": "B01N5IB20Q"}}}]}},
    )
    as_string = styles.parse_reply(
        config.GROQ70B,
        config.TOOL,
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "x", "function": {"name": "f", "arguments": '{"asin": "B01N5IB20Q"}'}}
                        ]
                    }
                }
            ]
        },
    )
    assert json.loads(decoded.text) == json.loads(as_string.text) == {"asin": "B01N5IB20Q"}


@pytest.mark.parametrize("target", [config.LLAMA32, config.GROQ70B])
def test_missing_tool_call_raises_rather_than_returning_empty_text(target):
    body = (
        {"message": {"content": "I think it is a shoe.", "tool_calls": []}}
        if target.provider == "ollama"
        else {"choices": [{"message": {"content": "I think it is a shoe."}}]}
    )
    with pytest.raises(styles.NoToolCall):
        styles.parse_reply(target, config.TOOL, body)


# --- repair turns ------------------------------------------------------------


BAD = '{"asin":"B01N5IB20Q","category":"sneakers"}'


def test_non_tool_repair_is_assistant_then_user():
    turns = styles.repair_turns(config.PROMPT, BAD, "errors", provider="ollama")
    assert [turn["role"] for turn in turns] == ["assistant", "user"]
    assert turns[0]["content"] == BAD
    assert turns[1]["content"] == "errors"


def test_native_style_repairs_like_prompt_style():
    turns = styles.repair_turns(config.NATIVE, BAD, "errors", provider="groq")
    assert [turn["role"] for turn in turns] == ["assistant", "user"]


def test_ollama_tool_repair_sends_arguments_as_an_object():
    """The 400 that killed three episodes on the first smoke run.

    Ollama answers a string-valued `arguments` with
    "Value looks like object, but can't find closing '}' symbol" — a parse error
    about a string that parses perfectly well. It wants the decoded object.
    """
    turns = styles.repair_turns(config.TOOL, BAD, "errors", provider="ollama")
    arguments = turns[0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, dict)
    assert arguments == json.loads(BAD)


def test_openai_tool_repair_sends_arguments_as_a_string():
    """Exactly the opposite, which is why this is not one code path."""
    turns = styles.repair_turns(config.TOOL, BAD, "errors", provider="groq")
    arguments = turns[0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert arguments == BAD


def test_tool_call_id_is_present_for_openai_and_absent_for_ollama():
    openai_turns = styles.repair_turns(config.TOOL, BAD, "errors", provider="groq")
    assert openai_turns[0]["tool_calls"][0]["id"] == openai_turns[1]["tool_call_id"]

    ollama_turns = styles.repair_turns(config.TOOL, BAD, "errors", provider="ollama")
    assert "id" not in ollama_turns[0]["tool_calls"][0]
    assert ollama_turns[1]["tool_name"] == styles.TOOL_NAME


def test_unencodable_tool_arguments_degrade_instead_of_400ing():
    """A repair that cannot be expressed still has to run.

    Losing the tool envelope costs fidelity on one attempt; sending Ollama a
    string it will reject costs the whole episode.
    """
    turns = styles.repair_turns(config.TOOL, "not json at all", "errors", provider="ollama")
    assert [turn["role"] for turn in turns] == ["assistant", "user"]
    assert "tool_calls" not in turns[0]
