from __future__ import annotations

from config import parse_sampler, seed_for


def test_sampler_values_keep_their_types():
    assert parse_sampler("top_k=1,repeat_penalty=1.0,min_p=0") == {
        "top_k": 1,
        "repeat_penalty": 1.0,
        "min_p": 0,
    }


def test_sampler_handles_booleans_and_blanks():
    assert parse_sampler("") == {}
    assert parse_sampler("  ,  ") == {}
    assert parse_sampler("mirostat=true") == {"mirostat": True}


def test_sampler_names_are_not_translated_between_providers():
    """The flag is a passthrough on purpose — `repeat_penalty` is llama.cpp's
    name and `frequency_penalty` is OpenAI's, and they are not the same knob."""
    assert "repeat_penalty" in parse_sampler("repeat_penalty=1.0")
    assert "frequency_penalty" in parse_sampler("frequency_penalty=0.5")


def test_seed_is_fixed_at_greedy_and_varies_above_it():
    """A fixed seed at temperature 1 would make all 20 samples identical and the
    variance table would then describe the harness, not the model."""
    assert seed_for(0.0, 0) == seed_for(0.0, 19)
    assert seed_for(1.0, 0) != seed_for(1.0, 19)
