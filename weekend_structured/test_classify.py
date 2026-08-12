"""Every rule must be reachable, and reachable by the payload that names it.

The point of these is not that `classify` returns a string. It is that the
`(loc, type)` map in `classify.py` stays wired to the validators in
`phase0/schema.py`. Deleting a validator, renaming a field or letting Pydantic
change an error type all break a test here — which is the failure mode day 5 hit
and could not see: a rule that silently stops matching reports zero violations,
and zero is indistinguishable from "the models got it right".
"""

from __future__ import annotations

import json

import pytest

from phase0.schema import ProductAttributes
from weekend_structured import classify

VALID = {
    "asin": "B01N5IB20Q",
    "brand": "Nike",
    "category": "footwear",
    "colour": "black",
    "size": "10",
    "pack_quantity": 1,
    "dimensions": None,
}


def verdict(payload: dict) -> classify.Verdict:
    return classify.classify(json.dumps(payload), ProductAttributes)


def test_valid_document_is_valid():
    result = verdict(VALID)
    assert result.valid
    assert result.rules == []
    assert result.document.asin == "B01N5IB20Q"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"asin": None}, classify.MISSING_FIELD),
        ({"asin": "not-an-asin"}, classify.BAD_ASIN),
        ({"category": "sneakers"}, classify.BAD_ENUM),
        ({"pack_quantity": "2 pack"}, classify.WRONG_TYPE),
        ({"pack_quantity": 5000}, classify.OUT_OF_RANGE),
        ({"colour": "unknown"}, classify.PLACEHOLDER),
        ({"brand": "n/a"}, classify.PLACEHOLDER),
        (
            {"dimensions": {"length_cm": None, "width_cm": None, "height_cm": None}},
            classify.EMPTY_DIMENSIONS,
        ),
        ({"category": "apparel", "size": None}, classify.SIZE_REQUIRED),
    ],
)
def test_each_rule_is_reachable(mutation, expected):
    payload = {**VALID, **mutation}
    if mutation.get("asin") is None and "asin" in mutation:
        payload.pop("asin")
    result = verdict(payload)
    assert not result.valid
    assert expected in result.rules, f"{expected} not in {result.rules}"


def test_nothing_lands_in_unclassified():
    """The catch-all bucket is a bug detector, not a category.

    If a mutation above ever starts producing `unclassified`, Pydantic changed an
    error type and the report has been quietly under-counting a real rule.
    """
    for mutation in (
        {"asin": "nope"},
        {"colour": "n/a"},
        {"category": "footwear", "size": None},
        {"dimensions": {"length_cm": None, "width_cm": None, "height_cm": None}},
    ):
        assert classify.UNKNOWN not in verdict({**VALID, **mutation}).rules


def test_every_violation_is_counted_not_just_the_first():
    """Day 5's under-count, pinned.

    Two independent violations in one response; a classifier that took
    `errors()[0]` reported one of them and the repair-rate denominator was wrong.
    """
    result = verdict(
        {
            **VALID,
            "colour": "unknown",
            "dimensions": {"length_cm": None, "width_cm": None, "height_cm": None},
        }
    )
    assert classify.PLACEHOLDER in result.rules
    assert classify.EMPTY_DIMENSIONS in result.rules


def test_placeholder_and_empty_dimensions_do_not_collide():
    """The exact bug day 5 documented: both messages contain "use null".

    Message-substring matching filed every empty-dimensions failure as a
    placeholder failure. Keying on `loc` makes the collision impossible, and this
    asserts the two land apart when only one of them is present.
    """
    only_dimensions = verdict(
        {**VALID, "dimensions": {"length_cm": None, "width_cm": None, "height_cm": None}}
    )
    assert only_dimensions.rules == [classify.EMPTY_DIMENSIONS]

    only_placeholder = verdict({**VALID, "colour": "unknown"})
    assert only_placeholder.rules == [classify.PLACEHOLDER]


# --- salvage ----------------------------------------------------------------


def test_clean_json_is_not_marked_salvaged():
    assert classify.classify(json.dumps(VALID), ProductAttributes).salvaged is False


def test_fenced_json_is_salvaged_and_still_a_violation():
    """Both halves of the decision in one assertion.

    Salvaged so the schema violations inside it are still visible; counted so the
    headline does not quietly forgive a response that ignored "nothing else".
    """
    text = f"Here you go:\n```json\n{json.dumps(VALID)}\n```\nHope that helps!"
    result = classify.classify(text, ProductAttributes)
    assert result.salvaged
    assert result.rules == [classify.JSON_IN_PROSE]
    assert not result.valid
    assert result.parsed["asin"] == "B01N5IB20Q"


def test_prose_wrapped_invalid_json_reports_both_problems():
    text = "```json\n" + json.dumps({**VALID, "colour": "unknown"}) + "\n```"
    result = classify.classify(text, ProductAttributes)
    assert classify.JSON_IN_PROSE in result.rules
    assert classify.PLACEHOLDER in result.rules


def test_truncated_json_is_not_repaired():
    """A response cut off by num_predict is a length failure, not a format one.

    Salvage takes first `{` to last `}`; with no closing brace there is nothing
    to take, and inventing one would turn a truncation into a fake violation of
    whatever field happened to be missing.
    """
    result = classify.classify('{"asin": "B01N5IB20Q", "categ', ProductAttributes)
    assert result.rules == [classify.NOT_JSON]


def test_json_array_is_not_an_object():
    result = classify.classify('[{"asin": "B01N5IB20Q"}]', ProductAttributes)
    assert result.rules == [classify.NOT_OBJECT]


def test_empty_response_is_not_json():
    assert classify.classify("   ", ProductAttributes).rules == [classify.NOT_JSON]


# --- the expressible/inexpressible split ------------------------------------


def test_cross_field_failure_is_inexpressible():
    """The claim table 2 rests on.

    `size_required` is a violation no JSON Schema can state, so no grammar can
    prevent it, so a constrained-decoding run that only fails this way has hit
    its ceiling.
    """
    result = verdict({**VALID, "category": "apparel", "size": None})
    assert result.only_inexpressible


def test_bad_enum_is_expressible():
    result = verdict({**VALID, "category": "sneakers"})
    assert not result.only_inexpressible


def test_asin_regex_is_inexpressible_because_we_never_send_it():
    """Not a claim about JSON Schema — a claim about what `model_json_schema` emits.

    The ASIN rule is a `field_validator`, so the emitted schema says
    `{"type": "string"}` and the pattern never reaches the provider. It would be
    expressible if it were declared as `Field(pattern=...)`, and it is not.
    """
    emitted = ProductAttributes.model_json_schema()["properties"]["asin"]
    assert "pattern" not in emitted
    assert verdict({**VALID, "asin": "nope"}).only_inexpressible


def test_rules_table_and_expressible_set_agree():
    assert classify.EXPRESSIBLE == {
        rule for rule, (ok, _) in classify.RULES.items() if ok
    }


# --- feedback ---------------------------------------------------------------


def test_feedback_names_the_field_and_the_rule():
    text = classify.feedback(verdict({**VALID, "colour": "unknown"}))
    assert "colour" in text
    assert "null" in text
    assert "Return the corrected JSON object only." in text


def test_feedback_carries_no_pydantic_urls():
    """Verbosity is the cost of every repair attempt, paid per token.

    Pydantic's `errors()` payload includes a docs URL per error, which invites
    the model to discuss Pydantic instead of the product.
    """
    text = classify.feedback(verdict({**VALID, "asin": "nope", "colour": "unknown"}))
    assert "https://" not in text
    assert text.count("\n") <= 5


def test_feedback_for_unparseable_says_so_without_a_field_list():
    text = classify.feedback(classify.classify("sorry, I cannot", ProductAttributes))
    assert "not JSON" in text
