"""What kind of wrong a response is, and whether a grammar could have stopped it.

Day 5 named violations by matching substrings of Pydantic's messages, and its own
docstring records what that cost: the placeholder rule and the empty-dimensions
rule both say "use null", the shorter phrase was tested first, and 22 nested-object
failures were filed as placeholder failures. A whole failure mode reported as zero.

This classifies on `(loc, type)` instead, which is structural. Probed against the
real model, every rule is uniquely determined:

    ('asin',)                 value_error   -> bad_asin
    ('brand'|'colour'|'size') value_error   -> placeholder_string
    ('dimensions',)           value_error   -> empty_dimensions
    ()                        value_error   -> size_required

Rewording a validator cannot silently re-bucket anything, because no message is
read. `test_classify.py` asserts each rule is reachable, so *deleting* one breaks
a test rather than quietly reporting zero of it.

The second column of `RULES` is the one the day is actually about. `expressible`
marks whether the rule survives `model_json_schema()` — whether it is a constraint
the provider's grammar ever sees. It is not a property of how hard the rule is:
`pack_quantity <= 1000` is expressible and "footwear needs a size" is not, and
that difference is the whole ceiling on constrained decoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

# --- the rules --------------------------------------------------------------

NOT_JSON = "not_json"
NO_TOOL_CALL = "no_tool_call"
JSON_IN_PROSE = "json_in_prose"
NOT_OBJECT = "not_object"
MISSING_FIELD = "missing_field"
WRONG_TYPE = "wrong_type"
BAD_ENUM = "bad_enum"
OUT_OF_RANGE = "out_of_range"
BAD_ASIN = "bad_asin"
PLACEHOLDER = "placeholder_string"
EMPTY_DIMENSIONS = "empty_dimensions"
SIZE_REQUIRED = "size_required"
UNKNOWN = "unclassified"

# rule -> (expressible in the JSON Schema we send, one-line gloss)
RULES: dict[str, tuple[bool, str]] = {
    NOT_JSON: (True, "no JSON object anywhere in the response"),
    NO_TOOL_CALL: (True, "forced tool choice, and the model answered in prose"),
    JSON_IN_PROSE: (True, "JSON wrapped in prose or a code fence"),
    NOT_OBJECT: (True, "parsed, but not a JSON object"),
    MISSING_FIELD: (True, "a required property is absent"),
    WRONG_TYPE: (True, "property has the wrong JSON type"),
    BAD_ENUM: (True, "invented an enum member"),
    OUT_OF_RANGE: (True, "integer outside minimum/maximum"),
    BAD_ASIN: (False, "ASIN fails its regex — sent as a bare string"),
    PLACEHOLDER: (False, "'unknown' where the contract says null"),
    EMPTY_DIMENSIONS: (False, "nested object present but entirely null"),
    SIZE_REQUIRED: (False, "cross-field: footwear/apparel without a size"),
    UNKNOWN: (False, "unmapped validator — classify.py needs updating"),
}

EXPRESSIBLE = frozenset(rule for rule, (ok, _) in RULES.items() if ok)

# Pydantic's built-in error types, grouped. Anything not here and not
# `value_error` lands in UNKNOWN, which is deliberately loud rather than
# silently swallowed into "wrong type".
_BUILTIN: dict[str, str] = {
    "missing": MISSING_FIELD,
    "enum": BAD_ENUM,
    "literal_error": BAD_ENUM,
    "greater_than": OUT_OF_RANGE,
    "greater_than_equal": OUT_OF_RANGE,
    "less_than": OUT_OF_RANGE,
    "less_than_equal": OUT_OF_RANGE,
}
_TYPE_ERRORS = (
    "_type",
    "_parsing",
    "_too_short",
    "_too_long",
    "_pattern_mismatch",
)

# value_error locations, most specific first. `()` is the model-level validator.
_CUSTOM: dict[tuple[str, ...], str] = {
    ("asin",): BAD_ASIN,
    ("brand",): PLACEHOLDER,
    ("colour",): PLACEHOLDER,
    ("size",): PLACEHOLDER,
    ("dimensions",): EMPTY_DIMENSIONS,
    (): SIZE_REQUIRED,
}


@dataclass
class Verdict:
    valid: bool
    rules: list[str] = field(default_factory=list)
    parsed: Any = None
    document: Any = None
    salvaged: bool = False
    detail: list[str] = field(default_factory=list)

    @property
    def only_inexpressible(self) -> bool:
        """Every failure here is one no grammar could have caught.

        The interesting cell in the report: a constrained-decoding run whose
        remaining violations are all of this kind has hit the mechanism's
        ceiling, and more constraint will not move it.
        """
        return bool(self.rules) and not any(r in EXPRESSIBLE for r in self.rules)


def salvage(text: str) -> tuple[Any, bool]:
    """Parse `text`, then try harder, and report which one worked.

    Strictness here would measure the harness, not the model. A response of
    "Here you go:\\n```json\\n{...}\\n```" is a real contract violation — the
    system prompt said a single JSON object and nothing else — but calling it
    *unparseable* throws away everything inside it, and then the prompt-style
    column is one opaque 100% failure instead of a breakdown.

    So: salvage for analysis, count for the headline. `JSON_IN_PROSE` is a
    violation like any other and the report gives it its own row, which is how
    you tell "the model cannot follow the schema" from "the model cannot stop
    saying hello".
    """
    stripped = text.strip()
    if not stripped:
        return None, False
    try:
        return json.loads(stripped), False
    except json.JSONDecodeError:
        pass

    # First `{` to last `}` — enough for fences, preambles and trailing sign-offs,
    # and it deliberately does not attempt to repair truncated JSON. A response
    # cut off by num_predict is a length failure, not a formatting one.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return None, False
    try:
        return json.loads(stripped[start : end + 1]), True
    except json.JSONDecodeError:
        return None, False


def name_error(error: dict[str, Any]) -> str:
    kind = str(error.get("type", ""))
    if kind in _BUILTIN:
        return _BUILTIN[kind]
    if kind == "value_error":
        loc = tuple(str(part) for part in error.get("loc", ()))
        return _CUSTOM.get(loc, UNKNOWN)
    if any(kind.endswith(suffix) for suffix in _TYPE_ERRORS):
        return WRONG_TYPE
    return UNKNOWN


def classify(text: str, model: type) -> Verdict:
    """Every distinct violation in one response, never just the first.

    Day 3's precedent, and it matters more here: the weekend measures which
    violations a repair loop recovers, and a response that both said "unknown"
    for a colour *and* emitted an all-null dimensions object has to count as two
    things, or the repair rate is computed against a denominator that was never
    right.
    """
    parsed, was_salvaged = salvage(text)
    if parsed is None:
        return Verdict(valid=False, rules=[NOT_JSON], detail=["no JSON object found"])
    if not isinstance(parsed, dict):
        return Verdict(
            valid=False,
            rules=[NOT_OBJECT],
            parsed=parsed,
            salvaged=was_salvaged,
            detail=[f"top level is {type(parsed).__name__}, not an object"],
        )

    rules = [JSON_IN_PROSE] if was_salvaged else []
    try:
        document = model.model_validate(parsed)
    except ValidationError as exc:
        errors = exc.errors()
        rules.extend(sorted({name_error(error) for error in errors}))
        return Verdict(
            valid=False,
            rules=sorted(set(rules)),
            parsed=parsed,
            salvaged=was_salvaged,
            detail=[_render(error) for error in errors],
        )

    return Verdict(
        valid=not rules,
        rules=rules,
        parsed=parsed,
        document=document,
        salvaged=was_salvaged,
        detail=[],
    )


def _render(error: dict[str, Any]) -> str:
    loc = ".".join(str(part) for part in error.get("loc", ())) or "(document)"
    return f"{loc}: {error.get('msg', '')}"


def feedback(verdict: Verdict) -> str:
    """What the repair arm sends back.

    Pydantic's own `errors()` payload carries `url`, `input` and Python type
    names — 300 tokens of noise per attempt, and the `url` invites the model to
    talk about Pydantic instead of about the product. This is `loc: msg` and one
    directive, which is the smallest thing that names both the place and the
    rule.
    """
    if verdict.rules == [NOT_JSON]:
        lines = ["The response was not JSON."]
    else:
        lines = [f"- {line}" for line in verdict.detail] or [
            f"- {rule}" for rule in verdict.rules
        ]
        if JSON_IN_PROSE in verdict.rules:
            lines.insert(0, "- the response contained text outside the JSON object")
        lines.insert(0, "Validation failed:")
    lines.append("Return the corrected JSON object only.")
    return "\n".join(lines)
