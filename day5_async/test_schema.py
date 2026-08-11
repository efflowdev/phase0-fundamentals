from __future__ import annotations

import pytest
from pydantic import ValidationError
from schema import Category, Dimensions, ProductAttributes


def valid(**overrides):
    base = {
        "asin": "B01N1G8OX8",
        "category": "electronics",
        "brand": "UGREEN",
        "pack_quantity": 2,
    }
    return ProductAttributes(**(base | overrides))


def test_a_well_formed_extraction_validates():
    product = valid()
    assert product.category is Category.ELECTRONICS
    assert product.brand == "UGREEN"
    assert product.size is None


def test_asin_is_normalised_and_checked():
    assert valid(asin=" b01n1g8ox8 ").asin == "B01N1G8OX8"
    with pytest.raises(ValidationError, match="not an ASIN"):
        valid(asin="B01")


def test_empty_string_becomes_absent():
    assert valid(brand="   ").brand is None


def test_placeholder_words_are_violations_not_absences():
    """The schema says null. `"unknown"` means the model did not follow it, and
    that has to be countable on Saturday rather than silently repaired."""
    for placeholder in ("unknown", "N/A", "Not Specified", "none", "-"):
        with pytest.raises(ValidationError, match="use null"):
            valid(brand=placeholder)


def test_the_validator_does_not_repair_bad_input():
    """If this ever starts returning None instead of raising, the violation-rate
    table stops measuring the model and starts measuring the parser."""
    with pytest.raises(ValidationError):
        valid(colour="unknown")


def test_an_invented_enum_member_is_rejected():
    with pytest.raises(ValidationError):
        valid(category="sneakers")


def test_pack_quantity_range_is_enforced():
    with pytest.raises(ValidationError):
        valid(pack_quantity=0)
    with pytest.raises(ValidationError):
        valid(pack_quantity=5000)


def test_pack_quantity_is_not_coerced_from_prose():
    """Pydantic will happily turn "2" into 2; it must not accept "2 pack"."""
    assert valid(pack_quantity="2").pack_quantity == 2
    with pytest.raises(ValidationError):
        valid(pack_quantity="2 pack")


def test_an_all_null_dimensions_object_is_rejected():
    """Structure without content — the model filled in the shape to look
    compliant. The correct answer is `dimensions: null`."""
    with pytest.raises(ValidationError, match="at least one measurement"):
        Dimensions(length_cm=None, width_cm=None, height_cm=None)
    assert valid(dimensions=None).dimensions is None
    assert valid(dimensions={"length_cm": 12.5}).dimensions.length_cm == 12.5


def test_dimensions_must_be_positive():
    with pytest.raises(ValidationError):
        Dimensions(length_cm=0)


def test_footwear_without_a_size_fails_on_the_combination():
    """Neither field is individually invalid, which is exactly why this needs a
    model_validator rather than a field_validator."""
    with pytest.raises(ValidationError, match="requires a size"):
        valid(category="footwear", size=None)
    assert valid(category="footwear", size="UK 8").size == "UK 8"


def test_json_schema_is_usable_as_a_tool_definition():
    schema = ProductAttributes.model_json_schema()
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"asin", "category"}
    assert "$defs" in schema and "Dimensions" in schema["$defs"]
    assert schema["$defs"]["Category"]["enum"][0] == "footwear"


def test_the_two_null_rules_have_distinguishable_messages():
    """Regression: both validators tell the model to "use null", and a
    classifier matching that shared phrase filed every empty-dimensions error
    as a placeholder error — 22 real failures reported as zero."""
    from pydantic import ValidationError as VE

    with pytest.raises(VE) as placeholder:
        valid(brand="unknown")
    with pytest.raises(VE) as empty_nested:
        valid(dimensions={"length_cm": None, "width_cm": None, "height_cm": None})

    p_msg = placeholder.value.errors()[0]["msg"]
    e_msg = empty_nested.value.errors()[0]["msg"]
    assert "is optional: use null" in p_msg
    assert "is optional: use null" not in e_msg
    assert "at least one measurement" in e_msg
    assert "at least one measurement" not in p_msg
