"""The contract an LLM has to satisfy when it extracts product attributes.

Deliberately awkward, because an easy schema produces a boring violation table
on Saturday. Four hard parts, one per thing models actually get wrong:

* a **nested object** (`dimensions`) they can produce empty,
* an **enum** (`category`) they can invent members of,
* **optionals that mean "absent"**, which they answer with the string
  `"unknown"`,
* a **ranged integer** they answer with `"2 pack"`.

`ProductAttributes.model_json_schema()` is not just documentation — it is
literally the `parameters` block of a tool definition in P2, and the
structured-output schema on Saturday. Run this file to see it.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

ASIN = re.compile(r"^[A-Z0-9]{10}$")

# The strings a model reaches for when it does not know. Every one of these is a
# contract violation: the schema says use null, and the difference matters —
# `null` means "this product has no colour", `"unknown"` means "I did not look".
PLACEHOLDERS = frozenset(
    {
        "unknown",
        "n/a",
        "na",
        "none",
        "null",
        "not specified",
        "not applicable",
        "unspecified",
        "not available",
        "-",
        "?",
    }
)


class Category(str, Enum):
    FOOTWEAR = "footwear"
    APPAREL = "apparel"
    ELECTRONICS = "electronics"
    HOME = "home"
    BEAUTY = "beauty"
    SUPPLEMENT = "supplement"
    BOOKS = "books"
    OTHER = "other"


class Dimensions(BaseModel):
    length_cm: float | None = Field(default=None, gt=0)
    width_cm: float | None = Field(default=None, gt=0)
    height_cm: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def at_least_one_measurement(self) -> Dimensions:
        """An all-null dimensions object is structure without content.

        This is a specific and common failure: the model sees a nested object in
        the schema and fills in its *shape* to look compliant. The correct answer
        when nothing is measurable is `"dimensions": null`, not
        `{"length_cm": null, "width_cm": null, "height_cm": null}`.
        """
        if self.length_cm is None and self.width_cm is None and self.height_cm is None:
            raise ValueError(
                "dimensions must carry at least one measurement; use null instead"
            )
        return self


class ProductAttributes(BaseModel):
    asin: str
    brand: str | None = None
    category: Category
    colour: str | None = None
    size: str | None = None
    pack_quantity: int = Field(default=1, ge=1, le=1000)
    dimensions: Dimensions | None = None

    @field_validator("asin")
    @classmethod
    def asin_is_well_formed(cls, value: str) -> str:
        value = value.strip().upper()
        if not ASIN.match(value):
            raise ValueError(f"not an ASIN: {value!r}")
        return value

    @field_validator("brand", "colour", "size", mode="before")
    @classmethod
    def absent_means_null(cls, value: object, info: ValidationInfo) -> object:
        """Empty string is absent; a placeholder word is a violation.

        Note `mode="before"`: this runs on the raw JSON value, ahead of type
        coercion, which is the only place a stray `""` can still be turned into
        `None` rather than a valid-but-meaningless empty string.

        And note what this deliberately does *not* do — it does not rewrite
        `"unknown"` to `None`. Coercing bad input into good output would make
        Saturday's violation-rate table measure this function instead of the
        model. A repair loop is allowed to fix things; a schema is not.
        """
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            if cleaned.casefold() in PLACEHOLDERS:
                raise ValueError(
                    f"{info.field_name} is optional: use null, not {value!r}"
                )
            return cleaned
        return value

    @model_validator(mode="after")
    def wearables_need_a_size(self) -> ProductAttributes:
        """Cross-field rules are why `model_validator` exists.

        No individual field is wrong here — `category` is a valid enum member and
        `size` is a valid optional. Only the combination is incoherent, and a
        field validator cannot see a combination.
        """
        if self.category in (Category.FOOTWEAR, Category.APPAREL) and not self.size:
            raise ValueError(f"{self.category.value} requires a size")
        return self


if __name__ == "__main__":
    import json

    print(json.dumps(ProductAttributes.model_json_schema(), indent=2))
