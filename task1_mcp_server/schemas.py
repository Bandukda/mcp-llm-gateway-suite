r"""Strict Pydantic input schemas for the billing MCP server.

Design notes that matter for review
-----------------------------------
* ``extra="forbid"``  -- an argument the tool does not understand is a bug in the
  caller, not something to ignore. Silently dropping ``{"ammount": 50}`` (typo)
  would make a refund of the wrong size feel like a success.
* ``strict=True`` on money -- Pydantic in lax mode happily turns ``"50"`` into
  ``50.0`` and ``True`` into ``1.0`` (``bool`` is a subclass of ``int``).
  For a field that moves money, an ambiguous type is rejected, not guessed.
* ``allow_inf_nan=False`` -- Python's ``json`` module accepts the non-standard
  literals ``NaN`` / ``Infinity``. Without this flag ``amount: Infinity`` passes
  a naive ``> 0`` check.
* Decimal-place check -- currency has two decimals; ``12.3456`` is a client bug.
* ``reason`` is validated *after* stripping, so ten spaces is not a reason.
* ``[0-9]`` not ``\d`` in the id pattern -- see the note on CUSTOMER_ID_PATTERN.

In production the amount would be ``Decimal`` (or minor units as ``int``) rather
than ``float``; ``float`` is kept here because the task specifies "positive
float" and because it keeps the generated JSON Schema a plain ``number``.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# NOTE: [0-9] rather than \d on purpose. In Python's ``re`` -- and in several
# JSON Schema validators -- ``\d`` is Unicode-aware and matches Arabic-Indic
# digits, Devanagari digits and friends, so "CUST-\u0661\u0660\u0660\u0664\u0662"
# would sail through a ``^CUST-\d{5}$`` check and then blow up downstream in a
# system that only understands ASCII ids. A test in tests/test_schemas.py pins
# this behaviour.
CUSTOMER_ID_PATTERN = r"^CUST-[0-9]{5}$"
MAX_REFUND_AMOUNT = 10_000.0
MIN_REASON_LENGTH = 10
MAX_REASON_LENGTH = 500


class StrictModel(BaseModel):
    """Base config shared by every tool input model."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=False,
        validate_assignment=True,
    )


class GetCustomerRecordInput(StrictModel):
    """Arguments for ``get_customer_record``."""

    customer_id: str = Field(
        ...,
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer identifier in the form CUST-XXXXX where X is a digit, e.g. CUST-10042.",
        examples=["CUST-10042"],
    )


class TriggerRefundInput(StrictModel):
    """Arguments for ``trigger_refund``."""

    customer_id: str = Field(
        ...,
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer identifier in the form CUST-XXXXX where X is a digit, e.g. CUST-10042.",
        examples=["CUST-10042"],
    )
    amount: float = Field(
        ...,
        gt=0,
        le=MAX_REFUND_AMOUNT,
        allow_inf_nan=False,
        description=(
            "Refund amount in USD. Must be a positive number with at most two "
            f"decimal places and at most {MAX_REFUND_AMOUNT:,.2f}."
        ),
        examples=[49.99],
    )
    reason: str = Field(
        ...,
        min_length=MIN_REASON_LENGTH,
        max_length=MAX_REASON_LENGTH,
        description=(
            "Human-readable justification for the refund, at least "
            f"{MIN_REASON_LENGTH} characters after trimming whitespace."
        ),
        examples=["Duplicate charge on the March invoice"],
    )
    idempotency_key: str | None = Field(
        default=None,
        min_length=8,
        max_length=64,
        description=(
            "Optional caller-supplied key. Replaying the same key returns the "
            "original refund instead of issuing a second one. Recommended for "
            "agent callers, which retry."
        ),
    )

    @field_validator("amount")
    @classmethod
    def _two_decimal_places(cls, value: float) -> float:
        try:
            exponent = Decimal(str(value)).normalize().as_tuple().exponent
        except InvalidOperation:  # pragma: no cover - guarded by allow_inf_nan
            raise ValueError("amount must be a finite decimal number")
        if isinstance(exponent, int) and exponent < -2:
            raise ValueError("amount must have at most 2 decimal places (currency precision)")
        return value

    @field_validator("reason")
    @classmethod
    def _reason_is_substantive(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < MIN_REASON_LENGTH:
            raise ValueError(
                f"reason must contain at least {MIN_REASON_LENGTH} non-whitespace characters"
            )
        return stripped


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Return the JSON Schema advertised in ``tools/list`` for a model.

    Pydantic emits ``$defs``/``allOf`` wrappers for some field types. MCP clients
    generally cope, but a flat schema is easier for a model to follow, so the
    schema is emitted in ``validation`` mode and the title is dropped.
    """
    schema = model.model_json_schema(mode="validation")
    schema.pop("title", None)
    return schema
