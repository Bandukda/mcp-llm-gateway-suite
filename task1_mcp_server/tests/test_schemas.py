"""Schema-level tests: every malformed-input path the reviewer will poke at."""

import pytest
from pydantic import ValidationError

from schemas import GetCustomerRecordInput, TriggerRefundInput


# --------------------------------------------------------------------------
# customer_id format
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "customer_id",
    ["CUST-10042", "CUST-00000", "CUST-99999"],
)
def test_accepts_well_formed_customer_id(customer_id):
    assert GetCustomerRecordInput(customer_id=customer_id).customer_id == customer_id


@pytest.mark.parametrize(
    ("customer_id", "why"),
    [
        ("CUST-1004", "four digits"),
        ("CUST-100425", "six digits"),
        ("cust-10042", "lowercase prefix"),
        ("CUST10042", "missing hyphen"),
        ("CUST-1004A", "non-digit in the numeric part"),
        (" CUST-10042", "leading whitespace"),
        ("CUST-10042 ", "trailing whitespace"),
        ("CUST-10042\n", "trailing newline: Pydantic anchors the whole string, but stdlib re $ would allow it"),
        ("CUST-10042\nCUST-10043", "embedded newline injection attempt"),
        ("", "empty"),
        ("CUST-١٠٠٤٢", "non-ASCII digits"),
    ],
)
def test_rejects_malformed_customer_id(customer_id, why):
    with pytest.raises(ValidationError):
        GetCustomerRecordInput(customer_id=customer_id)


@pytest.mark.parametrize("value", [10042, None, ["CUST-10042"], {"id": "CUST-10042"}, True])
def test_rejects_non_string_customer_id(value):
    with pytest.raises(ValidationError):
        GetCustomerRecordInput(customer_id=value)


def test_rejects_unknown_field():
    with pytest.raises(ValidationError) as exc:
        GetCustomerRecordInput(customer_id="CUST-10042", customerId="CUST-10042")
    assert "extra_forbidden" in {e["type"] for e in exc.value.errors()}


def test_rejects_missing_field():
    with pytest.raises(ValidationError) as exc:
        GetCustomerRecordInput()
    assert "missing" in {e["type"] for e in exc.value.errors()}


# --------------------------------------------------------------------------
# amount
# --------------------------------------------------------------------------
def test_accepts_valid_refund():
    args = TriggerRefundInput(
        customer_id="CUST-10042", amount=49.99, reason="Duplicate charge on the March invoice"
    )
    assert args.amount == 49.99


@pytest.mark.parametrize("amount", [0, -1, -0.01])
def test_rejects_non_positive_amount(amount):
    with pytest.raises(ValidationError):
        TriggerRefundInput(customer_id="CUST-10042", amount=amount, reason="Valid reason here")


def test_rejects_amount_above_ceiling():
    with pytest.raises(ValidationError):
        TriggerRefundInput(customer_id="CUST-10042", amount=10_000.01, reason="Valid reason here")


def test_rejects_string_amount_in_strict_mode():
    """Lax Pydantic would coerce "49.99" to 49.99. Money does not get guessed."""
    with pytest.raises(ValidationError):
        TriggerRefundInput(customer_id="CUST-10042", amount="49.99", reason="Valid reason here")


def test_rejects_bool_amount():
    """bool is a subclass of int; without strict mode True would become 1.0."""
    with pytest.raises(ValidationError):
        TriggerRefundInput(customer_id="CUST-10042", amount=True, reason="Valid reason here")


@pytest.mark.parametrize("amount", [float("inf"), float("-inf"), float("nan")])
def test_rejects_inf_and_nan(amount):
    """Python's json module parses the non-standard NaN/Infinity literals."""
    with pytest.raises(ValidationError):
        TriggerRefundInput(customer_id="CUST-10042", amount=amount, reason="Valid reason here")


def test_rejects_sub_cent_precision():
    with pytest.raises(ValidationError) as exc:
        TriggerRefundInput(customer_id="CUST-10042", amount=12.3456, reason="Valid reason here")
    assert "decimal places" in str(exc.value)


def test_accepts_integer_valued_float():
    assert TriggerRefundInput(
        customer_id="CUST-10042", amount=50.0, reason="Valid reason here"
    ).amount == 50.0


# --------------------------------------------------------------------------
# reason
# --------------------------------------------------------------------------
def test_rejects_short_reason():
    with pytest.raises(ValidationError):
        TriggerRefundInput(customer_id="CUST-10042", amount=10.0, reason="too short")


def test_rejects_whitespace_only_reason():
    """12 spaces satisfies min_length=10 but is not a reason."""
    with pytest.raises(ValidationError):
        TriggerRefundInput(customer_id="CUST-10042", amount=10.0, reason=" " * 12)


def test_reason_is_stored_stripped():
    args = TriggerRefundInput(
        customer_id="CUST-10042", amount=10.0, reason="  Duplicate charge on invoice  "
    )
    assert args.reason == "Duplicate charge on invoice"


def test_rejects_overlong_reason():
    with pytest.raises(ValidationError):
        TriggerRefundInput(customer_id="CUST-10042", amount=10.0, reason="x" * 501)


# --------------------------------------------------------------------------
# idempotency_key
# --------------------------------------------------------------------------
def test_idempotency_key_optional():
    assert TriggerRefundInput(
        customer_id="CUST-10042", amount=10.0, reason="Valid reason here"
    ).idempotency_key is None


def test_rejects_short_idempotency_key():
    with pytest.raises(ValidationError):
        TriggerRefundInput(
            customer_id="CUST-10042", amount=10.0, reason="Valid reason here", idempotency_key="short"
        )
