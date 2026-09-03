"""Policy: the prefix rule, and the ways a caller might try to slip past it."""

import pytest

from auth import Principal
from policy import ToolPolicy, is_wellformed_tool_name, normalize_tool_name

ADMIN = Principal(subject="ada@example.com", role="admin", tenant="acme")
VIEWER = Principal(subject="grace@example.com", role="viewer", tenant="acme")


@pytest.fixture
def policy():
    return ToolPolicy()


def test_viewer_may_call_ordinary_tools(policy):
    assert policy.evaluate("get_customer_record", VIEWER).allowed is True


def test_viewer_may_not_call_admin_tools(policy):
    decision = policy.evaluate("admin_reset_key", VIEWER)
    assert decision.allowed is False
    assert decision.rule == "prefix:admin_"


def test_admin_may_call_admin_tools(policy):
    assert policy.evaluate("admin_reset_key", ADMIN).allowed is True


@pytest.mark.parametrize(
    "name",
    [
        "Admin_reset_key",
        "ADMIN_RESET_KEY",
        "AdMiN_reset_key",
        " admin_reset_key",
        "admin_reset_key ",
    ],
)
def test_case_and_whitespace_variants_are_still_admin_tools(policy, name):
    """Evaluated on a normalised copy; the downstream may well be lenient."""
    assert policy.evaluate(name, VIEWER).allowed is False


def test_normalisation_folds_fullwidth_characters():
    assert normalize_tool_name("ａdmin_reset_key") == "admin_reset_key"


# -- charset guard ---------------------------------------------------------
@pytest.mark.parametrize(
    "name", ["get_customer_record", "admin_reset_key", "a", "tool.v2", "tool-name"]
)
def test_wellformed_names_accepted(name):
    assert is_wellformed_tool_name(name) is True


@pytest.mark.parametrize(
    ("name", "why"),
    [
        ("Аdmin_reset_key", "Cyrillic A homoglyph; no normal form folds it to Latin A"),
        ("admin_reset_key\n", "trailing newline: Python's $ allows it, \\Z does not"),
        ("admin reset key", "spaces"),
        ("admin/../reset", "path traversal shape"),
        ("admin_reset_key\x00", "embedded NUL"),
        ("", "empty"),
        ("x" * 129, "over length"),
        (None, "not a string at all"),
    ],
)
def test_malformed_names_rejected(name, why):
    assert is_wellformed_tool_name(name) is False


def test_homoglyph_is_rejected_by_charset_not_by_normalisation(policy):
    """The important half of the homoglyph story.

    normalize_tool_name() does NOT turn Cyrillic A into Latin A, and no Unicode
    normal form ever will -- they are distinct characters, not compatibility
    variants. So the policy alone would let it through; the charset guard is
    what stops it, before the policy is ever consulted.
    """
    sneaky = "Аdmin_reset_key"
    assert policy.evaluate(sneaky, VIEWER).allowed is True  # policy is blind to it
    assert is_wellformed_tool_name(sneaky) is False  # the charset guard is not


def test_deny_by_default_mode():
    strict = ToolPolicy(prefix_roles={"admin_": frozenset({"admin"})}, default_allow=False)
    decision = strict.evaluate("some_unlisted_tool", ADMIN)
    assert decision.allowed is False
    assert decision.rule == "default_deny"


def test_policy_is_data_not_code():
    """Adding a rule is a dict entry, not a branch in the request handler."""
    finance = ToolPolicy(
        prefix_roles={"admin_": frozenset({"admin"}), "billing_": frozenset({"admin", "finance"})}
    )
    assert finance.evaluate("billing_issue_credit", VIEWER).allowed is False
    assert finance.evaluate("billing_issue_credit", ADMIN).allowed is True
