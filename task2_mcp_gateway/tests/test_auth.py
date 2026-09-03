"""Authentication: who are you, and what happens when we cannot tell."""

import pytest

from auth import AuthError, extract_bearer_token, resolve_principal


@pytest.mark.parametrize(
    "header",
    ["Bearer admin-token-abc123", "bearer admin-token-abc123", "BEARER admin-token-abc123"],
)
def test_scheme_is_case_insensitive(header):
    """RFC 7235: the auth-scheme is case-insensitive."""
    assert resolve_principal(header).role == "admin"


def test_token_is_case_sensitive():
    with pytest.raises(AuthError):
        resolve_principal("Bearer ADMIN-TOKEN-ABC123")


@pytest.mark.parametrize(
    ("header", "reason"),
    [
        (None, "missing_header"),
        ("", "missing_header"),
        ("admin-token-abc123", "malformed_header"),
        ("Basic dXNlcjpwYXNz", "bad_scheme"),
        ("Bearer ", "malformed_header"),
        ("Bearer    ", "malformed_header"),
    ],
)
def test_malformed_headers_are_classified(header, reason):
    with pytest.raises(AuthError) as exc:
        extract_bearer_token(header)
    assert exc.value.reason == reason


def test_unknown_token_message_does_not_distinguish_cases():
    """An attacker must not learn whether a token existed and was revoked."""
    with pytest.raises(AuthError) as exc:
        resolve_principal("Bearer totally-made-up-token")
    assert str(exc.value) == "Invalid or expired bearer token"


def test_roles_and_tenants_resolve():
    assert resolve_principal("Bearer viewer-token-def456").tenant == "acme"
    assert resolve_principal("Bearer viewer-token-ghi789").tenant == "globex"
    assert resolve_principal("Bearer admin-token-abc123").is_admin is True
    assert resolve_principal("Bearer viewer-token-def456").is_admin is False
