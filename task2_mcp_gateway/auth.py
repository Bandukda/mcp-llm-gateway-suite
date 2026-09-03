"""Bearer token -> principal resolution for the MCP gateway.

The token table here is a stand-in. In production this function is the seam
where you drop in JWT verification (``PyJWT`` + JWKS) or RFC 7662 token
introspection; nothing else in the gateway changes, because everything
downstream of ``resolve_principal`` only sees a ``Principal``.

Two details that are easy to get wrong and are implemented properly here:

* **Constant-time comparison.** Tokens are looked up by SHA-256 digest rather
  than by raw string. A plain ``dict[token]`` lookup on a secret leaks timing
  information about the prefix; hashing first makes every lookup do the same
  work regardless of how close the guess was.
* **The scheme is case-insensitive but the token is not.** RFC 7235 defines the
  auth-scheme as case-insensitive (``bearer``, ``Bearer``, ``BEARER`` are all
  legal), while the credential itself is opaque and must be compared exactly.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass


class AuthError(Exception):
    """The caller did not present a credential we can resolve."""

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str
    tenant: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# Demo credentials. Never ship a table like this; see the module docstring.
_TOKEN_TABLE: dict[str, Principal] = {
    _digest("admin-token-abc123"): Principal(subject="ada@example.com", role="admin", tenant="acme"),
    _digest("viewer-token-def456"): Principal(subject="grace@example.com", role="viewer", tenant="acme"),
    _digest("viewer-token-ghi789"): Principal(subject="alan@example.com", role="viewer", tenant="globex"),
}

KNOWN_ROLES = frozenset({"admin", "viewer"})


def extract_bearer_token(authorization_header: str | None) -> str:
    """Pull the credential out of an ``Authorization`` header.

    Raises:
        AuthError: header missing, wrong scheme, or empty credential.
    """
    if not authorization_header:
        raise AuthError("Missing Authorization header", reason="missing_header")

    parts = authorization_header.split(None, 1)
    if len(parts) != 2:
        raise AuthError("Malformed Authorization header", reason="malformed_header")

    scheme, credential = parts
    if scheme.lower() != "bearer":
        raise AuthError(f"Unsupported authorization scheme: {scheme}", reason="bad_scheme")

    # split(None, 1) collapses runs of whitespace, so "Bearer " and "Bearer    "
    # both yield a single part and are classified as malformed_header above --
    # this branch is unreachable via that path. It stays because the function is
    # public and a caller may hand in a pre-split credential.
    credential = credential.strip()
    if not credential:
        raise AuthError("Empty bearer token", reason="empty_token")
    return credential


def resolve_principal(authorization_header: str | None) -> Principal:
    """Resolve an ``Authorization`` header to the identity behind it.

    Raises:
        AuthError: the header is unusable or the token is unknown.
    """
    token = extract_bearer_token(authorization_header)

    candidate = _digest(token)
    for known_digest, principal in _TOKEN_TABLE.items():
        if hmac.compare_digest(candidate, known_digest):
            return principal

    # Deliberately identical message for "no such token" and "revoked token":
    # a caller learns only that the credential did not work.
    raise AuthError("Invalid or expired bearer token", reason="unknown_token")
