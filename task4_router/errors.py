"""One error shape for the whole gateway, and one place that decides what leaks.

The rule: the client gets a stable, machine-readable envelope and a human
sentence. Everything else -- upstream response bodies, exception text, tracebacks,
internal hostnames, provider names, API keys that ended up in a URL -- goes to
the log, correlated by ``request_id``.

Why it matters beyond tidiness: upstream errors routinely embed the thing that
failed. ``httpx`` puts the target host and port in ``ConnectError``; provider
400s quote the offending request back, sometimes including the Authorization
header; a stack trace names your file layout and library versions. Each of those
is free reconnaissance, and the client can do nothing useful with any of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class GatewayErrorCode:
    RATE_LIMITED = "rate_limited"
    ALL_PROVIDERS_FAILED = "all_providers_failed"
    UPSTREAM_TIMEOUT = "upstream_timeout"
    UPSTREAM_ERROR = "upstream_error"
    BAD_REQUEST = "bad_request"
    UNAUTHENTICATED = "unauthenticated"
    INTERNAL = "internal_error"


# The only messages that ever reach a client.
SAFE_MESSAGES: dict[str, str] = {
    GatewayErrorCode.RATE_LIMITED: "Token rate limit exceeded for this API key.",
    GatewayErrorCode.ALL_PROVIDERS_FAILED: "All model providers are currently unavailable.",
    GatewayErrorCode.UPSTREAM_TIMEOUT: "The model provider did not respond in time.",
    GatewayErrorCode.UPSTREAM_ERROR: "The model provider returned an error.",
    GatewayErrorCode.BAD_REQUEST: "The request payload was invalid.",
    GatewayErrorCode.UNAUTHENTICATED: "A valid API key is required.",
    GatewayErrorCode.INTERNAL: "An internal error occurred.",
}

HTTP_STATUS: dict[str, int] = {
    GatewayErrorCode.RATE_LIMITED: 429,
    GatewayErrorCode.ALL_PROVIDERS_FAILED: 503,
    GatewayErrorCode.UPSTREAM_TIMEOUT: 504,
    GatewayErrorCode.UPSTREAM_ERROR: 502,
    GatewayErrorCode.BAD_REQUEST: 400,
    GatewayErrorCode.UNAUTHENTICATED: 401,
    GatewayErrorCode.INTERNAL: 500,
}


@dataclass
class GatewayError(Exception):
    code: str
    request_id: str
    detail: str | None = None          # logged, never serialised
    metadata: dict[str, Any] | None = None  # serialised; must be caller-safe

    def __post_init__(self) -> None:
        super().__init__(self.detail or SAFE_MESSAGES.get(self.code, "error"))

    @property
    def http_status(self) -> int:
        return HTTP_STATUS.get(self.code, 500)

    def to_payload(self) -> dict[str, Any]:
        """The exact JSON the client receives. ``detail`` is deliberately absent."""
        error: dict[str, Any] = {
            "type": "gateway_error",
            "code": self.code,
            "message": SAFE_MESSAGES.get(self.code, "An error occurred."),
            "request_id": self.request_id,
        }
        if self.metadata:
            error.update(self.metadata)
        return {"error": error}
