"""The property: no raw upstream detail reaches the client.

Every test here asserts a *negative* -- that something is absent from the
response body. Negatives are the only useful shape for this property: a leak is
never a failing assertion elsewhere, it is an extra string nobody looked for.
"""

import json

import httpx
import pytest
import pytest_asyncio

from app import create_app
from errors import GatewayError, GatewayErrorCode
from providers import MockProvider
from rate_limiter import SlidingWindowRateLimiter
from router import ModelRouter

SECRETS = [
    "10.4.2.19",
    "8443",
    "sk-live-SUPERSECRET0123456789",
    "/opt/gateway/providers.py",
    "Traceback",
    "line 91",
]


def build_app(primary="ok", secondary="ok", db_path=None, limit=50_000, **kwargs):
    limiter = SlidingWindowRateLimiter(db_path, limit_tokens_per_minute=limit)
    router = ModelRouter(
        MockProvider("primary", primary),
        MockProvider("secondary", secondary),
        limiter,
        **kwargs,
    )
    return create_app(router=router)


@pytest_asyncio.fixture
async def client_factory(db_path):
    clients = []

    def _build(**kwargs):
        app = build_app(db_path=db_path, **kwargs)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")
        clients.append(client)
        return client

    yield _build
    for client in clients:
        await client.aclose()


AUTH = {"Authorization": "Bearer tenant-a-key"}


# ---------------------------------------------------------------------------
# Happy path, so the negative tests mean something
# ---------------------------------------------------------------------------
async def test_successful_completion(client_factory):
    client = client_factory()
    response = await client.post(
        "/v1/completions", json={"prompt": "hello", "max_tokens": 64}, headers=AUTH
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "primary"
    assert body["usage"]["total_tokens"] > 0


async def test_failover_is_reported_to_the_client(client_factory):
    client = client_factory(primary="rate_limit")
    response = await client.post(
        "/v1/completions", json={"prompt": "hello", "max_tokens": 64}, headers=AUTH
    )
    assert response.status_code == 200
    assert response.json()["failed_over"] is True
    assert response.json()["provider"] == "secondary"


# ---------------------------------------------------------------------------
# Leak tests
# ---------------------------------------------------------------------------
async def test_upstream_exception_text_never_reaches_the_client(client_factory):
    """The primary raises with an internal IP, an API key and a file path in it."""
    client = client_factory(primary="leaky", secondary="leaky")
    response = await client.post(
        "/v1/completions", json={"prompt": "hello", "max_tokens": 64}, headers=AUTH
    )
    assert response.status_code == 503
    body = response.text
    for secret in SECRETS:
        assert secret not in body, f"leaked {secret!r}"


async def test_error_envelope_is_stable(client_factory):
    client = client_factory(primary="server_error", secondary="server_error")
    response = await client.post(
        "/v1/completions", json={"prompt": "hello"}, headers=AUTH
    )
    error = response.json()["error"]
    assert error["type"] == "gateway_error"
    assert error["code"] == GatewayErrorCode.ALL_PROVIDERS_FAILED
    assert error["message"] == "All model providers are currently unavailable."
    assert error["request_id"]


async def test_detail_field_is_never_serialised():
    """Unit-level guarantee: to_payload() cannot emit `detail`."""
    error = GatewayError(
        code=GatewayErrorCode.UPSTREAM_ERROR,
        request_id="req-1",
        detail="postgres://user:hunter2@10.0.0.5/prod blew up",
    )
    assert "hunter2" not in json.dumps(error.to_payload())
    assert "detail" not in error.to_payload()["error"]


async def test_attempt_metadata_carries_no_provider_internals(db_path):
    """Attempt records are useful to a client; they must stay generic.

    Named after the real provider names the app uses, because an earlier version
    of this test only re-ran the SECRETS loop and passed purely because the mocks
    were called "primary" and "secondary" -- it would have kept passing while the
    envelope shipped the whole vendor topology.
    """
    limiter = SlidingWindowRateLimiter(db_path, limit_tokens_per_minute=50_000)
    router = ModelRouter(
        MockProvider("primary-openai-us-east", "leaky"),
        MockProvider("secondary-anthropic-eu", "server_error"),
        limiter,
    )
    app = create_app(router=router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw"
    ) as client:
        response = await client.post("/v1/completions", json={"prompt": "hello"}, headers=AUTH)

    body = response.text
    for secret in SECRETS:
        assert secret not in body
    for vendor in ("openai", "anthropic", "us-east", "eu"):
        assert vendor not in body.lower(), f"leaked provider identity: {vendor}"
    assert "503" not in body, "leaked the raw upstream status code"

    error = response.json()["error"]
    assert error["providers_attempted"] == 2
    assert error["outcomes"] == ["retryable_error", "retryable_error"]


async def test_a_bug_in_the_gateway_returns_a_clean_500(db_path):
    """Even an unhandled exception must not put a traceback on the wire."""

    class ExplodingRouter:
        limiter = None

        async def route(self, **kwargs):
            raise RuntimeError("BOOM /opt/gateway/router.py secret=hunter2")

    app = create_app(router=ExplodingRouter())
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
        response = await client.post("/v1/completions", json={"prompt": "hi"}, headers=AUTH)

    assert response.status_code == 500
    assert "hunter2" not in response.text
    assert "Traceback" not in response.text
    assert response.json()["error"]["code"] == GatewayErrorCode.INTERNAL


# ---------------------------------------------------------------------------
# Status codes and headers
# ---------------------------------------------------------------------------
async def test_rate_limit_returns_429_with_retry_after(db_path):
    app = build_app(db_path=db_path, limit=100)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw"
    ) as client:
        response = await client.post(
            "/v1/completions", json={"prompt": "hello", "max_tokens": 500}, headers=AUTH
        )
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1
    assert response.json()["error"]["code"] == GatewayErrorCode.RATE_LIMITED


async def test_timeout_returns_504(client_factory):
    client = client_factory(primary="timeout", secondary="timeout", attempt_timeout_s=0.05)
    response = await client.post("/v1/completions", json={"prompt": "hello"}, headers=AUTH)
    assert response.status_code == 504


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "tenant-a-key"}, {"Authorization": "Bearer "}]
)
async def test_missing_credentials_return_401(client_factory, headers):
    client = client_factory()
    response = await client.post("/v1/completions", json={"prompt": "hello"}, headers=headers)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == GatewayErrorCode.UNAUTHENTICATED


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompt": ""},
        {"prompt": "hi", "max_tokens": 0},
        {"prompt": "hi", "max_tokens": -1},
        {"prompt": "hi", "max_tokens": 99_999},
        {"prompt": "hi", "temperature": 0.5},
        {"prompt": 42},
    ],
)
async def test_invalid_payloads_return_400(client_factory, payload):
    client = client_factory()
    response = await client.post("/v1/completions", json=payload, headers=AUTH)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == GatewayErrorCode.BAD_REQUEST


async def test_validation_errors_do_not_echo_the_payload(client_factory):
    """A pydantic error message quotes the offending input verbatim."""
    client = client_factory()
    response = await client.post(
        "/v1/completions",
        json={"prompt": "hi", "api_key_i_pasted_by_mistake": "sk-live-OOPS0123456789"},
        headers=AUTH,
    )
    assert response.status_code == 400
    assert "sk-live-OOPS0123456789" not in response.text


async def test_request_id_is_echoed_for_correlation(client_factory):
    client = client_factory(primary="server_error", secondary="server_error")
    response = await client.post(
        "/v1/completions",
        json={"prompt": "hello"},
        headers={**AUTH, "X-Request-Id": "trace-xyz"},
    )
    assert response.headers["X-Request-Id"] == "trace-xyz"
    assert response.json()["error"]["request_id"] == "trace-xyz"


async def test_usage_endpoint(client_factory):
    client = client_factory()
    await client.post("/v1/completions", json={"prompt": "hello"}, headers=AUTH)
    response = await client.get("/v1/usage", headers=AUTH)
    body = response.json()
    assert body["limit_tokens"] == 50_000
    assert 0 < body["used_tokens"] < 100
    assert body["remaining_tokens"] == body["limit_tokens"] - body["used_tokens"]


@pytest.mark.parametrize(
    ("body", "why"),
    [
        (b"[" * 20000, "deep nesting raises RecursionError, not ValueError"),
        (b'{"prompt":' + b"9" * 4401 + b"}", "int over the digit limit raises ValueError"),
        (b"{not json", "ordinary JSONDecodeError"),
    ],
)
async def test_unparseable_bodies_are_400_not_500(client_factory, body, why):
    client = client_factory()
    response = await client.post(
        "/v1/completions",
        content=body,
        headers={**AUTH, "Content-Type": "application/json"},
    )
    assert response.status_code == 400, why
    assert response.json()["error"]["code"] == GatewayErrorCode.BAD_REQUEST
