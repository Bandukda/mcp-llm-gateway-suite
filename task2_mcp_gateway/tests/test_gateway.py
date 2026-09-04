"""End-to-end proxy behaviour, driven over ASGI against a real downstream app."""

import json

import httpx
import pytest

from gateway_helpers import ADMIN, VIEWER, call_tool, rpc


# ---------------------------------------------------------------------------
# Authentication (transport layer)
# ---------------------------------------------------------------------------
async def test_missing_token_is_401_with_challenge(gateway):
    response = await gateway.post("/mcp", json=rpc("tools/list", {}))
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"].startswith("Bearer ")
    assert response.json()["error"]["code"] == -32002


async def test_bad_token_is_401(gateway):
    response = await gateway.post(
        "/mcp", json=rpc("tools/list", {}), headers={"Authorization": "Bearer nope"}
    )
    assert response.status_code == 401


async def test_unauthenticated_request_never_reaches_downstream(gateway, downstream_app):
    await gateway.post("/mcp", json=rpc("tools/list", {}))
    assert downstream_app.state.calls == []


# ---------------------------------------------------------------------------
# tools/list forwards transparently
# ---------------------------------------------------------------------------
async def test_tools_list_is_forwarded_transparently(gateway, downstream_app):
    response = await gateway.post(
        "/mcp", json=rpc("tools/list", {}), headers={"Authorization": VIEWER}
    )
    assert response.status_code == 200
    names = {tool["name"] for tool in response.json()["result"]["tools"]}
    assert names == {
        "get_customer_record",
        "search_orders",
        "admin_reset_key",
        "admin_delete_tenant",
    }
    assert downstream_app.state.calls[-1]["method"] == "tools/list"


async def test_tools_list_can_be_filtered_when_enabled(gateway_factory):
    """Opt-in hardening: do not advertise tools the caller cannot call."""
    gateway = gateway_factory(filter_tools_list=True)
    response = await gateway.post(
        "/mcp", json=rpc("tools/list", {}), headers={"Authorization": VIEWER}
    )
    names = {tool["name"] for tool in response.json()["result"]["tools"]}
    assert names == {"get_customer_record", "search_orders"}

    response = await gateway.post(
        "/mcp", json=rpc("tools/list", {}), headers={"Authorization": ADMIN}
    )
    assert len(response.json()["result"]["tools"]) == 4


# ---------------------------------------------------------------------------
# tools/call authorization -- the core of the task
# ---------------------------------------------------------------------------
async def test_viewer_may_call_ordinary_tool(gateway, downstream_app):
    response = await gateway.post(
        "/mcp",
        json=call_tool("get_customer_record", {"customer_id": "CUST-10042"}),
        headers={"Authorization": VIEWER},
    )
    assert response.status_code == 200
    assert response.json()["result"]["isError"] is False
    assert downstream_app.state.calls[-1]["params"]["name"] == "get_customer_record"


async def test_viewer_calling_admin_tool_is_32001(gateway):
    response = await gateway.post(
        "/mcp",
        json=call_tool("admin_reset_key", {"tenant": "acme"}),
        headers={"Authorization": VIEWER},
    )
    assert response.status_code == 200  # the request was answered; the answer is "no"
    body = response.json()
    assert body["error"]["code"] == -32001
    assert body["error"]["message"] == "Unauthorized Tool Call"
    assert body["id"] == 1  # the request id is preserved


async def test_denied_call_never_reaches_downstream(gateway, downstream_app):
    await gateway.post(
        "/mcp",
        json=call_tool("admin_reset_key", {"tenant": "acme"}),
        headers={"Authorization": VIEWER},
    )
    assert downstream_app.state.calls == []


async def test_admin_may_call_admin_tool(gateway, downstream_app):
    response = await gateway.post(
        "/mcp",
        json=call_tool("admin_reset_key", {"tenant": "acme"}),
        headers={"Authorization": ADMIN},
    )
    assert response.status_code == 200
    assert "error" not in response.json()
    assert downstream_app.state.calls[-1]["params"]["name"] == "admin_reset_key"


@pytest.mark.parametrize("name", ["Admin_reset_key", "ADMIN_RESET_KEY", "AdMiN_reset_key"])
async def test_case_evasion_is_denied_by_policy(gateway, downstream_app, name):
    """Case variants are well-formed names, so the *policy* is what stops them."""
    response = await gateway.post(
        "/mcp", json=call_tool(name, {"tenant": "acme"}), headers={"Authorization": VIEWER}
    )
    assert response.json()["error"]["code"] == -32001
    assert downstream_app.state.calls == []


@pytest.mark.parametrize("name", [" admin_reset_key", "admin_reset_key ", "admin_reset_key\n"])
async def test_whitespace_evasion_is_denied_by_the_charset_guard(gateway, downstream_app, name):
    """Padded names never reach the policy: the charset guard rejects them first.

    Two layers, two error codes, one outcome -- the call does not happen. -32602
    is the honest code here: the name is not a name, so there is nothing to
    authorize. Asserting the specific code (rather than "some error") is what
    keeps the layering from silently collapsing into one check later.
    """
    response = await gateway.post(
        "/mcp", json=call_tool(name, {"tenant": "acme"}), headers={"Authorization": VIEWER}
    )
    assert response.json()["error"]["code"] == -32602
    assert downstream_app.state.calls == []


async def test_homoglyph_name_is_rejected_as_invalid_params(gateway, downstream_app):
    response = await gateway.post(
        "/mcp",
        json=call_tool("Аdmin_reset_key", {"tenant": "acme"}),
        headers={"Authorization": VIEWER},
    )
    assert response.json()["error"]["code"] == -32602
    assert downstream_app.state.calls == []


async def test_original_tool_name_is_forwarded_unmodified(gateway, downstream_app):
    """The gateway judges a normalised copy but must not rewrite the request."""
    await gateway.post(
        "/mcp",
        json=call_tool("Search_Orders", {"customer_id": "CUST-10042"}),
        headers={"Authorization": VIEWER},
    )
    assert downstream_app.state.calls[-1]["params"]["name"] == "Search_Orders"


# ---------------------------------------------------------------------------
# Malformed payloads
# ---------------------------------------------------------------------------
async def test_invalid_json_is_parse_error(gateway):
    response = await gateway.post(
        "/mcp",
        content=b"{not json",
        headers={"Authorization": VIEWER, "Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32700


async def test_missing_jsonrpc_version_is_invalid_request(gateway):
    response = await gateway.post(
        "/mcp", json={"id": 1, "method": "tools/list"}, headers={"Authorization": VIEWER}
    )
    assert response.json()["error"]["code"] == -32600


@pytest.mark.parametrize("params", [None, [], "admin_reset_key", 42])
async def test_non_object_params_is_invalid_params(gateway, params):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params}
    response = await gateway.post("/mcp", json=body, headers={"Authorization": VIEWER})
    assert response.json()["error"]["code"] == -32602


@pytest.mark.parametrize("name", [None, "", 42, ["admin_reset_key"]])
async def test_bad_tool_name_type_is_invalid_params(gateway, name):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name}}
    response = await gateway.post("/mcp", json=body, headers={"Authorization": VIEWER})
    assert response.json()["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# Header hygiene -- the confused deputy problem
# ---------------------------------------------------------------------------
async def test_client_token_is_not_forwarded_downstream(gateway, downstream_app):
    await gateway.post(
        "/mcp",
        json=call_tool("get_customer_record", {"customer_id": "CUST-10042"}),
        headers={"Authorization": VIEWER},
    )
    forwarded = downstream_app.state.calls[-1]["headers"]
    assert "viewer-token-def456" not in json.dumps(forwarded)
    assert forwarded["authorization"] == "Bearer gateway-service-token"


async def test_identity_is_asserted_as_headers(gateway, downstream_app):
    await gateway.post(
        "/mcp",
        json=call_tool("get_customer_record", {"customer_id": "CUST-10042"}),
        headers={"Authorization": VIEWER},
    )
    forwarded = downstream_app.state.calls[-1]["headers"]
    assert forwarded["x-mcp-gateway-subject"] == "grace@example.com"
    assert forwarded["x-mcp-gateway-role"] == "viewer"
    assert forwarded["x-mcp-gateway-tenant"] == "acme"
    assert forwarded["x-request-id"]


async def test_request_id_is_propagated_when_supplied(gateway, downstream_app):
    response = await gateway.post(
        "/mcp",
        json=call_tool("get_customer_record", {"customer_id": "CUST-10042"}),
        headers={"Authorization": VIEWER, "X-Request-Id": "trace-abc-123"},
    )
    assert response.headers["X-Request-Id"] == "trace-abc-123"
    assert downstream_app.state.calls[-1]["headers"]["x-request-id"] == "trace-abc-123"


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
async def test_batch_rejected_by_default(gateway):
    response = await gateway.post(
        "/mcp", json=[rpc("tools/list", {}, 1)], headers={"Authorization": VIEWER}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32600
    assert "2025-06-18" in response.json()["error"]["message"]


async def test_batch_authorizes_each_entry_when_enabled(gateway_factory, downstream_app):
    gateway = gateway_factory(allow_batch=True)
    response = await gateway.post(
        "/mcp",
        json=[
            call_tool("get_customer_record", {"customer_id": "CUST-10042"}, 1),
            call_tool("admin_reset_key", {"tenant": "acme"}, 2),
            call_tool("search_orders", {"customer_id": "CUST-10042"}, 3),
        ],
        headers={"Authorization": VIEWER},
    )
    body = response.json()
    by_id = {entry["id"]: entry for entry in body}
    assert "result" in by_id[1]
    assert by_id[2]["error"]["code"] == -32001
    assert "result" in by_id[3]
    # Only the two permitted entries were forwarded.
    forwarded = [c["params"]["name"] for c in downstream_app.state.calls]
    assert forwarded == ["get_customer_record", "search_orders"]


# ---------------------------------------------------------------------------
# Upstream failure handling
# ---------------------------------------------------------------------------
async def test_upstream_timeout_is_sanitised(gateway_factory):
    class TimingOutClient:
        async def post(self, *args, **kwargs):
            raise httpx.ConnectTimeout("connect to 10.0.0.7:9001 timed out")

    from mcp_gateway import create_app

    app = create_app(downstream_url="http://downstream/mcp", client=TimingOutClient())
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")
    try:
        response = await client.post(
            "/mcp", json=rpc("tools/list", {}), headers={"Authorization": VIEWER}
        )
    finally:
        await client.aclose()

    assert response.status_code == 504
    assert response.json()["error"]["code"] == -32003
    # The internal address must not travel to the client.
    assert "10.0.0.7" not in response.text


async def test_health_endpoint_needs_no_auth(gateway):
    assert (await gateway.get("/healthz")).json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Downstream misbehaviour must not become a gateway crash
# ---------------------------------------------------------------------------
async def test_non_json_downstream_response_is_a_clean_502(gateway_factory):
    """An nginx 502 page or an auth portal is HTML, not JSON-RPC."""

    class HtmlClient:
        async def post(self, *args, **kwargs):
            return httpx.Response(
                502,
                headers={"content-type": "text/html"},
                content=b"<html><body>502 Bad Gateway</body></html>",
                request=httpx.Request("POST", "http://downstream/mcp"),
            )

    from mcp_gateway import create_app

    app = create_app(downstream_url="http://downstream/mcp", client=HtmlClient())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw"
    ) as client:
        response = await client.post(
            "/mcp", json=rpc("tools/list", {}), headers={"Authorization": VIEWER}
        )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == -32003
    assert "502 Bad Gateway" not in response.text


async def test_batch_survives_a_downstream_timeout(gateway_factory):
    """Batch mode had no error handling at all; one timeout crashed the request."""

    class TimingOutClient:
        async def post(self, *args, **kwargs):
            raise httpx.ConnectTimeout("connect to 10.0.0.7:9001 timed out")

    from mcp_gateway import create_app

    app = create_app(
        downstream_url="http://downstream/mcp", client=TimingOutClient(), allow_batch=True
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw"
    ) as client:
        response = await client.post(
            "/mcp",
            json=[
                call_tool("get_customer_record", {"customer_id": "CUST-10042"}, 1),
                call_tool("admin_reset_key", {"tenant": "acme"}, 2),
            ],
            headers={"Authorization": VIEWER},
        )
    assert response.status_code == 200
    by_id = {entry["id"]: entry for entry in response.json()}
    assert by_id[1]["error"]["code"] == -32003      # forwarded, upstream died
    assert by_id[2]["error"]["code"] == -32001      # denied locally, never forwarded
    assert "10.0.0.7" not in response.text


# ---------------------------------------------------------------------------
# Batch semantics the README claims
# ---------------------------------------------------------------------------
async def test_batch_preserves_order(gateway_factory):
    gateway = gateway_factory(allow_batch=True)
    response = await gateway.post(
        "/mcp",
        json=[
            call_tool("search_orders", {"customer_id": "CUST-10042"}, 10),
            call_tool("get_customer_record", {"customer_id": "CUST-10042"}, 20),
            call_tool("search_orders", {"customer_id": "CUST-20099"}, 30),
        ],
        headers={"Authorization": VIEWER},
    )
    assert [entry["id"] for entry in response.json()] == [10, 20, 30]


async def test_batch_notifications_produce_no_response(gateway_factory, downstream_app):
    """A JSON-RPC message with no id gets no reply, allowed or denied."""
    gateway = gateway_factory(allow_batch=True)
    response = await gateway.post(
        "/mcp",
        json=[
            call_tool("search_orders", {"customer_id": "CUST-10042"}, request_id=None),
            call_tool("admin_reset_key", {"tenant": "acme"}, request_id=None),
            call_tool("get_customer_record", {"customer_id": "CUST-10042"}, 7),
        ],
        headers={"Authorization": VIEWER},
    )
    body = response.json()
    assert [entry["id"] for entry in body] == [7]
    forwarded = [c["params"]["name"] for c in downstream_app.state.calls]
    assert forwarded == ["search_orders", "get_customer_record"]


async def test_batch_of_only_notifications_returns_202(gateway_factory):
    gateway = gateway_factory(allow_batch=True)
    response = await gateway.post(
        "/mcp",
        json=[call_tool("search_orders", {"customer_id": "CUST-10042"}, request_id=None)],
        headers={"Authorization": VIEWER},
    )
    assert response.status_code == 202


async def test_unexpected_client_exception_is_a_clean_500(gateway_factory):
    """A non-httpx failure is a bug in us, and still must not leak a traceback."""

    class Exploding:
        async def post(self, *args, **kwargs):
            raise RuntimeError("BOOM /opt/gateway/mcp_gateway.py secret=hunter2")

    from mcp_gateway import create_app

    app = create_app(downstream_url="http://downstream/mcp", client=Exploding())
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
        response = await client.post(
            "/mcp", json=rpc("tools/list", {}), headers={"Authorization": VIEWER}
        )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == -32603
    assert "hunter2" not in response.text
    assert "Traceback" not in response.text


async def test_batch_tools_list_is_filtered_when_filtering_is_enabled(gateway_factory):
    """Both flags on: the batch path must filter exactly like the single path."""
    gateway = gateway_factory(filter_tools_list=True, allow_batch=True)
    response = await gateway.post(
        "/mcp", json=[rpc("tools/list", {}, 1)], headers={"Authorization": VIEWER}
    )
    names = {tool["name"] for tool in response.json()[0]["result"]["tools"]}
    assert names == {"get_customer_record", "search_orders"}


# ---------------------------------------------------------------------------
# Method-name evasion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "method", ["Tools/Call", "TOOLS/CALL", "tools/Call", "ToOlS/cAlL"]
)
async def test_case_variant_of_tools_call_is_still_authorized(gateway, downstream_app, method):
    """The bypass: an exact compare on the method name skipped authorization.

    `authorize()` gated on `method != "tools/call"`, so any lexical variant fell
    straight through to the forward path and a viewer's `admin_reset_key` was
    delivered downstream, audited as `forwarded`. The mock downstream rejected it
    only because its own dispatch is exact-match — the downstream leniency this
    gateway is explicitly not allowed to depend on.
    """
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": {"name": "admin_reset_key", "arguments": {"tenant": "acme"}},
    }
    response = await gateway.post("/mcp", json=body, headers={"Authorization": VIEWER})
    assert response.json()["error"]["code"] == -32001
    assert downstream_app.state.calls == []


@pytest.mark.parametrize("method", ["tools/call ", " tools/call", "tools/call\n", "tools/call\t"])
async def test_padded_method_names_are_rejected(gateway, downstream_app, method):
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": {"name": "admin_reset_key", "arguments": {"tenant": "acme"}},
    }
    response = await gateway.post("/mcp", json=body, headers={"Authorization": VIEWER})
    assert response.json()["error"]["code"] == -32600
    assert downstream_app.state.calls == []


async def test_admin_case_variant_is_authorized_then_forwarded_verbatim(gateway, downstream_app):
    """Normalising the *decision* must not turn into rewriting the *request*.

    Two separate things happen here and the test pins both. The gateway
    normalises to decide, so an admin is allowed through. It then forwards the
    caller's original spelling, so the downstream is the one that decides what
    "Tools/Call" means -- and this downstream, being exact-match, answers -32601.
    That is the correct division of labour: -32601 from the downstream, never
    -32001 from the gateway, and never a silently corrected method name.
    """
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "Tools/Call",
        "params": {"name": "admin_reset_key", "arguments": {"tenant": "acme"}},
    }
    response = await gateway.post("/mcp", json=body, headers={"Authorization": ADMIN})

    assert downstream_app.state.calls[-1]["method"] == "Tools/Call"  # verbatim
    assert response.json()["error"]["code"] == -32601  # the downstream's answer
    assert response.json()["error"]["code"] != -32001  # not an authorization denial


async def test_batch_entries_also_normalise_the_method(gateway_factory, downstream_app):
    gateway = gateway_factory(allow_batch=True)
    response = await gateway.post(
        "/mcp",
        json=[
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "TOOLS/CALL",
                "params": {"name": "admin_reset_key", "arguments": {"tenant": "acme"}},
            }
        ],
        headers={"Authorization": VIEWER},
    )
    assert response.json()[0]["error"]["code"] == -32001
    assert downstream_app.state.calls == []


# ---------------------------------------------------------------------------
# Parse errors that are not JSONDecodeError
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("body", "why"),
    [
        (b'{"jsonrpc":"2.0","id":' + b"9" * 4401 + b',"method":"tools/list"}', "int over the digit limit raises ValueError"),
        (b"[" * 20000, "deep nesting raises RecursionError"),
        (b"{not json", "ordinary JSONDecodeError"),
    ],
)
async def test_unparseable_bodies_are_32700_not_500(gateway, body, why):
    response = await gateway.post(
        "/mcp",
        content=body,
        headers={"Authorization": VIEWER, "Content-Type": "application/json"},
    )
    assert response.status_code == 400, why
    assert response.json()["error"]["code"] == -32700
