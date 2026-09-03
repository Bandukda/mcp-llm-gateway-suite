"""
Task 2 -- MCP security gateway proxy.

An HTTP/JSON-RPC reverse proxy that sits between an agent client and a
downstream MCP server. It authenticates the caller, decides per *tool* whether
the call may proceed, forwards what is allowed, and answers what is not without
ever touching the downstream.

    agent ──Bearer──▶ gateway ──▶ downstream MCP server
                        │
                        └── denied: -32001, downstream never contacted

Run the pair locally::

    uvicorn downstream:downstream_app --port 9001
    MCP_DOWNSTREAM_URL=http://127.0.0.1:9001/mcp uvicorn mcp_gateway:app --port 9000

    curl -s localhost:9000/mcp \\
      -H 'Authorization: Bearer viewer-token-def456' \\
      -H 'Content-Type: application/json' \\
      -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
           "params":{"name":"admin_reset_key","arguments":{"tenant":"acme"}}}'
    # {"jsonrpc":"2.0","id":1,"error":{"code":-32001,"message":"Unauthorized Tool Call", ...}}

Design decisions worth reading
------------------------------

**HTTP 401 vs JSON-RPC -32001.** These answer different questions and the
gateway keeps them separate:

* *Who are you?* -- unanswerable (no token, bad token) is a transport-layer
  failure. It returns **HTTP 401** with a ``WWW-Authenticate`` header, which is
  what the MCP authorization spec expects and what lets a client know it should
  re-authenticate. A JSON-RPC error body is included as well so a client that
  only parses bodies still gets a structured answer.
* *May you do this?* -- a known caller attempting a tool they do not have is an
  application-layer decision. It returns **HTTP 200** carrying JSON-RPC error
  **-32001 Unauthorized Tool Call**, as the task specifies. The request was
  well-formed and was answered; the answer is "no".

**The client's token is never forwarded.** The gateway terminates the caller's
credential and mints its own for the downstream hop, passing identity as
``X-MCP-Gateway-Subject`` / ``-Role`` / ``-Tenant``. Forwarding the original
Authorization header is the confused-deputy setup: the downstream ends up
holding a credential it was never issued, and any downstream bug becomes a
credential leak. This also means the downstream can trust the identity headers
*only because* it trusts the gateway's own credential.

**Tool names are validated before they are judged.** See ``policy.py`` -- the
policy runs on a normalised copy, the charset is restricted, and the *original*
name is what gets forwarded, so the gateway never silently rewrites a request.

**Batching is rejected by default.** JSON-RPC batching was removed from MCP in
revision 2025-06-18. The per-entry authorization path is implemented and tested
(set ``MCP_GATEWAY_ALLOW_BATCH=1``) because plenty of pre-2025-06-18 clients
still send arrays, but the default matches the current spec.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from auth import AuthError, Principal, resolve_principal
from policy import (
    ToolPolicy,
    is_wellformed_method,
    is_wellformed_tool_name,
    normalize_method,
    normalize_tool_name,
)

# ---------------------------------------------------------------------------
# JSON-RPC error codes
# ---------------------------------------------------------------------------
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNAUTHORIZED_TOOL_CALL = -32001  # server-defined, per the task
UNAUTHENTICATED = -32002
UPSTREAM_UNAVAILABLE = -32003

DOWNSTREAM_URL = os.environ.get("MCP_DOWNSTREAM_URL", "http://127.0.0.1:9001/mcp")
DOWNSTREAM_TIMEOUT_S = float(os.environ.get("MCP_DOWNSTREAM_TIMEOUT_S", "10"))
DOWNSTREAM_TOKEN = os.environ.get("MCP_DOWNSTREAM_TOKEN", "gateway-service-token")
FILTER_TOOLS_LIST = os.environ.get("MCP_GATEWAY_FILTER_TOOLS_LIST", "0") == "1"
ALLOW_BATCH = os.environ.get("MCP_GATEWAY_ALLOW_BATCH", "0") == "1"

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s", force=True)
audit_log = logging.getLogger("mcp.gateway.audit")


def audit(**fields: Any) -> None:
    """One structured JSON line per decision. This is the artifact a security
    review asks for, so it is emitted for allows as well as denies."""
    audit_log.info(json.dumps({"ts": round(time.time(), 3), **fields}, default=str))


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------
def rpc_error(request_id: Any, code: int, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def looks_like_rpc_request(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("jsonrpc") == "2.0"
        and isinstance(payload.get("method"), str)
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
def create_app(
    downstream_url: str = DOWNSTREAM_URL,
    policy: ToolPolicy | None = None,
    client: httpx.AsyncClient | None = None,
    filter_tools_list: bool = FILTER_TOOLS_LIST,
    allow_batch: bool = ALLOW_BATCH,
) -> FastAPI:
    """Build the gateway.

    ``client`` is injectable so tests can point the gateway at an in-process
    downstream through ``httpx.ASGITransport`` -- no sockets, no ports, no
    flaky teardown.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = app.state.injected_client or httpx.AsyncClient(
            timeout=httpx.Timeout(DOWNSTREAM_TIMEOUT_S, connect=3.0)
        )
        try:
            yield
        finally:
            # Only close what we opened; an injected client belongs to the caller.
            if app.state.injected_client is None:
                await app.state.client.aclose()

    app = FastAPI(title="mcp-security-gateway", version="1.0.0", lifespan=lifespan)
    app.state.policy = policy or ToolPolicy()
    app.state.downstream_url = downstream_url
    app.state.filter_tools_list = filter_tools_list
    app.state.allow_batch = allow_batch
    app.state.injected_client = client
    app.state.client = client

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    # -- downstream hop ----------------------------------------------------
    def downstream_headers(principal: Principal, request_id: str) -> dict[str, str]:
        """Headers for the gateway -> downstream hop.

        The caller's Authorization header is *not* among them. The gateway
        authenticates itself and asserts the caller's identity separately.
        """
        return {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {DOWNSTREAM_TOKEN}",
            "X-MCP-Gateway-Subject": principal.subject,
            "X-MCP-Gateway-Role": principal.role,
            "X-MCP-Gateway-Tenant": principal.tenant,
            "X-Request-Id": request_id,
        }

    async def forward(payload: dict[str, Any], principal: Principal, request_id: str):
        """Send one JSON-RPC message downstream and return its httpx response."""
        return await app.state.client.post(
            app.state.downstream_url,
            content=json.dumps(payload),
            headers=downstream_headers(principal, request_id),
        )

    # -- per-message authorization ----------------------------------------
    def authorize(payload: dict[str, Any], principal: Principal, request_id: str) -> dict[str, Any] | None:
        """Return a JSON-RPC error to send back, or ``None`` to allow the call."""
        method = payload.get("method")
        rpc_id = payload.get("id")

        # A method name outside the ASCII set never reaches a routing decision.
        if not is_wellformed_method(method):
            audit(
                event="method_rejected",
                request_id=request_id,
                subject=principal.subject,
                reason="malformed_method",
                method_repr=repr(method)[:120],
            )
            return rpc_error(
                rpc_id,
                INVALID_REQUEST,
                "method contains characters outside the permitted set",
                {"permitted": "^[A-Za-z0-9_./-]{1,128}$"},
            )

        if normalize_method(method) != "tools/call":
            # tools/list, initialize, ping, resources/*, prompts/* pass through.
            #
            # Normalised, not compared exactly: see normalize_method(). An exact
            # compare let "Tools/Call" skip authorization and get forwarded.
            return None

        params = payload.get("params")
        if not isinstance(params, dict):
            return rpc_error(rpc_id, INVALID_PARAMS, "params must be an object for tools/call")

        name = params.get("name")
        if not isinstance(name, str) or not name:
            return rpc_error(rpc_id, INVALID_PARAMS, "params.name must be a non-empty string")

        if not is_wellformed_tool_name(name):
            audit(
                event="tool_call_rejected",
                request_id=request_id,
                subject=principal.subject,
                reason="malformed_tool_name",
                tool_repr=repr(name)[:120],
            )
            return rpc_error(
                rpc_id,
                INVALID_PARAMS,
                "params.name contains characters outside the permitted tool-name set",
                {"permitted": "^[A-Za-z0-9_.-]{1,128}$"},
            )

        decision = app.state.policy.evaluate(name, principal)
        audit(
            event="tool_call_allowed" if decision.allowed else "tool_call_denied",
            request_id=request_id,
            subject=principal.subject,
            role=principal.role,
            tenant=principal.tenant,
            tool=name,
            rule=decision.rule,
            reason=decision.reason,
        )
        if decision.allowed:
            return None

        return rpc_error(
            rpc_id,
            UNAUTHORIZED_TOOL_CALL,
            "Unauthorized Tool Call",
            {
                "tool": name,
                "required_role": sorted(
                    next(
                        (
                            roles
                            for prefix, roles in app.state.policy.prefix_roles.items()
                            if normalize_tool_name(name).startswith(prefix)
                        ),
                        frozenset(),
                    )
                ),
                "caller_role": principal.role,
                "request_id": request_id,
            },
        )

    def filter_tools_response(body: dict[str, Any], principal: Principal) -> dict[str, Any]:
        """Optionally drop tools the caller could never call from ``tools/list``.

        Off by default because the task says to forward ``tools/list``
        transparently. Turning it on is the stronger posture: a tool the model
        can see is a tool the model will try, and an admin tool listed to a
        viewer is both a wasted turn and a hint worth giving an attacker.
        """
        tools = body.get("result", {}).get("tools")
        if not isinstance(tools, list):
            return body
        body["result"]["tools"] = [
            tool
            for tool in tools
            if isinstance(tool, dict)
            and isinstance(tool.get("name"), str)
            and app.state.policy.visible_to(tool["name"], principal)
        ]
        return body

    # -- the endpoint ------------------------------------------------------
    @app.post("/mcp")
    async def proxy(request: Request):
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        started = time.perf_counter()

        # 1. Authenticate. Failure is HTTP 401, not a JSON-RPC-level answer.
        try:
            principal = resolve_principal(request.headers.get("authorization"))
        except AuthError as exc:
            audit(event="auth_failed", request_id=request_id, reason=exc.reason)
            return JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="mcp-gateway"'},
                content=rpc_error(None, UNAUTHENTICATED, str(exc), {"request_id": request_id}),
            )

        # 2. Parse.
        raw = await request.body()
        try:
            payload = json.loads(raw)
        except (ValueError, RecursionError):
            # Not just JSONDecodeError. An integer literal over
            # sys.get_int_max_str_digits() raises a plain ValueError, and deeply
            # nested arrays raise RecursionError -- both are ~4 KB of body that
            # turned into an unhandled 500 before this was widened.
            return JSONResponse(status_code=400, content=rpc_error(None, PARSE_ERROR, "Parse error"))

        # 3. Batch.
        if isinstance(payload, list):
            if not app.state.allow_batch:
                return JSONResponse(
                    status_code=400,
                    content=rpc_error(
                        None,
                        INVALID_REQUEST,
                        "JSON-RPC batching is not supported (removed in MCP revision 2025-06-18)",
                    ),
                )
            return await handle_batch(payload, principal, request_id)

        if not looks_like_rpc_request(payload):
            return JSONResponse(
                status_code=400,
                content=rpc_error(
                    payload.get("id") if isinstance(payload, dict) else None,
                    INVALID_REQUEST,
                    "Invalid JSON-RPC 2.0 request",
                ),
            )

        # 4. Authorize. A denial never reaches the downstream.
        denial = authorize(payload, principal, request_id)
        if denial is not None:
            return JSONResponse(status_code=200, content=denial)

        # 5. Forward.
        try:
            response = await forward(payload, principal, request_id)
        except httpx.TimeoutException:
            audit(event="upstream_timeout", request_id=request_id, subject=principal.subject)
            return JSONResponse(
                status_code=504,
                content=rpc_error(
                    payload.get("id"), UPSTREAM_UNAVAILABLE, "Upstream MCP server timed out",
                    {"request_id": request_id},
                ),
            )
        except httpx.HTTPError:
            # The exception text can carry internal hostnames and ports.
            audit_log.exception("downstream transport error request_id=%s", request_id)
            return JSONResponse(
                status_code=502,
                content=rpc_error(
                    payload.get("id"), UPSTREAM_UNAVAILABLE, "Upstream MCP server unavailable",
                    {"request_id": request_id},
                ),
            )
        except Exception:
            # Anything else is a bug in this gateway or in the client we were
            # handed. It still must not reach the caller as a traceback.
            audit_log.exception("unhandled gateway error request_id=%s", request_id)
            return JSONResponse(
                status_code=500,
                content=rpc_error(
                    payload.get("id"), INTERNAL_ERROR, "Internal gateway error",
                    {"request_id": request_id},
                ),
            )

        # 5b. Streamable HTTP: pass SSE through without buffering it.
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("text/event-stream"):
            return StreamingResponse(
                response.aiter_raw(),
                status_code=response.status_code,
                media_type="text/event-stream",
                headers={"X-Request-Id": request_id, "Cache-Control": "no-cache"},
            )

        # A downstream that answers with HTML -- an nginx 502, an auth portal, an
        # empty body -- must not turn into a 500 from the gateway.
        try:
            body = response.json()
        except ValueError:
            audit(
                event="upstream_non_json",
                request_id=request_id,
                subject=principal.subject,
                status=response.status_code,
                content_type=content_type or "(none)",
            )
            return JSONResponse(
                status_code=502,
                content=rpc_error(
                    payload.get("id"),
                    UPSTREAM_UNAVAILABLE,
                    "Upstream MCP server returned a malformed response",
                    {"request_id": request_id},
                ),
            )

        if (
            app.state.filter_tools_list
            and payload.get("method") == "tools/list"
            and isinstance(body, dict)
        ):
            body = filter_tools_response(body, principal)

        audit(
            event="forwarded",
            request_id=request_id,
            subject=principal.subject,
            method=payload.get("method"),
            status=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return JSONResponse(
            status_code=response.status_code,
            content=body,
            headers={"X-Request-Id": request_id},
        )

    async def handle_batch(entries: list[Any], principal: Principal, request_id: str):
        """Authorize each entry independently, forward only the survivors.

        Order is preserved, notifications (no ``id``) produce no response, and a
        denial in one entry does not stop the rest -- which is what makes this
        worth implementing rather than rejecting the whole array.
        """
        responses: list[dict[str, Any]] = []
        for entry in entries:
            if not looks_like_rpc_request(entry):
                responses.append(rpc_error(None, INVALID_REQUEST, "Invalid JSON-RPC 2.0 request"))
                continue
            denial = authorize(entry, principal, request_id)
            if denial is not None:
                if entry.get("id") is not None:
                    responses.append(denial)
                continue
            # Same failure handling as the single-message path: one bad entry
            # must not take down the whole batch, and no transport detail
            # travels back to the caller.
            try:
                response = await forward(entry, principal, request_id)
                entry_body = response.json()
            except httpx.TimeoutException:
                audit(event="upstream_timeout", request_id=request_id, batch=True)
                entry_body = rpc_error(
                    entry.get("id"), UPSTREAM_UNAVAILABLE, "Upstream MCP server timed out",
                    {"request_id": request_id},
                )
            except Exception:
                audit_log.exception("downstream failure in batch request_id=%s", request_id)
                entry_body = rpc_error(
                    entry.get("id"), UPSTREAM_UNAVAILABLE, "Upstream MCP server unavailable",
                    {"request_id": request_id},
                )
            if (
                app.state.filter_tools_list
                and entry.get("method") == "tools/list"
                and isinstance(entry_body, dict)
            ):
                # The single-message path filters; the batch path used to not,
                # so turning both flags on leaked the full tool list.
                entry_body = filter_tools_response(entry_body, principal)
            if entry.get("id") is not None:
                responses.append(entry_body)
        if not responses:
            return JSONResponse(status_code=202, content=None)
        return JSONResponse(status_code=200, content=responses)

    return app


app = create_app()

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9000)
