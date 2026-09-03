#!/usr/bin/env python3
"""
Task 1 -- Billing MCP server (stdio transport, strict validation).

Tools
-----
``get_customer_record(customer_id)``
``trigger_refund(customer_id, amount, reason, [idempotency_key])``

Run it::

    python server.py                    # speaks JSON-RPC on stdin/stdout
    python -m pytest tests -q           # unit + protocol tests
    python verify_stdout_purity.py      # proves stdout carries only JSON-RPC

Two decisions worth reading before the code
-------------------------------------------

1. **The tool handler is registered directly on ``request_handlers`` instead of
   through the ``@server.call_tool()`` decorator.** The decorator is convenient
   but it wraps the handler in ``except Exception: return isError result`` --
   every failure, including a malformed-arguments failure, comes back as a
   *successful* JSON-RPC response carrying ``isError: true``. The task requires
   "standard MCP JSON-RPC error codes", so validation failures must surface as
   real ``error`` objects with code ``-32602``. Registering the handler directly
   lets ``McpError`` propagate to the session, which serialises it as a JSON-RPC
   error (``Server._handle_request`` catches ``McpError`` and returns
   ``err.error``).

2. **Protocol errors and business errors are different things.**

   =============================== ============================ ==========
   Condition                       Response                     Code
   =============================== ============================ ==========
   ``tools/call`` unknown tool     JSON-RPC error               -32602
   arguments not a JSON object     JSON-RPC error               -32602
   schema violation                JSON-RPC error + ``data``    -32602
   customer id well-formed, absent ``CallToolResult(isError)``  n/a
   refund exceeds balance          ``CallToolResult(isError)``  n/a
   unexpected server bug           JSON-RPC error, sanitised    -32603
   =============================== ============================ ==========

   The line is "could the caller have known?". A malformed argument is a caller
   contract violation and belongs in the transport's error channel. "That
   customer does not exist" is a legitimate, expected answer to a legitimate
   call; it belongs in the tool result so the model can read it, apologise and
   try a different id, which is exactly what ``isError`` is for.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# fd 1 is claimed before anything else is imported or executed. Any import
# below that decides to print a deprecation banner writes to stderr.
# --------------------------------------------------------------------------
from stdio_guard import configure_stderr_logging, reserve_stdout_for_protocol

PROTOCOL_STDOUT = reserve_stdout_for_protocol()
log = configure_stderr_logging()

import asyncio  # noqa: E402
import json  # noqa: E402
from typing import Any  # noqa: E402

import anyio  # noqa: E402
import mcp.types as types  # noqa: E402
from mcp.server.lowlevel import NotificationOptions, Server  # noqa: E402
from mcp.server.models import InitializationOptions  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.shared.exceptions import McpError  # noqa: E402
from pydantic import BaseModel, ValidationError  # noqa: E402

from schemas import (  # noqa: E402
    GetCustomerRecordInput,
    TriggerRefundInput,
    json_schema_for,
)
from store import (  # noqa: E402
    BillingStore,
    CustomerNotFoundError,
    IdempotencyKeyConflict,
    RefundNotPermittedError,
    seeded_store,
)

SERVER_NAME = "billing-mcp"
SERVER_VERSION = "1.0.0"

TOOLS: dict[str, type[BaseModel]] = {
    "get_customer_record": GetCustomerRecordInput,
    "trigger_refund": TriggerRefundInput,
}


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------
def invalid_params(message: str, data: dict[str, Any] | None = None) -> McpError:
    """-32602 Invalid params."""
    return McpError(types.ErrorData(code=types.INVALID_PARAMS, message=message, data=data))


def internal_error(message: str = "Internal server error") -> McpError:
    """-32603 Internal error. The message is deliberately generic; the detail
    goes to stderr keyed by the log line, never to the client."""
    return McpError(types.ErrorData(code=types.INTERNAL_ERROR, message=message))


def _pydantic_error_payload(exc: ValidationError) -> dict[str, Any]:
    """Turn a Pydantic error into a compact, client-actionable ``data`` block.

    ``ValidationError`` objects contain the offending input. Echoing raw input
    back would leak whatever the caller sent (possibly a secret pasted into the
    wrong field), so only the location, the rule and the message travel.
    """
    return {
        "validation_errors": [
            {
                "field": ".".join(str(part) for part in err["loc"]) or "(root)",
                "rule": err["type"],
                "message": err["msg"],
            }
            for err in exc.errors()
        ]
    }


def _tool_error(message: str, **extra: Any) -> types.CallToolResult:
    """A business-level failure: a normal result the model can read and react to."""
    payload = {"ok": False, "error": message, **extra}
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, indent=2))],
        structuredContent=payload,
        isError=True,
    )


def _tool_ok(payload: dict[str, Any]) -> types.CallToolResult:
    body = {"ok": True, **payload}
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(body, indent=2))],
        structuredContent=body,
        isError=False,
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def handle_get_customer_record(args: GetCustomerRecordInput, store: BillingStore) -> types.CallToolResult:
    try:
        customer = store.get_customer(args.customer_id)
    except CustomerNotFoundError as exc:
        log.info("get_customer_record miss customer_id=%s", args.customer_id)
        return _tool_error(str(exc), customer_id=args.customer_id, code="customer_not_found")

    log.info("get_customer_record hit customer_id=%s", args.customer_id)
    return _tool_ok(
        {
            "customer": {
                "customer_id": customer.customer_id,
                "name": customer.name,
                "email": customer.email,
                "plan": customer.plan,
                "status": customer.status,
                "lifetime_value_usd": customer.lifetime_value_usd,
                "refundable_balance_usd": customer.refundable_balance_usd,
            }
        }
    )


def handle_trigger_refund(args: TriggerRefundInput, store: BillingStore) -> types.CallToolResult:
    try:
        refund, replayed = store.create_refund(
            customer_id=args.customer_id,
            amount_usd=args.amount,
            reason=args.reason,
            idempotency_key=args.idempotency_key,
        )
    except CustomerNotFoundError as exc:
        log.info("trigger_refund miss customer_id=%s", args.customer_id)
        return _tool_error(str(exc), customer_id=args.customer_id, code="customer_not_found")
    except RefundNotPermittedError as exc:
        log.info("trigger_refund rejected customer_id=%s reason=%s", args.customer_id, exc)
        return _tool_error(str(exc), customer_id=args.customer_id, code="refund_not_permitted")
    except IdempotencyKeyConflict as exc:
        log.warning("trigger_refund idempotency conflict key=%s", args.idempotency_key)
        return _tool_error(
            str(exc), customer_id=args.customer_id, code="idempotency_key_conflict"
        )

    log.info(
        "trigger_refund %s refund_id=%s customer_id=%s amount=%.2f",
        "replayed" if replayed else "created",
        refund.refund_id,
        refund.customer_id,
        refund.amount_usd,
    )
    return _tool_ok(
        {
            "replayed": replayed,
            "refund": {
                "refund_id": refund.refund_id,
                "customer_id": refund.customer_id,
                "amount_usd": refund.amount_usd,
                "reason": refund.reason,
                "status": refund.status,
                "created_at": refund.created_at,
            },
        }
    )


DISPATCH = {
    "get_customer_record": handle_get_customer_record,
    "trigger_refund": handle_trigger_refund,
}


# ---------------------------------------------------------------------------
# Server wiring
# ---------------------------------------------------------------------------
def build_server(store: BillingStore | None = None) -> Server:
    store = store if store is not None else seeded_store()
    server: Server = Server(SERVER_NAME, version=SERVER_VERSION)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="get_customer_record",
                title="Get customer record",
                description=(
                    "Look up a billing customer by id. Returns plan, account status, "
                    "lifetime value and the balance still eligible for refund."
                ),
                inputSchema=json_schema_for(GetCustomerRecordInput),
            ),
            types.Tool(
                name="trigger_refund",
                title="Trigger refund",
                description=(
                    "Issue a refund against a customer's refundable balance. The reason "
                    "is recorded on the ledger entry and shown to the customer. Pass an "
                    "idempotency_key when retrying so a retry cannot double-refund."
                ),
                inputSchema=json_schema_for(TriggerRefundInput),
            ),
        ]

    async def call_tool(req: types.CallToolRequest) -> types.ServerResult:
        name = req.params.name
        raw_arguments = req.params.arguments

        # 1. Unknown tool. tools/call exists, so this is a params problem.
        if name not in TOOLS:
            raise invalid_params(
                f"Unknown tool: {name!r}",
                {"available_tools": sorted(TOOLS)},
            )

        # 2. arguments must be a JSON object. In practice the SDK's own params
        #    model rejects a non-object first, with its own -32602, so this is
        #    belt and braces -- kept because the handler is registered directly
        #    on request_handlers and must not assume the SDK validated anything.
        if raw_arguments is None:
            raw_arguments = {}
        if not isinstance(raw_arguments, dict):
            raise invalid_params(
                "params.arguments must be a JSON object",
                {"received_type": type(raw_arguments).__name__},
            )

        # 3. Schema validation -> -32602 with a field-level breakdown.
        model = TOOLS[name]
        try:
            args = model.model_validate(raw_arguments)
        except ValidationError as exc:
            log.info("validation rejected tool=%s errors=%d", name, exc.error_count())
            raise invalid_params(
                f"Invalid arguments for tool {name!r}",
                _pydantic_error_payload(exc),
            ) from None

        # 4. Execute. Anything unexpected is a bug: log it in full, tell the
        #    client nothing beyond -32603.
        try:
            result = DISPATCH[name](args, store)
        except McpError:
            raise
        except Exception:
            log.exception("unhandled error in tool=%s", name)
            raise internal_error() from None

        return types.ServerResult(result)

    # Registered directly: see the module docstring, point 1.
    server.request_handlers[types.CallToolRequest] = call_tool
    return server


def initialization_options(server: Server) -> InitializationOptions:
    return server.create_initialization_options(
        notification_options=NotificationOptions(),
        experimental_capabilities={},
    )


def emit_chaos() -> None:
    """Deliberately misbehave, to prove the guard holds.

    Enabled with ``BILLING_MCP_CHAOS=1``. Each of these writes would corrupt the
    JSON-RPC stream on an unguarded server; ``verify_stdout_purity.py`` runs the
    server with the flag on and asserts stdout is still nothing but JSON-RPC.
    """
    import os as _os
    import subprocess as _subprocess

    # 1. The classic: a debug print somebody forgot to remove.
    print("DEBUG: about to serve requests")

    # 2. A library that bypasses sys.stdout and writes to the descriptor.
    _os.write(1, b"BANNER: some C extension says hello\n")

    # 3. A child process that inherits fd 1.
    _subprocess.run(["echo", "CHILD: subprocess stdout"], check=False)

    # 4. Something that is nearly JSON-RPC, which is worse than obvious garbage
    #    because a lenient parser may accept it.
    print('{"jsonrpc": "2.0", "id": 999, "result": {"spoofed": true}}')


async def main() -> None:
    import os as _os

    server = build_server()
    if _os.environ.get("BILLING_MCP_CHAOS"):
        log.warning("chaos mode on: writing junk to fd 1 and sys.stdout")
        emit_chaos()
    log.info("%s v%s starting on stdio", SERVER_NAME, SERVER_VERSION)
    async with stdio_server(stdout=anyio.wrap_file(PROTOCOL_STDOUT)) as (read_stream, write_stream):
        await server.run(read_stream, write_stream, initialization_options(server))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:  # pragma: no cover
        log.info("shutting down")
