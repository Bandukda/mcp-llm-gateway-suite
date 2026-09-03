"""A mock downstream MCP server speaking JSON-RPC over HTTP.

Stands in for the real MCP server the gateway protects. It is deliberately
*trusting*: it performs no authorization of its own, which is the point. If the
gateway lets an ``admin_`` call through, this server executes it. That makes the
gateway's tests meaningful -- a passing authorization test proves the gateway
blocked the call, not that the downstream happened to refuse it.

It also records every call it received, so a test can assert the negative:
"the downstream was never contacted".
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_customer_record",
        "description": "Look up a customer by id.",
        "inputSchema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string", "pattern": "^CUST-[0-9]{5}$"}},
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_orders",
        "description": "Search orders for a customer.",
        "inputSchema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}, "query": {"type": "string"}},
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "admin_reset_key",
        "description": "Rotate a tenant's API key. Destructive.",
        "inputSchema": {
            "type": "object",
            "properties": {"tenant": {"type": "string"}},
            "required": ["tenant"],
            "additionalProperties": False,
        },
    },
    {
        "name": "admin_delete_tenant",
        "description": "Permanently delete a tenant. Destructive.",
        "inputSchema": {
            "type": "object",
            "properties": {"tenant": {"type": "string"}},
            "required": ["tenant"],
            "additionalProperties": False,
        },
    },
]


def create_downstream_app() -> FastAPI:
    app = FastAPI(title="mock-downstream-mcp")
    app.state.calls: list[dict[str, Any]] = []

    def result(request_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": payload}

    def error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    @app.post("/mcp")
    async def handle(request: Request) -> JSONResponse:
        body = await request.json()
        app.state.calls.append(
            {
                "method": body.get("method"),
                "params": body.get("params"),
                "headers": dict(request.headers),
            }
        )

        method = body.get("method")
        request_id = body.get("id")

        if method == "initialize":
            return JSONResponse(
                result(
                    request_id,
                    {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "mock-downstream-mcp", "version": "1.0.0"},
                    },
                )
            )

        if method == "tools/list":
            return JSONResponse(result(request_id, {"tools": TOOLS}))

        if method == "tools/call":
            name = (body.get("params") or {}).get("name")
            arguments = (body.get("params") or {}).get("arguments") or {}
            known = {tool["name"] for tool in TOOLS}
            if name not in known:
                return JSONResponse(error(request_id, -32602, f"Unknown tool: {name!r}"))
            return JSONResponse(
                result(
                    request_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": f"executed {name} with {arguments}",
                            }
                        ],
                        "isError": False,
                    },
                )
            )

        if method == "ping":
            return JSONResponse(result(request_id, {}))

        return JSONResponse(error(request_id, -32601, f"Method not found: {method}"))

    return app


downstream_app = create_downstream_app()

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(downstream_app, host="127.0.0.1", port=9001)
