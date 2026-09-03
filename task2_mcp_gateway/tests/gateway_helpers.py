"""Request builders shared by the gateway tests.

Kept out of conftest.py deliberately: every task in this repo has its own
tests/conftest.py, and with no package __init__.py files pytest imports them all
as the top-level module name "conftest". Fixtures are fine -- pytest resolves
those per directory -- but a `from conftest import ...` in one task then
resolves against whichever task's conftest was imported first, and the whole
suite fails to collect. Uniquely named helper modules avoid the collision.
"""

ADMIN = "Bearer admin-token-abc123"
VIEWER = "Bearer viewer-token-def456"
VIEWER_OTHER_TENANT = "Bearer viewer-token-ghi789"


def rpc(method: str, params: dict | None = None, request_id: int | None = 1) -> dict:
    body: dict = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        body["id"] = request_id
    if params is not None:
        body["params"] = params
    return body


def call_tool(name: str, arguments: dict | None = None, request_id: int | None = 1) -> dict:
    return rpc("tools/call", {"name": name, "arguments": arguments or {}}, request_id)
