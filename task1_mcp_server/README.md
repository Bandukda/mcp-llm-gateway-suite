# Billing MCP server (stdio, strict validation)

A runnable MCP server exposing two tools, built on the official Python SDK
(`mcp`, pinned to the 1.x line) with Pydantic v2 for schema enforcement.

```
get_customer_record(customer_id)
trigger_refund(customer_id, amount, reason, [idempotency_key])
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt

python server.py                  # speaks JSON-RPC 2.0 on stdin/stdout
python -m pytest tests -q         # 58 tests
python verify_stdout_purity.py    # PASS - proves stdout carries only JSON-RPC
python verify_stdout_purity.py --unguarded   # FAIL, deliberately - see below
```

To wire it into Claude Desktop or any MCP client:

```json
{
  "mcpServers": {
    "billing": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/task1_mcp_server/server.py"]
    }
  }
}
```

## Files

| File | What it does |
| --- | --- |
| `stdio_guard.py` | Reserves fd 1 for JSON-RPC; diverts every other write to stderr |
| `schemas.py` | Strict Pydantic input models, one per tool |
| `store.py` | In-memory billing data; raises business errors |
| `server.py` | Tool definitions, error mapping, transport wiring |
| `verify_stdout_purity.py` | Spawns the server and asserts stdout is pure JSON-RPC |
| `tests/` | Schema tests, protocol tests, and the purity proof as pytest |

---

## The three things that make this non-trivial

### 1. STDIO isolation

An MCP stdio server speaks newline-delimited JSON-RPC on **file descriptor 1**.
Anything else that reaches fd 1 lands inside a frame and corrupts it. The
failure is ugly: the client dies with a parse error, or worse, silently accepts
a spoofed message.

"Never call `print()`" is a convention, not a control — it fails the first time
a dependency prints a deprecation banner. `stdio_guard.py` makes the failure
impossible at the OS level:

```python
wire_fd = os.dup(1)   # private duplicate: the only handle allowed to carry frames
os.dup2(2, 1)         # fd 1 now *is* stderr
```

Because this operates on the descriptor rather than on `sys.stdout`, it covers
every writer:

| Writer | Without the guard | With the guard |
| --- | --- | --- |
| `print("debug")` | corrupts the stream | stderr |
| `os.write(1, b"...")` from a C extension | corrupts the stream | stderr |
| `subprocess.run(["echo", ...])` inheriting fd 1 | corrupts the stream | stderr |
| A forgotten line that *looks* like JSON-RPC | silently accepted by the client | stderr |

`verify_stdout_purity.py` proves it rather than asserting it. It launches the
server with `BILLING_MCP_CHAOS=1`, which makes the server commit all four sins
above, then holds a real MCP conversation and checks every stdout line parses as
JSON-RPC 2.0.

```
$ python verify_stdout_purity.py
PASS
  5 stdout lines, all valid JSON-RPC 2.0
  all 4 deliberate stdout writes were diverted to stderr
  malformed arguments returned JSON-RPC error -32602

$ python verify_stdout_purity.py --unguarded
FAIL
  - line 1 is not JSON: 'DEBUG: about to serve requests'
  - line 4 ... {"jsonrpc": "2.0", "id": 999, "result": {"spoofed": true}}
```

That last line is the point of the exercise. A stray `print` of a JSON-shaped
string is not noise; it is a **forged response** the client will believe.

> **SDK note.** `mcp` v2 performs this same descriptor diversion inside
> `stdio_server()`. This project pins `mcp>=1.9,<2`, where `stdio_server()`
> wraps `sys.stdout.buffer` directly and does no diversion, so the guard is
> doing real work. It is kept regardless so the server behaves identically on
> either SDK line, and because the same reasoning applies to any custom
> transport.

### 2. Protocol compliance — where errors go

The handler is registered **directly** on `server.request_handlers`, not through
the `@server.call_tool()` decorator. The decorator wraps the handler in
`except Exception: return isError result`, so *every* failure — including
malformed arguments — comes back as a **successful** JSON-RPC response carrying
`isError: true` — so a client library has no way to distinguish a broken call
from a working one. Validation failures have to reach the transport's error
channel. Registering directly lets
`McpError` propagate; `Server._handle_request` catches it and serialises
`err.error` as a real JSON-RPC `error` object.

The mapping, and the reasoning behind it:

| Condition | Response | Code |
| --- | --- | --- |
| `tools/call` names an unknown tool | JSON-RPC error | `-32602` |
| `params.arguments` is not an object | JSON-RPC error | `-32602` |
| Schema violation | JSON-RPC error + per-field `data` | `-32602` |
| Customer id well-formed but absent | `CallToolResult(isError=true)` | — |
| Refund exceeds refundable balance | `CallToolResult(isError=true)` | — |
| Unexpected exception | JSON-RPC error, message sanitised | `-32603` |

The dividing line is **"could the caller have known?"**

- A malformed argument is a contract violation. It belongs in the protocol error
  channel, where a client library raises rather than hands the model a string.
- "That customer does not exist" is a legitimate answer to a legitimate call.
  It belongs in the tool result so the model can read it, apologise, and try a
  different id. That is exactly what `isError` is for.

Two supporting details:

- **`-32602` for an unknown tool, not `-32601`.** `-32601` means *method* not
  found; the method here is `tools/call` and it exists. The tool name is a
  parameter, so a bad one is invalid params. The error `data` lists the tools
  that do exist, which lets an agent self-correct in one turn.
- **`-32603` messages are generic.** The full traceback goes to stderr; the
  client is told only "Internal server error". A stack trace on the wire leaks
  file paths, library versions and sometimes credentials.

### 3. Validation and edge cases

`schemas.py` is deliberately paranoid. Each choice below has a test.

| Guard | Why |
| --- | --- |
| `extra="forbid"` | `{"ammount": 50}` (typo) must fail loudly, not silently refund `None` |
| `strict=True` | Lax Pydantic turns `"49.99"` into `49.99`. Money is not guessed |
| bool rejected for `amount` | `bool` subclasses `int`; lax mode makes `True` → `1.0` |
| `allow_inf_nan=False` | Python's `json` accepts the non-standard `NaN` / `Infinity` literals, and `Infinity > 0` is `True` |
| Two-decimal check | `12.3456` is a client bug, not a refund |
| `reason` stripped before length check | Ten spaces satisfies `min_length=10` but is not a reason |
| **`[0-9]{5}` not `\d{5}`** | `\d` is Unicode-aware. `CUST-١٠٠٤٢` (Arabic-Indic digits) passes `^CUST-\d{5}$` and then breaks every downstream ASCII-only system |

`test_rejects_malformed_customer_id` pins the Unicode-digit case specifically.

**`idempotency_key` (beyond the spec).** `trigger_refund` moves money and agents
retry: a tool call that times out at the client but succeeded at the server gets
re-sent, and the customer is refunded twice. The optional key makes the call
replay-safe, and the response reports `replayed: true` so the model knows not to
announce a second refund. Small addition, but it is the failure I would expect to
hit first in production.

It also shipped with a bug worth keeping visible: the key was stored without
being bound to the request it was issued for. Replaying a key with *different*
arguments returned the original refund and reported `ok`/`replayed` — so a call
asking to refund CUST-20099 $5 was answered with CUST-10042's $1000 refund, the
$5 never happened, and the model was told it had succeeded and not to
re-announce it. The key now carries a fingerprint of `(customer_id, amount,
reason)`, and reuse with a changed payload returns `idempotency_key_conflict`,
which is what Stripe and friends do. An idempotency key has to be a promise
about a specific request, or it is a promise about nothing.

## Test coverage

```
tests/test_schemas.py        39  every malformed-input path, field by field
tests/test_protocol.py       17  end-to-end through a real MCP ClientSession
tests/test_stdout_purity.py   2  guarded passes, unguarded fails
```
