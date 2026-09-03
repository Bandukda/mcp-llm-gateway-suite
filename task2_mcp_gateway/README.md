# MCP security gateway proxy

An HTTP/JSON-RPC reverse proxy between an agent client and a downstream MCP
server. It authenticates the caller, decides per *tool* whether the call may
proceed, forwards what is allowed, and answers what is not without ever
contacting the downstream.

```
agent ──Bearer token──▶ gateway ──service credential──▶ downstream MCP server
                           │
                           └── denied → -32001, downstream never contacted
```

## Run it

```bash
uvicorn downstream:downstream_app --port 9001
MCP_DOWNSTREAM_URL=http://127.0.0.1:9001/mcp uvicorn mcp_gateway:app --port 9000
```

```bash
# viewer calling an ordinary tool -> forwarded
curl -s localhost:9000/mcp -H 'Authorization: Bearer viewer-token-def456' \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"get_customer_record","arguments":{"customer_id":"CUST-10042"}}}'

# viewer calling an admin tool -> blocked at the gateway
curl -s localhost:9000/mcp -H 'Authorization: Bearer viewer-token-def456' \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call",
       "params":{"name":"admin_reset_key","arguments":{"tenant":"acme"}}}'
# {"jsonrpc":"2.0","id":2,"error":{"code":-32001,"message":"Unauthorized Tool Call",...}}
```

Demo credentials: `admin-token-abc123` (admin), `viewer-token-def456` (viewer,
tenant acme), `viewer-token-ghi789` (viewer, tenant globex).

| Env var | Default | Effect |
| --- | --- | --- |
| `MCP_DOWNSTREAM_URL` | `http://127.0.0.1:9001/mcp` | Where to forward |
| `MCP_DOWNSTREAM_TIMEOUT_S` | `10` | Downstream request timeout |
| `MCP_GATEWAY_FILTER_TOOLS_LIST` | `0` | Strip un-callable tools from `tools/list` |
| `MCP_GATEWAY_ALLOW_BATCH` | `0` | Accept JSON-RPC arrays |
| `MCP_DOWNSTREAM_TOKEN` | `gateway-service-token` | The credential the gateway presents downstream |

## Files

| File | What it does |
| --- | --- |
| `auth.py` | `Authorization` header → `Principal(subject, role, tenant)` |
| `policy.py` | Prefix rules, name normalisation, charset guard |
| `mcp_gateway.py` | The proxy: authenticate → authorize → forward |
| `downstream.py` | Mock MCP server that performs **no** authorization of its own |
| `tests/` | 91 tests |

---

## Design decisions

### HTTP 401 vs JSON-RPC -32001

Two different questions, two different layers:

| Question | Situation | Response |
| --- | --- | --- |
| *Who are you?* | No token, malformed header, unknown token | **HTTP 401** + `WWW-Authenticate: Bearer` |
| *May you do this?* | Known caller, tool they do not have | **HTTP 200** + JSON-RPC `-32001` |

An unauthenticated request has no established session, so there is nothing to
answer *within* the protocol — 401 with a challenge is what the MCP
authorization spec expects and what tells a client to re-authenticate rather
than retry. An authenticated-but-forbidden request was well-formed and *was*
answered; the answer is "no", which is application-level and belongs in the
JSON-RPC error channel. The 401 response also
carries a JSON-RPC error body, so a client that only parses bodies still gets
something structured.

### The client's token is never forwarded

The gateway terminates the caller's credential and mints its own for the
downstream hop:

```
Authorization: Bearer gateway-service-token   # the gateway's own identity
X-MCP-Gateway-Subject: grace@example.com      # who it is acting for
X-MCP-Gateway-Role: viewer
X-MCP-Gateway-Tenant: acme
X-Request-Id: 5f2c...                          # one id across both hops
```

Passing the original `Authorization` through is the classic **confused deputy**
setup: the downstream ends up holding a credential it was never issued and
cannot validate, any downstream logging bug becomes a credential leak, and a
compromised downstream can replay the user's token against *other* services.
Terminating it means the downstream trusts the identity headers precisely
because it first trusts the gateway's own credential — and `test_client_token_is_not_forwarded_downstream`
asserts the viewer's token appears nowhere in the forwarded headers.

### The mock downstream does no authorization

`downstream.py` executes whatever it is asked to execute. That is deliberate: if
it also checked roles, a passing authorization test would prove nothing about
the gateway. It records every call it receives, so tests assert the negative —
**the downstream was never contacted** — which is the actual security property.

### Policy as data, not as an `if`

```python
ToolPolicy(prefix_roles={"admin_": frozenset({"admin"})})
```

A policy you can print is a policy you can audit, and the next rule
(`billing_` needs finance) is a dict entry rather than a new branch in the
request handler. `default_allow=False` flips the whole gateway to
deny-by-default, which is what you would run once the tool inventory is known.

### The method name needs the same treatment as the tool name

The same reasoning applies one layer up, to the method name. Gating on
`method != "tools/call"` with an exact ASCII compare lets every lexical variant
skip the authorization branch and go straight to the forward path:

```
Tools/Call   TOOLS/CALL   tools/Call   "tools/call "   " tools/call"
```

A viewer's `admin_reset_key` would be **delivered to the downstream**, audited
as `forwarded` rather than `denied`. The mock downstream happens to reject those
spellings because its own dispatch is exact-match — which is precisely the
downstream leniency this gateway must not depend on.

The fix is the same two layers the tool name already had: `is_wellformed_method()`
rejects anything outside `^[A-Za-z0-9_./-]{1,128}\Z` with `-32600`, and the
routing decision compares `normalize_method(method)`. As with tool names, the
**original spelling is forwarded** — so an admin sending `Tools/Call` is
authorized by the gateway and then told `-32601` by the downstream, which is the
correct division of labour.

JSON-RPC method names are case-sensitive, so a strict downstream would call
`Tools/Call` unknown anyway. That is not the point: a gateway must not forward a
request whose meaning it did not resolve.

### Name normalisation, and its limits

`"admin_reset_key".startswith("admin_")` is the easy part. The attack is
everything that is *not* that string but that a downstream may still resolve to
that tool:

| Attempt | Stopped by | Code |
| --- | --- | --- |
| `Admin_reset_key`, `ADMIN_RESET_KEY` | policy, on a case-folded copy | `-32001` |
| `ａdmin_reset_key` (full-width) | charset guard (NFKC would fold it, but the guard runs first) | `-32602` |
| ` admin_reset_key`, `admin_reset_key\n` | charset guard | `-32602` |
| `Аdmin_reset_key` (**Cyrillic А**) | charset guard | `-32602` |

The Cyrillic case is the interesting one. NFKC folds *compatibility* equivalents,
but Cyrillic `А` and Latin `A` are genuinely different characters — no normal
form will ever unify them, and chasing homoglyphs with a lookalike table is
unwinnable. So the gateway constrains the input instead:
`^[A-Za-z0-9_.-]{1,128}\Z`, checked before the policy is consulted.

Two details in that regex:

- **`\Z`, not `$`.** In Python's `re`, `$` also matches immediately before a
  trailing newline, so `^[a-z_]+$` happily accepts `"admin_reset_key\n"` — a
  name shaped for log forging or for a downstream that strips whitespace before
  lookup. `test_malformed_names_rejected` pins it.
- **The *original* name is forwarded**, not the normalised one. A gateway that
  rewrites the caller's request is a gateway that will one day rewrite it wrong.

### `tools/list`: transparent by default, filterable by flag

Forwarding `tools/list` transparently is the default, and it is tested. `MCP_GATEWAY_FILTER_TOOLS_LIST=1` strips tools the caller could never
call, which is the stronger posture: a tool the model can see is a tool the
model will try, so listing `admin_delete_tenant` to a viewer buys a wasted turn,
a confusing error, and a free hint for an attacker. Both modes are tested.

### Batching is rejected by default

JSON-RPC batching was **removed from MCP in revision 2025-06-18**. The default is
therefore to reject arrays with `-32600` and say so in the message. The per-entry
path is implemented and tested behind `MCP_GATEWAY_ALLOW_BATCH=1` because plenty
of older clients still send arrays: each entry is authorized independently, only
survivors are forwarded, order is preserved, notifications produce no response,
and one denial does not abort the rest. All four of those are tested, as is a
downstream timeout mid-batch, which the first version of this code did not
handle at all.

### Errors do not leak infrastructure

A downstream `ConnectTimeout` carries text like `connect to 10.0.0.7:9001 timed
out`. The client gets `-32003 "Upstream MCP server timed out"` plus a
`request_id`; the detail goes to the audit log. `test_upstream_timeout_is_sanitised`
asserts the internal IP does not appear in the response body.

### Every decision is audited

One structured JSON line per decision, allows included:

```json
{"ts": 1788390833.242, "event": "tool_call_denied", "request_id": "88acd43f...",
 "subject": "grace@example.com", "role": "viewer", "tenant": "acme",
 "tool": "admin_reset_key", "rule": "prefix:admin_",
 "reason": "tool 'admin_reset_key' requires one of ['admin']; caller has role 'viewer'"}
```

Logging only denials tells you what was blocked but never what was *reachable*,
which is the question an incident review actually asks.

## Test coverage

```
tests/test_auth.py      12  header parsing, scheme case, token secrecy
tests/test_policy.py    25  prefix rules, normalisation, charset guard
tests/test_gateway.py   54  end-to-end proxy, batch semantics, downstream failure
```
