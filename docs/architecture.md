# How the four pieces fit together

The four services look independent. They are four layers of one system: two
halves of an agent's life, each with a gateway in front of it.

```
                        ┌──────────────────────────┐
                        │        AI agent          │
                        └───────────┬──────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
 ┌─────────────────┐     ┌────────────────────┐    ┌────────────────────┐
 │  MCP gateway    │     │  LLM gateway       │    │  Model router      │
 │                 │     │  streaming         │    │                    │
 │  authn + authz  │     │  guardrail         │    │  limits + fallback │
 └────────┬────────┘     └──────────┬─────────┘    └──────────┬─────────┘
          │                         │                         │
          ▼                         ▼                         ▼
 ┌─────────────────┐     ┌────────────────────┐    ┌────────────────────┐
 │  MCP server     │     │  model provider    │    │  primary provider  │
 │  tools + data   │     │  (SSE)             │    │  secondary         │
 └─────────────────┘     └────────────────────┘    └────────────────────┘
```

Read left to right, it is the two halves of an agent's life: **what it can do**
(tools, via MCP) and **what it can say** (tokens, via the model). Each half gets
a gateway, because the controls do not belong in every application.

---

## The four control points

| Layer | Controls | Fails by |
| --- | --- | --- |
| MCP server | What the tool accepts | Executing a malformed refund |
| MCP gateway | Who may invoke which tool | Letting a viewer reset a key |
| LLM gateway | What content leaves | Streaming an SSN to a screen |
| Model router | How much, and what happens when a provider dies | Unbounded spend, or a 3-second p99 |

Note that the MCP server and the MCP gateway both validate, and that is not redundancy by
accident. The gateway does not know the tool's schema — it knows *policy*. The
server does not know the caller's role — it knows *its own contract*. Each
enforces what it is authoritative for, and neither trusts the other to have done
it. If the gateway is bypassed (someone runs the server on stdio directly), the
schema still holds.

---

## One request, end to end

An agent asks to refund a customer:

1. **Agent → MCP gateway** with `Authorization: Bearer <user token>` and
   `tools/call {name: "trigger_refund"}`.
2. **Gateway authenticates.** No token or a bad one → HTTP 401 with a challenge.
3. **Gateway authorizes.** `trigger_refund` has no `admin_` prefix, so a viewer
   may call it. `admin_reset_key` would have stopped here with `-32001` and the
   MCP server would never have heard about it.
4. **Gateway forwards** — with its *own* credential, plus
   `X-MCP-Gateway-Subject/Role/Tenant` and a shared `X-Request-Id`. The user's
   token does not travel.
5. **MCP server validates** the arguments against its Pydantic schema. A bad
   amount → `-32602` with a field-level breakdown. A valid-but-impossible refund
   → `isError` result the model can read.
6. **The model is asked to summarise the outcome.** That request goes through the
   LLM gateway, which reserves budget, calls the primary under a 3-second
   deadline, fails over if needed, and redacts the streamed answer on the way
   back.

The `X-Request-Id` set at step 4 appears in the MCP gateway's audit log, the MCP
server's stderr, and the router's error envelope. One id, one incident, one
grep.

---

## What is deliberately not built

Worth naming explicitly.

- **No token verification.** The gateway's token table is a stand-in with a
  constant-time lookup. Production is JWT + JWKS or RFC 7662 introspection —
  `resolve_principal()` is the only function that changes.
- **No distributed state.** The rate limiter is SQLite, the breaker is
  per-process. Both are correct for one node and both have a stated Redis path.
- **No tool-description scanning.** An MCP server's *descriptions* are prompt
  injection surface — a malicious server can put instructions in the text the
  model reads. A real gateway scans them. Out of scope here, and the most
  obvious next addition.
- **No output-side tool-result inspection.** The MCP gateway inspects requests. A tool
  *result* can also carry PII or injected instructions; the natural fix is to run
  the redactor over the MCP gateway's responses, which is why they share a shape.

That last point is where the two gateways compose: the `StreamRedactor` drops
into the MCP gateway's response path essentially unchanged.

---

## Sequence: a denied call

```
agent                 MCP gateway              MCP server
  │                        │                        │
  │  tools/call            │                        │
  │  admin_reset_key       │                        │
  │  Bearer viewer-token   │                        │
  ├───────────────────────▶│                        │
  │                        │ resolve_principal      │
  │                        │   → role=viewer        │
  │                        │ is_wellformed_name     │
  │                        │   → ok                 │
  │                        │ policy.evaluate        │
  │                        │   → DENY               │
  │                        │ audit(tool_call_denied)│
  │  -32001                │                        │
  │◀───────────────────────┤                        │
  │                        │                    (never called)
```

The dotted half of that diagram is the property under test:
`assert downstream_app.state.calls == []`.
