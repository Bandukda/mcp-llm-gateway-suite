# Code walkthrough

Four services, each solving one problem in the path between an AI agent and the
systems it talks to. For each: the problem in plain words, why the obvious
approach fails, and the actual code that handles it. Every snippet is real code
from this repo, trimmed of comments for reading.

---

# 1. MCP server with strict validation and stdio handling

## The problem

A small server an AI model can call over stdio. It offers two tools: look up a
customer, and issue a refund.

Two things have to be true:

1. **Bad input gets rejected properly.** Not "handled gracefully" — *rejected*,
   with the correct error code, so the client library raises an exception rather
   than handing the model a confusing string.
2. **Only protocol data goes to stdout.** The server talks to its client through
   stdin/stdout. Any stray `print()` corrupts that conversation.

## Why the second one is a real problem

The server speaks JSON-RPC over **file descriptor 1**. One message per line:

```
{"jsonrpc":"2.0","id":1,"result":{"tools":[...]}}
```

If anything else writes there, it lands *inside* a message:

```
{"jsonrpc":"2.0","id":1,"reDEBUG: connecting to dbsult":{...}}
```

The client's parser dies. Worse: a stray `print` of a JSON-shaped string is a
**forged response** the client will believe.

## How the code does it

### Reserving stdout — `stdio_guard.py`

"Never call print" is a convention, not a control. It fails the first time a
dependency prints a banner. So the fix is at the operating-system level:

```python
def reserve_stdout_for_protocol() -> io.TextIOWrapper:
    sys.stdout.flush()

    wire_fd = os.dup(1)                    # private copy: the only handle
    os.set_inheritable(wire_fd, False)     # that may carry protocol frames

    os.dup2(2, 1)                          # fd 1 now IS stderr

    return io.TextIOWrapper(os.fdopen(wire_fd, "wb", buffering=0),
                            encoding="utf-8", newline="\n", write_through=True)
```

Three lines, and because it operates on the *descriptor* rather than on
`sys.stdout`, it covers every writer:

| Writer | Without the guard | With it |
| --- | --- | --- |
| `print("debug")` | corrupts the stream | stderr |
| `os.write(1, b"...")` from a C extension | corrupts the stream | stderr |
| a subprocess inheriting fd 1 | corrupts the stream | stderr |
| a forgotten line that *looks* like JSON-RPC | silently accepted | stderr |

`server.py` calls it before importing anything else, and hands the returned
stream to the transport:

```python
from stdio_guard import reserve_stdout_for_protocol
PROTOCOL_STDOUT = reserve_stdout_for_protocol()
...
async with stdio_server(stdout=anyio.wrap_file(PROTOCOL_STDOUT)) as (r, w):
    await server.run(r, w, initialization_options(server))
```

### Proving it — `verify_stdout_purity.py`

This is proved rather than asserted. `BILLING_MCP_CHAOS=1`
makes the server misbehave in all four ways above, then a real MCP conversation
runs and every stdout line is checked:

```
$ python verify_stdout_purity.py
PASS
  5 stdout lines, all valid JSON-RPC 2.0
  all 4 deliberate stdout writes were diverted to stderr

$ python verify_stdout_purity.py --unguarded
FAIL
  - line 1 is not JSON: 'DEBUG: about to serve requests'
  - line 4 ... {"jsonrpc": "2.0", "id": 999, "result": {"spoofed": true}}
```

That last line is the point of the exercise.

### Getting real error codes — the non-obvious part of `server.py`

The natural way to register a tool is the SDK's decorator. It doesn't work here:

```python
# What the SDK's @server.call_tool() decorator does internally:
except Exception as e:
    return self._make_error_result(str(e))     # a SUCCESSFUL response, isError: true
```

Every failure — including malformed arguments — comes back as a *successful*
response, so `-32602` is never emitted at all and the client library has no way
to distinguish a broken call from a working one. The fix is to register the
handler directly so
`McpError` propagates to the session, which serialises it as a real JSON-RPC
error:

```python
async def call_tool(req: types.CallToolRequest) -> types.ServerResult:
    if name not in TOOLS:
        raise invalid_params(f"Unknown tool: {name!r}", {"available_tools": sorted(TOOLS)})
    try:
        args = model.model_validate(raw_arguments)
    except ValidationError as exc:
        raise invalid_params(f"Invalid arguments for tool {name!r}",
                             _pydantic_error_payload(exc)) from None
    return types.ServerResult(DISPATCH[name](args, store))

server.request_handlers[types.CallToolRequest] = call_tool   # not the decorator
```

### Where each kind of error goes

The dividing line is **"could the caller have known?"**

| Condition | Response | Code |
| --- | --- | --- |
| unknown tool name | JSON-RPC error | `-32602` |
| schema violation | JSON-RPC error + per-field `data` | `-32602` |
| customer id valid but absent | `CallToolResult(isError=true)` | — |
| refund exceeds balance | `CallToolResult(isError=true)` | — |
| unexpected exception | JSON-RPC error, message sanitised | `-32603` |

A malformed argument is a contract violation → protocol error. "That customer
doesn't exist" is a legitimate answer to a legitimate call → tool result, so the
model can read it, apologise and try a different id.

### The validation — `schemas.py`

```python
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

CUSTOMER_ID_PATTERN = r"^CUST-[0-9]{5}$"     # [0-9], not \d

class TriggerRefundInput(StrictModel):
    customer_id: str = Field(..., pattern=CUSTOMER_ID_PATTERN)
    amount: float = Field(..., gt=0, le=10_000, allow_inf_nan=False)
    reason: str = Field(..., min_length=10, max_length=500)
```

Each choice has a test behind it:

| Guard | Why |
| --- | --- |
| `extra="forbid"` | `{"ammount": 50}` (typo) must fail loudly, not refund `None` |
| `strict=True` | lax Pydantic turns `"49.99"` into `49.99`. Money isn't guessed |
| bool rejected for amount | `bool` subclasses `int`; lax mode makes `True` → `1.0` |
| `allow_inf_nan=False` | Python's `json` accepts `Infinity`, and `Infinity > 0` is `True` |
| `reason` stripped first | ten spaces satisfies `min_length=10` but isn't a reason |
| **`[0-9]` not `\d`** | `\d` is Unicode-aware; `CUST-١٠٠٤٢` passed validation |

That last one was a real bug the tests caught.

---

# 2. MCP security gateway proxy

## The problem

A checkpoint between an AI agent and an MCP server. Read the caller's token to
work out their role. If they ask "what tools exist?", pass it through. If they
ask to *run* a tool whose name starts with `admin_`, they must be an admin —
otherwise reject it yourself, with error `-32001`, **without bothering the
downstream server at all**.

```
agent ──Bearer token──▶ gateway ──service credential──▶ downstream MCP server
                           │
                           └── denied → -32001, downstream never contacted
```

## How the code does it

### The decision, as data — `policy.py`

The rule lives in a dict, not in an `if` buried in the request handler. A policy
you can print is a policy you can audit, and the next rule is one line:

```python
@dataclass(frozen=True)
class ToolPolicy:
    prefix_roles: dict[str, frozenset[str]] = field(
        default_factory=lambda: {"admin_": frozenset({"admin"})})
    default_allow: bool = True

    def evaluate(self, tool_name: str, principal: Principal) -> Decision:
        normalized = normalize_tool_name(tool_name)
        for prefix, allowed_roles in self.prefix_roles.items():
            if normalized.startswith(prefix):
                if principal.role in allowed_roles:
                    return Decision(True, ..., rule=f"prefix:{prefix}")
                return Decision(False, ..., rule=f"prefix:{prefix}")
        return Decision(self.default_allow, ...)
```

### The hard part isn't the prefix check

`"admin_reset_key".startswith("admin_")` is easy. The attack is everything that
*isn't* that string but that the downstream may still resolve to that tool:

| Attempt | Stopped by | Code |
| --- | --- | --- |
| `Admin_reset_key`, `ADMIN_RESET_KEY` | policy, on a case-folded copy | `-32001` |
| `ａdmin_reset_key` (full-width) | charset guard | `-32602` |
| `" admin_reset_key"`, `"admin_reset_key\n"` | charset guard | `-32602` |
| `Аdmin_reset_key` (**Cyrillic А**) | charset guard | `-32602` |

The Cyrillic case is the instructive one. NFKC folds *compatibility* variants, so
full-width `ａ` becomes `a`, but Cyrillic `А` and Latin `A` are genuinely
different characters — no normal form will ever unify them, and a lookalike table
is a losing game. So constrain the input instead:

```python
TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}\Z")   # \Z, not $
```

`\Z` not `$`, because in Python `$` also matches before a trailing newline, so
`^[a-z_]+$` happily accepts `"admin_reset_key\n"` — a name shaped for log forging.
That was a real bug too.

### The bug an adversarial pass found

All that care went into the *tool* name. The *method* name, one line above, was
compared exactly:

```python
if method != "tools/call":     # ← Tools/Call skipped authorization entirely
    return None
```

So `Tools/Call`, `TOOLS/CALL` and `" tools/call"` never reached the policy, and a
viewer's `admin_reset_key` was **forwarded downstream**, audited as `forwarded`.
Same fix, one layer up:

```python
if not is_wellformed_method(method):
    return rpc_error(rpc_id, INVALID_REQUEST, "method contains characters outside...")
if normalize_method(method) != "tools/call":
    return None
```

The lesson generalises: the control had been applied to one field and not to the
field next to it.

### The client's token never travels

```python
def downstream_headers(principal: Principal, request_id: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {DOWNSTREAM_TOKEN}",   # the gateway's own
        "X-MCP-Gateway-Subject": principal.subject,      # who it acts for
        "X-MCP-Gateway-Role": principal.role,
        "X-Request-Id": request_id,
    }
```

Forwarding the caller's `Authorization` is the classic **confused deputy**: the
downstream ends up holding a credential it was never issued and cannot validate,
and a compromised downstream can replay the user's token elsewhere.

### Proving the denial

The mock downstream performs **no authorization of its own** and records every
call. That's deliberate — if it also checked roles, a passing test couldn't
distinguish "the gateway blocked it" from "the downstream refused it":

```python
async def test_denied_call_never_reaches_downstream(gateway, downstream_app):
    await gateway.post("/mcp", json=call_tool("admin_reset_key", ...),
                       headers={"Authorization": VIEWER})
    assert downstream_app.state.calls == []      # the actual security property
```

---

# 3. Streaming PII redaction guardrail

## The problem

Sit between an app and an LLM. As the model's answer streams back token by token,
spot emails, SSNs and card numbers and replace them with `[REDACTED]` — without
waiting for the whole answer, and without holding it in memory.

## Why this is the hardest of the four

The model doesn't emit PII in convenient pieces:

```
chunk 1: "Her SSN is 123"     ← forward this and it's on screen
chunk 2: "-45"
chunk 3: "-6789"              ← too late
```

`re.sub` on each chunk finds nothing, because no single chunk contains a complete
pattern. **You cannot un-send a token.**

Buffering the whole response and redacting at the end is correct and useless:
time-to-first-token becomes time-to-*last*-token, and memory grows with response
length. Streaming is the product; that approach throws it away.

## How the code does it

### The hold-back — `redactor.py`

Emit only what is provably safe. On each chunk:

```python
def feed(self, chunk: str) -> str:
    self._buffer += chunk
    whole  = self._left_context + self._buffer
    offset = len(self._left_context)

    matches = list(iter_matches(whole, offset))          # one scan

    hold  = holdback(self._buffer, self.max_holdback)    # what might still grow
    split = offset + len(self._buffer) - hold
    split = pull_back_behind_straddling_match(whole, split, offset, matches)

    safe_raw = self._buffer[:split - offset]
    emit = redact_range(whole, offset, split, self.stats, self.placeholder, matches)

    self._left_context = (self._left_context + safe_raw)[-LOOKBEHIND_CONTEXT:]
    self._buffer = self._buffer[split - offset:]
    return emit
```

Two rules compute the hold-back:

```python
TAIL_TOKEN  = re.compile(r"[A-Za-z0-9@._%+\-]+\Z")                    # emails, SSNs, keys
TAIL_DIGITS = re.compile(r"(?:[+(][\d ()\-]*|\d[\d ()\-]*)\Z")        # spaced cards, phones
```

Prose is unaffected because a space ends the first run — `"Hello there"` holds
back five characters, not the sentence.

### Step order is the whole trick

```python
# WRONG
buffer = redact_complete(buffer + chunk)   # "ada@example.co" is already a match
emit   = buffer[:len(buffer) - holdback(buffer)]
# client sees "[REDACTED]" ... then the "m" arrives after it
```

Holding back *first* means a pattern that is still growing is never judged.

### Both edges of the buffer lie to the regex

Several patterns are fenced with `(?<![\d\-])`. A lookbehind at position 0 of a
*truncated* string sees "start of input" and is satisfied even when the real
preceding character was a digit. End-of-string satisfies the trailing fence the
same way — and that one published a whole card number. So the scan sees the
complete buffer and only the safe window is emitted:

```python
def redact_range(text, start, end, stats=None, placeholder=..., matches=None):
    pieces, cursor = [], start
    for match in matches if matches is not None else iter_matches(text, start):
        if match.end() > end:
            break                    # still growing; belongs to the held tail
        pieces.append(text[cursor:match.start()])
        pieces.append(_record(match, stats, placeholder))
        cursor = match.end()
    pieces.append(text[cursor:end])
    return "".join(pieces)
```

### A rejected match must not consume the text it rejected

`CREDIT_CARD` matches any 13–19 digit run; Luhn decides whether it's really a
card. Returning the span verbatim made the scan resume at its *end*, so real PII
the run had swallowed was never examined:

```
"Invoice 12345 4111 1111 1111 1111 was charged."   →  unchanged
```

The five-digit invoice number shifts the greedy match one digit left, Luhn fails,
and the whole card is published. So a rejection advances a single character:

```python
def iter_matches(text: str, start: int = 0):
    pos = start
    while True:
        match = COMBINED.search(text, pos)
        if match is None:
            return
        if match.lastgroup == "CREDIT_CARD" and not _is_card(match.group(0)):
            pos = match.start() + 1          # do not consume the span
            continue
        yield match
        pos = match.end()
```

### False positives are a product bug too

A guardrail that redacts order numbers is one somebody turns off:

| Input | Result | Why |
| --- | --- | --- |
| `4111 1111 1111 1111` | `[REDACTED]` | Luhn-valid |
| `Order 1234567890123456` | untouched | fails Luhn |
| `123-45-6789` | `[REDACTED]` | valid SSN shape |
| `000-00-0000` | untouched | never-issued area number |

### What it costs

```
TTFT direct from mock  :    43.5 ms
TTFT through gateway   :    84.5 ms
a buffering proxy would:   210.0 ms

clean text streamed    : 3.24 MB
  peak hold-back       : 65 chars (bound 800)
  peak traced alloc    : 4.9 KiB
```

The ~40 ms is one inter-delta gap and it's explainable: the first delta `"Hello"`
is a trailing token with no word boundary, so it waits for the next delta to
prove it isn't the start of an email. That cost is a function of **token pacing,
not response length** — it doesn't compound, as the near-identical inter-frame
medians show.

---

# 4. Rate limiter and model fallback router

## The problem

Track how many tokens each customer used in the last 60 seconds and cut them off
above 50,000. Store the counts in a SQLite file on disk. If the main model
provider says "too many requests" or takes longer than 3 seconds, switch to a
backup. When things fail, return a tidy error that gives nothing away.

## How the code does it

### You don't know the token count in advance

You know the prompt going in; the completion size only exists afterwards. A
limiter that counts *after* lets unbounded concurrency start against the same
remaining budget. So the budget is taken up front and trued up after:

```python
reservation = await limiter.reserve(api_key, estimated_tokens)   # hold it now
completion  = await provider.complete(prompt, max_tokens)
await limiter.commit(reservation, completion.total_tokens)       # true it up
# or await limiter.release(reservation) if the call never happened
```

It's an airline seat hold. And the true-up is throughput, not just bookkeeping: a
request asking for 900 tokens holds 900 while in flight, but if the model returns
7, `commit()` frees the other 893 immediately.

`release()` in a `finally` means **a provider outage doesn't also exhaust the
tenant's quota.**

### Making SQLite safe under concurrency

```python
connection.execute("PRAGMA journal_mode=WAL")
connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
connection.execute("BEGIN IMMEDIATE")          # write lock at BEGIN, not on upgrade
used = connection.execute(
    "SELECT COALESCE(SUM(tokens),0) FROM token_usage WHERE api_key=? AND created_at>=?",
    (api_key, cutoff)).fetchone()[0]
if used + estimated_tokens > self.limit:
    connection.execute("ROLLBACK"); raise RateLimitExceeded(...)
connection.execute("INSERT INTO token_usage ... 'reserved')", ...)
connection.execute("DELETE FROM token_usage WHERE created_at < ?", (cutoff,))
connection.execute("COMMIT")
```

`BEGIN IMMEDIATE` is the one that matters, and it's measured rather than assumed.
20 racing 10k requests against a 50k budget, five runs each:

```
BEGIN IMMEDIATE : winners=5      used=50000        busy=0     (identical every run)
BEGIN           : winners=2-5    used=20000-50000  busy=15-18
```

A *deferred* transaction takes a read lock and upgrades on first write, so the
lock arrives **after** the budget check passed. Under WAL the failure is
`BUSY_SNAPSHOT`, which `busy_timeout` can't retry — an availability bug before
it's an accounting one.

Note the eviction `DELETE` runs *after* the check. Putting it first also takes the
write lock, which masks the isolation level entirely and makes the claim
untestable. That was the bug in the first draft: the guarantee was real, but
nothing proved it.

Every call goes through `asyncio.to_thread`, because SQLite is blocking and
calling it from the event loop stalls every other in-flight request.

### The timeout, and what a deadline promises

```python
try:
    completion = await asyncio.wait_for(provider.complete(prompt, max_tokens),
                                        timeout=timeout_s)
except asyncio.TimeoutError:
    return None, AttemptRecord(provider.name, "timeout", await elapsed_ms())
```

`wait_for` cancels, but cancellation is cooperative — the task gets
`CancelledError` at its next await point and may still be unwinding. So reserved
tokens are released in a `finally`, and a response arriving *after* the deadline
is discarded rather than raced into the reply. A deadline honoured late is still
broken.

### Not every failure deserves a retry

| Upstream | Action | Why |
| --- | --- | --- |
| 429 | fail over | the *provider* is saturated, not the request |
| 5xx, timeout, transport | fail over | provider-side fault |
| **4xx** | **return** | the request is wrong; a second provider returns a second 400 |

### The circuit breaker

Failover alone has a failure mode that looks like success: if the primary is
*down*, every request pays the full 3 s before switching. The fallback works, and
p99 is now 3 seconds.

```python
def allows_request(self, now=None) -> bool:
    state = self.state(now)
    if state == "closed":  return True
    if state == "open":    return False
    if self._probe_in_flight: return False      # half-open: exactly one probe
    self._probe_in_flight = True
    return True
```

Two ways it could wedge, both found by adversarial testing: a half-open probe
that got a **4xx** never cleared the flag, and a **client disconnect** re-raised
`CancelledError` past it. Either left the breaker refusing every probe forever. A
4xx now counts as provider health (the provider answered); cancellation calls
`abandon_probe()`, which releases the slot without claiming to know anything.

### One error envelope, one place that decides what leaks

```python
def to_payload(self) -> dict[str, Any]:
    error = {"type": "gateway_error",
             "code": self.code,
             "message": SAFE_MESSAGES.get(self.code, "An error occurred."),
             "request_id": self.request_id}
    if self.metadata:
        error.update(self.metadata)
    return {"error": error}          # `detail` is structurally unreachable
```

`GatewayError` carries a `detail` field that is **logged and never serialised**.
That matters because upstream errors embed the thing that failed: `httpx` puts
host and port in `ConnectError`, provider 400s quote your request back, a stack
trace names your file layout.

The tests assert *negatives* — a leak is never a failing assertion elsewhere, it's
an extra string nobody looked for:

```python
SECRETS = ["10.4.2.19", "8443", "sk-live-SUPERSECRET0123456789",
           "/opt/gateway/providers.py", "Traceback", "line 91"]

for secret in SECRETS:
    assert secret not in response.text
```

---

# The thread running through all four

Each service is the same question in a different costume:

> **When something goes wrong at a boundary, does the failure stay contained, and
> does the other side learn exactly what it needs and nothing more?**

- The MCP server: a stray print must not corrupt the protocol; a bad argument must not
  look like a tool result.
- The security gateway: a denied call must not reach the downstream; the caller's token must
  not either.
- The streaming guardrail: a partial SSN must not reach the screen; a long response must not reach
  memory.
- The router: one provider's outage must not consume the tenant's budget; its stack
  trace must not reach the client.
