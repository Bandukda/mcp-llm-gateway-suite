# MCP and LLM gateway suite

Four runnable services for the path between an AI agent and the systems it talks
to: an MCP tool server, a security gateway in front of MCP servers, a streaming
PII guardrail in front of a model provider, and a rate-limiting fallback router.

403 tests and two standalone proofs. Everything runs offline — mock providers and
a mock downstream MCP server are included, so there are no API keys to set.

```bash
./setup.sh                  # venv + dependencies
source .venv/bin/activate
./run_all_tests.sh          # everything, including the proofs
```

---

## What is here

| # | Directory | What it is | Tests |
| --- | --- | --- | --- |
| 1 | `task1_mcp_server/` | MCP tool server on stdio: strict Pydantic validation, JSON-RPC error mapping, OS-level stdout isolation | 58 |
| 2 | `task2_mcp_gateway/` | MCP security gateway: bearer auth, per-tool authorization, `-32001` denials | 91 |
| 3 | `task3_streaming_guardrail/` | LLM gateway: streaming PII redaction with a bounded hold-back buffer | 190 |
| 4 | `task4_router/` | Token-aware sliding window on SQLite, 3s deadline, failover, circuit breaker | 64 |

Each directory has its own README with the design reasoning. Start there.

## Documentation

| Doc | Read it when |
| --- | --- |
| [`docs/code-walkthrough.md`](docs/code-walkthrough.md) | **Start here.** Each service's problem in plain words, then the code that solves it |
| [`docs/architecture.md`](docs/architecture.md) | How the four pieces compose into one system, and what is deliberately not built |

---

## Running each service

```bash
# Task 1 — MCP server on stdio
cd task1_mcp_server && python server.py
python verify_stdout_purity.py              # proof that stdout stays clean
python verify_stdout_purity.py --unguarded  # the failure it prevents

# Task 2 — MCP gateway + mock downstream
cd task2_mcp_gateway
uvicorn downstream:downstream_app --port 9001 &
MCP_DOWNSTREAM_URL=http://127.0.0.1:9001/mcp uvicorn mcp_gateway:app --port 9000

# Task 3 — LLM gateway + mock provider
cd task3_streaming_guardrail
uvicorn mock_llm:mock_llm_app --port 9011 &
LLM_UPSTREAM_URL=http://127.0.0.1:9011/v1/chat/completions uvicorn llm_gateway:app --port 9010
python benchmark.py                         # TTFT and memory, over real sockets

# Task 4 — model router
cd task4_router && python app.py
```

---

## The four results worth looking at first

**Task 1 — stdout isolation, proved rather than asserted.**
The server is launched with `BILLING_MCP_CHAOS=1`, which makes it commit the four
sins that break stdio servers in production: a stray `print`, a raw `os.write(1, …)`,
a subprocess inheriting fd 1, and a `print` of a JSON-shaped string — a *forged
response* a client will accept.

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

**Task 2 — the denial is proved by a negative.**
The mock downstream performs no authorization of its own and records every call
it receives, so the assertion is `assert downstream_app.state.calls == []`. A
gateway that merely returned the right error while still forwarding the request
would fail.

**Task 3 — measured, not claimed.**

```
TTFT direct from mock  :    43.5 ms
TTFT through gateway   :    84.5 ms
a buffering proxy would:   210.0 ms

clean text streamed    : 3.24 MB
  peak hold-back       : 65 chars (bound 800)
  peak traced alloc    : 4.9 KiB
```

**Task 4 — concurrency, measured against the alternative.**
Twenty racing 10k-token requests against a 50k budget, five runs each:

```
BEGIN IMMEDIATE : winners=5    used=50000        busy=0     (identical every run)
BEGIN (deferred): winners=2-5  used=20000-50000  busy=15-18
```

Under WAL the deferred failure is `SQLITE_BUSY` on the lock upgrade, which
`busy_timeout` cannot retry — so most requests die with a database error instead
of a clean allow or deny. `test_deferred_transaction_is_not_safe` runs that
comparison rather than asserting the folklore.

---

## Bugs found during development

Kept here with their regression tests, because how a defect was found matters
more than the patch. Five independent adversarial review passes were run against
this code; everything below is fixed and pinned by a named test.

| Bug | Why it mattered |
| --- | --- |
| `\d` in `^CUST-\d{5}$` | Unicode-aware: `CUST-١٠٠٤٢` passed validation and broke everything downstream |
| `$` in a tool-name regex | Matches before a trailing newline, so `"admin_reset_key\n"` passed the charset check |
| SSN pattern ordered before credit card | Matched the first 9 digits of a card, emitting `[REDACTED]1111111` — publishing 7 digits |
| Redacting before computing hold-back | `"ada@example.co"` matched early; the `m` landed *after* the placeholder |
| Latency tests over `httpx.ASGITransport` | It buffers responses — the tests would have passed for a fully buffering gateway |
| `(` and `)` missing from the hold-back class | The phone pattern contains them: `"Call me at (555) 123-456"` + `"7 tomorrow"` streamed the number in the clear |
| Circuit breaker probe + a 4xx | `_probe_in_flight` was never cleared, so the primary was skipped for the life of the process even after recovering |
| `[a.__dict__ for a in attempts]` in the error envelope | Shipped provider names and upstream status codes to the client — the vendor topology the same file says stays in the log |
| Both edges of the buffer lied to the regex | Start- and end-of-string satisfy the `(?<![\d\-])` / `(?![\d\-])` fences, so a truncated buffer offered matches the full text never produces — the right edge published a whole card number |
| A Luhn rejection consumed the span it rejected | `"Invoice 12345 4111 1111 1111 1111"` left the card **entirely unredacted**: the digit prefix shifted the greedy match, Luhn failed, and the scan resumed past the card |
| An unbounded `EMAIL` pattern made the memory bound fiction | One match could be as long as the response — 3,044 characters buffered against a claimed 448 |
| The straddle guard silently broke the documented memory bound | The bound tests used input containing no match, so they could never see it |
| **`method != "tools/call"` compared exactly** | `Tools/Call` skipped authorization entirely and a viewer's `admin_reset_key` was **forwarded downstream**, audited as `forwarded` — the tool-name lesson, missed one layer up |
| An idempotency key bound to nothing | Reusing a key with different arguments replayed the first refund and reported success, so a $5 refund that never happened was announced as done |
| `json.loads` raises more than `JSONDecodeError` | A 4 KB body of digits or brackets became an unhandled 500 |
| `CancelledError` skipped the circuit breaker | A client disconnecting mid-probe wedged the breaker permanently, skipping a healthy primary for the life of the process |
| **The straddle guard and the emitter scanned from different origins** | They segmented adjacent matches out of phase, so the guard saw a clean split and the emitter published a real match's head — 41 SSNs in the clear from `"123-45-6789 " * 150` |
| `frame["choices"][0]` with no shape check | Six frame shapes aborted the stream, including OpenAI's own `include_usage` terminator |
| Only `choices[0]` was inspected | PII in `choices[1]` was forwarded unredacted, and the non-streaming path disagreed |

Several are worth dwelling on. The ASGI transport one was a bug in the *test*, which
is worse than a bug in the code because it gives false confidence. The paren one
was found by review rather than by tests, and the fix was not just widening the
character class: `pull_back_behind_straddling_match()` now makes the no-straddle
property structural, `assert_holdback_covers_patterns()` fails loudly if a new
pattern introduces an uncovered character, and the equivalence property test grew
from one text to nine across twelve chunk sizes.

---

## Notes on scope

- **The MCP SDK is pinned to `>=1.9,<2`.** v2 moved to constructor-based handler
  registration and now diverts fd 1 inside `stdio_server()` itself. This targets
  1.x — the API the published docs describe — and implements the stdout guard
  explicitly, where it is load-bearing.
- **Everything runs offline.** Mock providers and a mock downstream MCP server
  are included; no API keys, no network, no ports to configure for the tests.
- **Deliberately not built:** real JWT verification, distributed rate-limit
  state, and tool-description scanning for prompt injection. Each is called out
  in `docs/03-ARCHITECTURE.md` with the path to it.

## Layout

```
.
├── README.md                    this file
├── setup.sh                     venv + dependencies
├── run_all_tests.sh             every suite plus the standalone proofs
├── requirements.txt             pinned
├── pytest.ini
├── docs/
│   ├── 00-CONCEPTS-PRIMER.md
│   ├── 01-WHAT-EACH-TASK-ASKS.md
│   ├── 02-INTERVIEW-PREP.md
│   └── 03-ARCHITECTURE.md
├── task1_mcp_server/
├── task2_mcp_gateway/
├── task3_streaming_guardrail/
└── task4_router/
```
