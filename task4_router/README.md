# Rate-limiting and model fallback router

Token-aware sliding-window rate limiting per tenant API key on **on-disk
SQLite**, a hard 3000 ms deadline on the primary provider, automatic failover to
a secondary on 429 / timeout / 5xx, and a single sanitised error envelope that
never leaks upstream detail.

```
request ─▶ reserve budget ─▶ primary (3000 ms) ─▶ commit actual tokens
                                  │
                                  ├─ 429 / timeout / 5xx ─▶ secondary
                                  └─ 4xx ─────────────────▶ return, no failover
```

## Run it

```bash
python app.py    # or: uvicorn app:app --port 9020

curl -s localhost:9020/v1/completions \
  -H 'Authorization: Bearer tenant-a-key' -H 'Content-Type: application/json' \
  -d '{"prompt":"hello","max_tokens":64}'

curl -s localhost:9020/v1/usage -H 'Authorization: Bearer tenant-a-key'

python -m pytest tests -q    # 64 tests
```

| Env var | Default |
| --- | --- |
| `ROUTER_DB_PATH` | `router_usage.db` |
| `ROUTER_TOKENS_PER_MINUTE` | `50000` |
| `ROUTER_ATTEMPT_TIMEOUT_S` | `3.0` |

## Files

| File | What it does |
| --- | --- |
| `rate_limiter.py` | Sliding window over SQLite, reserve/commit/release |
| `router.py` | Deadlines, failover, circuit breaker |
| `providers.py` | Provider protocol + mocks with scriptable failure modes |
| `errors.py` | The one error envelope, and the one place that decides what leaks |
| `app.py` | FastAPI surface |

---

## Reserve / commit, not count-afterwards

A request's token cost is not known until the response exists. You know the
prompt going in; you learn the completion coming out. A limiter that only counts
*after* the fact lets unbounded concurrency start against the same remaining
budget — fine at low load, a stampede at high load, because every in-flight
request sees the same "plenty left".

```python
reservation = await limiter.reserve(key, estimated_tokens)   # take the budget now
completion  = await provider.complete(...)
await limiter.commit(reservation, completion.total_tokens)   # true it up
# or await limiter.release(reservation) if the call never happened
```

Between reserve and commit the tokens are **held**: they count against the
window, so concurrent requests see them. Same shape as an airline seat hold, for
the same reason.

The true-up is not just bookkeeping, it is throughput. A request asking for
`max_tokens=900` holds ~900 while in flight, but if the model returns 7 tokens,
`commit()` releases the other 893 immediately rather than billing the tenant
against a ceiling they never reached.

`release()` in a `finally` means a **provider outage does not also exhaust the
tenant's quota** — `test_failed_request_does_not_consume_budget`.

## Making SQLite safe under concurrency

| Concern | Handling |
| --- | --- |
| Readers blocking the writer | `PRAGMA journal_mode=WAL` |
| Check-then-act race | **`BEGIN IMMEDIATE`** around check-and-insert |
| Lock contention | `PRAGMA busy_timeout=5000` |
| Blocking the event loop | Every call via `asyncio.to_thread` |
| Unbounded growth | Lazy eviction of rows older than the window on each check |

`BEGIN IMMEDIATE` is the one that matters, and it is worth being precise about
why rather than repeating the folklore. A default *deferred* transaction takes a
read lock and upgrades on first write, so the write lock is acquired **after**
the budget check has already passed. Measured over five runs of a 20-way race
for a 50,000-token budget:

```
BEGIN IMMEDIATE : winners=5      used=50000        busy=0     (identical every run)
BEGIN           : winners=2-5    used=20000-50000  busy=15-18
```

Under WAL the deferred failure is `SQLITE_BUSY` / `BUSY_SNAPSHOT` on the
upgrade, which `busy_timeout` **cannot** retry — the reader's snapshot is already
stale — so about 16 of 20 requests die with a database error rather than getting
a clean allow or deny. It is an availability bug before it is an accounting one.
Under rollback-journal mode the same race surfaces as straightforward
oversubscription instead.

Two tests pin this: `test_concurrent_reservations_do_not_oversubscribe` asserts
exactly 5 of 20 win, and `test_deferred_transaction_is_not_safe` substitutes a
plain `BEGIN` and asserts it fails. One more detail makes the second test
possible at all — the eviction `DELETE` runs *after* the budget check. Ordering
it first also takes the write lock, which masks the isolation level completely
and makes the claim untestable. That was the bug in the first draft: the
guarantee was real, but nothing proved it.

`asyncio.to_thread` is the one that is easiest to skip and hardest to notice.
SQLite calls are blocking; making them from the event loop stalls every other
in-flight request in the process. That is how an async gateway becomes
mysteriously slow under load, with no single slow line to point at.
`test_limiter_does_not_block_the_event_loop` runs a heartbeat task during 50
writes and asserts it still got scheduled.

**Sliding, not fixed buckets.** A calendar-minute counter allows 2× the limit
across a boundary: 30k at 11:59:59 and 30k at 12:00:01. `test_window_is_rolling_not_fixed_buckets`
pins the difference.

SQLite is right for one node. For several gateway replicas the same protocol
moves to Redis with a Lua script — the interface does not change, only
`_reserve_sync`.

## Timeouts, and what a deadline actually promises

`asyncio.wait_for` cancels the wrapped task, but cancellation is cooperative: the
task receives `CancelledError` at its next await point and may still be unwinding
when the fallback starts. Two consequences are handled:

- **Reserved tokens are released in a `finally`**, so a cancelled attempt never
  permanently consumes budget it did not use.
- **A late response is discarded, not raced into the reply.** The deadline is a
  promise to the caller; honouring it late is still breaking it.

`test_timeout_is_honoured_within_tolerance` points the router at a provider that
sleeps 30 s with a 100 ms deadline and asserts the whole call finishes in under
500 ms — which only holds if the task was genuinely cancelled rather than
abandoned.

**Per-attempt deadline plus an optional total budget.** Giving the fallback a
fresh 3000 ms doubles the worst case to 6 s, which is rarely what "3 second
timeout" meant to whoever wrote the SLA. `total_deadline_s` caps the whole
operation; the per-attempt deadline becomes `min(per_attempt, time remaining)`.

## Not every failure deserves a retry

| Upstream | Action | Why |
| --- | --- | --- |
| 429 | fail over | the *provider* is saturated, not the request |
| 5xx | fail over | provider-side fault |
| timeout | fail over | provider-side fault |
| transport error | fail over | never reached the provider |
| **4xx** | **return** | the request is wrong; a second provider returns a second 400, one more bill, one more second of latency |

`test_4xx_does_not_trigger_failover` asserts the secondary is never called.

## The circuit breaker

Failover alone is not enough. If the primary is *down*, every request pays the
full 3 s deadline before failing over: the fallback works, and p99 becomes the
timeout value. After `failure_threshold` consecutive failures the breaker opens
and the primary is skipped entirely for `recovery_time_s`; then a single probe
decides whether to close it.

Two ways the breaker could get stuck, both found by adversarial testing rather
than by the tests: a half-open probe that got a **4xx** never cleared
`_probe_in_flight` (the client-error path raised before touching the breaker),
and a **client disconnect** mid-probe re-raised `CancelledError` past it. Either
left the breaker reporting half-open while refusing every probe, so a fully
recovered primary was skipped for the life of the process — exactly the "fallback
becomes the outage" failure the breaker exists to prevent. A 4xx now counts as
provider health (the provider answered); cancellation calls `abandon_probe()`,
which releases the slot without claiming to know anything.

`test_open_breaker_removes_the_timeout_penalty` is the one that shows the point:
with a 200 ms deadline and a dead primary, the third request completes in under
100 ms because the primary is never dialled.

## Error sanitisation

One envelope, one place that decides what leaks:

```json
{"error": {"type": "gateway_error",
           "code": "all_providers_failed",
           "message": "All model providers are currently unavailable.",
           "request_id": "8bc7998fc3224958aa3f20ce42fe78cf",
           "providers_attempted": 2,
           "outcomes": ["retryable_error", "timeout"]}}
```

`GatewayError` carries a `detail` field that is **logged and never serialised** —
`to_payload()` structurally cannot emit it. Everything a client might want for a
support ticket is in `request_id`; everything an attacker might want is in the
log.

The two extra fields are chosen, not incidental. A caller needs to know how hard
the gateway tried and whether retrying is sensible. It does not need
`[{"provider": "primary-openai-us-east", "status_code": 503}, ...]`, which is
what a naive `[a.__dict__ for a in attempts]` would have shipped — the vendor
topology and raw upstream status codes that `errors.py` says stay in the log.
`test_attempt_metadata_carries_no_provider_internals` names the providers after
the real ones the app uses and asserts neither vendor appears in the body.

This matters because upstream errors routinely embed the thing that failed.
`httpx` puts host and port in `ConnectError`. Provider 400s quote the request
back, sometimes including the Authorization header. A stack trace names your
file layout and library versions. None of it helps the caller.

`test_error_sanitisation.py` asserts negatives — a leak is never a failing
assertion elsewhere, it is an extra string nobody looked for. The `leaky` mock
provider raises with an internal IP, a live-looking API key and a file path in
one message, and the tests assert each is absent from the response. There is also
a test that an unhandled exception *inside the gateway itself* still produces a
clean 500, and one that a Pydantic validation error does not echo back a
misplaced API key from the request body.

## Test coverage

```
tests/test_rate_limiter.py         17  window, eviction, concurrency, persistence
tests/test_router.py               22  failover triggers, deadlines, breaker, budget
tests/test_error_sanitisation.py   25  leak negatives, status codes, envelope shape
```

## Production notes

- Multi-replica: move `_reserve_sync` to Redis + Lua. Everything else stands.
- Estimation quality drives how tight you can run the limit. The current
  4-chars-per-token heuristic is deliberately conservative; a real tokenizer
  (`tiktoken`) reduces the over-hold, at the cost of a dependency and some CPU.
- Streaming responses need `commit()` on stream completion, plus a `release()`
  on client disconnect. The reserve/commit protocol already accommodates it.
- The breaker is per-process. Across replicas, either accept N independent
  breakers (usually fine, they converge) or share state in Redis.
