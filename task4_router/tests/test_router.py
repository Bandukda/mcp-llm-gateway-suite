"""Failover mechanics: timeouts, retryable vs terminal errors, breaker, budget."""

import asyncio
import time

import pytest

from errors import GatewayError, GatewayErrorCode
from providers import MockProvider
from rate_limiter import SlidingWindowRateLimiter
from router import CircuitBreaker, ModelRouter


def build(primary_behaviour="ok", secondary_behaviour="ok", limiter=None, **kwargs):
    primary = MockProvider("primary", primary_behaviour, latency_s=kwargs.pop("primary_latency", 0.01))
    secondary = MockProvider("secondary", secondary_behaviour, latency_s=0.01)
    router = ModelRouter(primary, secondary, limiter, **kwargs)
    return router, primary, secondary


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
async def test_primary_is_used_when_healthy(limiter):
    router, primary, secondary = build(limiter=limiter)
    result = await router.route("tenant-a", "hello")
    assert result.completion.provider == "primary"
    assert result.failed_over is False
    assert secondary.call_count == 0


async def test_usage_is_committed_with_actual_tokens(limiter):
    router, _, _ = build(limiter=limiter)
    result = await router.route("tenant-a", "hello there", max_tokens=500)
    used = (await limiter.usage("tenant-a")).used_tokens
    # The estimate reserved ~502; the commit replaces it with the real total.
    assert used == result.completion.total_tokens
    assert used < 100


# ---------------------------------------------------------------------------
# Failover triggers
# ---------------------------------------------------------------------------
async def test_429_triggers_failover(limiter):
    router, primary, secondary = build("rate_limit", limiter=limiter)
    result = await router.route("tenant-a", "hello")
    assert result.completion.provider == "secondary"
    assert result.failed_over is True
    assert primary.call_count == 1 and secondary.call_count == 1


async def test_timeout_triggers_failover(limiter):
    router, _, secondary = build("timeout", limiter=limiter, attempt_timeout_s=0.05)
    result = await router.route("tenant-a", "hello")
    assert result.completion.provider == "secondary"
    assert result.attempts[0].outcome == "timeout"
    assert secondary.call_count == 1


async def test_5xx_triggers_failover(limiter):
    router, _, _ = build("server_error", limiter=limiter)
    assert (await router.route("tenant-a", "hello")).completion.provider == "secondary"


async def test_timeout_is_honoured_within_tolerance(limiter):
    router, _, _ = build("timeout", limiter=limiter, attempt_timeout_s=0.1)
    started = time.perf_counter()
    await router.route("tenant-a", "hello")
    elapsed = time.perf_counter() - started
    # 100ms of waiting on the primary plus a fast secondary. Nowhere near the
    # provider's 30s sleep, which proves the task was actually cancelled.
    assert 0.09 < elapsed < 0.5


async def test_4xx_does_not_trigger_failover(limiter):
    """A bad request is bad at every provider. Failing over buys a second bill."""
    router, _, secondary = build("bad_request", limiter=limiter)
    with pytest.raises(GatewayError) as exc:
        await router.route("tenant-a", "hello")
    assert exc.value.code == GatewayErrorCode.UPSTREAM_ERROR
    assert secondary.call_count == 0


async def test_both_providers_failing_is_a_single_clean_error(limiter):
    router, _, _ = build("server_error", "server_error", limiter=limiter)
    with pytest.raises(GatewayError) as exc:
        await router.route("tenant-a", "hello")
    assert exc.value.code == GatewayErrorCode.ALL_PROVIDERS_FAILED
    assert exc.value.http_status == 503


async def test_both_timing_out_reports_a_timeout(limiter):
    router, _, _ = build("timeout", "timeout", limiter=limiter, attempt_timeout_s=0.05)
    with pytest.raises(GatewayError) as exc:
        await router.route("tenant-a", "hello")
    assert exc.value.code == GatewayErrorCode.UPSTREAM_TIMEOUT
    assert exc.value.http_status == 504


# ---------------------------------------------------------------------------
# Budget interaction
# ---------------------------------------------------------------------------
async def test_failed_request_does_not_consume_budget(limiter):
    """A provider outage must not also exhaust the tenant's quota."""
    router, _, _ = build("server_error", "server_error", limiter=limiter)
    with pytest.raises(GatewayError):
        await router.route("tenant-a", "hello " * 100)
    assert (await limiter.usage("tenant-a")).used_tokens == 0


async def test_rate_limited_request_never_calls_a_provider(db_path):
    limiter = SlidingWindowRateLimiter(db_path, limit_tokens_per_minute=100)
    router, primary, secondary = build(limiter=limiter)
    with pytest.raises(GatewayError) as exc:
        await router.route("tenant-a", "hello", max_tokens=500)
    assert exc.value.code == GatewayErrorCode.RATE_LIMITED
    assert exc.value.http_status == 429
    assert primary.call_count == 0 and secondary.call_count == 0


async def test_rate_limit_error_carries_retry_guidance(db_path):
    limiter = SlidingWindowRateLimiter(db_path, limit_tokens_per_minute=1_000)
    router, _, _ = build(limiter=limiter)
    await limiter.reserve("tenant-a", 990)  # window nearly full
    with pytest.raises(GatewayError) as exc:
        await router.route("tenant-a", "hi", max_tokens=100)
    payload = exc.value.to_payload()["error"]
    assert payload["limit_tokens_per_minute"] == 1_000
    assert payload["used_tokens"] == 990
    assert "retry_after_seconds" in payload


async def test_overestimate_is_returned_to_the_budget_on_commit(db_path):
    """The reserve/commit true-up is not just bookkeeping, it is throughput.

    A request asking for max_tokens=900 reserves ~900 while in flight, so
    concurrent requests correctly see a nearly-full window. But if the model
    actually returns 7 tokens, commit() replaces the estimate and the other 893
    become available again immediately -- rather than the tenant being billed
    against a ceiling they never reached.
    """
    limiter = SlidingWindowRateLimiter(db_path, limit_tokens_per_minute=1_000)
    router, _, _ = build(limiter=limiter)

    result = await router.route("tenant-a", "hi", max_tokens=900)
    assert (await limiter.usage("tenant-a")).used_tokens == result.completion.total_tokens

    # The freed budget is immediately usable.
    await router.route("tenant-a", "hi", max_tokens=900)


# ---------------------------------------------------------------------------
# Total deadline
# ---------------------------------------------------------------------------
async def test_total_deadline_caps_the_whole_operation(limiter):
    """Without it, primary 3s + secondary 3s = a 6s worst case."""
    router, _, _ = build(
        "timeout", "timeout", limiter=limiter, attempt_timeout_s=1.0, total_deadline_s=0.15
    )
    started = time.perf_counter()
    with pytest.raises(GatewayError):
        await router.route("tenant-a", "hello")
    assert time.perf_counter() - started < 0.5


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
async def test_breaker_opens_after_repeated_primary_failures(limiter):
    breaker = CircuitBreaker(failure_threshold=3, recovery_time_s=60)
    router, primary, _ = build("server_error", limiter=limiter, breaker=breaker)

    for _ in range(3):
        await router.route("tenant-a", "hello")
    assert breaker.state() == "open"

    calls_before = primary.call_count
    result = await router.route("tenant-a", "hello")
    assert result.completion.provider == "secondary"
    assert primary.call_count == calls_before, "an open breaker must skip the primary"
    assert result.attempts[0].outcome == "circuit_open"


async def test_open_breaker_removes_the_timeout_penalty(limiter):
    """The point of the breaker: p99 stops being the timeout value."""
    breaker = CircuitBreaker(failure_threshold=2, recovery_time_s=60)
    router, _, _ = build("timeout", limiter=limiter, attempt_timeout_s=0.2, breaker=breaker)

    for _ in range(2):
        await router.route("tenant-a", "hello")

    started = time.perf_counter()
    await router.route("tenant-a", "hello")
    assert time.perf_counter() - started < 0.1, "still paying the primary's timeout"


async def test_breaker_half_opens_and_recovers(limiter):
    breaker = CircuitBreaker(failure_threshold=2, recovery_time_s=0.05)
    router, primary, _ = build("server_error", limiter=limiter, breaker=breaker)

    for _ in range(2):
        await router.route("tenant-a", "hello")
    assert breaker.state() == "open"

    await asyncio.sleep(0.06)
    assert breaker.state() == "half_open"

    primary.behaviour = "ok"
    result = await router.route("tenant-a", "hello")
    assert result.completion.provider == "primary"
    assert breaker.state() == "closed"


async def test_success_resets_the_failure_count(limiter):
    breaker = CircuitBreaker(failure_threshold=3)
    router, primary, _ = build("server_error", limiter=limiter, breaker=breaker)
    await router.route("tenant-a", "hello")
    await router.route("tenant-a", "hello")
    assert breaker.consecutive_failures == 2

    primary.behaviour = "ok"
    await router.route("tenant-a", "hello")
    assert breaker.consecutive_failures == 0


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
async def test_concurrent_requests_share_one_budget(db_path):
    limiter = SlidingWindowRateLimiter(db_path, limit_tokens_per_minute=1_000)
    router, _, _ = build(limiter=limiter)

    async def attempt():
        try:
            await router.route("tenant-a", "hello", max_tokens=200)
            return "ok"
        except GatewayError as exc:
            return exc.code

    outcomes = await asyncio.gather(*[attempt() for _ in range(20)])
    assert "ok" in outcomes
    assert GatewayErrorCode.RATE_LIMITED in outcomes
    assert (await limiter.usage("tenant-a")).used_tokens <= 1_000


async def test_half_open_probe_that_gets_a_4xx_does_not_wedge_the_breaker(limiter):
    """Regression: a client error on the probe used to strand the breaker.

    record_success/record_failure were the only places clearing
    _probe_in_flight, and the client_error path raised before reaching either.
    The breaker then reported half_open forever while refusing every probe, so
    the primary was skipped for the life of the process even once healthy.
    """
    breaker = CircuitBreaker(failure_threshold=2, recovery_time_s=0.05)
    router, primary, _ = build("server_error", limiter=limiter, breaker=breaker)

    for _ in range(2):
        await router.route("tenant-a", "hello")
    assert breaker.state() == "open"

    await asyncio.sleep(0.06)
    assert breaker.state() == "half_open"

    # The probe hits a 4xx: the provider is up, the request was wrong.
    primary.behaviour = "bad_request"
    with pytest.raises(GatewayError):
        await router.route("tenant-a", "hello")
    assert breaker._probe_in_flight is False
    assert breaker.state() == "closed", "a responding provider must close the breaker"

    primary.behaviour = "ok"
    result = await router.route("tenant-a", "hello")
    assert result.completion.provider == "primary"


async def test_client_error_counts_as_provider_health(limiter):
    """A 4xx must not push the breaker towards open: the provider is fine."""
    breaker = CircuitBreaker(failure_threshold=3)
    router, primary, _ = build("server_error", limiter=limiter, breaker=breaker)
    await router.route("tenant-a", "hello")
    assert breaker.consecutive_failures == 1

    primary.behaviour = "bad_request"
    with pytest.raises(GatewayError):
        await router.route("tenant-a", "hello")
    assert breaker.consecutive_failures == 0


async def test_client_disconnect_during_a_probe_does_not_wedge_the_breaker(limiter):
    """Cancellation says nothing about provider health, but must free the slot.

    _attempt re-raised CancelledError without touching the breaker, so a client
    that disconnected mid-probe left _probe_in_flight True forever: every later
    request saw a half-open breaker that refused every probe, and the primary was
    skipped for the life of the process even after it recovered.
    """
    breaker = CircuitBreaker(failure_threshold=2, recovery_time_s=0.05)
    router, primary, _ = build("server_error", limiter=limiter, breaker=breaker)

    for _ in range(2):
        await router.route("tenant-a", "hello")
    assert breaker.state() == "open"

    await asyncio.sleep(0.06)
    assert breaker.state() == "half_open"

    primary.behaviour = "timeout"
    task = asyncio.create_task(router.route("tenant-a", "hello"))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert breaker._probe_in_flight is False, "the probe slot was never released"

    primary.behaviour = "ok"
    result = await router.route("tenant-a", "hello")
    assert result.completion.provider == "primary"
