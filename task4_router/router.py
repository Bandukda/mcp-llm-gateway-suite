"""
Task 4 -- resilient model router.

Rate-limits per tenant, calls the primary provider under a hard deadline, and
fails over to a secondary on 429, timeout or 5xx. Every error the client sees is
a sanitised gateway envelope.

    request -> reserve budget -> primary (3000 ms deadline) -> commit
                                     |
                                     +-- 429 / timeout / 5xx --> secondary
                                     |
                                     +-- 4xx --> return, do NOT fail over

The parts that are easy to get subtly wrong
-------------------------------------------

**A timeout must cancel the work, not just stop waiting for it.**
``asyncio.wait_for`` cancels the wrapped task, but cancellation is cooperative:
the task gets a ``CancelledError`` at its next await point and may still be
unwinding when the fallback starts. Two consequences are handled here:

  * The reserved tokens are released in a ``finally``, so a cancelled attempt
    never permanently consumes budget it did not use.
  * A response arriving *after* the deadline is discarded rather than raced into
    the reply. The deadline is a promise to the caller; honouring it late is
    still breaking it.

**The deadline is per attempt, with an optional overall budget.** Giving the
fallback a fresh 3000 ms can double the worst case to 6 s, which is usually not
what "3 second timeout" meant to whoever wrote the SLA. ``total_deadline_s``
caps the whole operation; the per-attempt deadline is then
``min(per_attempt, time remaining)``.

**Not every failure deserves a retry.** A 400 means the request is wrong, and
sending a wrong request to a second provider produces a second 400, one more bill
and one more second of latency. Only 429, 5xx and transport failures fail over.

**A circuit breaker stops the fallback from becoming the outage.** If the
primary is down, every request pays the full deadline before failing over --
the fallback works, but p99 becomes the timeout value. After
``failure_threshold`` consecutive failures the breaker opens and the primary is
skipped entirely for ``recovery_time_s``, then a single probe decides whether to
close it.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from errors import GatewayError, GatewayErrorCode
from providers import Completion, Provider, ProviderError, estimate_tokens
from rate_limiter import RateLimitExceeded, SlidingWindowRateLimiter

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s", force=True)
log = logging.getLogger("llm.router")

DEFAULT_ATTEMPT_TIMEOUT_S = 3.0


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
@dataclass
class CircuitBreaker:
    """Three states: closed (normal), open (skip), half-open (one probe)."""

    failure_threshold: int = 5
    recovery_time_s: float = 30.0
    consecutive_failures: int = 0
    opened_at: float | None = None
    _probe_in_flight: bool = False

    def state(self, now: float | None = None) -> str:
        now = now if now is not None else time.monotonic()
        if self.opened_at is None:
            return "closed"
        if now - self.opened_at >= self.recovery_time_s:
            return "half_open"
        return "open"

    def allows_request(self, now: float | None = None) -> bool:
        state = self.state(now)
        if state == "closed":
            return True
        if state == "open":
            return False
        # half-open: exactly one probe at a time
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def abandon_probe(self) -> None:
        """Release a half-open probe slot without recording health either way.

        Used when an attempt ends for a reason that says nothing about the
        provider: the caller cancelled, or the overall deadline ran out before
        the provider was dialled.
        """
        self._probe_in_flight = False

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.opened_at = None
        self._probe_in_flight = False

    def record_failure(self, now: float | None = None) -> None:
        self.consecutive_failures += 1
        self._probe_in_flight = False
        if self.consecutive_failures >= self.failure_threshold:
            self.opened_at = now if now is not None else time.monotonic()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
@dataclass
class AttemptRecord:
    provider: str
    outcome: str
    duration_ms: float
    status_code: int | None = None


def _public_attempt_summary(attempts: list[AttemptRecord]) -> dict[str, Any]:
    """What a client may know about a failed route.

    Deliberately not ``[a.__dict__ for a in attempts]``. Those records carry
    provider names -- "primary-openai", "secondary-anthropic" -- and raw upstream
    status codes, which is exactly the vendor topology errors.py says stays in
    the log. A caller needs to know how hard we tried and whether retrying is
    sensible; it does not need our supply chain.
    """
    return {
        "providers_attempted": len(attempts),
        "outcomes": [attempt.outcome for attempt in attempts],
    }


@dataclass
class RoutedResponse:
    completion: Completion
    request_id: str
    attempts: list[AttemptRecord] = field(default_factory=list)

    @property
    def failed_over(self) -> bool:
        return len(self.attempts) > 1


class ModelRouter:
    def __init__(
        self,
        primary: Provider,
        secondary: Provider,
        limiter: SlidingWindowRateLimiter,
        attempt_timeout_s: float = DEFAULT_ATTEMPT_TIMEOUT_S,
        total_deadline_s: float | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.limiter = limiter
        self.attempt_timeout_s = attempt_timeout_s
        self.total_deadline_s = total_deadline_s
        self.breaker = breaker or CircuitBreaker()

    async def _attempt(
        self, provider: Provider, prompt: str, max_tokens: int, timeout_s: float
    ) -> tuple[Completion | None, AttemptRecord]:
        """Run one provider call under a deadline. Never raises for a provider fault."""
        started = time.perf_counter()

        async def elapsed_ms() -> float:
            return round((time.perf_counter() - started) * 1000, 2)

        try:
            completion = await asyncio.wait_for(
                provider.complete(prompt, max_tokens), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            # wait_for has already cancelled the task. Anything the provider
            # produces from here on is discarded: the deadline was a promise.
            return None, AttemptRecord(provider.name, "timeout", await elapsed_ms())
        except ProviderError as exc:
            outcome = "retryable_error" if exc.is_retryable else "client_error"
            return None, AttemptRecord(provider.name, outcome, await elapsed_ms(), exc.status_code)
        except asyncio.CancelledError:
            # A client disconnect mid-probe used to leave _probe_in_flight True
            # forever, so a half-open breaker refused every later probe and the
            # primary was skipped for the life of the process even once healthy
            # -- the "fallback becomes the outage" this breaker exists to
            # prevent. Cancellation says nothing about provider health, so the
            # probe slot is released without recording success or failure.
            if provider is self.primary:
                self.breaker.abandon_probe()
            raise
        except Exception:
            log.exception("unexpected provider failure provider=%s", provider.name)
            return None, AttemptRecord(provider.name, "retryable_error", await elapsed_ms())

        return completion, AttemptRecord(provider.name, "success", await elapsed_ms())

    async def route(
        self, api_key: str, prompt: str, max_tokens: int = 256, request_id: str | None = None
    ) -> RoutedResponse:
        request_id = request_id or uuid.uuid4().hex
        started = time.monotonic()
        attempts: list[AttemptRecord] = []

        # 1. Budget. Reserved up front so concurrent requests see the hold.
        estimated = estimate_tokens(prompt, max_tokens)
        try:
            reservation = await self.limiter.reserve(api_key, estimated)
        except RateLimitExceeded as exc:
            log.info(
                '{"event":"rate_limited","request_id":"%s","used":%d,"limit":%d}',
                request_id, exc.used, exc.limit,
            )
            raise GatewayError(
                code=GatewayErrorCode.RATE_LIMITED,
                request_id=request_id,
                detail=str(exc),
                metadata={
                    "limit_tokens_per_minute": exc.limit,
                    "used_tokens": exc.used,
                    "retry_after_seconds": exc.retry_after_s,
                },
            ) from None

        committed = False
        try:
            # 2. Provider order, respecting the breaker.
            candidates: list[Provider] = []
            if self.breaker.allows_request():
                candidates.append(self.primary)
            else:
                attempts.append(AttemptRecord(self.primary.name, "circuit_open", 0.0))
                log.info('{"event":"circuit_open","request_id":"%s"}', request_id)
            candidates.append(self.secondary)

            for provider in candidates:
                remaining = self._remaining_budget(started)
                if remaining <= 0:
                    attempts.append(AttemptRecord(provider.name, "deadline_exhausted", 0.0))
                    if provider is self.primary:
                        # Never dialled, so no health signal -- but the probe
                        # slot still has to come back.
                        self.breaker.abandon_probe()
                    break

                completion, record = await self._attempt(
                    provider, prompt, max_tokens, min(self.attempt_timeout_s, remaining)
                )
                attempts.append(record)

                if provider is self.primary:
                    if record.outcome in ("success", "client_error"):
                        # A 4xx means the provider is up and answering -- the
                        # request was wrong, not the provider. Counting it as a
                        # breaker success is not just bookkeeping: a half-open
                        # probe that got a 4xx used to return here without
                        # touching the breaker at all, leaving _probe_in_flight
                        # stuck True and the primary skipped for the life of the
                        # process, even after it fully recovered.
                        self.breaker.record_success()
                    elif record.outcome in ("timeout", "retryable_error"):
                        self.breaker.record_failure()

                if completion is not None:
                    await self.limiter.commit(reservation, completion.total_tokens)
                    committed = True
                    log.info(
                        '{"event":"routed","request_id":"%s","provider":"%s",'
                        '"failed_over":%s,"tokens":%d}',
                        request_id, provider.name,
                        "true" if len(attempts) > 1 else "false",
                        completion.total_tokens,
                    )
                    return RoutedResponse(completion, request_id, attempts)

                # A client error is the caller's problem; a second provider will
                # reject it identically. Stop.
                if record.outcome == "client_error":
                    raise GatewayError(
                        code=GatewayErrorCode.UPSTREAM_ERROR,
                        request_id=request_id,
                        detail=f"{provider.name} returned {record.status_code}",
                        metadata=_public_attempt_summary(attempts),
                    )

            # 3. Everything failed.
            timed_out = all(a.outcome in ("timeout", "deadline_exhausted") for a in attempts)
            raise GatewayError(
                code=(
                    GatewayErrorCode.UPSTREAM_TIMEOUT
                    if timed_out
                    else GatewayErrorCode.ALL_PROVIDERS_FAILED
                ),
                request_id=request_id,
                detail=f"all providers failed: {attempts}",
                metadata=_public_attempt_summary(attempts),
            )
        finally:
            # A reservation that never became a billable call is given back, so a
            # provider outage does not also exhaust the tenant's budget.
            if not committed:
                await self.limiter.release(reservation)

    def _remaining_budget(self, started: float) -> float:
        if self.total_deadline_s is None:
            return self.attempt_timeout_s
        return self.total_deadline_s - (time.monotonic() - started)
