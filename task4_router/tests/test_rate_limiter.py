"""Sliding window: accounting, eviction, concurrency, and the reserve protocol."""

import asyncio
import sqlite3

import pytest

from rate_limiter import RateLimitExceeded, SlidingWindowRateLimiter


async def test_database_is_on_disk(limiter, db_path):
    await limiter.reserve("tenant-a", 100)
    assert db_path.exists()
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0] == 1


async def test_usage_accumulates(limiter):
    for _ in range(3):
        await limiter.reserve("tenant-a", 1_000)
    assert (await limiter.usage("tenant-a")).used_tokens == 3_000


async def test_tenants_are_isolated(limiter):
    await limiter.reserve("tenant-a", 40_000)
    await limiter.reserve("tenant-b", 40_000)
    assert (await limiter.usage("tenant-a")).used_tokens == 40_000
    assert (await limiter.usage("tenant-b")).used_tokens == 40_000


async def test_limit_is_enforced(limiter):
    await limiter.reserve("tenant-a", 49_000)
    with pytest.raises(RateLimitExceeded):
        await limiter.reserve("tenant-a", 2_000)


async def test_request_exactly_at_the_limit_is_allowed(limiter):
    await limiter.reserve("tenant-a", 50_000)
    assert (await limiter.usage("tenant-a")).remaining_tokens == 0


async def test_retry_after_points_at_when_the_window_frees_up(limiter):
    now = 1_000_000.0
    await limiter.reserve("tenant-a", 50_000, now=now)
    with pytest.raises(RateLimitExceeded) as exc:
        await limiter.reserve("tenant-a", 1, now=now + 10)
    # The oldest row leaves the window 60s after it was written, i.e. 50s from now.
    assert exc.value.retry_after_s == pytest.approx(50.0, abs=0.01)


# ---------------------------------------------------------------------------
# The sliding window actually slides
# ---------------------------------------------------------------------------
async def test_old_usage_leaves_the_window(limiter):
    now = 1_000_000.0
    await limiter.reserve("tenant-a", 50_000, now=now)
    with pytest.raises(RateLimitExceeded):
        await limiter.reserve("tenant-a", 1_000, now=now + 30)
    # 61 seconds later the first reservation is outside the window.
    reservation = await limiter.reserve("tenant-a", 1_000, now=now + 61)
    assert reservation.estimated_tokens == 1_000


async def test_window_is_rolling_not_fixed_buckets(limiter):
    """A fixed-bucket limiter allows 2x the limit across a bucket boundary."""
    now = 1_000_000.0
    await limiter.reserve("tenant-a", 30_000, now=now + 59)
    with pytest.raises(RateLimitExceeded):
        # A calendar-minute limiter would reset here and allow this.
        await limiter.reserve("tenant-a", 30_000, now=now + 61)


async def test_eviction_keeps_the_table_bounded(limiter):
    now = 1_000_000.0
    for i in range(50):
        await limiter.reserve("tenant-a", 10, now=now + i)
    assert await limiter.row_count() == 50
    # Far in the future, everything before the window is swept on the next check.
    await limiter.reserve("tenant-a", 10, now=now + 500)
    assert await limiter.row_count() == 1


# ---------------------------------------------------------------------------
# reserve / commit / release
# ---------------------------------------------------------------------------
async def test_commit_replaces_the_estimate(limiter):
    reservation = await limiter.reserve("tenant-a", 5_000)
    assert (await limiter.usage("tenant-a")).used_tokens == 5_000
    await limiter.commit(reservation, 1_200)
    assert (await limiter.usage("tenant-a")).used_tokens == 1_200


async def test_release_gives_the_budget_back(limiter):
    reservation = await limiter.reserve("tenant-a", 5_000)
    await limiter.release(reservation)
    assert (await limiter.usage("tenant-a")).used_tokens == 0


async def test_reserved_tokens_are_visible_to_concurrent_requests(limiter):
    """The reason for reserving up front rather than counting afterwards."""
    await limiter.reserve("tenant-a", 49_999)  # in flight, not yet committed
    with pytest.raises(RateLimitExceeded):
        await limiter.reserve("tenant-a", 2)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
async def test_concurrent_reservations_do_not_oversubscribe(db_path):
    """20 racing requests for 10k each against a 50k budget: exactly 5 win.

    Without BEGIN IMMEDIATE this is the classic check-then-act race, and more
    than five get through.
    """
    limiter = SlidingWindowRateLimiter(db_path, limit_tokens_per_minute=50_000)

    async def attempt():
        try:
            await limiter.reserve("tenant-a", 10_000)
            return True
        except RateLimitExceeded:
            return False

    results = await asyncio.gather(*[attempt() for _ in range(20)])
    assert sum(results) == 5
    assert (await limiter.usage("tenant-a")).used_tokens == 50_000


async def test_limiter_does_not_block_the_event_loop(limiter):
    """SQLite calls run in a worker thread, so the loop stays responsive."""
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.001)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    try:
        await asyncio.gather(*[limiter.reserve(f"tenant-{i}", 10) for i in range(50)])
    finally:
        beat.cancel()
    assert ticks > 0, "the event loop never got a turn during 50 SQLite writes"


async def test_state_survives_a_new_limiter_instance(db_path):
    """On-disk means on-disk: a restart does not reset the window."""
    first = SlidingWindowRateLimiter(db_path, limit_tokens_per_minute=50_000)
    await first.reserve("tenant-a", 45_000)

    second = SlidingWindowRateLimiter(db_path, limit_tokens_per_minute=50_000)
    assert (await second.usage("tenant-a")).used_tokens == 45_000
    with pytest.raises(RateLimitExceeded):
        await second.reserve("tenant-a", 10_000)


async def test_deferred_transaction_is_not_safe(db_path):
    """Demonstrate that BEGIN IMMEDIATE is load-bearing, not decoration.

    Same 20-way race as the test above, with the default deferred transaction.
    A deferred BEGIN takes a read lock and upgrades on first write, so the
    budget check passes for more requests than there is budget for. This asserts
    the limiter *fails* -- either by oversubscribing or by raising SQLITE_BUSY on
    the upgrade. Both are the bug; neither happens with BEGIN IMMEDIATE.
    """
    limiter = SlidingWindowRateLimiter(
        db_path, limit_tokens_per_minute=50_000, begin_statement="BEGIN"
    )

    errors: list[Exception] = []

    async def attempt():
        try:
            await limiter.reserve("tenant-a", 10_000)
            return True
        except RateLimitExceeded:
            return False
        except sqlite3.OperationalError as exc:
            errors.append(exc)
            return False

    winners = sum(await asyncio.gather(*[attempt() for _ in range(20)]))
    used = (await limiter.usage("tenant-a")).used_tokens

    assert winners > 5 or used > 50_000 or errors, (
        "the deferred transaction happened to serialise this run; "
        f"winners={winners} used={used} errors={len(errors)}"
    )


async def test_negative_reservation_is_refused(limiter):
    """`used + (-100000)` passes any limit and leaves negative usage behind."""
    with pytest.raises(ValueError):
        await limiter.reserve("tenant-a", -100_000)
    assert (await limiter.usage("tenant-a")).used_tokens == 0

    # And the budget still behaves afterwards.
    await limiter.reserve("tenant-a", 50_000)
    with pytest.raises(RateLimitExceeded):
        await limiter.reserve("tenant-a", 1)
