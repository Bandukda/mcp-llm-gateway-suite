"""
Token-aware sliding-window rate limiter, backed by on-disk SQLite.

Why a reserve/commit protocol rather than a counter
---------------------------------------------------
A request's token cost is not known until the response exists. You know the
prompt size going in; you learn the completion size coming out. A limiter that
only counts after the fact lets an unbounded number of concurrent requests start
against the same remaining budget -- fine at low concurrency, a stampede at high
concurrency, because every in-flight request sees the same "plenty left".

So the flow is three steps:

    reservation = limiter.reserve(key, estimated_tokens)   # take the budget now
    ...call the model...
    limiter.commit(reservation, actual_tokens)             # true it up
    # or limiter.release(reservation) if the call never happened

Between reserve and commit the tokens are *held*: they count against the window,
so concurrent requests see them. This is the same shape as an airline seat hold,
and for the same reason.

Why SQLite, and how it is made safe
-----------------------------------
The task specifies on-disk SQLite. The care it needs:

* **WAL mode** so a reader does not block the writer.
* **BEGIN IMMEDIATE** around check-and-insert. The default deferred transaction
  takes a read lock first and upgrades on write, which under concurrency
  produces ``SQLITE_BUSY`` on the *upgrade* -- after the check has already
  passed. That is a check-then-act race that lets two requests both reserve the
  last of the budget. An immediate transaction takes the write lock up front.
  ``test_deferred_transaction_is_not_safe`` substitutes a plain ``BEGIN`` and
  shows it failing, so the claim is measured rather than asserted. Measured over
  five runs of a 20-way race for a 50,000-token budget:

      BEGIN IMMEDIATE : winners=5 used=50000 busy=0   (identical every run)
      BEGIN           : winners=2-5 used=20000-50000 busy=15-18

  Note what the deferred failure actually is. Under WAL the upgrade fails with
  ``SQLITE_BUSY``/``BUSY_SNAPSHOT``, which ``busy_timeout`` cannot retry -- the
  reader's snapshot is already stale -- so roughly 16 of 20 requests die with a
  database error instead of getting a clean allow or deny. It is an availability
  bug before it is an accounting one. Under rollback-journal mode the same race
  shows up as oversubscription instead.

  Note also that the eviction ``DELETE`` deliberately runs *after* the check.
  Putting it first also takes the write lock, which masks the isolation level
  entirely and makes the whole claim untestable.
* **``busy_timeout``** so contention waits rather than failing.
* **Every call runs in a worker thread** via ``asyncio.to_thread``. SQLite calls
  are blocking; making them directly from the event loop stalls every other
  in-flight request in the process. This is the single easiest way to make an
  async gateway mysteriously slow.

Eviction is lazy: rows older than the window are deleted on each check, which
keeps the table proportional to one window of traffic rather than to all of
history. A background vacuum would be the next step at scale.

SQLite is the right answer for one node. For several gateway replicas the same
protocol moves to Redis with a Lua script -- the interface below does not change,
only ``_reserve_sync``.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LIMIT_TOKENS_PER_MINUTE = 50_000
WINDOW_SECONDS = 60.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id           TEXT PRIMARY KEY,
    api_key      TEXT NOT NULL,
    tokens       INTEGER NOT NULL,
    created_at   REAL NOT NULL,
    state        TEXT NOT NULL CHECK (state IN ('reserved', 'committed'))
);
CREATE INDEX IF NOT EXISTS idx_usage_key_time ON token_usage (api_key, created_at);
"""


class RateLimitExceeded(Exception):
    """The tenant is over budget for the current window."""

    def __init__(self, used: int, limit: int, retry_after_s: float) -> None:
        super().__init__(f"{used}/{limit} tokens used in the last {WINDOW_SECONDS:.0f}s")
        self.used = used
        self.limit = limit
        self.retry_after_s = retry_after_s


@dataclass(frozen=True)
class Reservation:
    id: str
    api_key: str
    estimated_tokens: int


@dataclass(frozen=True)
class UsageSnapshot:
    api_key: str
    used_tokens: int
    limit_tokens: int

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.limit_tokens - self.used_tokens)


class SlidingWindowRateLimiter:
    """Token-aware sliding window over a rolling 60-second period."""

    def __init__(
        self,
        db_path: str | Path,
        limit_tokens_per_minute: int = DEFAULT_LIMIT_TOKENS_PER_MINUTE,
        window_seconds: float = WINDOW_SECONDS,
        busy_timeout_ms: int = 5_000,
        begin_statement: str = "BEGIN IMMEDIATE",
    ) -> None:
        # Configurable purely so a test can substitute the default deferred
        # "BEGIN" and demonstrate that the guarantee actually depends on it.
        # Nothing in production should change this.
        self.begin_statement = begin_statement
        self.db_path = str(db_path)
        self.limit = limit_tokens_per_minute
        self.window = window_seconds
        self.busy_timeout_ms = busy_timeout_ms
        self._init_schema()

    # -- connection handling ------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=self.busy_timeout_ms / 1000)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        # Take the write lock at BEGIN, not on upgrade. See the module docstring.
        connection.isolation_level = None
        return connection

    def _init_schema(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(SCHEMA)
        finally:
            connection.close()

    # -- synchronous core ---------------------------------------------------
    def _reserve_sync(self, api_key: str, estimated_tokens: int, now: float) -> Reservation:
        # A negative reservation is arithmetic, not a request: `used + (-100000)`
        # sails under any limit and leaves the window carrying negative usage,
        # which is free headroom for everything after it. Not reachable through
        # the HTTP surface today (estimate_tokens() is always >= 2), which is
        # exactly why it is worth pinning here rather than trusting the caller.
        if estimated_tokens < 0:
            raise ValueError("estimated_tokens must not be negative")
        connection = self._connect()
        try:
            connection.execute(self.begin_statement)
            cutoff = now - self.window
            used = connection.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage "
                "WHERE api_key = ? AND created_at >= ?",
                (api_key, cutoff),
            ).fetchone()[0]

            if used + estimated_tokens > self.limit:
                oldest = connection.execute(
                    "SELECT MIN(created_at) FROM token_usage "
                    "WHERE api_key = ? AND created_at >= ?",
                    (api_key, cutoff),
                ).fetchone()[0]
                connection.execute("ROLLBACK")
                # The window frees up when the oldest row falls out of it.
                retry_after = max(0.0, (oldest + self.window) - now) if oldest else 0.0
                raise RateLimitExceeded(int(used), self.limit, round(retry_after, 3))

            # Eviction runs after the check, not before it. Ordering it first
            # would make the DELETE take the write lock and mask whether
            # BEGIN IMMEDIATE is doing anything -- which is how the earlier
            # version of this file ended up with an untestable claim about it.
            connection.execute("DELETE FROM token_usage WHERE created_at < ?", (cutoff,))

            reservation_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO token_usage (id, api_key, tokens, created_at, state) "
                "VALUES (?, ?, ?, ?, 'reserved')",
                (reservation_id, api_key, estimated_tokens, now),
            )
            connection.execute("COMMIT")
            return Reservation(reservation_id, api_key, estimated_tokens)
        finally:
            connection.close()

    def _commit_sync(self, reservation_id: str, actual_tokens: int) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE token_usage SET tokens = ?, state = 'committed' WHERE id = ?",
                (actual_tokens, reservation_id),
            )
            connection.execute("COMMIT")
        finally:
            connection.close()

    def _release_sync(self, reservation_id: str) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM token_usage WHERE id = ?", (reservation_id,))
            connection.execute("COMMIT")
        finally:
            connection.close()

    def _usage_sync(self, api_key: str, now: float) -> UsageSnapshot:
        connection = self._connect()
        try:
            used = connection.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage "
                "WHERE api_key = ? AND created_at >= ?",
                (api_key, now - self.window),
            ).fetchone()[0]
            return UsageSnapshot(api_key, int(used), self.limit)
        finally:
            connection.close()

    def _row_count_sync(self) -> int:
        connection = self._connect()
        try:
            return connection.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0]
        finally:
            connection.close()

    # -- async surface ------------------------------------------------------
    async def reserve(
        self, api_key: str, estimated_tokens: int, now: float | None = None
    ) -> Reservation:
        """Hold ``estimated_tokens`` against the window, or raise RateLimitExceeded."""
        return await asyncio.to_thread(
            self._reserve_sync, api_key, estimated_tokens, now if now is not None else time.time()
        )

    async def commit(self, reservation: Reservation, actual_tokens: int) -> None:
        """Replace the estimate with what the call actually cost."""
        await asyncio.to_thread(self._commit_sync, reservation.id, actual_tokens)

    async def release(self, reservation: Reservation) -> None:
        """Give the hold back. Used when the call never produced a billable result."""
        await asyncio.to_thread(self._release_sync, reservation.id)

    async def usage(self, api_key: str, now: float | None = None) -> UsageSnapshot:
        return await asyncio.to_thread(
            self._usage_sync, api_key, now if now is not None else time.time()
        )

    async def row_count(self) -> int:
        """Rows currently stored. Used to prove eviction actually evicts."""
        return await asyncio.to_thread(self._row_count_sync)
