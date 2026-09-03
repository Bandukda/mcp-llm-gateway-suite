import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from rate_limiter import SlidingWindowRateLimiter  # noqa: E402


@pytest.fixture
def db_path(tmp_path):
    """On-disk SQLite, as the task requires -- a fresh file per test."""
    return tmp_path / "usage.db"


@pytest.fixture
def limiter(db_path):
    return SlidingWindowRateLimiter(db_path, limit_tokens_per_minute=50_000)
