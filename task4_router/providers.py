"""Provider abstraction plus mock providers with scriptable failure modes.

The mocks exist so the failover paths are testable without a network, an API
key, or a real provider being obliging enough to return 429 on demand.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Protocol


class ProviderError(Exception):
    """Any provider-side failure. Carries the upstream status when there is one."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def is_retryable(self) -> bool:
        """429 and 5xx are worth trying elsewhere; 4xx are the caller's fault."""
        if self.status_code is None:
            return True  # transport failure
        return self.status_code == 429 or self.status_code >= 500


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    provider: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Provider(Protocol):
    name: str

    async def complete(self, prompt: str, max_tokens: int) -> Completion: ...


@dataclass
class MockProvider:
    """A provider whose behaviour is scripted.

    ``behaviour`` is one of:
        "ok"        -- succeed after ``latency_s``
        "rate_limit"-- raise ProviderError(429)
        "timeout"   -- sleep far longer than any deadline
        "server_error" -- raise ProviderError(503)
        "bad_request"  -- raise ProviderError(400), which must NOT trigger failover
        "leaky"     -- raise with an internal address in the message
    """

    name: str
    behaviour: str = "ok"
    latency_s: float = 0.01
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete(self, prompt: str, max_tokens: int) -> Completion:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})

        if self.behaviour == "timeout":
            await asyncio.sleep(30)

        await asyncio.sleep(self.latency_s)

        if self.behaviour == "rate_limit":
            raise ProviderError(f"{self.name} rate limit exceeded", status_code=429)
        if self.behaviour == "server_error":
            raise ProviderError(f"{self.name} internal error", status_code=503)
        if self.behaviour == "bad_request":
            raise ProviderError(f"{self.name} rejected the prompt", status_code=400)
        if self.behaviour == "leaky":
            raise ProviderError(
                "connection to 10.4.2.19:8443 refused "
                "(api_key=sk-live-SUPERSECRET0123456789) "
                'File "/opt/gateway/providers.py", line 91, in complete',
                status_code=500,
            )

        completion = f"[{self.name}] response to: {prompt[:40]}"
        return Completion(
            text=completion,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(completion) // 4),
            provider=self.name,
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)


def estimate_tokens(prompt: str, max_tokens: int) -> int:
    """Pre-flight estimate: prompt length plus the caller's own ceiling.

    Roughly four characters per token for English. Being wrong is fine and
    expected -- ``commit()`` replaces the estimate with the real number once the
    response exists. What matters is that the estimate is never *low enough* to
    let a burst of concurrent requests through on a nearly-exhausted budget.
    """
    return max(1, len(prompt) // 4) + max_tokens


def jittered_backoff(attempt: int, base_s: float = 0.05, cap_s: float = 1.0) -> float:
    """Full-jitter backoff. The jitter is the point: synchronised retries from
    many clients are what turns one provider blip into a thundering herd."""
    return random.uniform(0, min(cap_s, base_s * (2**attempt)))
