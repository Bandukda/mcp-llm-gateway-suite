"""Latency proofs over real sockets, plus the memory bound.

These live apart from the other proxy tests for a reason worth stating: httpx's
``ASGITransport`` collects the whole response body before handing it back, so an
in-process test can never distinguish a streaming gateway from a buffering one.
Every *latency* test here therefore runs two real uvicorn servers on ephemeral
ports. If the gateway regresses into accumulating the response, these fail and
the ASGI-based tests do not. The memory test is the exception and says so: it
measures the redactor in process, where the number means something.
"""

import asyncio
import json
import socket
import time

import httpx
import pytest
import pytest_asyncio
import uvicorn

from llm_gateway import create_app
from mock_llm import create_mock_llm_app


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest_asyncio.fixture
async def live_gateway():
    upstream_port, gateway_port = free_port(), free_port()

    upstream = uvicorn.Server(
        uvicorn.Config(create_mock_llm_app(), host="127.0.0.1", port=upstream_port, log_level="error")
    )
    gateway = uvicorn.Server(
        uvicorn.Config(
            create_app(upstream_url=f"http://127.0.0.1:{upstream_port}/v1/chat/completions"),
            host="127.0.0.1",
            port=gateway_port,
            log_level="error",
        )
    )
    tasks = [asyncio.create_task(upstream.serve()), asyncio.create_task(gateway.serve())]

    deadline = time.monotonic() + 10
    while not (upstream.started and gateway.started):
        if time.monotonic() > deadline:  # pragma: no cover
            pytest.fail("uvicorn did not start")
        await asyncio.sleep(0.01)

    client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{gateway_port}", timeout=30.0)
    try:
        yield client
    finally:
        await client.aclose()
        upstream.should_exit = gateway.should_exit = True
        await asyncio.gather(*tasks)


async def stream_arrivals(client: httpx.AsyncClient, **body) -> tuple[list[float], str]:
    """Return (arrival offsets of content frames, reassembled text)."""
    arrivals: list[float] = []
    text = ""
    started = time.perf_counter()
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "mock-model-1", "stream": True, **body},
    ) as response:
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            content = json.loads(payload)["choices"][0].get("delta", {}).get("content")
            if content:
                arrivals.append(time.perf_counter() - started)
                text += content
    return arrivals, text


CHUNK_DELAY = 0.04
DELTAS_IN_CLEAN = 7


async def test_first_token_arrives_before_the_stream_finishes(live_gateway):
    """A buffering gateway would emit nothing until ~280ms; this emits far sooner."""
    arrivals, _ = await stream_arrivals(
        live_gateway, scenario="clean", chunk_delay_s=CHUNK_DELAY
    )
    assert arrivals, "no content frames arrived"
    time_if_buffered = DELTAS_IN_CLEAN * CHUNK_DELAY
    assert arrivals[0] < time_if_buffered * 0.5, (
        f"TTFT {arrivals[0]:.3f}s is close to the full generation time "
        f"({time_if_buffered:.3f}s), which means the response was buffered"
    )


async def test_frames_arrive_progressively(live_gateway):
    arrivals, _ = await stream_arrivals(
        live_gateway, scenario="clean", chunk_delay_s=CHUNK_DELAY
    )
    assert len(arrivals) >= 5
    spread = arrivals[-1] - arrivals[0]
    assert spread > CHUNK_DELAY * 2, "frames were bunched, so the stream was buffered"


async def test_added_latency_per_frame_is_small(live_gateway):
    """The guardrail's own cost, measured against the upstream's pacing."""
    arrivals, _ = await stream_arrivals(
        live_gateway, scenario="clean", chunk_delay_s=CHUNK_DELAY
    )
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    # Each gap should track the upstream delay, not accumulate on top of it.
    assert max(gaps) < CHUNK_DELAY * 3


def test_memory_does_not_grow_with_response_length():
    """Memory is bounded by the cap, not by response length.

    Deliberately *not* taking the live_gateway fixture. This measures the
    redactor directly, so the number is about the guardrail rather than about
    uvicorn's own buffers -- an earlier version took the fixture, spun up two
    servers and then ignored them, which was just slow and misleading.
    """
    import tracemalloc

    from redactor import MAX_HOLDBACK, StreamRedactor

    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]

    redactor = StreamRedactor()
    emitted = 0
    for i in range(20_000):
        emitted += len(redactor.feed(f"Paragraph {i} of ordinary prose with no PII at all. "))
    emitted += len(redactor.flush())

    peak_growth = tracemalloc.get_traced_memory()[1] - baseline
    tracemalloc.stop()

    assert redactor.stats.max_buffer <= MAX_HOLDBACK
    assert emitted == redactor.stats.characters_in
    # ~1MB of text through a redactor whose buffer is bounded by MAX_BUFFER.
    assert peak_growth < 512_000, f"peak growth {peak_growth} bytes suggests accumulation"


async def test_pii_still_redacted_over_the_wire(live_gateway):
    _, text = await stream_arrivals(live_gateway, scenario="split_pii")
    assert "123-45-6789" not in text
    assert "@example.com" not in text
    assert text.count("[REDACTED]") == 3
