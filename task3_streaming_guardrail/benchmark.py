#!/usr/bin/env python3
"""
Measure what the guardrail actually costs.

Runs the gateway and a mock provider on real sockets and reports, for a
deliberately slow upstream:

    * TTFT through the gateway vs straight from the provider
    * the per-frame gap, to show frames are not being bunched
    * peak hold-back, to show memory is bounded

Usage::

    python benchmark.py
"""

from __future__ import annotations

import asyncio
import json
import socket
import statistics
import time
import tracemalloc

import httpx
import uvicorn

from llm_gateway import create_app
from mock_llm import create_mock_llm_app
from redactor import MAX_BUFFER, StreamRedactor

CHUNK_DELAY = 0.03
SCENARIO = "clean"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def measure(client: httpx.AsyncClient, url: str) -> tuple[float, list[float], str]:
    arrivals: list[float] = []
    text = ""
    started = time.perf_counter()
    async with client.stream(
        "POST",
        url,
        json={
            "model": "mock-model-1",
            "stream": True,
            "scenario": SCENARIO,
            "chunk_delay_s": CHUNK_DELAY,
        },
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
    return arrivals[0] if arrivals else float("nan"), arrivals, text


async def main() -> None:
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
    while not (upstream.started and gateway.started):
        await asyncio.sleep(0.01)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            direct_ttft, direct_arrivals, _ = await measure(
                client, f"http://127.0.0.1:{upstream_port}/v1/chat/completions"
            )
            proxied_ttft, proxied_arrivals, _ = await measure(
                client, f"http://127.0.0.1:{gateway_port}/v1/chat/completions"
            )

        def gaps(arrivals: list[float]) -> float:
            deltas = [b - a for a, b in zip(arrivals, arrivals[1:])]
            return statistics.median(deltas) * 1000 if deltas else float("nan")

        buffered = len(direct_arrivals) * CHUNK_DELAY

        print(f"upstream pacing        : {CHUNK_DELAY * 1000:.0f} ms between deltas, "
              f"{len(direct_arrivals)} deltas")
        print(f"full generation time   : {buffered * 1000:.0f} ms")
        print()
        print(f"TTFT direct from mock  : {direct_ttft * 1000:7.1f} ms")
        print(f"TTFT through gateway   : {proxied_ttft * 1000:7.1f} ms")
        print(f"guardrail adds         : {(proxied_ttft - direct_ttft) * 1000:7.1f} ms to TTFT")
        print(f"a buffering proxy would: {buffered * 1000:7.1f} ms")
        print()
        print(f"median inter-frame gap : direct {gaps(direct_arrivals):5.1f} ms  "
              f"gateway {gaps(proxied_arrivals):5.1f} ms")

        # Memory, measured on the redactor directly so the number is about the
        # guardrail and not about uvicorn.
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        redactor = StreamRedactor()
        for i in range(50_000):
            redactor.feed(f"Paragraph {i} of ordinary prose with no sensitive data at all. ")
        redactor.flush()
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()

        # A second run over token-dense text, where the hold-back is non-zero.
        dense = StreamRedactor()
        for i in range(50_000):
            dense.feed(f"id-{i} ada.lovelace@example.com 4111111111111111")
        dense.flush()

        print()
        print("Why TTFT is not identical to the direct call: the first delta")
        print('("Hello") is a trailing token with no word boundary after it, so it')
        print("is held until the next delta proves it is not the start of an email.")
        print(f"That costs one inter-delta gap ({CHUNK_DELAY * 1000:.0f} ms here), not the response length.")
        print()
        print(f"clean text streamed    : {redactor.stats.characters_in / 1e6:.2f} MB")
        print(f"  peak hold-back       : {redactor.stats.max_buffer} chars (bound {MAX_BUFFER})")
        print(f"  peak traced alloc    : {peak / 1024:.1f} KiB")
        print(f"PII-dense streamed     : {dense.stats.characters_in / 1e6:.2f} MB, "
              f"{dense.stats.total_redactions:,} redactions")
        print(f"  peak hold-back       : {dense.stats.max_buffer} chars (bound {MAX_BUFFER})")
    finally:
        upstream.should_exit = gateway.should_exit = True
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
