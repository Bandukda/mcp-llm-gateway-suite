"""Stream-collection helper shared by the SSE proxy tests.

Kept out of conftest.py for the same reason as task 2's gateway_helpers: with no
package __init__.py files, every tests/conftest.py in this repo is imported
under the same top-level name, so cross-importing from "conftest" breaks
collection when the whole suite runs at once.
"""

import json

import httpx


async def collect_text(client: httpx.AsyncClient, scenario: str) -> tuple[str, list[dict]]:
    """Stream a scenario through the gateway and rebuild the visible text."""
    frames: list[dict] = []
    text = ""
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "mock-model-1", "stream": True, "scenario": scenario},
    ) as response:
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            frame = json.loads(payload)
            frames.append(frame)
            delta = frame.get("choices", [{}])[0].get("delta", {})
            text += delta.get("content") or ""
    return text, frames
