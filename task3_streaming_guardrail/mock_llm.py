"""A mock LLM provider that streams OpenAI-style SSE.

Exists so the guardrail can be tested without a network call or an API key, and
so a test can control exactly where a chunk boundary lands -- which is the
interesting variable. ``scenario`` picks a canned stream.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Iterable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Each scenario is a list of content deltas, split exactly where it hurts.
SCENARIOS: dict[str, list[str]] = {
    "clean": ["Hello", " there", ", how", " can I", " help", " you", " today?"],
    # Every PII value is torn across a boundary.
    "split_pii": [
        "Sure. The account holder is Ada, SSN 123",
        "-45",
        "-67",
        "89. Contact her at a",
        "da.lovelace",
        "@example",
        ".com or on 555",
        "-010",
        "-1234.",
    ],
    # A card number written with spaces, split mid-group.
    "split_card": [
        "Charge the card ending 1111: 4111 ",
        "1111 ",
        "1111 ",
        "1111 for the balance.",
    ],
    # PII arriving one character at a time -- the worst case for a naive
    # implementation, since no single chunk contains a complete pattern.
    "char_by_char": list("Email: ada@example.com now"),
    # Long clean prose, to show the hold-back does not grow with length.
    "long_clean": [f"Paragraph {i} of ordinary prose with no sensitive data. " for i in range(200)],
    # Numbers that look like PII but are not: an order id and a non-Luhn card.
    "false_positives": [
        "Order 1234567890123456 shipped. ",
        "Reference 000-00-0000 is a placeholder. ",
        "Build 9999999999999999 passed.",
    ],
    "api_key": ["Your key is sk-live-", "abcdefghij0123456789", " keep it secret."],
}


def sse_frame(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def content_frame(text: str, index: int = 0) -> dict[str, Any]:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "mock-model-1",
        "choices": [{"index": index, "delta": {"content": text}, "finish_reason": None}],
    }


def create_mock_llm_app() -> FastAPI:
    app = FastAPI(title="mock-llm-provider")

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        scenario = body.get("scenario", "split_pii")

        if not body.get("stream", True):
            # Non-streaming mode, so the gateway's non-streaming path is
            # exercised by tests rather than merely existing.
            return JSONResponse(
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "mock-model-1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "".join(SCENARIOS.get(scenario, SCENARIOS["split_pii"])),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            )

        delay_s = float(body.get("chunk_delay_s", 0.0))
        first_token_delay_s = float(body.get("first_token_delay_s", 0.0))
        deltas: Iterable[str] = SCENARIOS.get(scenario, SCENARIOS["split_pii"])

        async def generate():
            if first_token_delay_s:
                await asyncio.sleep(first_token_delay_s)
            # Role frame, as the real API sends.
            yield sse_frame(
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "mock-model-1",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
            )
            for delta in deltas:
                if delay_s:
                    await asyncio.sleep(delay_s)
                yield sse_frame(content_frame(delta))
            yield sse_frame(
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "mock-model-1",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            yield b"data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app


mock_llm_app = create_mock_llm_app()

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(mock_llm_app, host="127.0.0.1", port=9011)
