"""
Task 3 -- LLM gateway with a streaming PII guardrail.

Proxies ``POST /v1/chat/completions`` to an upstream provider and rewrites the
SSE stream on the way back, replacing emails, SSNs, card numbers, phone numbers
and API keys with ``[REDACTED]`` -- without ever holding the full response.

    client ──▶ gateway ──▶ provider
                  │
                  └── SSE in, redacted SSE out, O(1) memory

Run it::

    uvicorn mock_llm:mock_llm_app --port 9011
    LLM_UPSTREAM_URL=http://127.0.0.1:9011/v1/chat/completions \
        uvicorn llm_gateway:app --port 9010

    curl -N localhost:9010/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"mock-model-1","stream":true,"scenario":"split_pii"}'

What makes this non-trivial
---------------------------
The redaction itself is in ``redactor.py``; read its docstring for the hold-back
algorithm. This module handles the transport half:

* **Nothing is accumulated.** ``httpx.AsyncClient.stream`` plus an async
  generator into ``StreamingResponse`` means bytes flow through. The only memory
  that scales with anything is the redactor's hold-back buffer, which is capped
  at 800 characters regardless of response length.
* **Redaction spans frames.** A pattern torn across three SSE frames is one
  redaction, so output frames do not map one-to-one onto input frames. A frame
  whose content is entirely held back is *dropped*, not emitted empty: an empty
  delta is legal but wasteful, and some clients render it as a flicker.
* **The tail is flushed before the stream closes.** On ``finish_reason`` or
  ``[DONE]``, whatever is still held back is redacted and emitted as one final
  content frame. Without this, a response ending in PII would lose its last few
  characters -- silent truncation, the worst possible bug here.
* **Non-content frames pass through untouched.** Role frames, usage, and
  anything the provider adds later are forwarded verbatim. A guardrail that
  strips fields it does not recognise breaks clients on the provider's next
  release.
* **A frame that fails to parse is forwarded, not dropped.** Providers add
  fields. Failing open on *shape* while still failing closed on *content* is the
  right trade: the bytes were never buffered, so there is nothing to inspect
  anyway, and dropping unknown frames turns a provider change into an outage.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from redactor import StreamRedactor, redact_complete

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s", force=True)
log = logging.getLogger("llm.gateway")

UPSTREAM_URL = os.environ.get("LLM_UPSTREAM_URL", "http://127.0.0.1:9011/v1/chat/completions")
UPSTREAM_TIMEOUT_S = float(os.environ.get("LLM_UPSTREAM_TIMEOUT_S", "60"))


def sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _choices_of(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """Every well-formed choice in a frame, or an empty list.

    Providers send frames this code did not design for. OpenAI's
    ``stream_options={"include_usage": true}`` final frame carries
    ``"choices": []`` with a usage block; others omit ``choices`` entirely, or
    send it as a dict. Each of those used to raise IndexError / KeyError out of
    the async generator and abort the response mid-stream.
    """
    choices = frame.get("choices")
    if not isinstance(choices, list):
        return []
    return [choice for choice in choices if isinstance(choice, dict)]


def _content_of(choice: dict[str, Any]) -> tuple[str | None, Any]:
    """Return ``(content, finish_reason)`` for one choice."""
    delta = choice.get("delta")
    content = delta.get("content") if isinstance(delta, dict) else None
    return (content if isinstance(content, str) else None), choice.get("finish_reason")


def _finish_reason_of(frame: dict[str, Any]) -> Any:
    for choice in _choices_of(frame):
        if choice.get("finish_reason") is not None:
            return choice["finish_reason"]
    return None


def _with_content(frame: dict[str, Any], text: str) -> dict[str, Any]:
    """A frame carrying replacement content, preserving metadata.

    Never indexes into ``choices``: it may be empty, absent, or the wrong type.
    """
    clone = {key: value for key, value in frame.items() if key != "choices"}
    template = next(iter(_choices_of(frame)), {})
    choice = {key: value for key, value in template.items() if key not in ("delta", "finish_reason")}
    choice.setdefault("index", 0)
    choice["delta"] = {"content": text}
    choice["finish_reason"] = None
    clone["choices"] = [choice]
    return clone


async def redact_sse_stream(
    lines: AsyncIterator[str], redactor: StreamRedactor
) -> AsyncIterator[bytes]:
    """Transform an upstream SSE line stream into a redacted SSE byte stream."""
    last_frame: dict[str, Any] | None = None
    flushed = False

    async for line in lines:
        if not line.strip():
            continue  # frame separators are re-added when each frame is written

        if not line.startswith("data:"):
            # Comments (": keep-alive"), event:/id:/retry: fields -- pass through.
            yield f"{line}\n\n".encode()
            continue

        payload = line[len("data:") :].strip()

        if payload == "[DONE]":
            if not flushed:
                tail = redactor.flush()
                flushed = True
                if tail:
                    yield sse(_with_content(last_frame or {}, tail))
            yield b"data: [DONE]\n\n"
            continue

        try:
            frame = json.loads(payload)
        except json.JSONDecodeError:
            yield f"data: {payload}\n\n".encode()
            continue

        if not isinstance(frame, dict):
            yield f"data: {payload}\n\n".encode()
            continue

        last_frame = frame
        finish_reason = _finish_reason_of(frame)
        choices = _choices_of(frame)

        # Every choice, not just choices[0]. A frame whose first choice is a role
        # delta and whose second carries content used to forward that content
        # unredacted -- and the non-streaming path redacts every choice, so the
        # two disagreed about the same policy.
        emitted_any = False
        rebuilt: list[dict[str, Any]] = []
        for choice in choices:
            content, _ = _content_of(choice)
            if content is None:
                # Includes non-str content (a structured content-block list, or a
                # number), which must not be forwarded verbatim: drop the field
                # rather than pass PII through unexamined.
                delta = choice.get("delta")
                if isinstance(delta, dict) and "content" in delta and delta["content"] is not None:
                    choice = {**choice, "delta": {k: v for k, v in delta.items() if k != "content"}}
                rebuilt.append(choice)
                continue
            if flushed:
                # A content frame after [DONE] or after the terminal frame. The
                # redactor is closed, so this used to raise "feed() after
                # flush()". Redact it in one pass instead of dropping it.
                safe = redact_complete(content)
            else:
                safe = redactor.feed(content)
                if finish_reason is not None:
                    safe += redactor.flush()
                    flushed = True
            if safe:
                emitted_any = True
                rebuilt.append({**choice, "delta": {**(choice.get("delta") or {}), "content": safe}})
            elif finish_reason is not None:
                delta = {k: v for k, v in (choice.get("delta") or {}).items() if k != "content"}
                rebuilt.append({**choice, "delta": delta})

        if not choices:
            # No choices at all: a usage frame, or the include_usage terminator.
            if finish_reason is not None and not flushed:
                tail = redactor.flush()
                flushed = True
                if tail:
                    yield sse(_with_content(frame, tail))
            yield sse(frame)
            continue

        if emitted_any or finish_reason is not None or not any(
            _content_of(choice)[0] for choice in choices
        ):
            yield sse({**frame, "choices": rebuilt})
        # else: everything held back -- emit nothing rather than an empty delta.

    if not flushed:
        # Upstream ended without [DONE] or a finish frame (a dropped connection).
        tail = redactor.flush()
        if tail:
            yield sse(_with_content(last_frame or {}, tail))


def create_app(upstream_url: str = UPSTREAM_URL, client: httpx.AsyncClient | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = app.state.injected_client or httpx.AsyncClient(
            timeout=httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=5.0)
        )
        try:
            yield
        finally:
            if app.state.injected_client is None:
                await app.state.client.aclose()

    app = FastAPI(title="llm-gateway-guardrail", version="1.0.0", lifespan=lifespan)
    app.state.injected_client = client
    app.state.client = client
    app.state.upstream_url = upstream_url

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()

        # Non-streaming requests still get redacted, just in one pass. The error
        # handling here mirrors the streaming branch below: a transport failure
        # or a non-JSON body from the provider must not surface as a 500 with an
        # exception string in it, because httpx puts the upstream host and port
        # in that string.
        if not body.get("stream", True):
            try:
                response = await app.state.client.post(app.state.upstream_url, json=body)
            except httpx.TimeoutException:
                return JSONResponse(
                    status_code=504,
                    content={"error": {"message": "The model provider did not respond in time.",
                                       "type": "upstream_timeout"}},
                )
            except httpx.HTTPError:
                log.exception("upstream transport failure (non-streaming)")
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": "The model provider is unavailable.",
                                       "type": "upstream_error"}},
                )

            if response.status_code >= 400:
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": "Upstream provider error",
                                       "type": "upstream_error",
                                       "status": response.status_code}},
                )

            try:
                data = response.json()
            except ValueError:
                log.error("upstream returned a non-JSON body (non-streaming)")
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": "The model provider returned a malformed response.",
                                       "type": "upstream_error"}},
                )

            # Valid JSON of the wrong shape is still a malformed response; a
            # bare list or string would blow up on .get().
            if not isinstance(data, dict):
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": "The model provider returned a malformed response.",
                                       "type": "upstream_error"}},
                )
            choices = data.get("choices")
            for choice in choices if isinstance(choices, list) else []:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    message["content"] = redact_complete(message["content"])
            return JSONResponse(status_code=response.status_code, content=data)

        redactor = StreamRedactor()

        async def stream() -> AsyncIterator[bytes]:
            # Errors here cannot become an HTTP status: the response has already
            # started. They are reported as a terminal SSE error frame, which is
            # what an OpenAI-compatible client expects mid-stream. The message is
            # fixed text, never the exception -- httpx puts the upstream host and
            # port in ConnectError.
            try:
                async with app.state.client.stream(
                    "POST", app.state.upstream_url, json=body
                ) as upstream:
                    if upstream.status_code >= 400:
                        await upstream.aread()
                        yield sse({"error": {"message": "Upstream provider error",
                                             "type": "upstream_error",
                                             "status": upstream.status_code}})
                        yield b"data: [DONE]\n\n"
                        return
                    async for chunk in redact_sse_stream(upstream.aiter_lines(), redactor):
                        yield chunk
            except httpx.TimeoutException:
                yield sse({"error": {"message": "The model provider did not respond in time.",
                                     "type": "upstream_timeout"}})
                yield b"data: [DONE]\n\n"
            except httpx.HTTPError:
                log.exception("upstream transport failure (streaming)")
                yield sse({"error": {"message": "The model provider is unavailable.",
                                     "type": "upstream_error"}})
                yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9010)
