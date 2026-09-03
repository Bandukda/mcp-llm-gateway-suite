"""HTTP surface for the router: one endpoint, one error shape.

    POST /v1/completions
        Authorization: Bearer <tenant api key>
        {"prompt": "...", "max_tokens": 256}

Run it::

    python app.py            # or: uvicorn app:app --port 9020
    curl -s localhost:9020/v1/completions \
      -H 'Authorization: Bearer tenant-a-key' \
      -H 'Content-Type: application/json' \
      -d '{"prompt":"hello","max_tokens":64}'
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from errors import GatewayError, GatewayErrorCode
from providers import MockProvider
from rate_limiter import SlidingWindowRateLimiter
from router import CircuitBreaker, ModelRouter

DB_PATH = os.environ.get("ROUTER_DB_PATH", "router_usage.db")
LIMIT = int(os.environ.get("ROUTER_TOKENS_PER_MINUTE", "50000"))
ATTEMPT_TIMEOUT_S = float(os.environ.get("ROUTER_ATTEMPT_TIMEOUT_S", "3.0"))


class CompletionRequest(BaseModel):
    model_config = {"extra": "forbid"}

    prompt: str = Field(..., min_length=1, max_length=100_000)
    max_tokens: int = Field(default=256, gt=0, le=8192)


def create_app(router: ModelRouter | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.router is None:
            app.state.router = ModelRouter(
                primary=MockProvider("primary-openai", "ok"),
                secondary=MockProvider("secondary-anthropic", "ok"),
                limiter=SlidingWindowRateLimiter(DB_PATH, limit_tokens_per_minute=LIMIT),
                attempt_timeout_s=ATTEMPT_TIMEOUT_S,
                breaker=CircuitBreaker(),
            )
        yield

    app = FastAPI(title="llm-model-router", version="1.0.0", lifespan=lifespan)
    app.state.router = router

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/usage")
    async def usage(request: Request):
        api_key = _api_key(request)
        snapshot = await app.state.router.limiter.usage(api_key)
        return {
            "used_tokens": snapshot.used_tokens,
            "limit_tokens": snapshot.limit_tokens,
            "remaining_tokens": snapshot.remaining_tokens,
        }

    def _api_key(request: Request) -> str:
        header = request.headers.get("authorization", "")
        parts = header.split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
            raise GatewayError(
                code=GatewayErrorCode.UNAUTHENTICATED,
                request_id=request.headers.get("X-Request-Id") or uuid.uuid4().hex,
            )
        return parts[1].strip()

    @app.post("/v1/completions")
    async def completions(request: Request):
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        api_key = _api_key(request)

        try:
            body = CompletionRequest.model_validate(await request.json())
        except (ValidationError, ValueError, RecursionError) as exc:
            # RecursionError is not a ValueError: deeply nested JSON used to skip
            # this catch and land on the generic 500 handler. Sanitised either
            # way, but a malformed body is a 400.
            raise GatewayError(
                code=GatewayErrorCode.BAD_REQUEST, request_id=request_id, detail=str(exc)
            ) from None

        result = await app.state.router.route(
            api_key=api_key,
            prompt=body.prompt,
            max_tokens=body.max_tokens,
            request_id=request_id,
        )
        return JSONResponse(
            status_code=200,
            headers={"X-Request-Id": request_id},
            content={
                "id": request_id,
                "text": result.completion.text,
                "provider": result.completion.provider,
                "failed_over": result.failed_over,
                "usage": {
                    "prompt_tokens": result.completion.prompt_tokens,
                    "completion_tokens": result.completion.completion_tokens,
                    "total_tokens": result.completion.total_tokens,
                },
            },
        )

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError):
        """The only path from an exception to a response body.

        Centralising it is what makes "no raw upstream detail leaks" a property
        of the system rather than a habit: ``exc.detail`` is logged by whoever
        raised it and is never serialised here.
        """
        headers = {"X-Request-Id": exc.request_id}
        if exc.code == GatewayErrorCode.RATE_LIMITED and exc.metadata:
            headers["Retry-After"] = str(int(exc.metadata.get("retry_after_seconds", 1)) + 1)
        return JSONResponse(status_code=exc.http_status, headers=headers, content=exc.to_payload())

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        """Nothing escapes as a stack trace, including bugs in this gateway."""
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        return JSONResponse(
            status_code=500,
            content=GatewayError(
                code=GatewayErrorCode.INTERNAL, request_id=request_id, detail=repr(exc)
            ).to_payload(),
        )

    return app


app = create_app()

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9020)
