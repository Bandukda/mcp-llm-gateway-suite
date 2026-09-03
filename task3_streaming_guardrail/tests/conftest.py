import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
import pytest_asyncio  # noqa: E402

from llm_gateway import create_app  # noqa: E402
from mock_llm import create_mock_llm_app  # noqa: E402


@pytest_asyncio.fixture
async def gateway():
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_mock_llm_app()), base_url="http://upstream"
    )
    app = create_app(upstream_url="http://upstream/v1/chat/completions", client=upstream)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")
    try:
        yield client
    finally:
        await client.aclose()
        await upstream.aclose()
