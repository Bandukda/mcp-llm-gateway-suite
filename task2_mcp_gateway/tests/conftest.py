import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402

from downstream import create_downstream_app  # noqa: E402
from mcp_gateway import create_app  # noqa: E402

@pytest.fixture
def downstream_app():
    return create_downstream_app()


@pytest_asyncio.fixture
async def gateway_factory(downstream_app):
    """Build a gateway wired to the in-process downstream over ASGI.

    No sockets: httpx.ASGITransport calls the downstream app directly, so the
    tests are deterministic and there is no port to leak between runs.
    """
    clients: list[httpx.AsyncClient] = []

    def _build(**kwargs):
        downstream_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=downstream_app),
            base_url="http://downstream",
        )
        clients.append(downstream_client)
        app = create_app(
            downstream_url="http://downstream/mcp",
            client=downstream_client,
            **kwargs,
        )
        gateway_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        )
        clients.append(gateway_client)
        return gateway_client

    yield _build
    for client in clients:
        await client.aclose()


@pytest_asyncio.fixture
async def gateway(gateway_factory):
    return gateway_factory()
