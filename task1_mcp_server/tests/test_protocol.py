"""Protocol tests: drive the server through a real MCP ClientSession.

These assert the thing the task actually scores -- that a bad argument comes
back as a JSON-RPC *error* with code -32602, and that a legitimate-but-failing
call comes back as a *result* with isError, not as a protocol error.

Note on structure: the client session is opened inside each test rather than in
an async fixture. anyio cancel scopes must be entered and exited in the same
task, and pytest-asyncio runs async-generator fixtures in a separate task from
the test body, which trips that rule. A synchronous fixture handing back a
factory keeps everything in one task.
"""

import json

import mcp.types as types
import pytest
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session

from server import build_server
from store import seeded_store


@pytest.fixture
def store():
    return seeded_store()


@pytest.fixture
def session(store):
    def _open():
        return create_connected_server_and_client_session(build_server(store))

    return _open


# --------------------------------------------------------------------------
# tools/list
# --------------------------------------------------------------------------
async def test_lists_both_tools(session):
    async with session() as client:
        result = await client.list_tools()
    assert {tool.name for tool in result.tools} == {"get_customer_record", "trigger_refund"}


async def test_advertised_schema_is_strict(session):
    async with session() as client:
        result = await client.list_tools()
    schemas = {tool.name: tool.inputSchema for tool in result.tools}

    refund = schemas["trigger_refund"]
    assert refund["additionalProperties"] is False
    assert refund["properties"]["customer_id"]["pattern"] == r"^CUST-[0-9]{5}$"
    assert refund["properties"]["amount"]["exclusiveMinimum"] == 0
    assert refund["properties"]["reason"]["minLength"] == 10
    assert set(refund["required"]) == {"customer_id", "amount", "reason"}


# --------------------------------------------------------------------------
# Happy paths
# --------------------------------------------------------------------------
async def test_get_customer_record_success(session):
    async with session() as client:
        result = await client.call_tool("get_customer_record", {"customer_id": "CUST-10042"})
    assert result.isError is False
    payload = json.loads(result.content[0].text)
    assert payload["ok"] is True
    assert payload["customer"]["name"] == "Ada Lovelace"


async def test_trigger_refund_success(session, store):
    async with session() as client:
        result = await client.call_tool(
            "trigger_refund",
            {
                "customer_id": "CUST-10042",
                "amount": 49.99,
                "reason": "Duplicate charge on the March invoice",
            },
        )
    assert result.isError is False
    payload = json.loads(result.content[0].text)
    assert payload["refund"]["amount_usd"] == 49.99
    assert payload["refund"]["status"] == "pending_settlement"
    assert store.customers["CUST-10042"].refundable_balance_usd == pytest.approx(1150.01)


async def test_refund_idempotency_key_prevents_double_refund(session, store):
    args = {
        "customer_id": "CUST-10042",
        "amount": 100.0,
        "reason": "Agent retried after a network timeout",
        "idempotency_key": "retry-key-0001",
    }
    async with session() as client:
        first = json.loads((await client.call_tool("trigger_refund", args)).content[0].text)
        second = json.loads((await client.call_tool("trigger_refund", args)).content[0].text)

    assert first["replayed"] is False
    assert second["replayed"] is True
    assert first["refund"]["refund_id"] == second["refund"]["refund_id"]
    assert len(store.refunds) == 1
    assert store.customers["CUST-10042"].refundable_balance_usd == pytest.approx(1100.00)


# --------------------------------------------------------------------------
# Protocol errors: JSON-RPC -32602
# --------------------------------------------------------------------------
async def test_malformed_customer_id_is_invalid_params(session):
    async with session() as client:
        with pytest.raises(McpError) as exc:
            await client.call_tool("get_customer_record", {"customer_id": "12345"})
    assert exc.value.error.code == types.INVALID_PARAMS
    assert {e["field"] for e in exc.value.error.data["validation_errors"]} == {"customer_id"}


async def test_negative_amount_is_invalid_params(session):
    async with session() as client:
        with pytest.raises(McpError) as exc:
            await client.call_tool(
                "trigger_refund",
                {"customer_id": "CUST-10042", "amount": -5.0, "reason": "Refund the duplicate charge"},
            )
    assert exc.value.error.code == types.INVALID_PARAMS


async def test_short_reason_is_invalid_params(session):
    async with session() as client:
        with pytest.raises(McpError) as exc:
            await client.call_tool(
                "trigger_refund", {"customer_id": "CUST-10042", "amount": 5.0, "reason": "nope"}
            )
    assert exc.value.error.code == types.INVALID_PARAMS


async def test_unknown_field_is_invalid_params(session):
    async with session() as client:
        with pytest.raises(McpError) as exc:
            await client.call_tool(
                "trigger_refund",
                {
                    "customer_id": "CUST-10042",
                    "amount": 5.0,
                    "reason": "Refund the duplicate charge",
                    "currency": "GBP",
                },
            )
    assert exc.value.error.code == types.INVALID_PARAMS


async def test_unknown_tool_is_invalid_params(session):
    async with session() as client:
        with pytest.raises(McpError) as exc:
            await client.call_tool("delete_everything", {})
    assert exc.value.error.code == types.INVALID_PARAMS
    assert exc.value.error.data["available_tools"] == ["get_customer_record", "trigger_refund"]


async def test_multiple_bad_fields_are_all_reported(session):
    async with session() as client:
        with pytest.raises(McpError) as exc:
            await client.call_tool(
                "trigger_refund", {"customer_id": "nope", "amount": -1, "reason": "x"}
            )
    assert {e["field"] for e in exc.value.error.data["validation_errors"]} == {
        "customer_id",
        "amount",
        "reason",
    }


async def test_error_data_does_not_echo_input(session):
    """The validation payload must not reflect the caller's raw values back."""
    secret = "sk-live-THIS-SHOULD-NOT-COME-BACK"
    async with session() as client:
        with pytest.raises(McpError) as exc:
            await client.call_tool("get_customer_record", {"customer_id": secret})
    assert secret not in json.dumps(exc.value.error.model_dump())


# --------------------------------------------------------------------------
# Business errors: successful JSON-RPC response carrying isError
# --------------------------------------------------------------------------
async def test_unknown_customer_is_a_tool_error_not_a_protocol_error(session):
    async with session() as client:
        result = await client.call_tool("get_customer_record", {"customer_id": "CUST-99999"})
    assert result.isError is True
    assert json.loads(result.content[0].text)["code"] == "customer_not_found"


async def test_refund_over_balance_is_a_tool_error(session):
    async with session() as client:
        result = await client.call_tool(
            "trigger_refund",
            {"customer_id": "CUST-20099", "amount": 500.0, "reason": "Requested by the customer"},
        )
    assert result.isError is True
    assert json.loads(result.content[0].text)["code"] == "refund_not_permitted"


async def test_refund_for_suspended_customer_is_a_tool_error(session):
    async with session() as client:
        result = await client.call_tool(
            "trigger_refund",
            {"customer_id": "CUST-30007", "amount": 1.0, "reason": "Requested by the customer"},
        )
    assert result.isError is True
    assert json.loads(result.content[0].text)["code"] == "refund_not_permitted"


async def test_idempotency_key_reuse_with_different_arguments_is_refused(session, store):
    """An idempotency key bound to nothing is a promise about nothing.

    Reusing a key with different arguments returned the *original* refund and
    reported ok/replayed. So a call asking to refund CUST-20099 $5 was answered
    with CUST-10042's $1000 refund, the $5 refund never happened, and the model
    was told it had succeeded and not to re-announce it.
    """
    first = {
        "customer_id": "CUST-10042",
        "amount": 100.0,
        "reason": "Duplicate charge on the March invoice",
        "idempotency_key": "shared-key-0001",
    }
    second = {
        "customer_id": "CUST-20099",
        "amount": 5.0,
        "reason": "Completely different refund request",
        "idempotency_key": "shared-key-0001",
    }
    async with session() as client:
        ok = json.loads((await client.call_tool("trigger_refund", first)).content[0].text)
        clash = await client.call_tool("trigger_refund", second)

    assert ok["replayed"] is False
    assert clash.isError is True
    payload = json.loads(clash.content[0].text)
    assert payload["code"] == "idempotency_key_conflict"

    # Exactly one refund exists, and the other customer was left alone.
    assert len(store.refunds) == 1
    assert store.customers["CUST-20099"].refundable_balance_usd == pytest.approx(150.00)


async def test_identical_replay_still_returns_the_original(session, store):
    """The legitimate case must keep working: same key, same arguments."""
    args = {
        "customer_id": "CUST-10042",
        "amount": 100.0,
        "reason": "Agent retried after a network timeout",
        "idempotency_key": "retry-key-0002",
    }
    async with session() as client:
        first = json.loads((await client.call_tool("trigger_refund", args)).content[0].text)
        second = json.loads((await client.call_tool("trigger_refund", args)).content[0].text)

    assert second["replayed"] is True
    assert first["refund"]["refund_id"] == second["refund"]["refund_id"]
    assert len(store.refunds) == 1
