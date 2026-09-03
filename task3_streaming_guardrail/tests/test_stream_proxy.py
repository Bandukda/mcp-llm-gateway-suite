"""End-to-end: SSE in, redacted SSE out, without buffering the response."""

import pytest

from stream_helpers import collect_text


async def test_clean_stream_passes_through_unchanged(gateway):
    text, _ = await collect_text(gateway, "clean")
    assert text == "Hello there, how can I help you today?"


async def test_pii_split_across_frames_is_redacted(gateway):
    text, _ = await collect_text(gateway, "split_pii")
    assert "123-45-6789" not in text
    assert "ada.lovelace@example.com" not in text
    assert "555-010-1234" not in text
    assert text.count("[REDACTED]") == 3
    assert text.startswith("Sure. The account holder is Ada, SSN [REDACTED].")


async def test_no_fragment_of_an_ssn_escapes(gateway):
    """The failure mode that matters: a partial value reaching the client."""
    text, _ = await collect_text(gateway, "split_pii")
    for fragment in ("123-45", "123-4", "45-6789", "6789"):
        assert fragment not in text


async def test_card_split_mid_group_is_redacted(gateway):
    text, _ = await collect_text(gateway, "split_card")
    assert "4111" not in text
    assert "[REDACTED]" in text


async def test_character_by_character_stream(gateway):
    text, _ = await collect_text(gateway, "char_by_char")
    assert text == "Email: [REDACTED] now"


async def test_api_key_is_redacted(gateway):
    text, _ = await collect_text(gateway, "api_key")
    assert "sk-live-" not in text
    assert text == "Your key is [REDACTED] keep it secret."


async def test_false_positives_survive(gateway):
    """A 16-digit order id is not a card; a placeholder SSN is not an SSN."""
    text, _ = await collect_text(gateway, "false_positives")
    assert "1234567890123456" in text
    assert "000-00-0000" in text
    assert "[REDACTED]" not in text


async def test_long_clean_response_is_byte_identical(gateway):
    text, _ = await collect_text(gateway, "long_clean")
    expected = "".join(
        f"Paragraph {i} of ordinary prose with no sensitive data. " for i in range(200)
    )
    assert text == expected


# ---------------------------------------------------------------------------
# Frame-level behaviour
# ---------------------------------------------------------------------------
async def test_non_content_frames_are_preserved(gateway):
    _, frames = await collect_text(gateway, "clean")
    assert frames[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"


async def test_frame_metadata_is_preserved(gateway):
    _, frames = await collect_text(gateway, "clean")
    for frame in frames:
        assert frame["object"] == "chat.completion.chunk"
        assert frame["model"] == "mock-model-1"


async def test_no_empty_content_frames_are_emitted(gateway):
    _, frames = await collect_text(gateway, "split_pii")
    for frame in frames:
        content = frame["choices"][0].get("delta", {}).get("content")
        assert content != ""


async def test_stream_terminates_with_done(gateway):
    lines = []
    async with gateway.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "mock-model-1", "stream": True, "scenario": "split_pii"},
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.strip():
                lines.append(line)
    assert lines[-1].strip() == "data: [DONE]"


async def test_tail_pii_is_flushed_before_done(gateway):
    """A response ending in PII must not lose its final characters."""
    text, _ = await collect_text(gateway, "char_by_char")
    assert text.endswith(" now")


# ---------------------------------------------------------------------------
# Non-streaming path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["split_pii", "char_by_char", "split_card", "api_key"])
async def test_streaming_and_non_streaming_agree(gateway, scenario):
    """Two code paths, one policy. They must not drift.

    An earlier version of this test compared the streamed result against a local
    redact_complete() call and never touched llm_gateway's non-streaming branch
    at all -- so the name promised a guarantee nothing checked. This one issues a
    real stream:false request through the gateway.
    """
    streamed, _ = await collect_text(gateway, scenario)

    response = await gateway.post(
        "/v1/chat/completions",
        json={"model": "mock-model-1", "stream": False, "scenario": scenario},
    )
    assert response.status_code == 200
    non_streamed = response.json()["choices"][0]["message"]["content"]

    assert streamed == non_streamed


async def test_non_streaming_response_is_redacted(gateway):
    response = await gateway.post(
        "/v1/chat/completions",
        json={"model": "mock-model-1", "stream": False, "scenario": "split_pii"},
    )
    content = response.json()["choices"][0]["message"]["content"]
    assert "123-45-6789" not in content
    assert "ada.lovelace@example.com" not in content
    assert content.count("[REDACTED]") == 3


# ---------------------------------------------------------------------------
# Upstream failure on the non-streaming path
# ---------------------------------------------------------------------------
async def _gateway_with(client):
    import httpx

    from llm_gateway import create_app

    app = create_app(upstream_url="http://upstream/v1/chat/completions", client=client)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")


async def test_non_streaming_upstream_timeout_is_a_clean_504():
    import httpx

    class TimingOut:
        async def post(self, *args, **kwargs):
            raise httpx.ConnectTimeout("connect to 10.0.0.7:9011 timed out")

    client = await _gateway_with(TimingOut())
    try:
        response = await client.post(
            "/v1/chat/completions", json={"model": "m", "stream": False}
        )
    finally:
        await client.aclose()
    assert response.status_code == 504
    assert "10.0.0.7" not in response.text


async def test_non_streaming_non_json_upstream_is_a_clean_502():
    import httpx

    class HtmlBody:
        async def post(self, *args, **kwargs):
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b"<html>502 Bad Gateway</html>",
                request=httpx.Request("POST", "http://upstream/v1/chat/completions"),
            )

    client = await _gateway_with(HtmlBody())
    try:
        response = await client.post(
            "/v1/chat/completions", json={"model": "m", "stream": False}
        )
    finally:
        await client.aclose()
    assert response.status_code == 502
    assert "502 Bad Gateway" not in response.text


async def test_non_streaming_upstream_4xx_is_reported_not_parsed():
    import httpx

    class Rejects:
        async def post(self, *args, **kwargs):
            return httpx.Response(
                429,
                json={"error": {"message": "rate limited", "key": "sk-live-OOPS"}},
                request=httpx.Request("POST", "http://upstream/v1/chat/completions"),
            )

    client = await _gateway_with(Rejects())
    try:
        response = await client.post(
            "/v1/chat/completions", json={"model": "m", "stream": False}
        )
    finally:
        await client.aclose()
    assert response.status_code == 502
    assert "sk-live-OOPS" not in response.text


async def test_streaming_upstream_transport_failure_is_a_terminal_error_frame():
    """Mid-stream failures cannot be an HTTP status; they are an SSE error frame."""
    import httpx

    class Failing:
        def stream(self, *args, **kwargs):
            raise httpx.ConnectError("connect to 10.0.0.7:9011 refused")

    client = await _gateway_with(Failing())
    try:
        async with client.stream(
            "POST", "/v1/chat/completions", json={"model": "m", "stream": True}
        ) as response:
            body = "".join([line async for line in response.aiter_lines()])
    finally:
        await client.aclose()
    assert "upstream_error" in body
    assert "[DONE]" in body
    assert "10.0.0.7" not in body


async def test_non_streaming_json_of_the_wrong_shape_is_a_clean_502():
    import httpx

    class WrongShape:
        async def post(self, *args, **kwargs):
            return httpx.Response(
                200,
                json=["not", "an", "object"],
                request=httpx.Request("POST", "http://upstream/v1/chat/completions"),
            )

    client = await _gateway_with(WrongShape())
    try:
        response = await client.post(
            "/v1/chat/completions", json={"model": "m", "stream": False}
        )
    finally:
        await client.aclose()
    assert response.status_code == 502
