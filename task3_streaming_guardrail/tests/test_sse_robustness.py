"""Frames the provider sends that this code did not design for.

Every case here either crashed the response mid-stream or forwarded PII
unredacted before an adversarial pass found it. The first one is not exotic: it
is OpenAI's `stream_options={"include_usage": true}` terminator.
"""

import asyncio
import json

import pytest

from llm_gateway import redact_sse_stream
from redactor import StreamRedactor

SSN = "123-45-6789"
CARD = "4111111111111111"


async def drive(frames: list[str]) -> str:
    async def lines():
        for frame in frames:
            yield frame

    return "".join(
        [chunk.decode() async for chunk in redact_sse_stream(lines(), StreamRedactor())]
    )


def data(obj) -> str:
    return "data: " + json.dumps(obj)


def content_frame(text: str, finish=None) -> str:
    return data({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]})


# ---------------------------------------------------------------------------
# Frames that used to abort the stream
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("frames", "why"),
    [
        (
            [content_frame("ssn 123-45-678"), data({"choices": [], "usage": {"total_tokens": 9}}), "data: [DONE]"],
            "OpenAI include_usage terminator: choices:[] with a held-back tail -> IndexError",
        ),
        ([content_frame("hello there"), data({"usage": {"total_tokens": 9}}), "data: [DONE]"], "no choices key -> KeyError"),
        ([content_frame("hello there"), data({"choices": {"0": {}}}), "data: [DONE]"], "choices as a dict -> KeyError: 0"),
        ([content_frame("hello there"), data({"choices": ["oops"]}), "data: [DONE]"], "choices[0] a string -> ValueError"),
        ([content_frame("ssn 123-45-678"), data({"choices": []})], "choices:[] and no [DONE] -> IndexError"),
        ([content_frame("hi there"), "data: [DONE]", content_frame("ssn 123-45-6789")], "frame after [DONE] -> feed() after flush()"),
        (
            [content_frame("a", finish="stop"), content_frame("b", finish="stop"), "data: [DONE]"],
            "a second terminal frame -> feed() after flush()",
        ),
    ],
)
def test_malformed_frames_do_not_abort_the_stream(frames, why):
    out = asyncio.run(drive(frames))
    assert "[DONE]" in out or out, why
    assert SSN not in out, f"leaked while handling: {why}"


# ---------------------------------------------------------------------------
# Redaction bypasses in the frame layer, with no redactor involvement
# ---------------------------------------------------------------------------
def test_every_choice_is_redacted_not_just_the_first():
    """A role delta in choices[0] used to let choices[1] through untouched."""
    out = asyncio.run(
        drive(
            [
                data(
                    {
                        "choices": [
                            {"index": 0, "delta": {"role": "assistant"}},
                            {"index": 1, "delta": {"content": f"ssn {SSN} card {CARD}"}},
                        ]
                    }
                ),
                "data: [DONE]",
            ]
        )
    )
    assert SSN not in out
    assert CARD not in out


@pytest.mark.parametrize(
    "payload",
    [
        [{"type": "text", "text": f"ssn {SSN}"}],  # structured content blocks
        123456789,  # a number
        {"text": f"ssn {SSN}"},  # an object
    ],
)
def test_non_string_content_is_not_forwarded_verbatim(payload):
    """Unexamined content must be dropped, not passed through."""
    out = asyncio.run(
        drive([data({"choices": [{"index": 0, "delta": {"content": payload}}]}), "data: [DONE]"])
    )
    assert SSN not in out
    assert "123456789" not in out


def test_ordinary_stream_still_works():
    """The hardening must not break the normal path."""
    out = asyncio.run(
        drive(
            [
                data({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
                content_frame("Contact ada@example.com"),
                content_frame(" today.", finish="stop"),
                "data: [DONE]",
            ]
        )
    )
    text = "".join(
        json.loads(line[5:].strip())["choices"][0].get("delta", {}).get("content") or ""
        for line in out.splitlines()
        if line.startswith("data:") and "[DONE]" not in line
    )
    assert text == "Contact [REDACTED] today."
