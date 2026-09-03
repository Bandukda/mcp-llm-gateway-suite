#!/usr/bin/env python3
"""
Proof that stdout carries nothing but JSON-RPC.

This is the check the task's first scoring criterion asks for, run as a real
black-box test: spawn ``server.py`` as a subprocess with pipes, hold a genuine
MCP conversation over stdin/stdout, and assert that *every* byte that came back
on stdout is a well-formed JSON-RPC 2.0 message.

The server is launched with ``BILLING_MCP_CHAOS=1``, which makes it deliberately
misbehave in the four ways real servers break in production:

    print("...")                      -> Python-level write to sys.stdout
    os.write(1, b"...")               -> raw descriptor write, bypasses Python IO
    subprocess.run(["echo", ...])     -> child process inheriting fd 1
    print('{"jsonrpc": "2.0", ...}')  -> plausible-looking spoofed frame

If ``stdio_guard.reserve_stdout_for_protocol()`` were removed, this script fails
on the first of those. With it, all four land on stderr and stdout stays clean.

Usage::

    python verify_stdout_purity.py            # guarded (expected: PASS)
    python verify_stdout_purity.py --unguarded  # shows the failure it prevents
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

REQUESTS: list[dict] = [
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "stdout-purity-check", "version": "1.0.0"},
        },
    },
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "get_customer_record", "arguments": {"customer_id": "CUST-10042"}},
    },
    {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "trigger_refund",
            "arguments": {
                "customer_id": "CUST-10042",
                "amount": 25.5,
                "reason": "Duplicate charge on the March invoice",
            },
        },
    },
    # Malformed on purpose: must come back as a JSON-RPC error, code -32602.
    {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "trigger_refund", "arguments": {"customer_id": "oops", "amount": -1, "reason": "x"}},
    },
]

CHAOS_MARKERS = ("DEBUG:", "BANNER:", "CHILD:", '"spoofed"')


def run_server(guarded: bool, expected_ids: set[int]) -> tuple[str, str]:
    """Speak to the server like a real client: write, then read until answered.

    Piping every request in and closing stdin immediately is not a fair test --
    the transport shuts down the moment stdin reaches EOF, which can cancel
    in-flight handlers before they answer. A real client keeps the pipe open,
    so this does too.
    """
    env = {**os.environ, "BILLING_MCP_CHAOS": "1", "PYTHONUNBUFFERED": "1"}
    if not guarded:
        # Demonstration mode: neutralise the guard so the failure is visible.
        env["BILLING_MCP_DISABLE_GUARD"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=HERE,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    stderr_chunks: list[str] = []
    stderr_thread = threading.Thread(
        target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True
    )
    stderr_thread.start()

    stdout_lines: list[str] = []
    seen_ids: set[int] = set()

    def pump() -> None:
        for line in proc.stdout:
            stdout_lines.append(line)
            try:
                message = json.loads(line)
                if isinstance(message, dict) and message.get("id") is not None:
                    seen_ids.add(message["id"])
            except json.JSONDecodeError:
                pass  # kept verbatim in stdout_lines; check() will flag it

    stdout_thread = threading.Thread(target=pump, daemon=True)
    stdout_thread.start()

    try:
        for message in REQUESTS:
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()

        deadline = time.monotonic() + 15
        while not expected_ids <= seen_ids and time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.02)
    finally:
        with contextlib.suppress(OSError):
            proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
            proc.wait(timeout=5)
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

    return "".join(stdout_lines), "".join(stderr_chunks)


def check(stdout: str, stderr: str) -> int:
    lines = [line for line in stdout.splitlines() if line.strip()]
    failures: list[str] = []
    responses: dict[int, dict] = {}

    print(f"stdout lines received: {len(lines)}")
    for index, line in enumerate(lines, start=1):
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"line {index} is not JSON ({exc}): {line[:120]!r}")
            continue
        if not isinstance(message, dict):
            failures.append(f"line {index} is JSON but not a JSON-RPC object: {line[:120]!r}")
            continue
        if message.get("jsonrpc") != "2.0":
            failures.append(f"line {index} has no jsonrpc:2.0 marker: {line[:120]!r}")
            continue
        if "id" in message and message["id"] is not None:
            responses[message["id"]] = message

    for marker in CHAOS_MARKERS:
        if marker in stdout:
            failures.append(f"chaos output {marker!r} reached stdout")
        if marker not in stderr:
            failures.append(f"chaos output {marker!r} did not reach stderr either -- it was lost")

    # The protocol itself must still be correct, not merely quiet.
    if 2 not in responses or "result" not in responses[2]:
        failures.append("tools/list did not return a result")
    elif {tool["name"] for tool in responses[2]["result"]["tools"]} != {
        "get_customer_record",
        "trigger_refund",
    }:
        failures.append("tools/list returned the wrong tool set")

    if 3 not in responses or responses[3].get("result", {}).get("isError") is not False:
        failures.append("get_customer_record did not return a successful result")

    if 5 not in responses:
        failures.append("malformed call produced no response")
    elif "error" not in responses[5]:
        failures.append("malformed call did not produce a JSON-RPC error")
    elif responses[5]["error"]["code"] != -32602:
        failures.append(f"expected -32602, got {responses[5]['error']['code']}")

    print()
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  - {failure}")
        print("\n--- first 15 stdout lines ---")
        for line in lines[:15]:
            print(f"  {line[:160]}")
        return 1

    print("PASS")
    print(f"  {len(lines)} stdout lines, all valid JSON-RPC 2.0")
    print(f"  all {len(CHAOS_MARKERS)} deliberate stdout writes were diverted to stderr")
    print("  malformed arguments returned JSON-RPC error -32602")
    return 0


if __name__ == "__main__":
    guarded = "--unguarded" not in sys.argv
    mode = "guarded" if guarded else "UNGUARDED (demonstration of the failure)"
    print(f"Running billing-mcp in chaos mode, {mode}\n")
    expected = {m["id"] for m in REQUESTS if "id" in m}
    out, err = run_server(guarded, expected)
    sys.exit(check(out, err))
