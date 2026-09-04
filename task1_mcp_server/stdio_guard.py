"""
STDIO isolation guard for MCP stdio servers.

Why this file exists
--------------------
An MCP stdio server speaks newline-delimited JSON-RPC over file descriptor 1.
Anything else that reaches fd 1 -- a stray ``print()``, a library that writes a
banner, a progress bar, a C extension writing straight to the descriptor, or a
subprocess that inherits fd 1 -- lands in the middle of the byte stream and
corrupts the frame the client is parsing. The usual symptom is a client that
dies with "Unexpected token" or silently hangs.

Discipline ("never call print") is not a control: it fails the moment a
third-party dependency is added. This module makes the failure mode
*impossible* at the operating-system level:

    1. Duplicate the real fd 1 onto a private high descriptor. That private
       duplicate becomes the only handle allowed to carry protocol frames.
    2. ``dup2(2, 1)`` -- point fd 1 at stderr. Every existing and future writer
       to fd 1 (Python's ``sys.stdout``, C code, forked children) now writes to
       stderr, which is free-form and safe.
    3. Hand the private duplicate to the MCP transport as its output stream.

Because step 2 operates on the descriptor and not on ``sys.stdout``, it also
covers writers that never go through Python's IO stack.

Note on SDK versions: ``mcp`` v2 performs this same descriptor diversion inside
``stdio_server()``. This project pins ``mcp>=1.9,<2`` (see requirements.txt),
where ``stdio_server()`` wraps ``sys.stdout.buffer`` directly and does not
divert fd 1, so the guard is doing real work here. Keeping it also means the
server behaves identically on either SDK line.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import warnings

__all__ = ["reserve_stdout_for_protocol", "configure_stderr_logging"]

_ALREADY_RESERVED = False

# Both streams below live for the life of the process, so the module keeps a
# reference to each. Without it the diverted sys.stdout is collectable the moment
# anything replaces sys.stdout -- which is exactly what pytest's capture plugin
# does after every test -- and its finaliser raises ResourceWarning for a file
# that was never meant to be closed in the first place.
_RETAINED: list[io.TextIOWrapper] = []


def reserve_stdout_for_protocol() -> io.TextIOWrapper:
    """Divert fd 1 to stderr and return the private stream that owns the wire.

    Call this once, as early as possible in ``main()``, before importing or
    constructing anything that might write to stdout.

    Returns:
        A UTF-8 text stream bound to a private duplicate of the original fd 1.
        Pass it to ``stdio_server(stdout=anyio.wrap_file(stream))``.
    """
    global _ALREADY_RESERVED
    if _ALREADY_RESERVED:
        raise RuntimeError("reserve_stdout_for_protocol() must only be called once per process")

    if os.environ.get("BILLING_MCP_DISABLE_GUARD"):
        # Escape hatch used only by verify_stdout_purity.py --unguarded, so the
        # failure this module prevents can be demonstrated rather than asserted.
        _ALREADY_RESERVED = True
        return sys.stdout  # type: ignore[return-value]

    # Flush before diverting. Anything already sitting in Python's stdout buffer
    # was written while fd 1 still pointed at the wire, so it belongs there --
    # and at this point in startup the buffer is empty anyway, since nothing has
    # run yet. Doing it explicitly keeps that assumption honest if the call ever
    # moves later in main().
    try:
        sys.stdout.flush()
    except (ValueError, OSError):  # pragma: no cover - closed stdout
        pass

    # 1. Private duplicate of the real stdout. This is the protocol wire.
    wire_fd = os.dup(1)
    os.set_inheritable(wire_fd, False)  # child processes must not get the wire

    # 2. fd 1 now points at stderr. print(), C writes and inherited fd 1 in
    #    subprocesses all land on stderr from here on.
    os.dup2(2, 1)

    # 3. Rebuild sys.stdout on the diverted descriptor with line buffering, so
    #    interleaved debug output stays readable next to stderr logging.
    try:
        diverted = io.TextIOWrapper(
            os.fdopen(os.dup(1), "wb", buffering=0),
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
        _RETAINED.append(diverted)
        sys.stdout = diverted
    except OSError:  # pragma: no cover - exotic environments
        sys.stdout = sys.stderr

    # Warnings default to stderr already, but make it explicit and non-fatal.
    warnings.simplefilter("default")

    _ALREADY_RESERVED = True

    # buffering=0 + write_through: the transport writes one JSON frame then
    # flushes; no partial frame may ever sit in a buffer if the process dies.
    wire = io.TextIOWrapper(
        os.fdopen(wire_fd, "wb", buffering=0),
        encoding="utf-8",
        newline="\n",
        write_through=True,
    )
    _RETAINED.append(wire)
    return wire


def configure_stderr_logging(level: int = logging.INFO, name: str = "mcp-billing") -> logging.Logger:
    """Send every log record to stderr in a structured, greppable format.

    ``logging.basicConfig()`` defaults to stderr, but only if no handler has been
    installed yet by an imported library. Forcing the configuration removes that
    ordering dependency.
    """
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        force=True,
    )
    return logging.getLogger(name)
