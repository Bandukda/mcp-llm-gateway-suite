"""Run the black-box stdout purity proof as part of the normal test suite."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_guarded_server_keeps_stdout_pure():
    proc = subprocess.run(
        [sys.executable, "verify_stdout_purity.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


def test_unguarded_server_corrupts_stdout():
    """The guard is load-bearing: without it the same server corrupts the stream.

    This asserts a *successful* run, because the unguarded script's job is to
    detect corruption and it exits 0 when it finds it. If it ever came back
    clean, the guard would have stopped mattering and that is the failure worth
    catching.
    """
    proc = subprocess.run(
        [sys.executable, "verify_stdout_purity.py", "--unguarded"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "corrupts the stream, as expected" in proc.stdout
    assert "spoofed" in proc.stdout
