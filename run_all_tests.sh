#!/usr/bin/env bash
# Run every test suite, plus the two standalone proofs.
set -uo pipefail

cd "$(dirname "$0")"
# Prefer the venv ./setup.sh built, so this works whether or not it was
# activated first. An absolute path is required: the checks below run inside
# ( cd task1_mcp_server && ... ) subshells, where a relative one would not
# resolve. Falls back to python3 -- never bare `python`, which macOS has not
# shipped since Catalina.
if [ -z "${PYTHON:-}" ] && [ -x "$PWD/.venv/bin/python" ]; then
    PYTHON="$PWD/.venv/bin/python"
fi
PYTHON="${PYTHON:-python3}"

if ! "$PYTHON" -c "import pytest, mcp" 2>/dev/null; then
    echo "ERROR: $PYTHON cannot import pytest and mcp." >&2
    echo "       Run ./setup.sh first, then re-run this script." >&2
    exit 1
fi
status=0

echo "=============================================================="
echo " Full test suite"
echo "=============================================================="
"$PYTHON" -m pytest || status=1

echo
echo "=============================================================="
echo " Task 1 - stdout purity proof (black box, real subprocess)"
echo "=============================================================="
( cd task1_mcp_server && "$PYTHON" verify_stdout_purity.py ) || status=1

echo
echo "=============================================================="
echo " Task 1 - the same server WITHOUT the guard, for contrast"
echo "=============================================================="
( cd task1_mcp_server && "$PYTHON" verify_stdout_purity.py --unguarded ) || status=1

echo
echo "=============================================================="
echo " Task 3 - streaming latency and memory benchmark"
echo "=============================================================="
( cd task3_streaming_guardrail && "$PYTHON" benchmark.py ) || status=1

echo
if [ "$status" -eq 0 ]; then
  echo "ALL CHECKS PASSED"
else
  echo "SOMETHING FAILED (exit $status)"
fi
exit "$status"
