#!/usr/bin/env bash
# Run every test suite, plus the two standalone proofs.
set -uo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
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
