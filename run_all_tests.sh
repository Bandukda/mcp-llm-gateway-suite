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
echo " Task 1 - THE SAME SERVER WITH THE GUARD DISABLED"
echo ""
echo " The run below is SUPPOSED to fail. It shows what the guard"
echo " prevents. Seeing FAIL here means the check is working."
echo " If this run ever PASSES, that is the real problem."
echo "=============================================================="
if ( cd task1_mcp_server && "$PYTHON" verify_stdout_purity.py --unguarded ); then
  echo
  echo "UNEXPECTED: the unguarded server passed. The guard is not load-bearing."
  status=1
else
  echo
  echo "Good - the unguarded server failed, as it must."
fi

echo
echo "=============================================================="
echo " Task 3 - streaming latency and memory benchmark"
echo "=============================================================="
( cd task3_streaming_guardrail && "$PYTHON" benchmark.py ) || status=1

echo
if [ "$status" -eq 0 ]; then
  echo "ALL CHECKS PASSED"
  echo "(the FAIL above is the deliberate unguarded demo - see the banner)"
else
  echo "SOMETHING FAILED (exit $status)"
fi
exit "$status"
