#!/usr/bin/env bash
# Create a virtualenv and install everything. Run once, from this directory.
#
# Needs Python 3.10 or newer (the MCP SDK requires it). If your default python3
# is older, either install a newer one or point this at an existing one:
#
#     PYTHON=python3.12 ./setup.sh
#
set -euo pipefail
cd "$(dirname "$0")"

MIN="3.10"

version_ok() {
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null
}

# An explicit choice wins; otherwise take the newest suitable interpreter found.
if [ -n "${PYTHON:-}" ]; then
    if ! command -v "$PYTHON" >/dev/null 2>&1; then
        echo "ERROR: PYTHON=$PYTHON is not on your PATH." >&2
        exit 1
    fi
    if ! version_ok "$PYTHON"; then
        echo "ERROR: $PYTHON is $("$PYTHON" -V 2>&1), but Python $MIN or newer is required." >&2
        exit 1
    fi
    INTERPRETER="$PYTHON"
else
    INTERPRETER=""
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && version_ok "$candidate"; then
            INTERPRETER="$candidate"
            break
        fi
    done
fi

if [ -z "$INTERPRETER" ]; then
    CURRENT="$(python3 -V 2>&1 || echo 'not found')"
    cat >&2 <<MSG

ERROR: Python $MIN or newer is required. The MCP SDK does not publish builds for
       anything older, which is why pip reports "No matching distribution found".

Your default python3 is: $CURRENT

Fix it one of these ways:

  1. Install a newer Python with Homebrew:
         brew install python@3.12
         PYTHON=python3.12 ./setup.sh

  2. Or, if you already have a suitable one installed, point at it directly:
         PYTHON=/full/path/to/python3.12 ./setup.sh

  3. No Homebrew? Download an installer from https://www.python.org/downloads/
     then re-run ./setup.sh

MSG
    exit 1
fi

echo "Using $INTERPRETER ($("$INTERPRETER" -V 2>&1))"

# --clear rebuilds in place if a .venv is already sitting here.
"$INTERPRETER" -m venv --clear .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then run:             ./run_all_tests.sh"
