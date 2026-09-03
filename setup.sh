#!/usr/bin/env bash
# Create a virtualenv and install everything. Run once, from this directory.
set -euo pipefail

cd "$(dirname "$0")"

# --clear rebuilds in place if a .venv is already sitting here.
python3 -m venv --clear .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo "Done. Activate with:  source .venv/bin/activate"
echo "Then run:             ./run_all_tests.sh"
