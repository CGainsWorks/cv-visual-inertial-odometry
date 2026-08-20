#!/usr/bin/env bash
# Create a virtual environment, install the dependencies and run the tests.
#
# Nothing here needs the dataset.  `scripts/fetch_euroc.py` downloads that
# separately, and it is about 19 GB.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

if [ ! -d "$VENV" ]; then
    echo "creating $VENV"
    "$PYTHON" -m venv "$VENV"
fi

if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$VENV/bin/activate"
else
    # shellcheck disable=SC1091
    . "$VENV/Scripts/activate"
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "running the tests"
python -m pytest tests/ -q

echo
echo "setup complete.  Next:"
echo "  make demo                 both configurations on a generated sequence"
echo "  python scripts/fetch_euroc.py     download EuRoC (about 19 GB)"
echo "  make experiments          every sequence, both configurations"
