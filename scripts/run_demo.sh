#!/usr/bin/env bash
# The one command the README promises: generate a sequence, run visual
# odometry and visual-inertial odometry over it, and print the comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/demo.py "$@"
