#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
# --list and --dry-run need only Python's standard library. Executing Python
# tests still uses the repository's .venv, as recorded by the runner.
exec python3 scripts/run_tests.py "$@"
