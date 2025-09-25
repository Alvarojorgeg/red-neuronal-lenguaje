#!/usr/bin/env bash
set -euo pipefail

# Determine repository root (directory containing this script's parent)
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON:-${PYTHON3:-python3}}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python
fi

exec "$PYTHON_BIN" "$REPO_DIR/scripts/run_web.py" "$@"
