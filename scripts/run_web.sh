#!/usr/bin/env bash
set -euo pipefail

# Determine repository root (directory containing this script's parent)
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"

# Ensure the src/ package is importable when running Flask
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# Configure Flask entry point
export FLASK_APP=web.app

if ! command -v flask >/dev/null 2>&1; then
    echo "[run_web] No se encontró el comando 'flask'. Instalando dependencias..."
    pip install --user -r requirements.txt
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "[run_web] Lanzando la aplicación Flask en http://127.0.0.1:5000"
exec flask run --host=127.0.0.1 --port=5000 "$@"
