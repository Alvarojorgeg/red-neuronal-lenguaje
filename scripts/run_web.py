#!/usr/bin/env python3
"""Launch the Flask web interface with cross-platform support."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path


def _ensure_dependencies(requirements_path: Path) -> None:
    try:
        import flask  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        print("[run_web] Flask no se encontró en el entorno actual. Instalando dependencias...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
    else:
        return


def _launch_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:  # pragma: no cover - apertura del navegador es mejor-effort
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lanza la interfaz web de entrenamiento")
    parser.add_argument("--host", default="127.0.0.1", help="Host donde exponer la aplicación Flask")
    parser.add_argument("--port", type=int, default=5000, help="Puerto para la aplicación Flask")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="No abrir el navegador automáticamente tras iniciar el servidor",
    )
    args = parser.parse_args(argv)

    repo_dir = Path(__file__).resolve().parent.parent
    requirements_path = repo_dir / "requirements.txt"

    _ensure_dependencies(requirements_path)

    src_path = repo_dir / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    os.environ.setdefault("PYTHONPATH", str(src_path))

    # Cargar la aplicación Flask y lanzar el servidor
    from web.app import app  # import dentro de la función para respetar PYTHONPATH

    url = f"http://{args.host}:{args.port}"
    print(f"[run_web] Lanzando la aplicación Flask en {url}")

    if not args.no_browser:
        _launch_browser(url)

    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
