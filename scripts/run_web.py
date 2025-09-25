"""Cross-platform launcher for the Flask demo application."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

CONFLICT_MARKER = "<<<<<<< "


def _check_merge_conflicts(repo_dir: Path) -> None:
    """Fail fast if the working tree still has unresolved conflicts."""

    conflict_files = []
    for path in repo_dir.rglob("*.py"):
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith(CONFLICT_MARKER):
                        conflict_files.append(path)
                        break
        except OSError:
            continue

    if conflict_files:
        rel = ", ".join(str(p.relative_to(repo_dir)) for p in conflict_files)
        raise SystemExit(
            "Se detectaron marcadores de merge sin resolver en: "
            f"{rel}. Ejecuta 'git restore --source=origin/main <archivo>' o usa scripts/doctor.py."
        )


def ensure_dependencies(requirements: Path) -> None:
    """Install Flask (and the rest of the requirements) if it is missing."""

    try:  # pragma: no cover - runtime dependency check
        import flask  # type: ignore  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    print("[run_web] Instalando dependencias listadas en requirements.txt...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements),
    ])


def main() -> None:
    repo_dir: Final[Path] = Path(__file__).resolve().parents[1]
    requirements = repo_dir / "requirements.txt"
    src_dir = repo_dir / "src"

    ensure_dependencies(requirements)
    _check_merge_conflicts(repo_dir)

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    new_pythonpath = str(src_dir)
    if pythonpath:
        new_pythonpath = f"{new_pythonpath}{os.pathsep}{pythonpath}"
    env["PYTHONPATH"] = new_pythonpath
    env.setdefault("FLASK_APP", "web.app")

    print("[run_web] Lanzando la aplicación Flask en http://127.0.0.1:5000")
    subprocess.check_call([
        sys.executable,
        "-m",
        "flask",
        "run",
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
    ], env=env)


if __name__ == "__main__":
    main()
