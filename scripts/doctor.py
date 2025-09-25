"""Herramienta de diagnóstico para el proyecto.

Ejecuta varias comprobaciones habituales (versión de Python, dependencias,
conflictos de merge, importaciones) y ofrece mensajes de ayuda en caso de
problemas. Está pensada para facilitar el soporte a usuarios en Windows.
"""
from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
PYTHON_MIN = (3, 9)


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: Optional[str] = None


def _format_status(result: CheckResult) -> str:
    icon = "✅" if result.passed else "❌"
    if result.message:
        return f"{icon} {result.name}: {result.message}"
    return f"{icon} {result.name}"


def check_python_version() -> CheckResult:
    if sys.version_info >= PYTHON_MIN:
        return CheckResult("Versión de Python", True, f"{sys.version.split()[0]}")
    expected = ".".join(map(str, PYTHON_MIN))
    return CheckResult(
        "Versión de Python",
        False,
        f"Detectada {sys.version.split()[0]}. Instala Python {expected} o superior.",
    )


def check_virtualenv() -> CheckResult:
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return CheckResult("Entorno virtual", True, f"Activo: {Path(sys.prefix).name}")
    return CheckResult(
        "Entorno virtual",
        False,
        "No se detecta venv activo. Usa 'python -m venv .venv' y actívalo si es posible.",
    )


def check_requirements_installed() -> CheckResult:
    try:
        import flask  # type: ignore  # noqa: F401
        import tensorflow  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        return CheckResult(
            "Dependencias",
            False,
            "Faltan librerías. Ejecuta 'pip install -r requirements.txt'.",
        )
    return CheckResult("Dependencias", True, "Flask y TensorFlow disponibles")


def _iter_merge_conflicts(paths: Iterable[Path]) -> List[Path]:
    conflict_files: List[Path] = []
    pattern = re.compile(r"^<<<<<<< ")
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if pattern.match(line):
                        conflict_files.append(path)
                        break
        except OSError:
            continue
    return conflict_files


def check_merge_conflicts() -> CheckResult:
    candidate_files = list(REPO_ROOT.rglob("*.py"))
    conflicts = _iter_merge_conflicts(candidate_files)
    if not conflicts:
        return CheckResult("Conflictos de merge", True, "Sin marcadores en archivos .py")

    rel_paths = ", ".join(str(path.relative_to(REPO_ROOT)) for path in conflicts)
    message = (
        "Se detectaron marcadores de merge en: "
        f"{rel_paths}. Ejecuta 'git restore --source=origin/main <archivo>' para repararlo."
    )
    return CheckResult("Conflictos de merge", False, message)


def check_pythonpath() -> CheckResult:
    pythonpath = os.environ.get("PYTHONPATH", "")
    expected = str(SRC_DIR)
    if expected in pythonpath.split(os.pathsep):
        return CheckResult("PYTHONPATH", True, "Incluye el directorio src/")
    return CheckResult(
        "PYTHONPATH",
        False,
        "No incluye src/. Ejecuta scripts/run_web.py o añade la ruta manualmente.",
    )


def check_imports() -> CheckResult:
    sys.path.insert(0, str(SRC_DIR))
    try:
        importlib.import_module("mlp_compiler.compiler")
    except SyntaxError as exc:
        return CheckResult(
            "Importación de mlp_compiler",
            False,
            f"Error de sintaxis al importar: {exc}. Revisa archivos modificados.",
        )
    except ModuleNotFoundError as exc:
        return CheckResult(
            "Importación de mlp_compiler",
            False,
            f"No se encontró el paquete: {exc}. Asegura que la ruta src/ existe.",
        )
    finally:
        try:
            sys.path.remove(str(SRC_DIR))
        except ValueError:
            pass
    return CheckResult("Importación de mlp_compiler", True, "Módulo importable correctamente")


def check_git_status() -> CheckResult:
    if shutil.which("git") is None:
        return CheckResult("Git", False, "Git no está disponible en el PATH")

    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        return CheckResult("Git", False, f"No se pudo ejecutar git status: {exc}")

    if result.stdout.strip():
        return CheckResult(
            "Git",
            False,
            "Hay cambios locales sin confirmar. Usa 'git stash' o 'git restore --staged --worktree .'",
        )
    return CheckResult("Git", True, "Working tree limpio")


def main() -> None:
    checks = [
        check_python_version(),
        check_virtualenv(),
        check_requirements_installed(),
        check_git_status(),
        check_merge_conflicts(),
        check_pythonpath(),
        check_imports(),
    ]

    print("Diagnóstico del proyecto red-neuronal-lenguaje\n")
    for result in checks:
        print(_format_status(result))

    print(
        "\nSi alguna comprobación falla, revisa el mensaje anterior y aplica la sugerencia. "
        "Puedes volver a ejecutar este script tantas veces como necesites."
    )


if __name__ == "__main__":
    main()

