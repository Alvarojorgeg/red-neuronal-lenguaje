"""Command line utility to train a compiled model on MNIST."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from mlp_compiler.training import TrainingResult, build_and_train


DEFAULT_ARCHITECTURE = "Dense(300, relu) -> Dropout(0.2) -> Dense(100, relu) -> Dense(10, softmax)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        type=str,
        default=DEFAULT_ARCHITECTURE,
        help="Arquitectura en el mini-lenguaje textual.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas de entrenamiento.")
    parser.add_argument("--batch-size", type=int, default=128, help="Tamaño de batch.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fracción del set de entrenamiento dedicada a validación.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Limitar la cantidad de ejemplos de entrenamiento (para pruebas rápidas).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Limitar la cantidad de ejemplos de test.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Ruta para guardar las curvas de entrenamiento en PNG.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=784,
        help="Dimensión de entrada para la primera capa Dense.",
    )
    return parser.parse_args()


def _maybe_plot(history: TrainingResult, plot_path: Optional[Path]) -> None:
    if plot_path is None:
        return

    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history.get("val_accuracy", []), label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history.get("val_loss", []), label="val")
    axes[1].set_title("Pérdida")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(plot_path)
    print(f"Curvas de entrenamiento guardadas en {plot_path}")


def main() -> None:
    args = parse_args()

    print("Arquitectura:", args.architecture)
    result = build_and_train(
        args.architecture,
        input_dim=args.input_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        limit_train=args.train_size,
        limit_test=args.test_size,
        verbose=2,
    )

    print(f"\nPrecisión en test: {result.test_accuracy:.4f}")
    print(f"Pérdida en test: {result.test_loss:.4f}")

    _maybe_plot(result, args.plot_path)


if __name__ == "__main__":
    main()
