"""Pure NumPy implementation of a minimal MLP forward pass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .activations import ACTIVATIONS


def _assert_ndarray(x: np.ndarray, name: str) -> None:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} debe ser np.ndarray, recibido: {type(x)}")


def neuron_forward(x: np.ndarray, w: np.ndarray, b: float, activation: str = "relu") -> np.ndarray:
    """Compute the forward pass of a single neuron.

    Parameters
    ----------
    x:
        Input batch with shape ``(batch, in_features)``.
    w:
        Weight vector with shape ``(in_features,)``.
    b:
        Bias term as scalar.
    activation:
        Name of the activation function to apply.
    """

    _assert_ndarray(x, "x")
    _assert_ndarray(w, "w")
    if activation not in ACTIVATIONS:
        raise ValueError(f"Activación desconocida: {activation}")
    z = x @ w + b
    return ACTIVATIONS[activation](z)


@dataclass
class Layer:
    """Simple fully connected layer using NumPy arrays."""

    in_features: int
    out_features: int
    activation_name: str = "relu"
    weight_scale: float = 0.01

    def __post_init__(self) -> None:
        if self.activation_name not in ACTIVATIONS:
            raise ValueError(f"Activación desconocida: {self.activation_name}")
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("in_features y out_features deben ser positivos")

        if self.activation_name == "relu":
            scale = np.sqrt(2.0 / self.in_features)
        else:
            scale = np.sqrt(1.0 / self.in_features)
        scale *= self.weight_scale / 0.01

        self.W = np.random.randn(self.in_features, self.out_features) * scale
        self.b = np.zeros((self.out_features,), dtype=np.float64)

    def forward(self, X: np.ndarray) -> np.ndarray:
        _assert_ndarray(X, "X")
        if X.shape[1] != self.in_features:
            raise ValueError(
                f"Dimensión de entrada esperada {self.in_features}, recibida {X.shape[1]}"
            )
        Z = X @ self.W + self.b
        return ACTIVATIONS[self.activation_name](Z)


class MLP:
    """Minimalist Multi Layer Perceptron composed of :class:`Layer` objects."""

    def __init__(self, layers: Iterable[Layer]):
        self.layers: List[Layer] = list(layers)
        if not self.layers:
            raise ValueError("Se requiere al menos una capa")

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
