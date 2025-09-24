"""Activation functions used in the NumPy-based MLP implementation."""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-z))


def relu(z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit (ReLU) activation."""
    return np.maximum(0.0, z)


def linear(z: np.ndarray) -> np.ndarray:
    """Identity activation."""
    return z


ACTIVATIONS = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": np.tanh,
    "linear": linear,
}


SUPPORTED_ACTIVATIONS = frozenset(ACTIVATIONS.keys())


def get_activation(name: str):
    """Return an activation function by name.

    Parameters
    ----------
    name:
        Name of the activation. The lookup is case-insensitive.

    Returns
    -------
    Callable activation function.

    Raises
    ------
    ValueError
        If the activation name is unknown.
    """

    key = name.lower()
    try:
        return ACTIVATIONS[key]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Activación desconocida: {name}") from exc
