"""Utilities to transform a textual architecture into a ``tf.keras.Sequential`` model."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Sequence

from tensorflow import keras
from tensorflow.keras import layers

__all__ = ["ArchitectureError", "compile_model"]

_LAYER_REGEX = re.compile(r"(?P<name>[A-Za-z]+)\s*\((?P<args>[^)]*)\)\s*$")
_SUPPORTED_ACTIVATIONS = {"relu", "sigmoid", "tanh", "softmax", "linear"}


class ArchitectureError(ValueError):
    """Raised when the architecture description is invalid."""


@dataclass
class ParsedLayer:
    name: str
    args: Sequence[object]


def _parse_args(arg_str: str) -> List[object]:
    if arg_str.strip() == "":
        return []

    parts = [part.strip() for part in arg_str.split(",") if part.strip()]
    typed: List[object] = []
    for part in parts:
        if re.fullmatch(r"[+-]?\d+", part):
            typed.append(int(part))
        elif re.fullmatch(r"[+-]?\d*\.\d+", part):
            typed.append(float(part))
        else:
            typed.append(part.lower())
    return typed


def _parse_layer(token: str) -> ParsedLayer:
    match = _LAYER_REGEX.match(token)
    if not match:
        raise ArchitectureError(f"Capa inválida: '{token}'")

    name = match.group("name").lower()
    args = _parse_args(match.group("args"))
    return ParsedLayer(name=name, args=args)


def _normalise_tokens(architecture_string: str) -> Iterable[str]:
    for token in architecture_string.split("->"):
        clean = token.strip()
        if clean:
            yield clean


def compile_model(architecture_string: str, input_dim: int | None = None) -> keras.Sequential:
    """Compile a textual architecture description into a ``tf.keras.Sequential`` model."""

    tokens = list(_normalise_tokens(architecture_string))
    if not tokens:
        raise ArchitectureError("La arquitectura no puede estar vacía")

    parsed = [_parse_layer(token) for token in tokens]

    model_layers: List[layers.Layer] = []
    inferred_input_dim: int | None = None

    for layer in parsed:
        if layer.name == "input":
            if len(layer.args) != 1 or not isinstance(layer.args[0], int):
                raise ArchitectureError("Input(dim) requiere un entero")
            inferred_input_dim = layer.args[0]
            continue

        if layer.name == "dense":
            if len(layer.args) < 1:
                raise ArchitectureError(
                    "Dense(units, [activation]) requiere al menos el argumento 'units'."
                )

            units = layer.args[0]
            if not isinstance(units, int):
                raise ArchitectureError("El argumento 'units' debe ser entero")

            activation = None
            if len(layer.args) >= 2:
                activation = layer.args[1]
                if activation not in _SUPPORTED_ACTIVATIONS:
                    raise ArchitectureError(f"Activación no soportada: {activation}")

            if not model_layers:
                first_input = input_dim if input_dim is not None else inferred_input_dim
                if first_input is None:
                    raise ArchitectureError(
                        "La primera capa Dense requiere 'input_dim' o un nodo Input(dim)."
                    )
                model_layers.append(
                    layers.Dense(units, activation=activation, input_shape=(first_input,))
                )
            else:
                model_layers.append(layers.Dense(units, activation=activation))
            continue

        if layer.name == "dropout":
            if len(layer.args) != 1 or not isinstance(layer.args[0], (int, float)):
                raise ArchitectureError("Dropout(rate) requiere un único valor numérico")

            rate = float(layer.args[0])
            if not 0 <= rate < 1:
                raise ArchitectureError("Dropout rate debe estar entre 0 y 1")

            model_layers.append(layers.Dropout(rate))
            continue

        raise ArchitectureError(f"Tipo de capa no soportado: {layer.name}")

    return keras.Sequential(model_layers, name="compiled_from_text")
