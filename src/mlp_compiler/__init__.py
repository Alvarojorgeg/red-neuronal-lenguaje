"""Utilities for building simple MLPs and compiling textual architectures."""
from .activations import ACTIVATIONS, SUPPORTED_ACTIVATIONS, get_activation
from .numpy_mlp import Layer, MLP, neuron_forward
from .compiler import compile_model, ArchitectureError

__all__ = [
    "ACTIVATIONS",
    "SUPPORTED_ACTIVATIONS",
    "get_activation",
    "Layer",
    "MLP",
    "neuron_forward",
    "compile_model",
    "ArchitectureError",
]
