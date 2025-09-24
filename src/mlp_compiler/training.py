"""Helper utilities for training models built from the textual compiler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist

from .compiler import compile_model


@dataclass
class TrainingResult:
    model: keras.Model
    history: Dict[str, list]
    test_loss: float
    test_accuracy: float


DatasetSplit = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[DatasetSplit, DatasetSplit]


def load_mnist(
    *,
    normalize: bool = True,
    flatten: bool = True,
    one_hot: bool = True,
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
) -> Dataset:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if limit_train is not None:
        x_train = x_train[:limit_train]
        y_train = y_train[:limit_train]
    if limit_test is not None:
        x_test = x_test[:limit_test]
        y_test = y_test[:limit_test]

    if normalize:
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
    if flatten:
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))
    if one_hot:
        num_classes = int(y_train.max() + 1)
        y_train = utils.to_categorical(y_train, num_classes)
        y_test = utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


def build_and_train(
    architecture: str,
    *,
    input_dim: int,
    epochs: int = 5,
    batch_size: int = 128,
    validation_split: float = 0.1,
    verbose: int = 1,
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
) -> TrainingResult:
    (x_train, y_train), (x_test, y_test) = load_mnist(
        limit_train=limit_train, limit_test=limit_test
    )

    model = compile_model(architecture, input_dim=input_dim)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(
        x_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    return TrainingResult(
        model=model,
        history=history.history,
        test_loss=float(test_loss),
        test_accuracy=float(test_accuracy),
    )
