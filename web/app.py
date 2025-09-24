"""Flask application to experiment with the MLP compiler through a web UI."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from flask import Flask, render_template, request
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mlp_compiler.training import TrainingResult, build_and_train

app = Flask(__name__)

DEFAULT_ARCHITECTURE = "Dense(300, relu) -> Dropout(0.2) -> Dense(100, relu) -> Dense(10, softmax)"


@dataclass
class FormData:
    architecture: str
    epochs: int
    batch_size: int
    validation_split: float
    train_size: int | None


@dataclass
class TrainingView:
    accuracy_plot: str
    loss_plot: str
    test_accuracy: float
    test_loss: float


def _get_form_data() -> FormData:
    def _int(value: str, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _float(value: str, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    architecture = request.form.get("architecture", DEFAULT_ARCHITECTURE)
    epochs = _int(request.form.get("epochs"), 3)
    batch_size = _int(request.form.get("batch_size"), 128)
    validation_split = _float(request.form.get("validation_split"), 0.1)
    train_size_raw = request.form.get("train_size")
    train_size = _int(train_size_raw, 5000) if train_size_raw else 5000
    if train_size <= 0:
        train_size = None

    return FormData(
        architecture=architecture,
        epochs=max(1, epochs),
        batch_size=max(1, batch_size),
        validation_split=min(max(validation_split, 0.05), 0.4),
        train_size=train_size,
    )


def _plot_history(history: TrainingResult) -> tuple[str, str]:
    def _plot(metric: str, val_metric: str, title: str, ylabel: str) -> str:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(history.history.get(metric, []), label="train")
        if val_metric in history.history:
            ax.plot(history.history.get(val_metric, []), label="val")
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("ascii")

    acc_plot = _plot("accuracy", "val_accuracy", "Evolución de Accuracy", "Accuracy")
    loss_plot = _plot("loss", "val_loss", "Evolución de la Pérdida", "Loss")
    return acc_plot, loss_plot


@app.route("/", methods=["GET", "POST"])
def index():
    error: str | None = None
    view: TrainingView | None = None
    form_data = FormData(
        architecture=DEFAULT_ARCHITECTURE,
        epochs=3,
        batch_size=128,
        validation_split=0.1,
        train_size=5000,
    )

    if request.method == "POST":
        form_data = _get_form_data()
        try:
            result = build_and_train(
                form_data.architecture,
                input_dim=784,
                epochs=form_data.epochs,
                batch_size=form_data.batch_size,
                validation_split=form_data.validation_split,
                limit_train=form_data.train_size,
                limit_test=1000,
                verbose=0,
            )
            acc_plot, loss_plot = _plot_history(result)
            view = TrainingView(
                accuracy_plot=acc_plot,
                loss_plot=loss_plot,
                test_accuracy=result.test_accuracy,
                test_loss=result.test_loss,
            )
        except Exception as exc:  # pragma: no cover - web runtime
            error = str(exc)

    return render_template("index.html", form_data=form_data, view=view, error=error)


if __name__ == "__main__":
    app.run(debug=True)
