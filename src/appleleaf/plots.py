from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_history(history, output_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for axis, (metric, title) in zip(
        axes,
        [("loss", "Weighted loss"), ("accuracy", "Accuracy"), ("macro_f1", "Macro F1")],
    ):
        axis.plot(history["epoch"], history[f"train_{metric}"], label="Train")
        axis.plot(history["epoch"], history[f"val_{metric}"], label="Validation")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.3)
    _save(fig, output_path)


def plot_confusion_matrix(labels, predictions, class_names, output_path) -> None:
    matrix = confusion_matrix(
        labels, predictions, labels=list(range(len(class_names)))
    )
    fig, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title("Baseline CNN - Test Confusion Matrix")
    _save(fig, output_path)


def _save(fig, output_path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

